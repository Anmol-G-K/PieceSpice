"""Tier-4 loss LUT: measured golden-DPT switching energy and Curve-3/7 devices.

The LUT is a data container plus strictly in-grid interpolation.  Switching
energies come only from golden clamped-inductive DPT sweeps (recorded at the
identical measured condition, otherwise measured live by loss_lut_bench);
conduction resistance comes from the extracted Curve-3 table at the recorded
Curve-3 extraction gate drive and the gate threshold comes from Curve-7.
Solver-stabilizing device inputs are measured too: ``coss_f`` is the vendor
Curve-4 output capacitance at the recorded reference Vds (a linear C that only
prevents singular/no-cap solver steps, not a voltage-dependent waveform model),
and ``r_off_ohms`` is the off-state resistance from a Vgs=0 vendor DC sweep.
A failed Roff measurement is NaN with a source gap; the emitter then falls back
to a project-decision Roff with a loud comment and JSON disclosure.
Unmeasured grid nodes are NaN with a source gap -- never an invented datasheet
point, a filled-in number, or an extrapolated value.

There is no extrapolation path: an out-of-grid query is always NaN.  The
recorded config-derived grids have 1-3 finite measured nodes and no complete
2x2 interpolation cell; ``interpolation_capability()`` reports each grid's
true status (constant_node / single_node / grid_interpolation) and
``grid_provenance()`` states that limitation without assuming a fixed shape.
The capability is published in ``source["interpolation"]`` and ``to_dict()``.

PLECS (P-TC) documents interpolation of E_on/E_off(v, i, T) tables and warns
that dissipated thermal power is not the electrical power consumed by the
device.  The emitted Tier-4 switch has no loss element or thermal network:
losses are computed offline and injected thermally.  Efficiency therefore
requires an explicitly supplied useful output power; Vbus*Iload is input
power and is never used as p_out.  The DPT waveform gate is retired.
"""

from __future__ import annotations

import copy
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from config.yaml_io import load_input, load_output

SCHEMA = "piecespice.loss_lut/1.0"
NAN = float("nan")

# PLECS P-TC, notes under "Conduction Losses" and "Switching Losses": the
# dissipated thermal power does not correspond to the electrical power that is
# consumed by the device; take this into account when estimating efficiency.
# Carried into every efficiency result and into LUT provenance so the Tier-4
# offline gate never presents thermal losses as electrical input power.
THERMAL_LOSS_NOTE = (
    "PLECS P-TC: dissipated thermal power is not the electrical power consumed "
    "by the device; take this into account when estimating efficiency. The "
    "emitted Tier-4 switch has no loss element and no thermal RC network: "
    "E_on/E_off/r_on are evaluated offline, so efficiency needs an explicit "
    "useful output-power definition (Vbus*Iload is input power, not output)."
)


def _node_key(vbus_v: float, current_a: float) -> str:
    """Stable key for one measured grid node in source provenance."""
    return f"{float(vbus_v)!r}|{float(current_a)!r}"


def _finite_or_nan(value: Any) -> float:
    if value is None:
        return NAN
    f = float(value)
    return f if math.isfinite(f) else NAN


def _matrix(value: Any) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(_finite_or_nan(x) for x in row) for row in value)


def _axis_position(points: tuple[float, ...],
                   value: float) -> tuple[int, int, float, str] | None:
    """Return (low index, high index, weight, mode); None when unavailable.

    Strictly in-grid: no first/last-pair extrapolation is implemented, so a
    value outside ``points`` (or a non-finite value) always returns None.  A
    single-point axis is exact only at that point.
    """
    if not math.isfinite(float(value)) or not points:
        return None
    v = float(value)
    if len(points) == 1:
        return (0, 0, 0.0, "exact") if v == points[0] else None
    if v < points[0] or v > points[-1]:
        return None
    for k in range(len(points) - 1):
        if v == points[k]:
            return (k, k, 0.0, "exact")
        if points[k] < v < points[k + 1]:
            w = (v - points[k]) / (points[k + 1] - points[k])
            return (k, k + 1, w, "interpolate")
    return (len(points) - 1, len(points) - 1, 0.0, "exact")


def _grid_value(
    matrix: tuple[tuple[float, ...], ...],
    vi: int, vj: int, wv: float, ci: int, cj: int, wc: float,
) -> float:
    """Bilinear (or degenerate linear) value; NaN if any corner is unmeasured."""
    corners = (float(matrix[vi][ci]), float(matrix[vi][cj]),
               float(matrix[vj][ci]), float(matrix[vj][cj]))
    if not all(math.isfinite(x) for x in corners):
        return NAN
    if vi == vj and ci == cj:
        return corners[0]
    if vi == vj:
        return (1.0 - wc) * corners[0] + wc * corners[1]
    if ci == cj:
        return (1.0 - wv) * corners[0] + wv * corners[2]
    return ((1.0 - wv) * ((1.0 - wc) * corners[0] + wc * corners[1])
            + wv * ((1.0 - wc) * corners[2] + wc * corners[3]))


def _linear_value(values: tuple[float, ...], axis: tuple[float, ...],
                  value: float) -> float:
    pos = _axis_position(axis, float(value))
    if pos is None:
        return NAN
    i, j, w, _mode = pos
    a, b = values[i], values[j]
    if not (math.isfinite(a) and math.isfinite(b)):
        return NAN
    return (1.0 - w) * a + w * b


def _measured_node_count(e_on: Any, e_off: Any) -> int:
    """Count nodes where both switching energies are finite measurements."""
    return sum(
        1
        for row_on, row_off in zip(e_on, e_off)
        for on, off in zip(row_on, row_off)
        if math.isfinite(float(on)) and math.isfinite(float(off))
    )


def _complete_cell_count(e_on: Any, e_off: Any) -> int:
    """Count 2x2 cells whose four corners carry both measured energies."""
    rows_on, rows_off = list(e_on), list(e_off)
    cells = 0
    for i in range(len(rows_on) - 1):
        for j in range(len(rows_on[i]) - 1):
            corners = ((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1))
            if all(
                math.isfinite(float(rows_on[a][b]))
                and math.isfinite(float(rows_off[a][b]))
                for a, b in corners
            ):
                cells += 1
    return cells


def _interpolation_capability(e_on: Any, e_off: Any,
                              n_vbus: int, n_current: int) -> dict[str, Any]:
    """Report what the measured grid can actually support (D7/M4)."""
    finite = _measured_node_count(e_on, e_off)
    if n_vbus == 1 and n_current == 1:
        status = "constant_node"
        rule = ("Single 1x1 measured grid: no interpolation is defined; the "
                "only node is a constant measured value and every off-node "
                "query is NaN.")
    elif finite <= 1:
        status = "single_node"
        rule = ("Only one finite measured node: bilinear interpolation is "
                "impossible; every off-node query is NaN.")
    else:
        status = "grid_interpolation"
        rule = ("Bilinear interpolation only where all four corners are "
                "finite; no extrapolation; otherwise NaN.")
    return {
        "status": status,
        "finite_nodes": finite,
        "complete_cells": _complete_cell_count(e_on, e_off),
        "vbus_points": n_vbus,
        "current_points": n_current,
        "rule": rule,
    }


def interpolation_provenance_text(cap: Mapping[str, Any]) -> str:
    """Truthful grid provenance derived from interpolation_capability().

    No fixed node count or grid shape is assumed: the sentence follows the
    measured capability dict, so a grid with several finite nodes but no full
    2x2 cell is never described as a single-node grid.
    """
    finite = cap.get("finite_nodes")
    complete = cap.get("complete_cells")
    nv, nc = cap.get("vbus_points"), cap.get("current_points")
    status = cap.get("status")
    shape = (f"{nv}x{nc}" if nv is not None and nc is not None
             else "config-derived")
    nodes = (f"{finite} finite measured node(s)" if finite is not None
             else "finite measured nodes")
    if complete == 0:
        detail = ("no 2x2 interpolation cell has all four corners finite, so "
                  "off-node queries are NaN / insufficient_measured_nodes and "
                  "no bilinear interpolation is currently exercised")
    elif complete is None:
        detail = ("bilinear interpolation requires all four measured corners "
                  "finite; off-node queries without them are NaN")
    else:
        detail = (f"{complete} complete 2x2 cell(s) allow bilinear "
                  "interpolation; other off-node queries are NaN")
    return (f"The current {shape} grid has {nodes} (capability '{status}'); "
            f"{detail}.")


def _data_limitations(temperature_c: float, e_on: Any, e_off: Any,
                      n_vbus: int, n_current: int) -> list[dict[str, Any]]:
    """Measured-data limitations (not project decisions) for JSON/meta (M3)."""
    cap = _interpolation_capability(e_on, e_off, n_vbus, n_current)
    return [{
        "kind": "data_limitation",
        "name": "single_temperature_single_measured_node",
        "temperature_c": float(temperature_c),
        "finite_nodes": cap["finite_nodes"],
        "interpolation_status": cap["status"],
        "rule": (
            f"E_on/E_off and r_on are measured only at conditions.temperature = "
            f"{float(temperature_c):g} C; the LUT has no temperature axis and "
            "is valid only at that condition. It is a measured data slice, "
            "not a physical guarantee at other junction temperatures. The "
            f"measured grid has {cap['finite_nodes']} finite node(s) "
            f"(capability '{cap['status']}'); off-node queries are NaN. "
            "Losses are evaluated offline and injected thermally, not "
            "electrical power consumed by the emitted switch (PLECS P-TC)."
        ),
        "basis": THERMAL_LOSS_NOTE,
    }]


@dataclass(frozen=True)
class LossLUT:
    """Config- and measurement-derived Tier-4 loss table.

    ``e_on_uj`` / ``e_off_uj`` are indexed [vbus_point][current_point] in uJ.
    ``r_on_ohms`` and ``v_on_v`` (when present) align with ``current_points``.
    ``vth_v`` is the Curve-7 threshold used by the emitted ideal switch.
    ``source`` carries every provenance rule, node measurement and gap.
    ``coss_f`` (F) is the measured Curve-4 Coss reference; ``r_off_ohms`` is the
    measured Vgs=0 off-state resistance (NaN when the sweep is not measurable).
    """

    vendor: str
    temperature_c: float
    vbus_points: tuple[float, ...]
    current_points: tuple[float, ...]
    e_on_uj: tuple[tuple[float, ...], ...]
    e_off_uj: tuple[tuple[float, ...], ...]
    r_on_ohms: tuple[float, ...] | None = None
    v_on_v: tuple[float, ...] | None = None
    vth_v: float = NAN
    vgs_on_v: float = NAN
    source: dict[str, Any] = field(default_factory=dict)
    coss_f: float = NAN
    r_off_ohms: float = NAN

    def __post_init__(self) -> None:
        if not self.vbus_points or not self.current_points:
            raise ValueError("LossLUT axes must be non-empty")
        if not all(math.isfinite(float(v)) for v in self.vbus_points + self.current_points):
            raise ValueError("LossLUT grid axes must be finite")
        n_i = len(self.current_points)
        for name in ("e_on_uj", "e_off_uj"):
            rows = getattr(self, name)
            if len(rows) != len(self.vbus_points) or any(len(r) != n_i for r in rows):
                raise ValueError(f"{name} shape must be [vbus][current]")
        for name in ("r_on_ohms", "v_on_v"):
            values = getattr(self, name)
            if values is not None and len(values) != n_i:
                raise ValueError(f"{name} must align with current_points")

        # D13: a finite negative energy is a PLECS-style diagnostic, not a
        # construction-time failure.  Normalize it to NaN and record a gap so
        # ``energy_at`` can report it; never crop it to zero, which would
        # fabricate a measured loss.  Shape/axis errors above still raise --
        # those are malformed containers, not data diagnostics.
        negative: dict[str, list[tuple[int, int, float]]] = {}
        for name in ("e_on_uj", "e_off_uj"):
            cleaned: list[tuple[float, ...]] = []
            bad: list[tuple[int, int, float]] = []
            for vi, row in enumerate(getattr(self, name)):
                new_row: list[float] = []
                for ci, raw in enumerate(row):
                    value = _finite_or_nan(raw)
                    if math.isfinite(value) and value < 0.0:
                        bad.append((vi, ci, value))
                        new_row.append(NAN)
                    else:
                        new_row.append(value)
                cleaned.append(tuple(new_row))
            if bad:
                object.__setattr__(self, name, tuple(cleaned))
                negative[name] = bad
        if negative:
            source = copy.deepcopy(self.source)
            gaps = list(source.get("gaps") or [])
            for name, bad in negative.items():
                for vi, ci, value in bad:
                    gaps.append({
                        "point": {"vbus_v": self.vbus_points[vi],
                                  "current_a": self.current_points[ci]},
                        "metric": name, "status": "negative_loss",
                        "recorded_value": value,
                        "reason": "PLECS negative-loss diagnostic: finite "
                                  "negative energy normalized to NaN, never "
                                  "cropped to zero",
                    })
            source["gaps"] = gaps
            object.__setattr__(self, "source", source)


    def to_dict(self) -> dict[str, Any]:
        """Plain-Python record for reports (NaN is kept, not turned into 0)."""
        return {
            "schema": SCHEMA,
            "vendor": self.vendor,
            "temperature_c": self.temperature_c,
            "vbus_points": list(self.vbus_points),
            "current_points": list(self.current_points),
            "e_on_uj": [list(r) for r in self.e_on_uj],
            "e_off_uj": [list(r) for r in self.e_off_uj],
            "r_on_ohms": None if self.r_on_ohms is None else list(self.r_on_ohms),
            "v_on_v": None if self.v_on_v is None else list(self.v_on_v),
            "vth_v": self.vth_v,
            "vgs_on_v": self.vgs_on_v,
            "coss_f": self.coss_f,
            "r_off_ohms": self.r_off_ohms,
            "source": copy.deepcopy(self.source),
            # Honest capability and measured-data limitations, mirrored in
            # source for provenance consumers that read only that sub-tree.
            "interpolation": self.interpolation_capability(),
            "data_limitations": self.data_limitations(),
            "device_values_finite": self.device_values_finite(),
            "thermal_loss_note": THERMAL_LOSS_NOTE,
        }

    def interpolation_capability(self) -> dict[str, Any]:
        """Honest description of what the measured grid can actually support."""
        return _interpolation_capability(
            self.e_on_uj, self.e_off_uj,
            len(self.vbus_points), len(self.current_points))

    def grid_provenance(self) -> str:
        """Capability-derived grid provenance for emitted/JSON comments."""
        return interpolation_provenance_text(self.interpolation_capability())

    def data_limitations(self) -> list[dict[str, Any]]:
        """Measured-data limitations (not project decisions) for JSON/meta."""
        return _data_limitations(
            self.temperature_c, self.e_on_uj, self.e_off_uj,
            len(self.vbus_points), len(self.current_points))

    def device_values_finite(self) -> dict[str, Any]:
        """Guard inputs for the emitted .MODEL; NaN must never reach SPICE."""
        vth_ok = math.isfinite(float(self.vth_v))
        r_on_ok = (self.r_on_ohms is not None and len(self.r_on_ohms) > 0
                   and all(math.isfinite(float(r)) for r in self.r_on_ohms))
        selected = None
        if self.r_on_ohms:
            index, _rule = self.representative_current_index()
            candidate = float(self.r_on_ohms[index])
            selected = candidate if math.isfinite(candidate) else None
        return {
            "finite": bool(vth_ok and r_on_ok),
            "vth_finite": vth_ok,
            "r_on_all_finite": r_on_ok,
            "r_on_representative_ohms": selected,
            "coss_f_finite": math.isfinite(float(self.coss_f)),
            "r_off_measured": math.isfinite(float(self.r_off_ohms)),
            "r_off_note": ("r_off_ohms NaN is allowed: emitter falls back to a "
                           "project-decision Roff with loud comment+JSON "
                           "disclosure; never write NaN into a SPICE netlist"),
            "rule": ("emitter must refuse to emit .MODEL SW(Ron=..., Vt=...) "
                     "when any selected device value is non-finite; never "
                     "write NaN into a SPICE netlist"),
        }

    def representative_current_index(self, value: float | None = None
                                     ) -> tuple[int, str]:
        """Index of the representative measured current point.

        ``statistics.median`` is the midpoint of the two middle values for an
        even-length axis and therefore generally is NOT itself a measured
        current.  The nearest measured current is selected; an exact tie (the
        median lies halfway between two currents) resolves to the LOWER index
        because ``min`` keeps the first key with the smallest distance.
        """
        if value is None:
            target = float(statistics.median(self.current_points))
            kind = ("median of grid axis; nearest measured point, lower "
                    "index on ties")
        else:
            target = float(value)
            kind = ("explicit load current; nearest measured point, lower "
                    "index on ties")
        index = min(
            range(len(self.current_points)),
            key=lambda k: (abs(float(self.current_points[k]) - target), k),
        )
        return index, kind

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LossLUT":
        """Rebuild a LUT from a JSON record (null in a matrix means NaN)."""
        schema = data.get("schema", SCHEMA)
        if schema != SCHEMA:
            raise ValueError(f"unsupported loss LUT schema: {schema}")
        source = copy.deepcopy(dict(data.get("source") or {}))
        return cls(
            vendor=str(data["vendor"]),
            temperature_c=float(data["temperature_c"]),
            vbus_points=tuple(float(v) for v in data["vbus_points"]),
            current_points=tuple(float(i) for i in data["current_points"]),
            e_on_uj=_matrix(data["e_on_uj"]),
            e_off_uj=_matrix(data["e_off_uj"]),
            r_on_ohms=(None if data.get("r_on_ohms") is None else
                       tuple(_finite_or_nan(v) for v in data["r_on_ohms"])),
            v_on_v=(None if data.get("v_on_v") is None else
                    tuple(_finite_or_nan(v) for v in data["v_on_v"])),
            vth_v=_finite_or_nan(data.get("vth_v")),
            vgs_on_v=_finite_or_nan(data.get("vgs_on_v")),
            coss_f=_finite_or_nan(data.get("coss_f")),
            r_off_ohms=_finite_or_nan(data.get("r_off_ohms")),
            source=source,
        )

    def energy_at(self, vbus_v: float, current_a: float) -> dict[str, Any]:
        """E_on/E_off at a measured node or by in-grid interpolation only.

        Finite results carry a status that states the actual capability:
        ``measured``, ``single_node``, ``constant_node`` or ``interpolated``.
        There is no extrapolation path; out-of-grid, missing and negative-loss
        results are NaN with a diagnostic status, never zero.
        E_on/E_off are a pair: an unmeasured corner in either channel makes
        both NaN (``insufficient_measured_nodes``).
        """
        vp = _axis_position(self.vbus_points, float(vbus_v))
        cp = _axis_position(self.current_points, float(current_a))
        out: dict[str, Any] = {
            "vbus_v": float(vbus_v), "current_a": float(current_a),
            "e_on_uj": NAN, "e_off_uj": NAN, "status": "outside_grid",
            "method": None, "indices": None, "source": None,
        }
        if vp is None or cp is None:
            out["status"] = ("invalid_value" if not
                             (math.isfinite(float(vbus_v))
                              and math.isfinite(float(current_a)))
                             else "outside_grid")
            return out
        vi, vj, wv, vm = vp
        ci, cj, wc, cm = cp
        out["indices"] = {"vbus": [vi, vj], "current": [ci, cj]}
        if vm == "exact" and cm == "exact":
            e_on = float(self.e_on_uj[vi][ci])
            e_off = float(self.e_off_uj[vi][ci])
            node = self.source.get("nodes", {}).get(
                _node_key(float(vbus_v), float(current_a)))
            out["method"] = "exact_grid_node"
            out["source"] = node or {"method": "exact_grid_node"}
            finite = math.isfinite(e_on) and math.isfinite(e_off)
            if not finite:
                out["status"] = (
                    str(node.get("status")) if isinstance(node, dict) and node
                    and node.get("status") else "not_evaluable")
            else:
                capability = self.interpolation_capability()["status"]
                out["status"] = (capability if capability in ("constant_node",
                                                              "single_node")
                                 else "measured")
        else:
            e_on = _grid_value(self.e_on_uj, vi, vj, wv, ci, cj, wc)
            e_off = _grid_value(self.e_off_uj, vi, vj, wv, ci, cj, wc)
            out["method"] = "bilinear_interpolation"
            if not (math.isfinite(e_on) and math.isfinite(e_off)):
                # A measured node supplies both energies or neither: a missing
                # corner in either channel invalidates the whole grid cell.
                e_on = e_off = NAN
            out["status"] = ("interpolated" if
                             math.isfinite(e_on) and math.isfinite(e_off)
                             else "insufficient_measured_nodes")
            out["source"] = {
                "method": out["method"],
                "rule": "linear interpolation on measured grid nodes only; "
                        "no extrapolation",
                "corners": out["indices"],
            }
        if (math.isfinite(e_on) and e_on < 0.0) or (math.isfinite(e_off) and e_off < 0.0):
            out.update(e_on_uj=NAN, e_off_uj=NAN, status="negative_loss")
            return out
        out["e_on_uj"] = e_on
        out["e_off_uj"] = e_off
        return out


    def r_on_at(self, current_a: float) -> float:
        """Interpolated Curve-3 on-resistance in ohms; NaN outside the grid."""
        if self.r_on_ohms is None:
            return NAN
        return _linear_value(self.r_on_ohms, self.current_points, float(current_a))

    def v_on_at(self, current_a: float) -> float:
        """Conduction drop: measured v_on table when given, else i * r_on."""
        if self.v_on_v is not None:
            return _linear_value(self.v_on_v, self.current_points, float(current_a))
        r = self.r_on_at(current_a)
        return r * float(current_a) if math.isfinite(r) else NAN

    def losses_at(self, vbus_v: float, current_a: float, *,
                  f_sw_hz: float | None = None) -> dict[str, Any]:
        """Conduction and switching losses computed separately from the LUT.

        No efficiency is produced here.  PLECS P-TC states that thermal losses
        are not the electrical power consumed by the device; ``note`` carries
        that caveat into every caller and into the JSON record.
        """
        energy = self.energy_at(vbus_v, current_a)
        out: dict[str, Any] = {
            "p_cond_w": NAN, "p_switch_w": NAN, "p_total_w": NAN,
            "f_sw_hz": None if f_sw_hz is None else float(f_sw_hz),
            "status": "invalid_input", "energy": energy,
            "note": THERMAL_LOSS_NOTE,
        }
        r = self.r_on_at(current_a)
        if not math.isfinite(r):
            out["status"] = "conduction_resistance_unavailable"
            return out
        p_cond = r * float(current_a) ** 2
        out["p_cond_w"] = p_cond
        if not (math.isfinite(energy["e_on_uj"])
                and math.isfinite(energy["e_off_uj"])):
            out["status"] = energy["status"]
            return out
        if f_sw_hz is None:
            out["status"] = "frequency_required"
            return out
        f = float(f_sw_hz)
        if not math.isfinite(f) or f < 0.0:
            out["status"] = "invalid_input"
            return out
        p_switch = (energy["e_on_uj"] + energy["e_off_uj"]) * 1e-6 * f
        if not math.isfinite(p_switch):
            out["p_switch_w"] = p_switch
            out["status"] = energy["status"]
            return out
        out.update(p_switch_w=p_switch, p_total_w=p_cond + p_switch,
                   status="computed")
        return out

    def efficiency_at(self, vbus_v: float, current_a: float,
                      p_out_w: float | None = None, *,
                      f_sw_hz: float | None = None) -> dict[str, Any]:
        """Efficiency only for an explicit *useful output* power.

        ``p_out_w`` must be the converter's useful output power supplied by
        the caller.  Vbus*Iload is bus input power, so this method never
        derives p_out from it; with no p_out it returns NaN efficiency and
        status ``output_power_required`` together with the separately
        computed loss terms.  Losses are thermal/offline data (PLECS P-TC),
        not electrical power consumed by the emitted switch.
        """
        losses = self.losses_at(vbus_v, current_a, f_sw_hz=f_sw_hz)
        out: dict[str, Any] = dict(losses)
        out.update(efficiency=NAN, p_out_w=None, reason=None)
        if p_out_w is None:
            out.update(status="output_power_required",
                       reason=("No explicit converter useful output power was "
                               "supplied; Vbus*Iload is bus input power, not "
                               "output. Efficiency is N/A. " + THERMAL_LOSS_NOTE))
            return out
        p_out = float(p_out_w)
        if not (math.isfinite(p_out) and p_out > 0.0):
            out.update(status="invalid_input",
                       reason="p_out_w must be finite and positive")
            return out
        out["p_out_w"] = p_out
        if losses["status"] != "computed":
            out["reason"] = "loss inputs unavailable: " + str(losses["status"])
            return out
        out["efficiency"] = p_out / (p_out + losses["p_total_w"])
        return out


def _r_on_from_curve(vendor: str, config: Mapping[str, Any],
                     currents: tuple[float, ...]) -> tuple[list[float], dict[str, Any]]:
    """Interpolate the measured Curve-3 Rds(on) table at each grid current."""
    from simulation.testbenches import loss_lut_bench as bench

    meta: dict[str, Any] = {
        "source_file": bench.curve_provenance(vendor, "curve3_rds_on"),
        "measured_key": "extracted.curve_id / extracted.curve_rds_mohm",
        "gate_drive_v": float(config["conditions"]["Vgs_fixed"]),
        "gate_drive_rule": "conditions.Vgs_fixed (Curve-3 extraction drive)",
        "method": "linear interpolation of measured curve; no extrapolation",
        "unit_conversion": "mOhm / 1000 -> Ohm",
        "kind": "measured",
        "finite_guard": ("currents outside the measured Curve-3 range stay "
                         "NaN with a gap; never write NaN into a SPICE .MODEL"),
    }
    data = load_output(vendor, "curve3_rds_on")
    extracted = data["extracted"]
    ids = np.asarray(extracted["curve_id"], dtype=float)
    rs = np.asarray(extracted["curve_rds_mohm"], dtype=float) / 1000.0
    order = np.argsort(ids)
    ids, rs = ids[order], rs[order]
    meta["curve_id_range_a"] = [float(ids[0]), float(ids[-1])]
    meta["reference_rds_on_at_Id_load_ohms"] = (
        float(extracted["rds_on_mohm_at_Id_load"]) / 1000.0)
    meta["reference_condition_a"] = float(config["conditions"]["Id_load"])
    values: list[float] = []
    for current in currents:
        if current < ids[0] or current > ids[-1]:
            values.append(NAN)
        else:
            values.append(float(np.interp(current, ids, rs)))
    return values, meta


def _coss_from_output(vendor: str,
                      config: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    """Measured Curve-4 Coss at the recorded cap reference voltage, in farads.

    This is a solver-stabilizing linear capacitor for the emitted Tier-4
    subcircuit, not a voltage-dependent or waveform-fidelity Coss model.  The
    value and its reference voltage/temperature are copied from the extracted
    output file; nothing is recomputed or invented here.
    """
    from simulation.testbenches import loss_lut_bench as bench

    meta: dict[str, Any] = {
        "kind": "measured",
        "file": bench.curve_provenance(vendor, "curve4_capacitances")["file"],
        "key": "extracted.coss_f",
        "value_f": NAN,
        "ref_voltage_v": NAN,
        "ref_voltage_key": "extracted.cap_ref_vds",
        "temperature_c": float(config["conditions"]["temperature"]),
        "temperature_key": "conditions.temperature",
        "unit": "F",
        "rule": ("measured small-signal Coss at the Curve-4 cap_ref_Vds "
                 "reference; solver-stabilizing linear C only, not a "
                 "voltage-dependent/waveform-fidelity model"),
    }
    try:
        extracted = load_output(vendor, "curve4_capacitances")["extracted"]
        coss = _finite_or_nan(extracted["coss_f"])
        ref = _finite_or_nan(extracted.get("cap_ref_vds"))
        if not (math.isfinite(coss) and coss > 0.0):
            raise ValueError("extracted.coss_f is missing or not positive")
        if not math.isfinite(ref):
            raise ValueError("extracted.cap_ref_vds is missing")
        meta.update(value_f=coss, ref_voltage_v=ref)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        meta.update(kind="missing", value_f=NAN, ref_voltage_v=NAN,
                    reason=str(exc))
    return meta["value_f"], meta

def build_lut(vendor: str, *, config: Mapping[str, Any] | None = None,
              simulate: bool = True, reproduce_recorded: bool = False,
              points: tuple[Any, ...] | None = None) -> LossLUT:
    """Build one vendor's Tier-4 LUT from config-derived golden DPT sweeps.

    ``simulate=False`` is the offline path: only identical-condition recorded
    golden DPT metrics are used and every other grid node is an explicit gap.
    """
    from simulation.testbenches import loss_lut_bench as bench

    cfg = load_input(vendor) if config is None else config
    grid, rules = bench.derive_grid(cfg)
    if points is not None:
        # D11: config-derived rules do not describe caller-supplied points.
        # Republish them for the grid actually used instead of silently
        # attaching the derived prose to an explicit override.
        grid = points
        rules = {
            "vbus": "caller-supplied points=... vbus values (not config-derived)",
            "current": "caller-supplied points=... current values (not config-derived)",
            "l_load": "caller-supplied GridPoint.l_load_h (no new numeric value "
                      "derived here)",
            "temperature": "conditions.temperature",
            "vgs_on": "dpt.vgs_on (DPT gate drive)",
            "kind": "project_decision_explicit_points",
            "rationale": "explicit points override the config-derived grid; "
                         "config derivation rules are not republished",
        }

    sweeps = bench.run_golden_sweep(
        vendor, config=cfg, points=grid, simulate=simulate,
        reproduce_recorded=reproduce_recorded)

    current_axis = tuple(sorted({float(p.current_a) for p in grid}))
    vbus_axis = tuple(sorted({float(p.vbus_v) for p in grid}))
    i_index = {value: k for k, value in enumerate(current_axis)}
    v_index = {value: k for k, value in enumerate(vbus_axis)}
    e_on = [[NAN] * len(current_axis) for _ in vbus_axis]
    e_off = [[NAN] * len(current_axis) for _ in vbus_axis]

    node_provenance: dict[str, Any] = {}
    gaps: list[dict[str, Any]] = []
    for result in sweeps:
        key = _node_key(result.point.vbus_v, result.point.current_a)
        node_provenance[key] = result.as_dict()
        iv, ic = v_index[float(result.point.vbus_v)], i_index[float(result.point.current_a)]
        e_on[iv][ic] = float(result.e_on_uj)
        e_off[iv][ic] = float(result.e_off_uj)
        if result.status != "measured":
            gaps.append({
                "point": result.point.as_dict(), "metric": "e_on_uj,e_off_uj",
                "status": result.status, "reason": result.reason,
                "source": result.source, "guard": result.guard,
            })

    r_on, r_meta = _r_on_from_curve(vendor, cfg, current_axis)
    for current, resistance in zip(current_axis, r_on):
        if not math.isfinite(resistance):
            gaps.append({"point": {"current_a": current}, "metric": "r_on_ohms",
                         "status": "outside_measured_curve_range",
                         "reason": "Curve-3 sweep does not cover this grid current",
                         "source": r_meta["source_file"]["file"]})
    try:
        vth = float(load_output(vendor, "curve7_threshold_voltage")["extracted"]["vth_v"])
        if not math.isfinite(vth):
            raise ValueError("Curve-7 vth_v is not finite")
        vth_meta = {
            "source_file": bench.curve_provenance(vendor, "curve7_threshold_voltage"),
            "measured_key": "extracted.vth_v",
            "kind": "measured",
        }
    except (FileNotFoundError, KeyError, ValueError) as exc:
        vth = NAN
        vth_meta = {"source_file": bench.curve_provenance(vendor, "curve7_threshold_voltage"),
                    "measured_key": "extracted.vth_v", "kind": "missing",
                    "reason": str(exc)}
        gaps.append({"point": {}, "metric": "vth_v", "status": "missing",
                     "reason": str(exc), "source": vth_meta["source_file"]["file"]})

    coss_f, coss_meta = _coss_from_output(vendor, cfg)
    if not math.isfinite(coss_f):
        gaps.append({"point": {}, "metric": "coss_f", "status": "missing",
                     "reason": coss_meta.get("reason", "no measured Curve-4 Coss"),
                     "source": coss_meta["file"]})

    r_off_ohms, r_off_meta = bench.measure_r_off_ohms(
        vendor, cfg, simulate=simulate)
    if not math.isfinite(r_off_ohms):
        gaps.append({
            "point": {}, "metric": "r_off_ohms", "status": r_off_meta["kind"],
            "reason": r_off_meta.get("reason", "no measured off-state resistance"),
            "source": r_off_meta["bench"],
        })

    cond = cfg["conditions"]
    capability = _interpolation_capability(
        e_on, e_off, len(vbus_axis), len(current_axis))
    source: dict[str, Any] = {
        "schema": SCHEMA,
        "method": "clamped-inductive golden DPT E_on/E_off at config-derived "
                  "(Vbus, load) nodes; Curve-3 r_on, Curve-7 Vth, Curve-4 Coss "
                  "and Vgs=0 DC-sweep Roff",
        "config": bench.config_provenance(vendor),
        "config_fields": {
            "dpt": {k: cfg["dpt"][k] for k in (
                "vbus", "vgs_on", "i_load_target", "l_load", "t1_width",
                "t2_on", "t2_width", "edge_time", "t_end", "max_step")},
            "conditions": {k: cond[k] for k in (
                "Vbus", "Id_load", "Vgs_fixed", "Vgs_sweep", "temperature")},
        },
        "grid": {
            "vbus_points": list(vbus_axis), "current_points": list(current_axis),
            "rules": rules,
            "kind": "project_decision",
        },
        "r_on": r_meta,
        "vth": vth_meta,
        "coss_f": coss_meta,
        "r_off_ohms": r_off_meta,
        "nodes": node_provenance,
        "gaps": gaps,
        "golden_evaluable": all(r.status == "measured" for r in sweeps),
        "dpt_crossing_evaluable": (
            node_provenance.get(_node_key(float(cfg["dpt"]["vbus"]),
                                          float(cfg["dpt"]["i_load_target"])), {})
            .get("status") == "measured"),
        "waveform_gate": "retired_by_design",
        "interpolation": capability,
        "data_limitations": _data_limitations(
            float(cond["temperature"]), e_on, e_off,
            len(vbus_axis), len(current_axis)),
        "thermal_loss_note": THERMAL_LOSS_NOTE,
        "device_value_guard": {
            "vth_finite": math.isfinite(vth),
            "r_on_all_finite": all(math.isfinite(float(r)) for r in r_on),
            "rule": ("emitter must refuse to emit .MODEL when vth or the "
                     "selected r_on is NaN; never write NaN into SPICE"),
            "coss_f_finite": math.isfinite(coss_f),
            "r_off_measured": math.isfinite(r_off_ohms),
            "r_off_note": ("r_off_ohms NaN is allowed: emitter falls back to a "
                           "project-decision Roff with loud comment+JSON "
                           "disclosure; never write NaN into SPICE"),
        },
        "decisions": [
            {"name": "grid_axes", "kind": "project_decision",
             "rule": rules["vbus"] + "; " + rules["current"]},
            {"name": "interpolation", "kind": "project_decision",
             "rule": ("bilinear interpolation applies only when all four "
                      "measured corners are finite; no extrapolation is "
                      "implemented; off-node queries are NaN"),
             "basis": ("PLECS P-TC documents linear table interpolation; "
                       + interpolation_provenance_text(capability)),
             "capability": capability},

            {"name": "missing_crossings", "kind": "project_decision",
             "rule": "NaN/N/A with a source gap; never zero or a filled value"},
            {"name": "completion_guard", "kind": "project_decision",
             "rule": f"t_last >= {bench.T_LAST_FRACTION:g}*t_end and "
                     f"trace_points >= {bench.TRACE_POINT_FRACTION:g}*golden points "
                     "(reused from pipeline/tier3.py)"},
            {"name": "thermal_loss_accounting", "kind": "project_decision",
             "rule": THERMAL_LOSS_NOTE,
             "basis": "PLECS P-TC notes under Conduction/Switching Losses"},
            {"name": "solver_stabilization", "kind": "measured_with_fallback",
             "rule": ("emitted Tier-4 switch uses the measured Curve-4 Coss as "
                      "one active linear C_QOSS and the measured Vgs=0 Roff; "
                      "C_QOSS only slows drain dV/dt to avoid singular/no-cap "
                      "steps and adds L-C overshoot, so the DPT waveform gate "
                      "stays retired. Voltage-dependent capacitors and the "
                      "body diode remain retired. A NaN Roff falls back to a "
                      "project-decision emitter value with loud comment+JSON "
                      "disclosure; no fallback number is invented here."),
             "basis": "local://tier4_stabilization_contract.md"},
        ],
    }
    return LossLUT(
        vendor=vendor, temperature_c=float(cond["temperature"]),
        vbus_points=vbus_axis, current_points=current_axis,
        e_on_uj=tuple(tuple(row) for row in e_on),
        e_off_uj=tuple(tuple(row) for row in e_off),
        r_on_ohms=tuple(r_on), vth_v=vth,
        vgs_on_v=float(cfg["dpt"]["vgs_on"]),
        coss_f=coss_f, r_off_ohms=r_off_ohms, source=source,
    )
