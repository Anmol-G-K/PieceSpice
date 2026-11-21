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
    "emitted Tier-4 switch has no loss element: E_on/E_off/r_on are evaluated "
    "offline and an optional caller-configured thermal network may consume "
    "that loss estimate, so efficiency needs an explicit useful output-power "
    "definition (Vbus*Iload is input power, not output)."
)


def _node_key(vbus_v: float, current_a: float) -> str:
    """Stable key for one measured grid node in source provenance."""
    return f"{float(vbus_v)!r}|{float(current_a)!r}"


def _node_key_with_temperature(vbus_v: float, current_a: float,
                               temperature_c: float) -> str:
    """Temperature-qualified node key for opt-in 3-D grids."""
    return f"{_node_key(vbus_v, current_a)}|{float(temperature_c)!r}"


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
                      n_vbus: int, n_current: int, *,
                      temperature_points: tuple[float, ...] = (),
                      has_temperature_tables: bool = False
                      ) -> list[dict[str, Any]]:
    """Measured-data limitations (not project decisions) for JSON/meta (M3)."""
    cap = _interpolation_capability(e_on, e_off, n_vbus, n_current)
    if has_temperature_tables:
        points = tuple(float(t) for t in temperature_points)
        return [{
            "kind": "data_limitation",
            "name": "temperature_interpolated_grid",
            "temperature_c": float(temperature_c),
            "temperature_points": list(points),
            "finite_nodes": cap["finite_nodes"],
            "interpolation_status": cap["status"],
            "rule": (
                "E_on/E_off are measured at T = "
                + ", ".join(f"{t:g}" for t in points)
                + f" C with reference slice T = {float(temperature_c):g} C. "
                "Trilinear interpolation is used only when all eight corners "
                "of the two adjacent measured temperature slices are finite; "
                "off-grid, NaN-corner and temperature-extrapolated queries are "
                "NaN with a diagnostic status, never zero or extrapolated. "
                f"r_on is measured at conditions.temperature = "
                f"{float(temperature_c):g} C only and has no temperature axis. "
                "Losses are evaluated offline and injected thermally, not "
                "electrical power consumed by the emitted switch (PLECS P-TC)."
            ),
            "basis": THERMAL_LOSS_NOTE,
        }]
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
    # Opt-in temperature axis.  ``None`` normalizes to
    # ``(temperature_c,)`` in __post_init__, so every existing 2-D LUT keeps
    # its exact reference-table behaviour.
    temperature_points: tuple[float, ...] | None = None
    e_on_uj_by_temperature: tuple[tuple[tuple[float, ...], ...], ...] | None = None
    e_off_uj_by_temperature: tuple[tuple[tuple[float, ...], ...], ...] | None = None
    # Config/LUT switching threshold.  NaN means "use the Curve-7 JEDEC Vth";
    # the emitter labels that fallback source=curve7_jedec_vth and never
    # writes NaN into a SPICE .MODEL.
    vt_switch_v: float = NAN
    # Measured Curve-3 Rds(on) family: gate drive (V) -> r_on per current
    # point (ohm).  When present, the emitter selects the measured point at
    # the actual vgs_on instead of the median-current fallback.
    r_on_by_vgs: Mapping[float, tuple[float, ...]] | None = None

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

        # Opt-in temperature axis: a bare 2-D LUT normalizes to its single
        # reference temperature, preserving the historical behaviour.
        raw_temps = self.temperature_points
        if raw_temps is None:
            temps = (float(self.temperature_c),)
        else:
            temps = tuple(float(t) for t in raw_temps)
            if not temps:
                raise ValueError("temperature_points must be non-empty")
            if not all(math.isfinite(t) for t in temps):
                raise ValueError("temperature_points must be finite")
            if any(b <= a for a, b in zip(temps, temps[1:])):
                raise ValueError(
                    "temperature_points must be strictly increasing")
        object.__setattr__(self, "temperature_points", temps)
        for name in ("e_on_uj_by_temperature", "e_off_uj_by_temperature"):
            matrices = getattr(self, name)
            if matrices is None:
                continue
            if len(matrices) != len(temps):
                raise ValueError(
                    f"{name} first axis must align with temperature_points")
            normalized = []
            for rows in matrices:
                matrix = _matrix(rows)
                if (len(matrix) != len(self.vbus_points)
                        or any(len(row) != n_i for row in matrix)):
                    raise ValueError(
                        f"{name} shape must be [temperature][vbus][current]")
                normalized.append(matrix)
            object.__setattr__(self, name, tuple(normalized))

        table = self.r_on_by_vgs
        if table is not None:
            cleaned: dict[float, tuple[float, ...]] = {}
            for key, row in table.items():
                gate = float(key)
                if not math.isfinite(gate):
                    raise ValueError("r_on_by_vgs gate-drive keys must be finite")
                values = tuple(_finite_or_nan(v) for v in row)
                if len(values) != n_i:
                    raise ValueError(
                        "r_on_by_vgs rows must align with current_points")
                cleaned[gate] = values
            object.__setattr__(self, "r_on_by_vgs", cleaned)
        object.__setattr__(self, "vt_switch_v", _finite_or_nan(self.vt_switch_v))

        # D13: a finite negative energy is a PLECS-style diagnostic, not a
        # construction-time failure.  Normalize it to NaN and record a gap so
        # ``energy_at`` can report it; never crop it to zero, which would
        # fabricate a measured loss.  Shape/axis errors above still raise --
        # those are malformed containers, not data diagnostics.
        negative: list[dict[str, Any]] = []

        def _clean_matrix(name: str, matrix: Any,
                          temperature: float | None
                          ) -> tuple[tuple[tuple[float, ...], ...], bool]:
            """Return (normalized matrix, changed); negatives become NaN."""
            changed = False
            cleaned: list[tuple[float, ...]] = []
            for vi, row in enumerate(matrix):
                new_row: list[float] = []
                for ci, raw in enumerate(row):
                    value = _finite_or_nan(raw)
                    if math.isfinite(value) and value < 0.0:
                        negative.append({
                            "metric": name, "vi": vi, "ci": ci,
                            "value": value, "temperature": temperature,
                        })
                        new_row.append(NAN)
                        changed = True
                    else:
                        new_row.append(value)
                cleaned.append(tuple(new_row))
            return tuple(cleaned), changed

        for name in ("e_on_uj", "e_off_uj"):
            cleaned_matrix, changed = _clean_matrix(
                name, getattr(self, name), None)
            if changed:
                object.__setattr__(self, name, cleaned_matrix)
        for name in ("e_on_uj_by_temperature", "e_off_uj_by_temperature"):
            matrices = getattr(self, name)
            if matrices is None:
                continue
            cleaned_slices: list[tuple[tuple[float, ...], ...]] = []
            changed_any = False
            for ti, matrix in enumerate(matrices):
                cleaned_matrix, changed = _clean_matrix(
                    name, matrix, temps[ti])
                cleaned_slices.append(cleaned_matrix)
                changed_any = changed_any or changed
            if changed_any:
                object.__setattr__(self, name, tuple(cleaned_slices))
        if negative:
            source = copy.deepcopy(self.source)
            gaps = list(source.get("gaps") or [])
            for item in negative:
                point = {"vbus_v": self.vbus_points[item["vi"]],
                         "current_a": self.current_points[item["ci"]]}
                if item["temperature"] is not None:
                    point["temperature_c"] = item["temperature"]
                gaps.append({
                    "point": point,
                    "metric": item["metric"], "status": "negative_loss",
                    "recorded_value": item["value"],
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
            "temperature_points": list(self.temperature_points or ()),
            "e_on_uj_by_temperature": (
                None if self.e_on_uj_by_temperature is None else
                [[list(row) for row in matrix]
                 for matrix in self.e_on_uj_by_temperature]),
            "e_off_uj_by_temperature": (
                None if self.e_off_uj_by_temperature is None else
                [[list(row) for row in matrix]
                 for matrix in self.e_off_uj_by_temperature]),
            "vt_switch_v": self.vt_switch_v,
            "r_on_by_vgs": (
                None if self.r_on_by_vgs is None else
                {str(float(gate)): list(row)
                 for gate, row in sorted(self.r_on_by_vgs.items())}),
            "source": copy.deepcopy(self.source),
            # Honest capability and measured-data limitations, mirrored in
            # source for provenance consumers that read only that sub-tree.
            "interpolation": self.interpolation_capability(),
            "temperature_capability": self.temperature_capability(),
            "data_limitations": self.data_limitations(),
            "device_values_finite": self.device_values_finite(),
            "thermal_loss_note": THERMAL_LOSS_NOTE,
        }

    def interpolation_capability(self) -> dict[str, Any]:
        """Honest description of what the measured grid can actually support."""
        return _interpolation_capability(
            self.e_on_uj, self.e_off_uj,
            len(self.vbus_points), len(self.current_points))

    def temperature_capability(self) -> dict[str, Any]:
        """What the (opt-in) temperature axis can support; no extrapolation."""
        points = tuple(self.temperature_points or ())
        if self.e_on_uj_by_temperature is None:
            status = ("single_temperature" if len(points) == 1
                      else "reference_table_only")
            rule = ("Only the reference 2-D E_on/E_off table exists; a query "
                    "at any other temperature is NaN, never extrapolated.")
        else:
            finite_slices = sum(
                1 for matrix in self.e_on_uj_by_temperature
                if any(math.isfinite(float(v)) for row in matrix for v in row))
            status = ("trilinear_interpolation" if finite_slices >= 2
                      else "insufficient_measured_temperatures")
            rule = ("Trilinear interpolation requires all eight corners of the "
                    "two temperature slices to be finite; no extrapolation.")
        return {
            "status": status,
            "temperature_points": list(points),
            "reference_temperature_c": float(self.temperature_c),
            "has_3d_tables": self.e_on_uj_by_temperature is not None,
            "rule": rule,
        }

    def grid_provenance(self) -> str:
        """Capability-derived grid provenance for emitted/JSON comments."""
        return interpolation_provenance_text(self.interpolation_capability())

    def data_limitations(self) -> list[dict[str, Any]]:
        """Measured-data limitations (not project decisions) for JSON/meta."""
        return _data_limitations(
            self.temperature_c, self.e_on_uj, self.e_off_uj,
            len(self.vbus_points), len(self.current_points),
            temperature_points=tuple(self.temperature_points or ()),
            has_temperature_tables=self.e_on_uj_by_temperature is not None)

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
        by_temperature = {
            name: (None if data.get(name) is None else
                   tuple(_matrix(matrix) for matrix in data[name]))
            for name in ("e_on_uj_by_temperature", "e_off_uj_by_temperature")
        }
        table = data.get("r_on_by_vgs")
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
            temperature_points=(None if data.get("temperature_points") is None
                                else tuple(float(t) for t in
                                           data["temperature_points"])),
            e_on_uj_by_temperature=by_temperature["e_on_uj_by_temperature"],
            e_off_uj_by_temperature=by_temperature["e_off_uj_by_temperature"],
            vt_switch_v=_finite_or_nan(data.get("vt_switch_v")),
            r_on_by_vgs=(None if table is None else
                         {float(gate): tuple(_finite_or_nan(v) for v in row)
                          for gate, row in dict(table).items()}),
            source=source,
        )

    def _node(self, vbus_v: float, current_a: float,
              temperature_c: float | None) -> dict[str, Any] | None:
        """Measured-node provenance, temperature-qualified when available."""
        nodes = self.source.get("nodes", {})
        if temperature_c is not None:
            key = _node_key_with_temperature(
                vbus_v, current_a, temperature_c)
            if isinstance(nodes, Mapping) and key in nodes:
                return nodes[key]
        return nodes.get(_node_key(vbus_v, current_a)) \
            if isinstance(nodes, Mapping) else None

    def _energy_2d(self, e_on_mat: Any, e_off_mat: Any,
                   vbus_v: float, current_a: float, *,
                   temperature_c: float | None = None) -> dict[str, Any]:
        """Reference 2-D node/bilinear lookup for one energy table pair."""
        vp = _axis_position(self.vbus_points, float(vbus_v))
        cp = _axis_position(self.current_points, float(current_a))
        out: dict[str, Any] = {
            "vbus_v": float(vbus_v), "current_a": float(current_a),
            "e_on_uj": NAN, "e_off_uj": NAN, "status": "outside_grid",
            "method": None, "indices": None, "source": None,
            "temperature_c": (None if temperature_c is None
                              else float(temperature_c)),
            "temperature_method": ("reference_2d" if temperature_c is None
                                   else "exact_temperature_slice"),
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
            e_on = float(e_on_mat[vi][ci])
            e_off = float(e_off_mat[vi][ci])
            node = self._node(float(vbus_v), float(current_a), temperature_c)
            out["method"] = "exact_grid_node"
            out["source"] = node or {"method": "exact_grid_node"}
            finite = math.isfinite(e_on) and math.isfinite(e_off)
            if not finite:
                out["status"] = (
                    str(node.get("status")) if isinstance(node, dict) and node
                    and node.get("status") else "not_evaluable")
            else:
                capability = _interpolation_capability(
                    e_on_mat, e_off_mat,
                    len(self.vbus_points), len(self.current_points))["status"]
                out["status"] = (capability if capability in ("constant_node",
                                                              "single_node")
                                 else "measured")
        else:
            e_on = _grid_value(e_on_mat, vi, vj, wv, ci, cj, wc)
            e_off = _grid_value(e_off_mat, vi, vj, wv, ci, cj, wc)
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
        if ((math.isfinite(e_on) and e_on < 0.0)
                or (math.isfinite(e_off) and e_off < 0.0)):
            out.update(e_on_uj=NAN, e_off_uj=NAN, status="negative_loss")
            return out
        out["e_on_uj"] = e_on
        out["e_off_uj"] = e_off
        return out

    def temperature_at(self, temperature_c: float) -> dict[str, Any]:
        """Position of a query on the (opt-in) temperature axis.

        Statuses are explicit: ``exact_reference``, ``exact_slice``,
        ``interpolated``, ``outside_grid``, ``invalid_value`` or
        ``temperature_table_unavailable``.  No extrapolation is implemented.
        """
        points = tuple(self.temperature_points or ())
        out: dict[str, Any] = {
            "temperature_c": None, "temperature_points": list(points),
            "status": "invalid_value", "indices": None, "weight": None,
            "method": None, "extrapolation": False,
        }
        try:
            value = float(temperature_c)
        except (TypeError, ValueError):
            return out
        out["temperature_c"] = value
        if not math.isfinite(value):
            return out
        if self.e_on_uj_by_temperature is not None:
            pos = _axis_position(points, value)
            if pos is None:
                out["status"] = "outside_grid"
                return out
            i, j, weight, mode = pos
            out["indices"] = [i, j]
            out["weight"] = weight
            if mode == "exact":
                out["status"] = "exact_slice"
                out["method"] = "measured_temperature_slice"
            else:
                out["status"] = "interpolated"
                out["method"] = "trilinear_interpolation"
            return out
        if len(points) == 1 and value == points[0]:
            out["status"] = "exact_reference"
            out["indices"] = [0, 0]
            out["weight"] = 0.0
            out["method"] = "reference_2d"
            return out
        out["status"] = ("temperature_table_unavailable"
                         if len(points) == 1 else "outside_grid")
        return out

    def energy_at(self, vbus_v: float, current_a: float,
                  temperature_c: float | None = None) -> dict[str, Any]:
        """E_on/E_off at a measured node or by in-grid interpolation only.

        Finite results carry a status that states the actual capability:
        ``measured``, ``single_node``, ``constant_node`` or ``interpolated``.
        There is no extrapolation path; out-of-grid, missing and negative-loss
        results are NaN with a diagnostic status, never zero.
        E_on/E_off are a pair: an unmeasured corner in either channel makes
        both NaN (``insufficient_measured_nodes``).

        ``temperature_c=None`` uses the reference 2-D table, preserving the
        original API.  A supplied temperature is resolved on the opt-in
        temperature axis: an exact slice reuses the 2-D rules, and an interior
        query uses trilinear interpolation only when all eight corners of the
        two adjacent slices are finite.  Off-axis/extrapolated queries are NaN.
        """
        if temperature_c is None:
            out = self._energy_2d(self.e_on_uj, self.e_off_uj,
                                  vbus_v, current_a)
            out["temperature_c"] = float(self.temperature_c)
            out["temperature_method"] = "reference_2d"
            return out

        position = self.temperature_at(temperature_c)
        out: dict[str, Any] = {
            "vbus_v": float(vbus_v), "current_a": float(current_a),
            "temperature_c": position["temperature_c"],
            "temperature_method": position["method"],
            "e_on_uj": NAN, "e_off_uj": NAN,
            "status": position["status"], "method": None,
            "indices": None, "source": None,
            "temperature": position,
        }
        if position["status"] in ("invalid_value", "outside_grid",
                                  "temperature_table_unavailable"):
            return out
        if position["status"] == "exact_reference":
            reference = self._energy_2d(self.e_on_uj, self.e_off_uj,
                                        vbus_v, current_a)
            reference["temperature_c"] = float(self.temperature_c)
            reference["temperature_method"] = "exact_reference"
            return reference
        matrices_on = self.e_on_uj_by_temperature
        matrices_off = self.e_off_uj_by_temperature
        if matrices_on is None or matrices_off is None:  # defensive
            return out
        ti, tj = position["indices"]
        if ti == tj:
            out = self._energy_2d(matrices_on[ti], matrices_off[ti],
                                  vbus_v, current_a,
                                  temperature_c=float(position["temperature_c"]))
            out["temperature"] = position
            return out

        vp = _axis_position(self.vbus_points, float(vbus_v))
        cp = _axis_position(self.current_points, float(current_a))
        if vp is None or cp is None:
            out["status"] = ("invalid_value" if not
                             (math.isfinite(float(vbus_v))
                              and math.isfinite(float(current_a)))
                             else "outside_grid")
            return out
        vi, vj, wv, _vm = vp
        ci, cj, wc, _cm = cp
        weight = float(position["weight"] or 0.0)
        corners = []
        for index, matrices_on_slice, matrices_off_slice in (
            (ti, matrices_on[ti], matrices_off[ti]),
            (tj, matrices_on[tj], matrices_off[tj]),
        ):
            corners.append((
                index,
                _grid_value(matrices_on_slice, vi, vj, wv, ci, cj, wc),
                _grid_value(matrices_off_slice, vi, vj, wv, ci, cj, wc),
            ))
        out["indices"] = {"vbus": [vi, vj], "current": [ci, cj],
                          "temperature": [ti, tj]}
        if not all(math.isfinite(on) and math.isfinite(off)
                   for _index, on, off in corners):
            out["status"] = "insufficient_measured_nodes"
            out["source"] = {
                "method": "trilinear_interpolation",
                "rule": ("all eight corners (two temperature slices x four "
                         "vbus/current corners) must be finite; no "
                         "extrapolation"),
                "temperature_slices": [ti, tj],
            }
            return out
        low = corners[0]
        high = corners[1]
        e_on = ((1.0 - weight) * low[1] + weight * high[1])
        e_off = ((1.0 - weight) * low[2] + weight * high[2])
        out.update(
            e_on_uj=e_on, e_off_uj=e_off, status="interpolated",
            method="trilinear_interpolation",
            source={
                "method": "trilinear_interpolation",
                "rule": ("bilinear within each measured temperature slice, "
                         "then linear in temperature; no extrapolation"),
                "temperature_slices": [ti, tj],
                "temperature_weight": weight,
                "corners": out["indices"],
            },
        )
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
                  f_sw_hz: float | None = None,
                  temperature_c: float | None = None) -> dict[str, Any]:
        """Conduction and switching losses computed separately from the LUT.

        No efficiency is produced here.  PLECS P-TC states that thermal losses
        are not the electrical power consumed by the device; ``note`` carries
        that caveat into every caller and into the JSON record.
        """
        energy = self.energy_at(vbus_v, current_a,
                                temperature_c=temperature_c)
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
                      f_sw_hz: float | None = None,
                      temperature_c: float | None = None) -> dict[str, Any]:
        """Efficiency only for an explicit *useful output* power.

        ``p_out_w`` must be the converter's useful output power supplied by
        the caller.  Vbus*Iload is bus input power, so this method never
        derives p_out from it; with no p_out it returns NaN efficiency and
        status ``output_power_required`` together with the separately
        computed loss terms.  Losses are thermal/offline data (PLECS P-TC),
        not electrical power consumed by the emitted switch.
        """
        losses = self.losses_at(vbus_v, current_a, f_sw_hz=f_sw_hz,
                                temperature_c=temperature_c)
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

def _r_on_by_vgs_from_config(
    vendor: str, config: Mapping[str, Any], currents: tuple[float, ...],
) -> tuple[dict[float, tuple[float, ...]] | None, dict[str, Any]]:
    """Parse an optional measured Curve-3 Rds(on) family from config.

    Accepted shapes (units are ohm, aligned with ``current_points``)::

        r_on_by_vgs:
          vgs_points: [5.0, 10.0]
          r_on_ohms: [[...], [...]]

        r_on_by_vgs:
          5.0: [...]
          10.0: [...]

    The table is a measured family: each row is the Curve-3 Rds(on) at one
    gate drive for every grid current.  Nothing is interpolated or invented
    here; the emitter selects the nearest measured gate-drive row.
    """
    raw = config.get("r_on_by_vgs")
    meta: dict[str, Any] = {
        "kind": "absent",
        "source": f"config/inputs/{vendor}.yaml#r_on_by_vgs",
        "unit": "Ohm",
        "rule": ("no measured Rds(on)-vs-Vgs family in config; the emitter "
                 "keeps the median/current-point Curve-3 selection"),
    }
    if raw is None:
        return None, meta
    if not isinstance(raw, Mapping):
        raise ValueError("config r_on_by_vgs must be a mapping")
    if "r_on_ohms" in raw:
        rows = raw["r_on_ohms"]
        gates = raw.get("vgs_points")
        if gates is None:
            raise ValueError("r_on_by_vgs.vgs_points is required with r_on_ohms")
        if not isinstance(rows, (list, tuple)) or not isinstance(
                gates, (list, tuple)):
            raise ValueError("r_on_by_vgs rows and vgs_points must be sequences")
        if len(rows) != len(gates):
            raise ValueError(
                "r_on_by_vgs r_on_ohms rows must match vgs_points length")
        pairs = list(zip(gates, rows))
        shape = "vgs_points/r_on_ohms matrix"
    else:
        pairs = []
        for key, row in raw.items():
            try:
                gate = float(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "r_on_by_vgs keys must be numeric gate drives (V), or "
                    "use vgs_points/r_on_ohms") from exc
            pairs.append((gate, row))
        shape = "gate-drive -> row mapping"
    table: dict[float, tuple[float, ...]] = {}
    for gate, row in pairs:
        gate_value = float(gate)
        if not math.isfinite(gate_value):
            raise ValueError("r_on_by_vgs gate drives must be finite")
        values = tuple(_finite_or_nan(v) for v in row)
        if len(values) != len(currents):
            raise ValueError(
                "r_on_by_vgs rows must align with the LUT current_points")
        table[gate_value] = values
    meta.update({
        "kind": "measured",
        "shape": shape,
        "vgs_points": sorted(table),
        "current_points": list(currents),
        "rule": ("measured Curve-3 Rds(on) family; the emitter selects the "
                 "nearest measured gate-drive row to the actual vgs_on "
                 "(exact match preferred, no interpolation)"),
        "finite_rows": {str(gate): all(math.isfinite(v) for v in row)
                        for gate, row in sorted(table.items())},
    })
    return table, meta


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
            "temperature": ("caller-supplied GridPoint.temperature_c values "
                            "(not config-derived)"),
            "vgs_on": "dpt.vgs_on (DPT gate drive)",
            "source": "explicit_points",
            "kind": "project_decision_explicit_points",
            "rationale": "explicit points override the config-derived grid; "
                         "config derivation rules are not republished",
        }

    sweeps = bench.run_golden_sweep(
        vendor, config=cfg, points=grid, simulate=simulate,
        reproduce_recorded=reproduce_recorded)

    cond = cfg["conditions"]
    config_temperature = float(cond["temperature"])

    def _point_temperature(point: Any) -> float:
        try:
            raw = float(getattr(point, "temperature_c", NAN))
        except (TypeError, ValueError):
            raw = NAN
        # A caller-built GridPoint may omit the opt-in temperature; use the
        # config condition rather than passing NaN into a simulator or axis.
        return raw if math.isfinite(raw) else config_temperature

    current_axis = tuple(sorted({float(p.current_a) for p in grid}))
    vbus_axis = tuple(sorted({float(p.vbus_v) for p in grid}))
    temperature_axis = tuple(sorted({_point_temperature(p) for p in grid}))
    multi_temperature = len(temperature_axis) > 1
    i_index = {value: k for k, value in enumerate(current_axis)}
    v_index = {value: k for k, value in enumerate(vbus_axis)}
    e_on = {t: [[NAN] * len(current_axis) for _ in vbus_axis]
            for t in temperature_axis}
    e_off = {t: [[NAN] * len(current_axis) for _ in vbus_axis]
             for t in temperature_axis}

    node_provenance: dict[str, Any] = {}
    gaps: list[dict[str, Any]] = []
    for result in sweeps:
        temperature = _point_temperature(result.point)
        key = (_node_key_with_temperature(result.point.vbus_v,
                                          result.point.current_a, temperature)
               if multi_temperature else
               _node_key(result.point.vbus_v, result.point.current_a))
        node_provenance[key] = result.as_dict()
        iv = v_index[float(result.point.vbus_v)]
        ic = i_index[float(result.point.current_a)]
        e_on[temperature][iv][ic] = float(result.e_on_uj)
        e_off[temperature][iv][ic] = float(result.e_off_uj)
        if result.status != "measured":
            gaps.append({
                "point": result.point.as_dict(), "metric": "e_on_uj,e_off_uj",
                "status": result.status, "reason": result.reason,
                "source": result.source, "guard": result.guard,
            })

    reference_temperature = (config_temperature
                             if config_temperature in temperature_axis
                             else temperature_axis[0])
    e_on_reference = e_on[reference_temperature]
    e_off_reference = e_off[reference_temperature]
    e_on_by_temperature = (
        tuple(tuple(tuple(row) for row in e_on[t])
              for t in temperature_axis) if multi_temperature else None)
    e_off_by_temperature = (
        tuple(tuple(tuple(row) for row in e_off[t])
              for t in temperature_axis) if multi_temperature else None)

    r_on, r_meta = _r_on_from_curve(vendor, cfg, current_axis)
    for current, resistance in zip(current_axis, r_on):
        if not math.isfinite(resistance):
            gaps.append({"point": {"current_a": current}, "metric": "r_on_ohms",
                         "status": "outside_measured_curve_range",
                         "reason": "Curve-3 sweep does not cover this grid current",
                         "source": r_meta["source_file"]["file"]})
    r_on_by_vgs, r_on_by_vgs_meta = _r_on_by_vgs_from_config(
        vendor, cfg, current_axis)
    for gate, row in (r_on_by_vgs or {}).items():
        for current, resistance in zip(current_axis, row):
            if not math.isfinite(resistance):
                gaps.append({
                    "point": {"current_a": current, "vgs_on_v": gate},
                    "metric": "r_on_by_vgs", "status": "unmeasured_point",
                    "reason": ("measured Rds(on)-vs-Vgs family has no finite "
                               "value at this gate drive/current"),
                    "source": r_on_by_vgs_meta["source"],
                })
    raw_vt = cfg.get("vt_switch_v")
    vt_switch_v = NAN
    vt_meta: dict[str, Any] = {
        "kind": "absent",
        "value_v": NAN,
        "source": f"config/inputs/{vendor}.yaml#vt_switch_v",
        "rule": ("no config switching threshold; the emitter keeps the "
                 "Curve-7 JEDEC Vth and labels source=curve7_jedec_vth"),
    }
    if raw_vt is not None:
        candidate = _finite_or_nan(raw_vt)
        if not math.isfinite(candidate) or candidate <= 0.0:
            raise ValueError(
                "config vt_switch_v must be a finite positive voltage")
        vt_switch_v = candidate
        vt_meta.update({
            "kind": "config_derived",
            "value_v": candidate,
            "rule": ("config-provided switching threshold used by the "
                     "emitted ideal switch; no Curve-7 fallback"),
        })
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
    if vt_meta["kind"] == "absent":
        vt_meta.update({
            "kind": "curve7_jedec_vth",
            "value_v": vth,
            "source": "config/outputs/curve7_threshold_voltage#extracted.vth_v",
            "rule": ("no config switching threshold; the emitter uses the "
                     "Curve-7 JEDEC Vth and labels source=curve7_jedec_vth"),
        })

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

    capability = _interpolation_capability(
        e_on_reference, e_off_reference,
        len(vbus_axis), len(current_axis))
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
            "temperature_points": list(temperature_axis),
            "reference_temperature_c": reference_temperature,
            "source": rules.get("source", "config_derived_default"),
            "rules": rules,
            "kind": "project_decision",
        },
        "r_on": r_meta,
        "r_on_by_vgs": r_on_by_vgs_meta,
        "vth": vth_meta,
        "vt_switch": vt_meta,
        "coss_f": coss_meta,
        "r_off_ohms": r_off_meta,
        "nodes": node_provenance,
        "gaps": gaps,
        "golden_evaluable": all(r.status == "measured" for r in sweeps),
        "dpt_crossing_evaluable": (
            node_provenance.get(
                _node_key_with_temperature(
                    float(cfg["dpt"]["vbus"]),
                    float(cfg["dpt"]["i_load_target"]),
                    reference_temperature)
                if multi_temperature else
                _node_key(float(cfg["dpt"]["vbus"]),
                          float(cfg["dpt"]["i_load_target"])), {})
            .get("status") == "measured"),
        "waveform_gate": "retired_by_design",
        "interpolation": capability,
        "temperature": {
            "points": list(temperature_axis),
            "reference_temperature_c": reference_temperature,
            "has_3d_tables": multi_temperature,
            "rule": ("opt-in temperature axis from config loss_grid; trilinear "
                     "interpolation only with all eight corners finite, no "
                     "extrapolation" if multi_temperature else
                     "single reference temperature; no temperature axis"),
        },
        "data_limitations": _data_limitations(
            reference_temperature, e_on_reference, e_off_reference,
            len(vbus_axis), len(current_axis),
            temperature_points=temperature_axis,
            has_temperature_tables=multi_temperature),
        "thermal_loss_note": THERMAL_LOSS_NOTE,
        "device_value_guard": {
            "vth_finite": math.isfinite(vth),
            "vt_switch_finite": math.isfinite(vt_switch_v),
            "r_on_all_finite": all(math.isfinite(float(r)) for r in r_on),
            "rule": ("emitter must refuse to emit .MODEL when vth or the "
                     "selected r_on is NaN; never write NaN into SPICE. A "
                     "missing vt_switch_v falls back to Curve-7 Vth."),
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
            {"name": "temperature_axis", "kind": "project_decision",
             "rule": rules.get(
                 "temperature", "conditions.temperature"),
             "temperature_points": list(temperature_axis),
             "reference_temperature_c": reference_temperature,
             "interpolation": ("trilinear only with all eight corners finite; "
                               "no extrapolation" if multi_temperature else
                               "single measured temperature slice")},
            {"name": "r_on_selection", "kind": "measured",
             "rule": r_on_by_vgs_meta["rule"],
             "table_available": r_on_by_vgs is not None,
             "vgs_points": r_on_by_vgs_meta.get("vgs_points"),
             "source": r_on_by_vgs_meta["source"]},
            {"name": "vt_switch_selection", "kind": vt_meta["kind"],
             "rule": vt_meta["rule"],
             "value_v": vt_meta["value_v"],
             "source": vt_meta["source"]},
        ],
    }
    return LossLUT(
        vendor=vendor, temperature_c=reference_temperature,
        vbus_points=vbus_axis, current_points=current_axis,
        e_on_uj=tuple(tuple(row) for row in e_on_reference),
        e_off_uj=tuple(tuple(row) for row in e_off_reference),
        r_on_ohms=tuple(r_on), vth_v=vth,
        vgs_on_v=float(cfg["dpt"]["vgs_on"]),
        coss_f=coss_f, r_off_ohms=r_off_ohms,
        temperature_points=temperature_axis,
        e_on_uj_by_temperature=e_on_by_temperature,
        e_off_uj_by_temperature=e_off_by_temperature,
        vt_switch_v=vt_switch_v, r_on_by_vgs=r_on_by_vgs,
        source=source,
    )
