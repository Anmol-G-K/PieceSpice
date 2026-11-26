"""The solver-facing ROM: one ``PWLModelSpec``, two emitters.

``models.netlist_builder`` turns a fitted spec into a *SPICE* subcircuit --
smooth ``tanh`` knee, ``Q={..}`` expressions -- because that is what a general
SPICE engine can consume.  This module turns the same spec into the *solver's*
form: tables and planes, no expressions.  The model error is therefore shared
between the two arms by construction, and any difference between them is engine
error rather than a different fit.

Node structure mirrors the emitted subcircuit one-for-one (``RG``/``RD``/``RS``
parasitics, channel across ``di``/``ss`` controlled by ``gi``/``ss``, the three
charge elements, the body diode) so an element-by-element comparison is
possible.  What changes is only how each element is described:

============================  ==========================  ==================
emitted .lib                  solver ROM                  why
============================  ==========================  ==================
``G_CH ... VALUE={..tanh..}``  ``PWL_CHANNEL`` 2-D table  a plane per triangle
                                                        instead of a Jacobian
                                                        per step
``C_QOSS ... Q={..}``          ``PWL_CAP`` Q(V) table     charge is the state,
                                                        not an expression
``D_BODY ss di D_BD_x``        PWL diode from Curve 6     region set, no
                                                        Newton on exp()
============================  ==========================  ==================
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np
from dataclasses import dataclass
from typing import Any

from engine.ir.circuit import (
    AnalysisDirective,
    Circuit,
    Element,
    ElementKind,
    NodeIndex,
)
from engine.solver.channel import ChannelTable
from engine.solver.device import ChargeTable, PwlTable

#: Version of the solver ROM artifact.  Bumped when the payload shape changes.
SCHEMA = "piecespice.solver_rom/1.0"

#: Parasitic defaults taken from ``models.netlist_builder`` so the solver ROM
#: reproduces the emitted subcircuit rather than inventing its own.
_DEFAULT_R_GATE = 1.5
_DEFAULT_R_DRAIN = 1e-4
_DEFAULT_R_SOURCE = 1e-4

#: Charge tables are zero-anchored in the emitted ROMs; a spec whose Q(0) is not
#: zero would inject a constant charge offset, so it is refused rather than
#: silently shifted.
_Q_ZERO_TOL = 1e-15


#: Grid the ANN channel is sampled onto.  Both axes are dense where the device
#: actually operates: Vgs from zero up to the top of the fitted range, Vds
#: quadratically spaced so the first cell is a fraction of a volt wide instead
#: of the whole knee.  This is a property of the emitter, not of a vendor.
ANN_VGS_ROWS = 17
ANN_VDS_POINTS = 151


@dataclass(frozen=True, slots=True)
class SolverRom:
    """A fitted device in the form the custom solver consumes."""

    device: str
    channel: ChannelTable
    qoss: ChargeTable
    body_diode: PwlTable
    c_iss_f: float
    c_rss_f: float
    r_gate: float = _DEFAULT_R_GATE
    r_drain: float = _DEFAULT_R_DRAIN
    r_source: float = _DEFAULT_R_SOURCE
    vth: float = 0.0
    rds_on: float = 0.0
    #: Which model basis the tables were built from.  The solver consumes
    #: tables, so the basis is a free choice; leaving it implicit is what made
    #: every early measurement Tier-2 without saying so.
    basis: str = "tier2"

    # -- construction ------------------------------------------------------

    @classmethod
    def from_spec(cls, spec: Any) -> SolverRom:
        """Build from a :class:`common.types.PWLModelSpec`."""
        meta: Mapping[str, Any] = dict(getattr(spec, "metadata", {}) or {})
        qoss = _charge_table(spec.q_oss_vs_v, spec.q_oss_charge_c, "Qoss")
        body = PwlTable.from_pairs(zip(spec.body_diode_vsd, spec.body_diode_isd))
        vgs, rows = _channel_grid_with_anchors(spec)
        return cls(
            device=str(spec.device_name),
            channel=ChannelTable.build(vgs, spec.breakpoints_vds, rows),
            qoss=qoss,
            body_diode=body,
            c_iss_f=float(meta.get("c_iss_lin", 0.0)),
            c_rss_f=float(meta.get("c_rss_lin", 0.0)),
            r_gate=float(meta.get("r_gate", _DEFAULT_R_GATE)),
            r_drain=float(meta.get("r_drain", _DEFAULT_R_DRAIN)),
            r_source=float(meta.get("r_source", _DEFAULT_R_SOURCE)),
            vth=float(spec.v_th),
            rds_on=float(spec.rds_on),
        )

    @classmethod
    def from_ann_fit(cls, vendor: str, *, spec: Any | None = None,
                     vgs_rows: int = ANN_VGS_ROWS,
                     vds_points: int = ANN_VDS_POINTS,
                     vds_max: float | None = None) -> "SolverRom":
        """Build the tables from the ANN basis instead of the Tier-2 spec.

        The ANN surface is smooth, which is why it is not emitted as PWL for
        SPICE.  The solver does not care: it consumes a table, so the fitted
        network is **evaluated** on a grid and tabulated.  That is table
        construction, not SPICE-expression evaluation, so the solver's
        no-expressions rule still holds.

        Charge and the body diode stay on the spec's tables: the ANN charge fit
        is a separate model and mixing the two without measuring the mixture
        would blur which one the error belongs to.
        """
        from engine.solver.channel import ChannelTable
        from pipeline.ann_fit import ann_fit_for_vendor
        from pipeline.fitting import build_pwl_spec

        fit_payload = ann_fit_for_vendor(vendor)
        channel_fit = fit_payload["channel_fit"]
        spec = spec if spec is not None else build_pwl_spec(vendor)

        train = getattr(channel_fit, "train_range", {}) or {}
        vgs_hi = float(train.get("vgs", [2.5, 6.0])[1])
        # The table must cover the range the device is actually used at, not
        # only the range it was fitted on.  Capping it at the training maximum
        # makes the solver clamp where the emitted expression keeps
        # evaluating -- that is a table-truncation error, not a model one.
        train_vds = float(train.get("vds", [0.0, spec.breakpoints_vds[-1]])[1])
        vds_hi = max(train_vds, float(vds_max) if vds_max else train_vds)

        # Vgs: zero, the fitted threshold, then evenly spaced to the top.  Two
        # anchors below the fitted range are what let the device block.
        vgs = [0.0]
        v_th = float(channel_fit.v_th)
        if 0.0 < v_th < vgs_hi:
            vgs.append(v_th)
        # Spread the rows from the threshold upward, not from the bottom of the
        # fitted range: the gap between Vth and the lowest driven Vgs is where
        # the device turns on, and leaving it as one cell makes the turn-on
        # plane a secant across a volt.  The network is smooth, so evaluating
        # it below its training range is interpolation of its own form, not
        # invention of data.
        vgs.extend(np.linspace(v_th, vgs_hi, vgs_rows).tolist())
        vgs = sorted({round(v, 9) for v in vgs if 0.0 <= v <= vgs_hi})

        # Vds: quadratic spacing, so the first cells resolve the knee.
        fractions = (np.arange(vds_points) / (vds_points - 1.0)) ** 2
        vds = (fractions * vds_hi).tolist()
        vds[0] = 0.0
        vds = sorted({round(v, 12) for v in vds})

        rows = []
        for value in vgs:
            current, _, _ = channel_fit.evaluate(
                np.full(len(vds), value), np.asarray(vds))
            rows.append(np.maximum(np.asarray(current, dtype=float), 0.0)
                        .tolist())

        base = cls.from_spec(spec)
        return cls(
            device=base.device,
            channel=ChannelTable.build(vgs, vds, rows),
            qoss=base.qoss,
            body_diode=base.body_diode,
            c_iss_f=base.c_iss_f,
            c_rss_f=base.c_rss_f,
            r_gate=base.r_gate,
            r_drain=base.r_drain,
            r_source=base.r_source,
            vth=base.vth,
            rds_on=base.rds_on,
            basis="ann",
        )

    # -- the circuit -------------------------------------------------------

    def to_circuit(self, *, title: str | None = None) -> Circuit:
        """Flatten the ROM into a circuit the solver can step.

        Node names match the emitted subcircuit's internals, so a trace named
        ``V(gi)`` means the same thing in both arms.
        """
        name = self.device
        elements = (
            Element(name="RG", kind=ElementKind.RESISTOR, nodes=("g", "gi"),
                    value=self.r_gate),
            Element(name="RD", kind=ElementKind.RESISTOR, nodes=("d", "di"),
                    value=self.r_drain),
            Element(name="RS", kind=ElementKind.RESISTOR, nodes=("s", "ss"),
                    value=self.r_source),
            Element(name=f"CH_{name}", kind=ElementKind.PWL_CHANNEL,
                    nodes=("di", "ss", "gi"), data=self.channel),
            Element(name=f"C_QOSS_{name}", kind=ElementKind.PWL_CAP,
                    nodes=("di", "ss"), data=self.qoss),
            Element(name=f"D_BODY_{name}", kind=ElementKind.DIODE,
                    nodes=("ss", "di"), data=self.body_diode),
        )
        extra = []
        if self.c_iss_f > 0.0:
            extra.append(Element(name=f"C_ISS_{name}",
                                 kind=ElementKind.CAPACITOR,
                                 nodes=("gi", "ss"), value=self.c_iss_f))
        if self.c_rss_f > 0.0:
            extra.append(Element(name=f"C_RSS_{name}",
                                 kind=ElementKind.CAPACITOR,
                                 nodes=("gi", "di"), value=self.c_rss_f))
        all_elements = elements + tuple(extra)
        nodes: list[str] = []
        for element in all_elements:
            nodes.extend(element.nodes)
        return Circuit(
            title=title or f"ROM {name}",
            elements=all_elements,
            models={},
            node_index=NodeIndex.build(nodes),
            directives=(),
        )

    def subcircuit(self, *, name: str | None = None) -> tuple[tuple[str, ...], Circuit]:
        """Pin order and circuit for one instance in a bench."""
        return ("d", "g", "s"), self.to_circuit(title=f"ROM {self.device}")

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "device": self.device,
            "basis": self.basis,
            "channel": {
                "vgs": list(self.channel.vgs),
                "vds": list(self.channel.vds),
                "id": [list(row) for row in self.channel.id],
            },
            "qoss": {
                "v": list(self.qoss.table.x),
                "q": list(self.qoss.table.y),
            },
            "body_diode": {
                "vsd": list(self.body_diode.x),
                "isd": list(self.body_diode.y),
            },
            "capacitances": {"c_iss_f": self.c_iss_f, "c_rss_f": self.c_rss_f},
            "parasitics": {"r_gate": self.r_gate, "r_drain": self.r_drain,
                           "r_source": self.r_source},
            "scalars": {"v_th": self.vth, "rds_on": self.rds_on},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SolverRom:
        schema = payload.get("schema")
        if schema != SCHEMA:
            raise ValueError(
                f"unsupported solver ROM schema {schema!r} (expected {SCHEMA!r})")
        channel = payload["channel"]
        qoss = payload["qoss"]
        body = payload["body_diode"]
        caps = payload.get("capacitances", {})
        par = payload.get("parasitics", {})
        scal = payload.get("scalars", {})
        return cls(
            device=str(payload["device"]),
            basis=str(payload.get("basis", "tier2")),
            channel=ChannelTable.build(channel["vgs"], channel["vds"],
                                       channel["id"]),
            qoss=ChargeTable(PwlTable(tuple(qoss["v"]), tuple(qoss["q"]))),
            body_diode=PwlTable(tuple(body["vsd"]), tuple(body["isd"])),
            c_iss_f=float(caps.get("c_iss_f", 0.0)),
            c_rss_f=float(caps.get("c_rss_f", 0.0)),
            r_gate=float(par.get("r_gate", _DEFAULT_R_GATE)),
            r_drain=float(par.get("r_drain", _DEFAULT_R_DRAIN)),
            r_source=float(par.get("r_source", _DEFAULT_R_SOURCE)),
            vth=float(scal.get("v_th", 0.0)),
            rds_on=float(scal.get("rds_on", 0.0)),
        )


def _channel_grid_with_anchors(spec: Any) -> tuple[list[float], list[list[float]]]:
    """Prepend the two zero-current anchors the emitted transfer curve carries.

    The extracted Curve-1 families start at the lowest driven Vgs (2.5 V on
    EPC2216), but the device must block below threshold.  Without an anchor the
    first table cell is the 2.5 V curve and its plane extrapolates *downwards*
    into a large negative current at Vgs = 0 -- which is exactly how a ROM fails
    to block.  The emitted subcircuit solves this with an origin ramp plus a
    threshold anchor, so the solver table carries the same two rows:
    ``Vgs = 0`` and ``Vgs = Vth``, both at zero current.
    """
    vgs = [float(v) for v in spec.vgs_values]
    rows = [[float(i) for i in row] for row in spec.breakpoints_id]
    width = len(spec.breakpoints_vds)
    anchors = [v for v in (0.0, float(spec.v_th)) if v < vgs[0]]
    for value in sorted(anchors, reverse=True):
        vgs.insert(0, value)
        rows.insert(0, [0.0] * width)
    return vgs, rows


def _charge_table(voltages: Sequence[float], charges: Sequence[float],
                  label: str) -> ChargeTable:
    if len(voltages) < 2 or len(voltages) != len(charges):
        raise ValueError(f"{label}: need matching V/Q tables of length >= 2")
    pairs = list(zip(voltages, charges))
    table = PwlTable.from_pairs(pairs)
    if abs(table.y[0]) > _Q_ZERO_TOL:
        raise ValueError(
            f"{label}: Q(0) must be 0, got {table.y[0]:g} C; an absolute charge "
            "offset would be integrated as loss")
    return ChargeTable(table)


def circuit_with_rom(rom: SolverRom, *, end_time: float | None = None,
                     max_step: float | None = None,
                     title: str | None = None) -> Circuit:
    """The ROM plus an explicit ``.tran`` card, ready for ``solve_transient``."""
    base = rom.to_circuit(title=title)
    directive = AnalysisDirective(kind="tran", raw=".tran (synthetic)",
                                  end_time=end_time, start_time=0.0,
                                  max_step=max_step)
    return Circuit(title=base.title, elements=base.elements, models=base.models,
                   node_index=base.node_index, directives=(directive,))
