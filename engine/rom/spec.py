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
import math

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from engine.ir.circuit import (
    AnalysisDirective,
    Circuit,
    DeviceModel,
    Element,
    ElementKind,
    NodeIndex,
)
from engine.solver.channel import ChannelTable
from engine.solver.device import ChargeTable, PwlTable

#: Version of the solver ROM artifact.  Bumped when the payload shape changes.
SCHEMA = "piecespice.solver_rom/1.0"

ROOT = Path(__file__).resolve().parent.parent.parent

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
#: Rows placed below the fitted threshold, so the turn-on region is resolved
#: by interpolation rather than by one chord across it.
ANN_VGS_BELOW_THRESHOLD = 9
#: Half-width, in rows, of the graded band around the threshold, and the row
#: spacing there.
ANN_KNEE_ROWS = 8
ANN_KNEE_STEP_V = 0.05
#: Volts per row for the charge tables; the row count scales with the span so
#: one physical resolution is held however far the device blocks.
CHARGE_STEP_V = 0.05
ANN_VDS_POINTS = 151


def _blocking_voltage(vendor: str) -> float | None:
    """The voltage the DPT bench asks *vendor* to block, from its config."""
    import yaml

    path = ROOT / "config" / "inputs" / f"{vendor}.yaml"
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    bench = (data.get("dpt") or {}).get("vbus") if isinstance(data, dict) else None
    if bench is None:
        return None
    try:
        return float(bench)
    except (TypeError, ValueError):
        return None


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
    #: Gate charge tables.  ``None`` means the linear equivalents above are
    #: used; a table means the basis supplied a nonlinear gate charge and the
    #: solver should carry it, because switching edges are charge-dominated.
    qiss: ChargeTable | None = None
    qrss: ChargeTable | None = None
    #: Node pair the rss charge is defined across.  The Tier-2 spec uses
    #: gate->drain (Vgd); the Tier-4 ANN emits drain->gate (Vdg).  Carrying the
    #: wrong orientation flips the sign of the whole gate-drain charge and no
    #: amount of grid or step refinement recovers it.
    qrss_nodes: tuple[str, str] = ("gi", "di")
    #: Gate-source bleeder resistance, emitted by the Tier-4 ANN as `RGS`.
    r_gs: float | None = None
    #: Fold series parasitics at or below this value into a direct connection.
    #: Table-model practice is terminal-only: a 0.1 mohm source/drain resistor
    #: carries no physics a terminal table needs, but it does add two unknowns
    #: and two 1e4 S entries that dominate the matrix condition number.
    merge_below_ohm: float | None = 1e-3
    #: Parametric body diode ``(IS, N, RS)`` as the emitter declares it.  The
    #: emitted `.MODEL` is a Shockley fit of Curve 6, not the sampled curve, and
    #: the two conduct differently through the clamp -- which is where a turn-off
    #: spends its energy.
    body_diode_params: tuple[float, float, float] | None = None

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
        # Resolve the turn-on region between 0 V and the threshold.  Without
        # these rows the first cell is one chord across the whole sub-threshold
        # span -- and for a device whose threshold is 4.35 V that chord carries
        # about an amp at Vgs = 3 V where the fit says zero.  It is a *value*
        # error, so no time step removes it: IRFS7730PBF showed exactly that
        # signature, sitting at 2.5-3 % through a 20x refinement of both engines
        # while the other three converged below 1.4 %.  The rows are evenly
        # spaced at the resolution the fitted region already uses above v_th.
        for k in range(1, ANN_VGS_BELOW_THRESHOLD):
            below = v_th * k / (ANN_VGS_BELOW_THRESHOLD + 1.0)
            if below > 0.0:
                vgs.append(below)
        # The knee itself is exponential: Id turns over four decades within a
        # few hundred millivolts of v_th, so a linear chord across it misplaces
        # the turn-on instant by a large relative error even when the cells are
        # only 0.35 V wide.  50 mV is the scale over which the exponential
        # turns, so the band is graded at that resolution on both sides of the
        # threshold.  It is a physical length, not a per-device parameter.
        for k in range(-ANN_KNEE_ROWS, ANN_KNEE_ROWS + 1):
            near = v_th + k * ANN_KNEE_STEP_V
            if 0.0 < near <= vgs_hi:
                vgs.append(near)
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

        # Charge comes from the same basis as the channel.  Keeping the
        # Tier-2 charge while taking only the ANN channel makes the ROM a
        # hybrid, and edges are charge-dominated: that hybrid is why the
        # solver's IRFS rise was 3 ns against 25.8.
        charge = fit_payload["charge_fit"]
        qoss = _resampled_charge(charge, "oss", "Qoss")
        qiss = _resampled_charge(charge, "iss", "Qiss")
        qrss = _resampled_charge(charge, "rss", "Qrss")

        base = cls.from_spec(spec)
        return cls(
            device=base.device,
            channel=ChannelTable.build(vgs, vds, rows),
            qoss=qoss,
            qiss=qiss,
            qrss=qrss,
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

    @classmethod
    def from_t4_ann(cls, vendor: str, *, spec: Any | None = None,
                    calibration: Mapping[str, float] | None = None,
                    **kwargs: Any) -> "SolverRom":
        """ANN channel and charge, with the Tier-4 ANN's dynamic calibration.

        Tier-4 ANN is the same channel fit as the plain ANN basis; what it adds
        is a small set of factors (``k_cgs``, ``k_cdg``, ``k_cds``, ``k_rg``)
        that the Tier-4 pipeline fits against the golden double pulse's edges.
        Switching edges are charge- and gate-resistance-dominated, so carrying
        those factors is what makes the solver reproduce the emitted waveform
        instead of merely the static curves.

        The factors are read from the recorded calibration, never re-fitted
        here: this builder reproduces an existing model, it does not tune one.
        """
        import json

        if spec is None:
            from pipeline.fitting import build_pwl_spec

            spec = build_pwl_spec(vendor)
        base = cls.from_ann_fit(vendor, spec=spec, **kwargs)
        factors = dict(calibration or {})
        if not factors:
            record = (ROOT / "reports" / "comparison" / vendor
                      / "tier4_validation_ann.json")
            if not record.is_file():
                raise FileNotFoundError(
                    f"{record} is missing; run the Tier-4 ANN build for {vendor}")
            factors = json.loads(record.read_text(encoding="utf-8")).get(
                "dynamic_calibration", {}) or {}

        # The Tier-4 ANN does not emit the fit's charge tables -- it emits a
        # C-inf parameterisation of them (a constant Cgs, a rational/tanh Qoss
        # in V(di,ss), a rational/tanh Qgd in V(di,gi)) plus a gate bleeder.
        # Reproducing the *tables* therefore reproduces a different model; the
        # charge is rebuilt from the emitted parameterisation instead.
        cinf = (getattr(base, "_charge_provenance", {}) or {}).get("cinf_caps", {})
        if not cinf:
            from pipeline.ann_fit import ann_fit_for_vendor

            charge_fit = ann_fit_for_vendor(vendor)["charge_fit"]
            cinf = (getattr(charge_fit, "provenance", {}) or {}).get(
                "cinf_caps", {}) or {}
        if not cinf:
            raise ValueError(f"{vendor}: the ANN charge fit carries no C-inf spec")

        # Body diode: the emitter declares a Shockley `.MODEL`, so the solver
        # must take the fitted parameters, not the sampled Curve-6 points.
        from models.netlist_builder import fit_body_diode

        diode = fit_body_diode(spec.body_diode_vsd, spec.body_diode_isd,
                               rds_on=spec.rds_on)

        k_cds = float(factors.get("k_cds", 1.0))
        k_cdg = float(factors.get("k_cdg", 1.0))
        k_cgs = float(factors.get("k_cgs", 1.0))
        # The charge must be integrated to the voltage the device actually
        # blocks.  Tied to the channel table's span it truncated at 10 V and
        # clamped there: IRFS7730PBF carried 10.99 nC at the bench's 38 V where
        # the emitted expression carries 23.0 nC, and its off-state ringing came
        # out at 1.14 A against the reference's 2.63 A -- 59 % of that device's
        # residual, and a *value* error, which is why it never converged with
        # the time step.
        #
        # The span cannot simply be stretched: the tables are uniformly spaced,
        # so covering 38 V with the default 201 rows gives 0.19 V per row where
        # the 10 V table gave 0.05 V, and measuring that alone made IRFS worse
        # (2.49 -> 3.84 %).  The row count therefore scales with the span to
        # hold one physical resolution, the same 50 mV the knee band uses.
        charge_top = max(base.channel.vds[-1], _blocking_voltage(vendor) or 0.0,
                         1e-9)
        charge_points = max(CHARGE_RESAMPLE_POINTS,
                            int(np.ceil(charge_top / CHARGE_STEP_V)) + 1)
        vds_top = max(base.channel.vds[-1], 1e-9)
        qoss = _cinf_charge(cinf["cds"], charge_top, k_cds, "Qoss",
                            points=charge_points)
        # Gate-drain charge is a function of V(di,gi), evaluated on the same span.
        qrss = _cinf_charge(cinf["cdg"], charge_top, k_cdg, "Qrss",
                            points=charge_points,
                            excess=float(cinf.get("cdg_neg_excess_f") or 0.0))

        return cls(
            device=base.device,
            channel=base.channel,
            qoss=qoss,
            body_diode=base.body_diode,
            c_iss_f=float(cinf["cgs_f"]) * k_cgs,
            c_rss_f=0.0,
            # The Tier-4 ANN takes its gate/source parasitics from the C-inf
            # spec, not from the Tier-2 metadata: Tier-2's 1.5 ohm against the
            # emitted 0.58 ohm is a 3x error in the gate path.
            r_gate=float(cinf.get("r_gate_ohm") or base.r_gate)
            * float(factors.get("k_rg", 1.0)),
            r_drain=base.r_drain,
            r_source=float(cinf.get("r_source_ohm") or base.r_source),
            vth=base.vth,
            rds_on=base.rds_on,
            basis="t4ann",
            qrss=qrss,
            qrss_nodes=("di", "gi"),
            r_gs=1e7,
            body_diode_params=(float(diode.isat), float(diode.n),
                               float(diode.rs)),
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
                    nodes=("ss", "di"), data=self.body_diode,
                    model=(None if self.body_diode_params is None
                           else f"D_BD_{name}")),
        )
        extra = []
        if self.qiss is not None:
            extra.append(Element(name=f"C_ISS_{name}",
                                 kind=ElementKind.PWL_CAP,
                                 nodes=("gi", "ss"), data=self.qiss))
        elif self.c_iss_f > 0.0:
            extra.append(Element(name=f"C_ISS_{name}",
                                 kind=ElementKind.CAPACITOR,
                                 nodes=("gi", "ss"), value=self.c_iss_f))
        if self.qrss is not None:
            extra.append(Element(name=f"C_RSS_{name}",
                                 kind=ElementKind.PWL_CAP,
                                 nodes=self.qrss_nodes, data=self.qrss))
        elif self.c_rss_f > 0.0:
            extra.append(Element(name=f"C_RSS_{name}",
                                 kind=ElementKind.CAPACITOR,
                                 nodes=self.qrss_nodes, value=self.c_rss_f))
        if self.r_gs is not None and self.r_gs > 0.0:
            extra.append(Element(name="RGS", kind=ElementKind.RESISTOR,
                                 nodes=("gi", "ss"), value=self.r_gs))
        all_elements = elements + tuple(extra)
        models: dict[str, DeviceModel] = {}
        if self.body_diode_params is not None:
            isat, n, rs = self.body_diode_params
            models[f"D_BD_{name}"] = DeviceModel(
                name=f"D_BD_{name}", family="D",
                params={"IS": repr(float(isat)), "N": repr(float(n)),
                        "RS": repr(float(rs))})
        all_elements = _merge_series(all_elements, self.merge_below_ohm)
        nodes: list[str] = []
        for element in all_elements:
            nodes.extend(element.nodes)
        return Circuit(
            title=title or f"ROM {name}",
            elements=all_elements,
            models=models,
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
            "qiss": (None if self.qiss is None else
                     {"v": list(self.qiss.table.x),
                      "q": list(self.qiss.table.y)}),
            "qrss": (None if self.qrss is None else
                     {"v": list(self.qrss.table.x),
                      "q": list(self.qrss.table.y)}),
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
            qiss=(None if payload.get("qiss") is None else
                  ChargeTable(PwlTable(tuple(payload["qiss"]["v"]),
                                       tuple(payload["qiss"]["q"])))),
            qrss=(None if payload.get("qrss") is None else
                  ChargeTable(PwlTable(tuple(payload["qrss"]["v"]),
                                       tuple(payload["qrss"]["q"])))),
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


#: Points the fitted charge curve is resampled onto.  The fit stores a short
#: monotone table (7-31 points) and the emitted expression interpolates it
#: smoothly; tabulating the raw points linearly is coarser than the .lib and
#: costs waveform accuracy.  Resampling the fit's own curve keeps the solver
#: table on the same curve as the emitter.
CHARGE_RESAMPLE_POINTS = 201


def _cinf_charge_at(cap_spec: Any, v: float) -> float:
    """Q(v) exactly as the emitted ``Q = {...}`` expression defines it.

    The expression is ``cmin*v + sum(dc*v/(1 + sqrt(v^2+1e-4)/vk))`` for the
    rational form and ``cmin*v + sum(dc*vk*tanh(v/vk))`` for the tanh form.
    Q is written **directly** -- it is not the integral of a C(v) -- so
    integrating a capacitance here would tabulate a different model.
    """
    if isinstance(cap_spec, dict) and cap_spec.get("kind") == "rational":
        value = float(cap_spec["cmin"]) * v
        for dc, vk in cap_spec["terms"]:
            if dc > 0.0:
                value += float(dc) * v / (1.0 + math.sqrt(v * v + 1.0e-4)
                                          / float(vk))
        return value
    cmin, terms = cap_spec
    value = float(cmin) * v
    for dc, vk in terms:
        if dc > 0.0:
            value += float(dc) * float(vk) * math.tanh(v / float(vk))
    return value


def _cinf_charge(cap_spec: Any, v_max: float, scale: float,
                 label: str, points: int = CHARGE_RESAMPLE_POINTS,
                 excess: float = 0.0) -> ChargeTable:
    """Tabulate the emitted Q(v) curve itself.

    ``excess`` is the emitter's gate-drain accumulation term for ``Vgs > Vds``
    (``0.5*C*(v - sqrt(v^2+1e-2) + 1e-1)``).  It is zero for devices whose Cgd
    does not rise below zero Vdg and several nanofarads for those where it
    does, so omitting it is a *device-dependent* error, not a small one.
    """
    grid = np.linspace(0.0, v_max, points)
    charge = np.array([_cinf_charge_at(cap_spec, float(v)) for v in grid])
    if excess:
        charge = charge + 0.5 * excess * (
            grid - np.sqrt(grid * grid + 1.0e-2) + 1.0e-1)
    charge = charge * scale
    charge = np.maximum.accumulate(np.maximum(charge, 0.0))
    charge -= charge[0]
    return ChargeTable(PwlTable(tuple(grid.tolist()), tuple(charge.tolist())))


def _resampled_charge(charge_fit: Any, name: str, label: str,
                      points: int = CHARGE_RESAMPLE_POINTS) -> ChargeTable:
    """Sample the fitted charge curve, not its control points."""
    table = charge_fit.tables[name]
    top = float(max(table["v"]))
    grid = np.linspace(0.0, top, points)
    values = np.asarray(charge_fit.evaluate(name, grid), dtype=float)
    values = np.maximum.accumulate(np.maximum(values, 0.0))   # monotone by
    values -= values[0]                                      # construction
    return ChargeTable(PwlTable(tuple(grid.tolist()), tuple(values.tolist())))


def _merge_series(elements: tuple[Element, ...],
                  threshold: float | None) -> tuple[Element, ...]:
    """Collapse resistors below *threshold* into a direct connection.

    Union-find over the element list, then rename.  The dropped resistor is
    kept nowhere: it is below the resolution any measurement in this project
    resolves, and keeping two 1e4 S conductances in a 9-node matrix costs
    conditioning -- and therefore step size -- for no physical content.
    """
    if not threshold:
        return elements
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    keep: list[Element] = []
    for element in elements:
        if (element.kind is ElementKind.RESISTOR and element.value is not None
                and element.value <= threshold):
            a, b = element.nodes
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
            continue
        keep.append(element)
    if not parent:
        return elements
    return tuple(
        Element(name=e.name, kind=e.kind,
                nodes=tuple(find(n) for n in e.nodes),
                value=e.value, waveform=e.waveform, model=e.model,
                expression=e.expression, data=e.data)
        for e in keep)


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
