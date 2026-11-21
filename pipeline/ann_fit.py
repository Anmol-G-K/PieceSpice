"""Offline ANN fit orchestration for one vendor -- task T2.

Reads only existing extracted curve YAMLs (and, when explicitly requested and
available, the already-captured golden DPT trajectory).  It fits the guarded
ANN channel and the log-scale ANN charge nets, and returns fit objects plus a
serialisable record for T4 to place in the validation JSON.  It writes no
repository artifact and runs no fidelity/speed gate.

Provenance follows spec section 5: every fitted scalar traces to a curve YAML
key or to a recorded project decision.  Curve-4 stores only cap equivalents at
one reference bias, so a full 2-D Q(Vgs,Vds) surface is not learnable from the
current extracted data; the charge fit records that limitation explicitly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from approximation.ann_charge import AnnChargeFit, fit_charge_qv
from approximation.ann_channel import AnnChannelFit, DptPoints, fit_channel_iv
from approximation.ann_residual import fit_channel_residual
from common.types import CurveData, trim_trailing_invalid
from config.yaml_io import load_input, load_output
from approximation.smoother import eps_from_grid

__all__ = ["ann_fit_for_vendor", "hybrid_ann_fit_for_vendor"]

_THRESHOLD_FACTOR = 0.5          # dI/dVds drop defining the ohmic/sat knee
_DPT_MAX_POINTS = 60             # cap on extracted high-V/I training points
_DPT_TIMEOUT_S = 30.0            # extraction guard; failure is recorded, not hidden


def _family_value(curve: CurveData) -> float:
    """CurveData family values are optional; fit families must declare one."""
    value = curve.family_value
    if value is None:
        raise ValueError(f"{curve.name}: missing family_value")
    return float(value)


def _extract(vendor: str, bench: str) -> Dict[str, Any]:
    return load_output(vendor, bench)["extracted"]


def _channel_families(vendor: str, cond: Dict[str, Any]
                      ) -> Tuple[List[CurveData], float]:
    """Curve-1 output family as regular-grid CurveData (Vds, Id per Vgs)."""
    c1 = _extract(vendor, "curve1_output_characteristics")
    prefix = "curve_id_vgs_"
    pairs = sorted((float(k[len(prefix):-2]), k) for k in c1
                   if k.startswith(prefix))
    if not pairs:
        raise RuntimeError(f"{vendor}: Curve 1 has no Vgs family")
    vmax = float(cond["Vds_sweep_max"])
    families: List[CurveData] = []
    for vgs, key in pairs:
        y = np.asarray(c1[key], dtype=float)
        x = np.linspace(0.0, vmax, y.size)
        x, y = trim_trailing_invalid(x, y)
        families.append(CurveData(
            name=f"curve1_vgs_{vgs:g}",
            x_name="Vds", x_unit="V", y_name="Id", y_unit="A",
            x_data=[float(v) for v in x], y_data=[float(v) for v in y],
            family_param="Vgs", family_value=float(vgs),
            metadata={"source": f"curve1_output_characteristics:{key}"}))
    return families, vmax


def _v_sat(families: List[CurveData]) -> float:
    """Data-derived ohmic/saturation boundary from the top Vgs curve.

    Chosen where dId/dVds first falls below half its initial value; no vendor
    constant or fixed volts are embedded (E3's Vsat=2 V is a source hint, not
    a PieceSpice default).
    """
    top = max(families, key=_family_value)
    x, y = top.to_numpy()
    if x.size < 4:
        return float(x[-1])
    g = np.gradient(y, x, edge_order=2)
    g0 = g[0]
    if not np.isfinite(g0) or g0 <= 0.0:
        return float(x[len(x) // 2])
    below = np.flatnonzero(g < _THRESHOLD_FACTOR * g0)
    return float(x[below[0]]) if below.size else float(x[len(x) // 2])


def _static_guard_width(families: List[CurveData]
                        ) -> Tuple[float, Dict[str, Any]]:
    """Low-VDS guard width from one static Curve-1 grid (before DPT append).

    Curve-1 families share a uniform Vds axis; the top-Vgs curve is also the
    one used by :func:`_v_sat`.  Deriving the width here keeps it a measured
    grid property (half the grid spacing) and prevents dense DPT trajectory
    samples from collapsing eps_from_grid's half-minimum cap.
    """
    top = max(families, key=_family_value)
    x, _ = top.to_numpy()
    xs = np.unique(x[np.isfinite(x)])
    width = eps_from_grid(xs.tolist())
    if not np.isfinite(width) or width <= 0.0:
        raise RuntimeError(
            f"{top.name}: static Curve-1 Vds grid has no positive spacing")
    prov: Dict[str, Any] = {
        "value": float(width),
        "rule": "approximation.smoother.eps_from_grid (median positive "
                "spacing, capped at half minimum)",
        "source": f"{top.metadata.get('source') or top.name}:Vds axis",
        "family": top.name,
        "n_grid_points": int(xs.size),
        "dpt_points_excluded": True,
    }
    return width, prov


def _charge_curves(vendor: str, cond: Dict[str, Any]
                   ) -> Tuple[CurveData, CurveData, Dict[str, Any]]:
    """Curve-4 capacitance anchor + Curve-5 gate-charge plateau."""
    c4 = _extract(vendor, "curve4_capacitances")
    c5 = _extract(vendor, "curve5_gate_charge")
    try:
        vref = float(c4["cap_ref_vds"])
    except (KeyError, TypeError, ValueError):
        vref = 0.25 * (float(cond["Vds_cap_max"])
                       - float(cond["Vds_cap_min"]))
    meta: Dict[str, Any] = {k: (float(c4[k]) if c4.get(k) is not None else None)
            for k in ("ciss_f", "coss_f", "crss_f")}
    meta["cap_ref_vds"] = float(vref)
    meta["source"] = "curve4_capacitances:extracted"
    coss_value = meta.get("coss_f")
    if coss_value is None or float(coss_value) <= 0.0:
        raise RuntimeError(f"{vendor}: curve4 has no positive coss_f")
    coss = float(coss_value)

    curve4 = CurveData(
        name="curve4_capacitances_coss",
        x_name="Vds", x_unit="V", y_name="Coss_eq", y_unit="F",
        x_data=[0.0, float(vref)], y_data=[coss, coss],
        metadata=meta)
    xg = np.asarray(c5["curve_vgs"], dtype=float)
    qg = np.asarray(c5["curve_qg_nc"], dtype=float) * 1e-9
    plateau = CurveData(
        name="curve5_gate_charge",
        x_name="Vgs", x_unit="V", y_name="Qg", y_unit="C",
        x_data=[float(v) for v in xg], y_data=[float(v) for v in qg],
        metadata={"source": "curve5_gate_charge:curve_qg_nc [nC] -> C",
                  "v_plateau_v": c5.get("v_plateau_v")})
    return curve4, plateau, meta


def _dpt_points(vendor: str, vds_max: float
                ) -> Tuple[Optional[DptPoints], Dict[str, Any]]:
    """Best-effort high-V/I points from the already-captured golden DPT.

    Extraction is offline fitting data, not a fairness-bench row.  If LTspice
    is unavailable or the run aborts, the reason is recorded and no point is
    fabricated; the fit then uses Curve 1 alone.
    """
    prov: Dict[str, Any] = {
        "source": "dpt_trajectory", "vendor": vendor,
        "status": "attempted", "sample_indices": [],
        "n_points": 0, "sample_selection": None, "reason": None,
        "external_source_status": "conditional_until_url_and_section_pinned",
    }
    try:
        from simulation.testbenches.dpt_trajectory import capture_trajectory
        traj = capture_trajectory(vendor, timeout_s=_DPT_TIMEOUT_S,
                                  require_complete=True)
        vgs = np.asarray(traj.input[0], dtype=float)
        vds = np.asarray(traj.input[1], dtype=float)
        ids = np.asarray(traj.output[0], dtype=float)
        mask = (np.isfinite(vgs) & np.isfinite(vds) & np.isfinite(ids)
                & (vds > vds_max) & (ids > 0.0))
        all_idx = np.flatnonzero(mask)
        vgs, vds, ids = vgs[mask], vds[mask], ids[mask]
        if vgs.size == 0:
            prov.update(status="no_high_v_samples", sample_indices=[],
                        reason=f"no DPT points with Vds > {vds_max:g} V")
            return None, prov
        stride = max(1, int(np.ceil(vgs.size / _DPT_MAX_POINTS)))
        kept_idx = all_idx[::stride]
        vgs, vds, ids = vgs[::stride], vds[::stride], ids[::stride]
        prov.update(status="ok", n_points=int(vgs.size),
                    sample_indices=[int(i) for i in kept_idx],
                    sample_selection=f"Vds > {vds_max:g} V, stride {stride}",
                    pin_roles=list(traj.pin_roles))
        return DptPoints(vgs, vds, ids, prov), prov
    except Exception as exc:  # noqa: BLE001 - report, never fabricate
        prov.update(status="unavailable", sample_indices=[],
                    reason=f"{type(exc).__name__}: {str(exc)[:200]}")
        return None, prov


def ann_fit_for_vendor(vendor: str, *, include_dpt_points: bool = True
                       ) -> Dict[str, Any]:
    """Fit channel + charge ANN candidates for *vendor* (no file writes).

    Returns a dict with the two fit objects, a serialisable ``record`` and the
    data provenance.  T4 consumes the fit objects to emit the UROM library and
    copies the record into ``unified_validation.json#ann_fit``.
    """
    cfg = load_input(vendor)
    cond = cfg["conditions"]
    families, vmax = _channel_families(vendor, cond)
    guard_width, guard_prov = _static_guard_width(families)

    v_th = float(_extract(vendor, "curve7_threshold_voltage")["vth_v"])
    v_sat = _v_sat(families)
    c4, plateau, cap_meta = _charge_curves(vendor, cond)

    dpt, dpt_prov = (None, {"source": "dpt_trajectory", "status": "disabled",
                            "sample_indices": [], "n_points": 0,
                            "sample_selection": None,
                            "reason": "include_dpt_points=False"})
    if include_dpt_points:
        dpt, dpt_prov = _dpt_points(vendor, vmax)

    channel_inputs: List[Any] = list(families)
    if dpt is not None:
        channel_inputs.append(dpt)
    channel_fit: AnnChannelFit = fit_channel_iv(
        channel_inputs, v_th, v_sat,
        guard_width=guard_width, guard_width_provenance=guard_prov)
    charge_fit: AnnChargeFit = fit_charge_qv(c4, plateau)

    q0_used = dict(charge_fit.provenance.get("q0_used") or {})
    q0_oss = q0_used.get("oss")
    if q0_oss is None and q0_used:
        q0_oss = float(np.median(list(q0_used.values())))
    record: Dict[str, Any] = {
        "activation": channel_fit.activation,
        "neurons": channel_fit.neurons,
        "loss_weights": dict(channel_fit.loss_weights),
        "gm_gds_q_derivative_terms":
            channel_fit.diagnostics["derivative_terms"]
            + charge_fit.provenance["derivative_terms"],
        "train_range": dict(channel_fit.train_range),
        "q0_offset": (float(q0_oss) if q0_oss is not None else None),
        "q0_offset_by_table": q0_used,
        "q0_offset_source":
            "approximation.ann_charge.fit_charge_qv provenance.q0_used "
            "(log-Q0 offsets actually used per table); q0_offset is the "
            "q_oss entry; charge_fit.q0 is the caller override (None here)",
        "guard_width_v": float(guard_width),
        "out_of_range_check": channel_fit.out_of_range_check,
    }
    provenance: Dict[str, Any] = {
        "guard_width": guard_prov,
        "curves": [
            "curve1_output_characteristics.yaml",
            "curve2_transfer_characteristic.yaml",
            "curve3_rds_on.yaml",
            "curve4_capacitances.yaml",
            "curve5_gate_charge.yaml",
            "curve7_threshold_voltage.yaml",
        ],
        "dpt_high_vi": dpt_prov,
        "v_th_source": "curve7_threshold_voltage:vth_v",
        "v_sat_source": "curve1 top-Vgs curve, dI/dVds half-initial rule",
        "v_sat_value": v_sat,
        "cap_equivalents": cap_meta,
        "channel_fit": channel_fit.provenance,
        "charge_fit": charge_fit.provenance,
        "external_source_status": "conditional_until_url_and_section_pinned",
    }
    limitations = [
        "Curve-4 extracts only cap equivalents at one reference bias; charge "
        "tables are source-derived Q(Cref)=Cref*Vref anchors where no C(V) "
        "sweep exists, not a learned 2-D Q(Vgs,Vds) surface.",
        "Physics guards (VDS=0 transform, tanh Vth cutoff, low-VDS guard, "
        "alpha=10) are adopted from CONDITIONAL external sources; their "
        "numeric values are not verified PieceSpice results.",
        "Static/offline fit quality is not a fidelity gate; G1-G6 remain "
        "with T5 and no speed number is implied by this record.",
    ]
    return {
        "vendor": vendor,
        "device": cfg["device"],
        "channel_fit": channel_fit,
        "charge_fit": charge_fit,
        "record": record,
        "provenance": provenance,
        "limitations": limitations,
    }


# Hybrid fit parameters.  The hybrid basis fits only the signed residual
# current; C0 charge tables come from the closed Tier-2 spec unchanged.
_CURVE4_SWEEP_KEYS = ("curve_vds", "ciss_curve_f", "coss_curve_f",
                      "crss_curve_f")
_HYBRID_ACTIVATION = "isru"
_HYBRID_NEURONS = 6


def _curve4_data_richness(vendor: str) -> Dict[str, Any]:
    """Record full Curve-4 array presence/length as a diagnostic only.

    The persisted sweep is consumed by the simplis charge basis.  The hybrid
    basis neither requires it nor trains ANN charge nets from it, so an
    absent or malformed array is reported, never raised.
    """
    note: Dict[str, Any] = {
        "source": "curve4_capacitances.yaml",
        "role": "data-richness diagnostic only; hybrid keeps C0 charge and "
                "trains no ANN charge net",
        "keys": {}, "n_samples": {}, "missing_keys": [],
        "status": "unavailable",
    }
    try:
        doc = load_output(vendor, "curve4_capacitances") or {}
    except Exception as exc:  # diagnostic only; never gate the hybrid fit
        note["reason"] = str(exc)
        return note
    if not isinstance(doc, dict):
        note["reason"] = "curve4_capacitances.yaml is not a mapping"
        return note
    source = dict(doc)
    extracted = doc.get("extracted") or {}
    if isinstance(extracted, dict):
        source.update(extracted)
    present = {}
    for key in _CURVE4_SWEEP_KEYS:
        value = source.get(key)
        present[key] = value is not None
        note["keys"][key] = bool(present[key])
        if isinstance(value, (list, tuple, np.ndarray)):
            note["n_samples"][key] = len(value)
    note["missing_keys"] = [k for k, ok in present.items() if not ok]
    note["status"] = "available" if not note["missing_keys"] else "partial"
    return note


def hybrid_ann_fit_for_vendor(vendor: str, *, include_dpt_points: bool = True,
                              dpt_points: Optional[DptPoints] = None,
                              activation: str = _HYBRID_ACTIVATION,
                              neurons: int = _HYBRID_NEURONS) -> Dict[str, Any]:
    """Fit the hybrid C0+signed-residual basis for *vendor* (no file writes).

    The closed Tier-2 channel is evaluated by
    :func:`models.netlist_builder.evaluate_tier2_channel`; the residual is
    fitted to golden-minus-C0 on Curve-1 and, when supplied or captured, the
    DPT samples.  C0 charge tables are copied unchanged by the emitter; this
    fit trains no ANN charge nets and does not require the full Curve-4 C(V)
    sweep (its array presence is only a data-richness note).  Returns the C0
    spec, the residual fit object, a serialisable record and provenance.
    No gate runs here.
    """
    from pipeline.fitting import build_pwl_spec

    cfg = load_input(vendor)
    cond = cfg["conditions"]
    families, vmax = _channel_families(vendor, cond)
    guard_width, guard_prov = _static_guard_width(families)
    v_th = float(_extract(vendor, "curve7_threshold_voltage")["vth_v"])
    v_sat = _v_sat(families)
    curve4_note = _curve4_data_richness(vendor)
    spec = build_pwl_spec(vendor)

    if dpt_points is not None:
        dpt, dpt_prov = dpt_points, {
            "source": "caller-supplied recorded DPT samples",
            "status": "ok", "n_points": int(np.size(dpt_points.id)),
            **dict(dpt_points.provenance),
        }
    elif include_dpt_points:
        dpt, dpt_prov = _dpt_points(vendor, vmax)
    else:
        dpt, dpt_prov = None, {"source": "dpt_trajectory", "status": "disabled",
                               "sample_indices": [], "n_points": 0,
                               "reason": "include_dpt_points=False"}
    channel_inputs: List[Any] = list(families)
    if dpt is not None:
        channel_inputs.append(dpt)
    residual_fit = fit_channel_residual(
        channel_inputs, spec, v_th=v_th, activation=activation,
        neurons=neurons, guard_width=guard_width,
        guard_width_provenance=guard_prov,
        spec_source=f"pipeline.fitting.build_pwl_spec({vendor})")

    record: Dict[str, Any] = {
        "basis": "hybrid",
        "vendor": vendor,
        "device": cfg["device"],
        "c0_channel": "closed Tier-2 G_CH from pipeline.fitting.build_pwl_spec",
        "residual_expression": "Vds*(b2+sum(w2_i*phi(w1_i0*Vgs+"
                               "w1_i1*Vds+b1_i)))*0.5*(1+tanh(alpha*(Vgs-Vth)))",
        "activation": residual_fit.activation,
        "neurons": residual_fit.neurons,
        "guard_width_v": float(guard_width),
        "residual_train_samples": residual_fit.diagnostics["train_samples"],
        "residual_curve1_samples": residual_fit.diagnostics["curve1_samples"],
        "residual_dpt_samples": residual_fit.diagnostics["dpt_samples"],
        "residual_rel_rmse": residual_fit.diagnostics["residual_rel_rmse"],
        "residual_zero_vds_max_abs_a":
            residual_fit.diagnostics["zero_vds_max_abs_a"],
        "residual_cutoff_attenuation":
            residual_fit.diagnostics["cutoff_attenuation"],
        "residual_derivative_diagnostics":
            residual_fit.diagnostics["analytic_vs_finite_difference"],
        "charge_basis": "unchanged closed Tier-2 C0 charge tables; "
                        "no ANN charge fit in the hybrid basis",
        "curve4_data_richness": curve4_note,
        "optimizer": {
            "residual": {k: residual_fit.diagnostics[k] for k in (
                "optimizer_cost", "optimizer_starts_run", "optimizer_nfev",
                "optimizer_njev", "optimizer_success",
                "optimizer_termination_reason")},
        },
    }
    provenance: Dict[str, Any] = {
        "curves": ["curve1_output_characteristics.yaml",
                   "curve7_threshold_voltage.yaml"],
        "curve4_full_sweep": curve4_note,
        "charge_basis": "closed Tier-2 C0 Q tables; no ANN charge fit",
        "guard_width": guard_prov,
        "v_th_source": "curve7_threshold_voltage:vth_v",
        "v_sat_source": "curve1 top-Vgs curve, dI/dVds half-initial rule",
        "v_sat_value": v_sat,
        "dpt_samples": dpt_prov,
        "residual_fit": residual_fit.provenance,
        "c0_evaluator": "models.netlist_builder.evaluate_tier2_channel "
                        "(closed Tier-2 G_CH, eps=0)",
        "external_source_status": "conditional_until_url_and_section_pinned",
    }
    limitations = [
        "C0 charge tables are emitted unchanged; no ANN charge net is "
        "trained or attached by the hybrid basis.  Full Curve-4 arrays are "
        "persisted for simplis and recorded only as a data-richness note.",
        "Residual is fitted offline to golden-minus-C0; static fit quality "
        "is not a fidelity gate and no speed number is implied here.",
        "Physics guards (VDS=0 factor, tanh Vth cutoff) follow the existing "
        "ANN study and remain CONDITIONAL until source URLs are pinned.",
    ]
    return {
        "vendor": vendor,
        "device": cfg["device"],
        "c0_spec": spec,
        "residual_fit": residual_fit,
        "record": record,
        "provenance": provenance,
        "limitations": limitations,
    }
