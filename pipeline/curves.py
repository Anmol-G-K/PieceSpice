"""PieceSpice curve-extraction pipeline.

Single vendor-parameterised implementation of the 7-curve extraction flow
(Critical fix C1 / Major M11 / Minor m20): the vendor script and notebook 01
both delegate here instead of carrying their own copies.

Usage:
    from pipeline.curves import run_all_curves
    outputs = run_all_curves("irfs7730pbf")
"""

import math
from pathlib import Path

import numpy as np

from config.yaml_io import load_input, save_output
from simulation.testbenches.static_sweep import StaticSweepTestbench
from simulation.testbenches.capacitance import CapacitanceTestbench
from simulation.testbenches.gate_charge import reconstruct_gate_charge


def _interp(x, y, x0):
    return float(np.interp(x0, x, y))


def _cap_ref_vds(cond: dict) -> float:
    """Datasheet reference Vds for capacitance reporting.

    Per-vendor key ``Cap_ref_Vds`` (e.g. CPM2: 7.5 V); falls back to a quarter
    of the cap sweep ceiling so the reported point always lies inside the
    extracted envelope.
    """
    ref = cond.get("Cap_ref_Vds")
    if ref is not None:
        return float(ref)
    return float(cond["Vds_cap_min"]) + 0.25 * (
        float(cond["Vds_cap_max"]) - float(cond["Vds_cap_min"]))



def _finite_float_list(values, label: str) -> list:
    """Copy *values* to plain finite Python floats without changing any value."""
    try:
        out = [float(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Curve-4 {label}: non-numeric sample") from exc
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"Curve-4 {label}: non-finite sample")
    return out


def _curve4_extracted(ciss, coss, crss, cap_ref: float) -> dict:
    """Curve-4 extracted dict: legacy scalars plus the full AC sweep arrays.

    Scalar keys are computed exactly as before. The sweep arrays are copied
    verbatim (same values, same order) from the bench CurveData objects.
    Missing, ragged, non-finite or unordered sweeps raise ValueError so a
    partial array is never written.
    """
    vds = None
    caps = []
    for label, curve in (("Ciss", ciss), ("Coss", coss), ("Crss", crss)):
        xs = getattr(curve, "x_data", None)
        ys = getattr(curve, "y_data", None)
        if xs is None or ys is None:
            raise ValueError(f"Curve-4 {label}: missing x_data/y_data")
        x_list = _finite_float_list(xs, f"{label}.x_data")
        y_list = _finite_float_list(ys, f"{label}.y_data")
        if len(x_list) != len(y_list):
            raise ValueError(f"Curve-4 {label}: x/y length mismatch")
        if vds is None:
            vds = x_list
        elif x_list != vds:
            raise ValueError(f"Curve-4 {label}: Vds axis differs from Ciss")
        caps.append(y_list)

    if len(vds) < 2:
        raise ValueError("Curve-4 sweep needs >= 2 points; refusing partial data")
    if any(b <= a for a, b in zip(vds, vds[1:])):
        raise ValueError("Curve-4 Vds sweep is not strictly ascending")

    ciss_y, coss_y, crss_y = caps
    return {
        "ciss_f": _interp(vds, ciss_y, cap_ref),
        "coss_f": _interp(vds, coss_y, cap_ref),
        "crss_f": _interp(vds, crss_y, cap_ref),
        "cap_ref_vds": cap_ref,
        "curve_vds": vds,
        "ciss_curve_f": ciss_y,
        "coss_curve_f": coss_y,
        "crss_curve_f": crss_y,
    }


_CV_ROW_KEYS = ("curve_vds", "ciss_curve_f", "coss_curve_f",
                "crss_curve_f")


def _cap_bias_points(cond: dict) -> list:
    """Gate biases requested for the multi-bias C-V sweep.

    ``conditions.Cap_bias_Vgs`` is opt-in: absent (or empty) keeps the
    historical single Vgs=0 record. Values are device-family gate drives,
    not measurements; the bench's capacitance-validity guard decides
    whether each bias can actually be measured.
    """
    raw = cond.get("Cap_bias_Vgs")
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError("Cap_bias_Vgs must be a list of gate voltages")
    out = []
    for value in raw:
        try:
            bias = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cap_bias_Vgs entry {value!r} is not numeric") from exc
        if not math.isfinite(bias):
            raise ValueError(f"Cap_bias_Vgs entry {value!r} is not finite")
        if bias in out:
            raise ValueError(f"Cap_bias_Vgs lists Vgs={bias:g} V twice")
        out.append(bias)
    return out


def _cv_sweep_row(extracted: dict, vgs_bias: float,
                  temperature: float) -> dict:
    """One ``cv_sweeps`` row, shaped as tier3_simplis expects."""
    row = {key: extracted[key] for key in _CV_ROW_KEYS}
    row["vgs_bias_v"] = float(vgs_bias)
    row["temperature_c"] = float(temperature)
    return row


def _run_curve4(cfg: dict, roles) -> tuple:
    """Extract and persist Curve 4; return (path, base Ciss, base Crss).

    The base Vgs=0 sweep always runs and supplies the legacy fields. Each
    ``Cap_bias_Vgs`` bias beyond 0 V is measured independently; a bias the
    bench's validity guard rejects is omitted from ``cv_sweeps`` and its
    failure is disclosed in ``notes`` (never silently dropped). Curve 5
    reuses the base Ciss/Crss, so they are returned alongside the path.
    """
    cond = cfg["conditions"]
    lib = Path(cfg["library"])
    subckt = cfg["subcircuit"]
    device = cfg["device"]
    temperature = float(cond.get("temperature", 25.0))

    cap_tb = CapacitanceTestbench(f_meas=cond.get("F_meas"))
    sweep_args = dict(
        vds_min=cond["Vds_cap_min"], vds_max=cond["Vds_cap_max"],
        points=cond["Cap_points"], pin_roles=roles)
    ciss, coss, crss = cap_tb.run(lib, subckt, **sweep_args)
    cap_ref = _cap_ref_vds(cond)
    extracted = _curve4_extracted(ciss, coss, crss, cap_ref)

    conditions_used = {
        "Vds_cap_min": cond["Vds_cap_min"],
        "Vds_cap_max": cond["Vds_cap_max"],
        "Cap_points": cond["Cap_points"],
        "F_meas": cond["F_meas"],
    }
    notes = (f"Small-signal AC @ {cond['F_meas']:.0e} Hz; "
             f"reported at Vds={cap_ref} V.")

    biases = _cap_bias_points(cond)
    if biases:
        conditions_used["Cap_bias_Vgs"] = biases
        sweeps = [_cv_sweep_row(extracted, 0.0, temperature)]
        failures = []
        for bias in biases:
            if bias == 0.0:
                continue
            try:
                b_ciss, b_coss, b_crss = cap_tb.run(
                    lib, subckt, vgs_bias=bias, **sweep_args)
                b_extracted = _curve4_extracted(
                    b_ciss, b_coss, b_crss, cap_ref)
            except Exception as exc:  # invalid point, solver abort, or
                # malformed array: omit this bias and disclose why.
                failures.append(f"Vgs={bias:g} V ({exc})")
                continue
            sweeps.append(_cv_sweep_row(b_extracted, bias, temperature))

        if len(sweeps) > 1:
            extracted["cv_sweeps"] = sorted(
                sweeps, key=lambda row: row["vgs_bias_v"])
            notes += (f" Multi-bias C-V at Vgs="
                      f"{[row['vgs_bias_v'] for row in extracted['cv_sweeps']]}"
                      f" V; Vgs=0 legacy fields kept.")
        else:
            notes += (" No nonzero Cap_bias_Vgs sweep passed the capacitance "
                      "validity guard; legacy single Vgs=0 record retained.")
        if failures:
            notes += " Omitted invalid bias sweeps: " + "; ".join(failures) + "."

    path = save_output(
        device, "curve4_capacitances", "pass", conditions_used, extracted,
        notes=notes)
    return path, ciss, crss


def run_curve4(vendor: str) -> Path:
    """Run only Curve 4 (capacitances) for *vendor* and persist it.

    Used to refresh bias-aware C-V data without re-running the six
    unrelated curves.
    """
    cfg = load_input(vendor)
    from parser.netlist import resolve_pin_connections
    roles = resolve_pin_connections(
        Path(cfg["library"]), cfg["subcircuit"],
        pin_order_semantic=cfg.get("pins", {}).get("pin_order_semantic"))
    path, _, _ = _run_curve4(cfg, roles)
    print(f"[pipeline] {vendor}: Curve 4 extracted -> {path}")
    return path


def run_all_curves(vendor: str, keep_notebook: bool = False) -> dict:
    """Run all seven datasheet curves for *vendor* from its input YAML.

    Returns the dict of written output paths keyed by testbench name.
    """
    cfg = load_input(vendor)
    cond = cfg["conditions"]
    lib = Path(cfg["library"])
    subckt = cfg["subcircuit"]
    device = cfg["device"]
    temperature = float(cond.get("temperature", 25.0))

    from parser.netlist import resolve_pin_connections
    roles = resolve_pin_connections(
        lib, subckt, pin_order_semantic=cfg.get("pins", {}).get("pin_order_semantic"))

    tb = StaticSweepTestbench()
    interp = _interp
    outputs: dict = {}

    # ---- Curve 1: Output characteristics --------------------------------
    oc = tb.run_output_characteristics(
        lib, subckt, vgs_list=cond["Vgs_sweep"],
        vds_max=cond["Vds_sweep_max"], vds_points=cond["Vds_points"],
        pin_roles=roles)
    outputs["curve1_output_characteristics"] = save_output(
        device, "curve1_output_characteristics", "pass",
        {"Vgs_sweep": cond["Vgs_sweep"], "Vds_sweep_max": cond["Vds_sweep_max"],
         "Vds_points": cond["Vds_points"], "temperature": temperature},
        {vgs_key(float(v)): c.y_data
         for v, c in zip(cond["Vgs_sweep"], oc)},
        notes="Curve 1: Id-Vds family.")

    # ---- Curve 2: Transfer characteristic -------------------------------
    tr = tb.run_transfer_characteristics(
        lib, subckt, vds_fixed=cond["Vds_fixed"],
        vgs_max=cond["Vgs_transfer_max"], vgs_points=cond["Vgs_transfer_points"],
        pin_roles=roles)
    vx, iy = tr.to_numpy()
    g = np.gradient(iy, vx)
    vp = interp(iy, vx, cond["Id_load"])
    outputs["curve2_transfer_characteristic"] = save_output(
        device, "curve2_transfer_characteristic", "pass",
        {"Vds_fixed": cond["Vds_fixed"], "Vgs_transfer_max": cond["Vgs_transfer_max"],
         "Vgs_transfer_points": cond["Vgs_transfer_points"]},
        {"curve_vgs": vx.tolist(), "curve_id": iy.tolist(),
         "gfs_s": float(np.interp(vp, vx, g)), "v_plateau_v": vp},
        notes="Curve 2: transfer characteristic; gfs at plateau.")

    # ---- Curve 3: Rds(on) ------------------------------------------------
    rds = tb.run_rds_on(lib, subckt, vgs=cond["Vgs_fixed"],
                        vds_max=cond["Vds_rds_max"],
                        vds_points=cond["Vds_rds_points"],
                        min_current=cond["Id_rds_min"], pin_roles=roles)
    rx, ry = rds.to_numpy()
    outputs["curve3_rds_on"] = save_output(
        device, "curve3_rds_on", "pass",
        {"Vgs_fixed": cond["Vgs_fixed"], "Vds_rds_max": cond["Vds_rds_max"],
         "Id_rds_min": cond["Id_rds_min"]},
        {"rds_on_mohm_at_Id_load": _interp(rx, ry, cond["Id_load"]),
         "curve_id": rx.tolist(), "curve_rds_mohm": ry.tolist()},
        notes="Curve 3: Rds(on) vs Id in ohmic region (Id-x axis by design).")

    # ---- Curve 4: Capacitances ------------------------------------------
    outputs["curve4_capacitances"], ciss, crss = _run_curve4(cfg, roles)

    # ---- Curve 5: Gate charge (quasi-static) -----------------------------
    df_qg, bp = reconstruct_gate_charge(tr, ciss, crss,
        v_bus=cond["Vbus"], i_load=cond["Id_load"],
        v_gs_max=cond["Vgs_qg_max"], v_ds_on=cond["Vds_on"],
        qgd_cal_nC=cond.get("Qgd_typ_nC"))
    outputs["curve5_gate_charge"] = save_output(
        device, "curve5_gate_charge", "pass",
        {"Id_load": cond["Id_load"], "Vbus": cond["Vbus"],
         "Vgs_qg_max": cond["Vgs_qg_max"], "Vds_on": cond["Vds_on"],
         "Qgd_typ_nC": cond.get("Qgd_typ_nC")},
        {"v_plateau_v": bp["v_plateau_V"], "qgs_nc": bp["q_gs_nC"],
         "qgd_nc": bp["q_gd_nC"], "q_tail_nc": bp["q_tail_nC"],
         "qg_total_nc": bp["q_total_nC"],
         "curve_qg_nc": df_qg["Qg_nC"].tolist(),
         "curve_vgs": df_qg["Vgs_V"].tolist()},
        notes="Quasi-static Option A (VM-02 workaround); Qgd calibrated to "
              "datasheet where provided (VM-03).")

    # ---- Curve 6: Body diode ---------------------------------------------
    diode = tb.run_body_diode(lib, subckt, vsd_max=cond["Vsd_max"],
                              points=cond["Vsd_points"], pin_roles=roles)
    dx, dy = diode.to_numpy()
    isd_ref = cond.get("Isd_ref", 10.0)
    outputs["curve6_body_diode"] = save_output(
        device, "curve6_body_diode", "pass",
        {"Vsd_max": cond["Vsd_max"], "Vsd_points": cond["Vsd_points"]},
        {"body_diode_vsd_at_Isd_ref_v": interp(dy, dx, isd_ref),
         "Isd_ref": isd_ref,
         "curve_isd": dy.tolist(), "curve_vsd": dx.tolist()},
        notes="Reverse conduction / body-diode forward characteristic.")

    # ---- Curve 7: Threshold voltage ---------------------------------------
    vth_curve, v_th = tb.run_threshold_voltage(lib, subckt,
        i_target=cond["Id_th_target"], v_max=cond["Vgs_th_max"],
        pin_roles=roles)
    outputs["curve7_threshold_voltage"] = save_output(
        device, "curve7_threshold_voltage", "pass",
        {"Id_th_target": cond["Id_th_target"], "Vgs_th_max": cond["Vgs_th_max"]},
        {"vth_v": v_th, "Id_th_target": cond["Id_th_target"],
         "curve_vgs": vth_curve.x_data, "curve_id": vth_curve.y_data},
        notes="JEDEC constant-current method at per-vendor Id_th_target.")

    print(f"[pipeline] {vendor}: all 7 curves extracted -> "
          f"config/outputs/{device}/")
    return outputs


def vgs_key(v: float) -> str:
    """Canonical Curve-1 family key for a Vgs drive value.

    Uses %g so 4.0->'4' but 4.5->'4.5' (int() truncation collided 4.0/4.5).
    """
    return f"curve_id_vgs_{v:g}_v"
