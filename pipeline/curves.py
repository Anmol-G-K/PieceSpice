"""PieceSpice curve-extraction pipeline.

Single vendor-parameterised implementation of the 7-curve extraction flow
(Critical fix C1 / Major M11 / Minor m20): the vendor script and notebook 01
both delegate here instead of carrying their own copies.

Usage:
    from pipeline.curves import run_all_curves
    outputs = run_all_curves("irfs7730pbf")
"""

from pathlib import Path

import numpy as np

from config.yaml_io import load_input, save_output
from simulation.testbenches.static_sweep import StaticSweepTestbench, \
    normalized_lib_copy, discard_normalized_lib
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
    cap_tb = CapacitanceTestbench(f_meas=cond.get("F_meas"))
    ciss, coss, crss = cap_tb.run(lib, subckt,
        vds_min=cond["Vds_cap_min"], vds_max=cond["Vds_cap_max"],
        points=cond["Cap_points"], pin_roles=roles)
    cap_ref = _cap_ref_vds(cond)
    outputs["curve4_capacitances"] = save_output(
        device, "curve4_capacitances", "pass",
        {"Vds_cap_min": cond["Vds_cap_min"], "Vds_cap_max": cond["Vds_cap_max"],
         "Cap_points": cond["Cap_points"], "F_meas": cond["F_meas"]},
        {"ciss_f": interp(ciss.x_data, ciss.y_data, cap_ref),
         "coss_f": interp(coss.x_data, coss.y_data, cap_ref),
         "crss_f": interp(crss.x_data, crss.y_data, cap_ref),
         "cap_ref_vds": cap_ref},
        notes=f"Small-signal AC @ {cond['F_meas']:.0e} Hz; "
              f"reported at Vds={cap_ref} V.")

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
