"""Write output YAMLs for all currently-passing IRFS7730PBF curves.

Runs each testbench with conditions from config/inputs/irfs7730pbf.yaml and
records results via config.yaml_io.save_output(). No testbench logic changes.
This is the canonical example of the config-driven pipeline; a future CLI
command will supersede this script.
"""

import sys
sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")

import numpy as np
from pathlib import Path

from config.yaml_io import load_input, save_output
from simulation.testbenches.static_sweep import StaticSweepTestbench
from simulation.testbenches.capacitance import CapacitanceTestbench
from simulation.testbenches.gate_charge import reconstruct_gate_charge

cfg = load_input("irfs7730pbf")
cond = cfg["conditions"]
lib = Path(cfg["library"])
subckt = cfg["subcircuit"]
V = cfg["device"]

tb = StaticSweepTestbench()
interp = lambda x, y, x0: float(np.interp(x0, x, y))

# Curve 1: Output characteristics
oc = tb.run_output_characteristics(lib, subckt, vgs_list=cond["Vgs_sweep"],
    vds_max=cond["Vds_sweep_max"], vds_points=cond["Vds_points"])
save_output(V, "curve1_output_characteristics", "pass",
    {"Vgs_sweep": cond["Vgs_sweep"], "Vds_sweep_max": cond["Vds_sweep_max"],
     "Vds_points": cond["Vds_points"], "temperature": cond["temperature"]},
    {f"curve_id_vgs_{int(v)}_v": c.y_data for v, c in zip(cond["Vgs_sweep"], oc)},
    notes="VM-04: kA-domain currents at full overdrive are model-inherent.")
print("curve1 done")

# Curve 2: Transfer characteristic
tr = tb.run_transfer_characteristics(lib, subckt, vds_fixed=10.0,
    vgs_max=cond["Vgs_transfer_max"], vgs_points=cond["Vgs_transfer_points"])
vx, iy = tr.to_numpy()
g = np.gradient(iy, vx)
vp = interp(iy, vx, cond["Id_load"])
save_output(V, "curve2_transfer_characteristic", "pass",
    {"Vds_fixed": 10.0, "Vgs_transfer_max": cond["Vgs_transfer_max"],
     "Vgs_transfer_points": cond["Vgs_transfer_points"]},
    {"curve_vgs": vx.tolist(), "curve_id": iy.tolist(),
     "gfs_s": float(np.interp(vp, vx, g)), "v_plateau_v": vp},
    notes="VM-01: no subthreshold in LEVEL=1 core.")
print("curve2 done")

# Curve 3: Rds(on)
rds = tb.run_rds_on(lib, subckt, vgs=cond["Vgs_fixed"],
                    vds_max=cond["Vds_rds_max"],
                    vds_points=cond["Vds_rds_points"],
                    min_current=cond["Id_rds_min"])
rx, ry = rds.to_numpy()
save_output(V, "curve3_rds_on", "pass",
    {"Vgs_fixed": cond["Vgs_fixed"], "Vds_rds_max": cond["Vds_rds_max"],
     "Id_rds_min": cond["Id_rds_min"]},
    {"rds_on_10v_mohm": interp(rx, ry, 50.0),
     "curve_id": rx.tolist(), "curve_vds": ry.tolist()},
    notes="Interpolated at Id=50A per reference table.")
print("curve3 done")

# Curve 4: Capacitances
ciss, coss, crss = CapacitanceTestbench().run(lib, subckt,
    vds_min=cond["Vds_cap_min"], vds_max=cond["Vds_cap_max"],
    points=cond["Cap_points"])
save_output(V, "curve4_capacitances", "pass",
    {"Vds_cap_min": cond["Vds_cap_min"], "Vds_cap_max": cond["Vds_cap_max"],
     "Cap_points": cond["Cap_points"], "F_meas": cond["F_meas"]},
    {"ciss_f": interp(ciss.x_data, ciss.y_data, 25.0),
     "coss_f": interp(coss.x_data, coss.y_data, 25.0),
     "crss_f": interp(crss.x_data, crss.y_data, 25.0)},
    notes=f"Small-signal AC @ {cond['F_meas']:.0e} Hz; values reported at Vds=25V.")
print("curve4 done")

# Curve 5: Gate charge (quasi-static)
df_qg, bp = reconstruct_gate_charge(tr, ciss, crss,
    v_bus=cond["Vbus"], i_load=cond["Id_load"],
    v_gs_max=cond["Vgs_qg_max"], v_ds_on=cond["Vds_on"],
    qgd_cal_nC=cond.get("Qgd_typ_nC"))
save_output(V, "curve5_gate_charge", "pass",
    {"Id_load": cond["Id_load"], "Vbus": cond["Vbus"],
     "Vgs_qg_max": cond["Vgs_qg_max"], "Vds_on": cond["Vds_on"],
     "Qgd_typ_nC": cond.get("Qgd_typ_nC")},
    {"v_plateau_v": bp["v_plateau_V"], "qgs_nc": bp["q_gs_nC"],
     "qgd_nc": bp["q_gd_nC"], "q_tail_nc": bp["q_tail_nC"],
     "qg_total_nc": bp["q_total_nC"],
     "curve_qg_nc": df_qg["Qg_nC"].tolist(), "curve_vgs": df_qg["Vgs_V"].tolist()},
    notes="Quasi-static Option A (VM-02 workaround); Qgd calibrated to datasheet (VM-03).")
print("curve5 done")

# Curve 6: Body diode
diode = tb.run_body_diode(lib, subckt, vsd_max=cond["Vsd_max"],
                          points=cond["Vsd_points"])
dx, dy = diode.to_numpy()
save_output(V, "curve6_body_diode", "pass",
    {"Vsd_max": cond["Vsd_max"], "Vsd_points": cond["Vsd_points"]},
    {"body_diode_vsd_10a_v": interp(dy, dx, 10.0),
     "curve_isd": dy.tolist(), "curve_vds": dx.tolist()},
    notes="Conservative voltage cap avoids diode-exponential divergence.")
print("curve6 done")

# Curve 7: Threshold voltage
vth_curve, v_th = tb.run_threshold_voltage(lib, subckt,
    i_target=cond["Id_th_target"], v_max=cond["Vgs_th_max"])
save_output(V, "curve7_threshold_voltage", "partial",
    {"Id_th_target": cond["Id_th_target"], "Vgs_th_max": cond["Vgs_th_max"]},
    {"vth_v": v_th, "curve_vgs": vth_curve.x_data, "curve_id": vth_curve.y_data},
    notes="VM-01: fitted effective VTO, not physical JEDEC Vth (datasheet 2.1-3.7 V).")
print("curve7 done")

# Blocked: raw transient gate charge
save_output(V, "gate_charge_transient", "blocked",
    {"Ig_drive": cond["Ig_drive"], "Vbus": cond["Vbus"], "Id_load": cond["Id_load"]},
    {}, notes="VM-02 MODPEX sense-and-mirror pathology; replaced by quasi-static Curve 5.")

for p in sorted(Path("config/outputs", V).glob("*.yaml")):
    print(" ", p)
