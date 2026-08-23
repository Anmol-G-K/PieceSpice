"""Smoke test: run all curves for EPC2216 with config-driven pin order."""
import sys
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np

from config.yaml_io import load_input
from parser.netlist import resolve_pin_connections
from simulation.testbenches.static_sweep import StaticSweepTestbench
from simulation.testbenches.capacitance import CapacitanceTestbench
from simulation.testbenches.gate_charge import reconstruct_gate_charge

cfg = load_input('epc2216')
cond = cfg['conditions']
lib = Path(cfg['library']).resolve()
sub = cfg['subcircuit']

pin_roles = resolve_pin_connections(
    lib, sub, pin_order_semantic=cfg['pins'].get('pin_order_semantic'))
print(f"SMOKE pin_roles = {pin_roles}")

static = StaticSweepTestbench()

# Curve 1: output characteristics
oc = static.run_output_characteristics(lib, sub,
    vgs_list=cond['Vgs_sweep'], vds_max=cond['Vds_sweep_max'],
    vds_points=cond['Vds_points'], pin_roles=pin_roles)
last = oc[-1]
print(f"SMOKE curve1 @Vgs={last.family_value}: Id(Vds=15) = {last.y_data[-1]:.2f} A")

# Curve 2: transfer
tr = static.run_transfer_characteristics(lib, sub,
    vds_fixed=cond['Vds_fixed'], vgs_max=cond['Vgs_transfer_max'],
    vgs_points=cond['Vgs_transfer_points'], pin_roles=pin_roles)
vx, iy = tr.to_numpy()
print(f"SMOKE curve2: Id @ Vgs=6V, Vds={cond['Vds_fixed']}V = {iy[-1]:.2f} A")

# Curve 3: Rds(on)
rds = static.run_rds_on(lib, sub, vgs=cond['Vgs_fixed'],
    vds_max=cond['Vds_rds_max'], vds_points=cond['Vds_rds_points'],
    min_current=cond['Id_rds_min'], pin_roles=pin_roles)
rx, ry = rds.to_numpy()
print(f"SMOKE curve3: Rds(on) @ Vgs=5V (Id=25A) = {np.interp(25, rx, ry):.2f} mOhm"
      if len(rx) else "SMOKE curve3: no points above mask")

# Curve 4: capacitances
cap = CapacitanceTestbench().run(lib, sub, vds_min=cond['Vds_cap_min'],
    vds_max=cond['Vds_cap_max'], points=cond['Cap_points'],
    pin_roles=pin_roles)
for cc in cap:
    print(f"SMOKE curve4 {cc.name} @ Vds=5V = "
          f"{np.interp(5, cc.x_data, cc.y_data)*1e12:.0f} pF")

# Curve 6: reverse conduction / body diode
dio = static.run_body_diode(lib, sub, vsd_max=cond['Vsd_max'],
    points=cond['Vsd_points'], pin_roles=pin_roles)
dx, dy = dio.to_numpy()
print(f"SMOKE curve6: Vsd @ Isd=10A = {np.interp(10, dy, dx):.3f} V")

# Curve 7: Vth
vth_curve, v_th = static.run_threshold_voltage(lib, sub,
    i_target=cond['Id_th_target'], v_max=cond['Vgs_th_max'],
    pin_roles=pin_roles)
print(f"SMOKE curve7: Vth @ {cond['Id_th_target']*1e3:.0f} mA = {v_th:.3f} V")
