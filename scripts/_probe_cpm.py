"""Probe: cap values per point for CPM2."""
import sys
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np

from config.yaml_io import load_input
from parser.netlist import resolve_pin_connections
from simulation.testbenches.capacitance import CapacitanceTestbench

cfg = load_input('cpm2-1200')
cond = cfg['conditions']
lib = Path(cfg['library']).resolve()
sub = cfg['subcircuit']
roles = resolve_pin_connections(lib, sub,
    pin_order_semantic=cfg['pins'].get('pin_order_semantic'))

cap = CapacitanceTestbench().run(lib, sub, vds_min=0.1, vds_max=50,
    points=25, pin_roles=roles)
for cc in cap:
    print("CAP", cc.name)
    for x, y in zip(cc.x_data[::4], cc.y_data[::4]):
        flag = " <-- garbage" if y > 1e-3 else ""
        print(f"   Vds={x:8.2f}  C={y:.4e}{flag}")
