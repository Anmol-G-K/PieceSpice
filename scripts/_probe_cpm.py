"""Probe: options aren't it either. Remaining bench-vs-standalone difference:
the notebook runs curves 1..N in the SAME kernel - earlier curves' state.
But my earlier smoke script also ran curve1 first and failed at curve1 itself
(Vgs=10 output char). And standalone passes. One more difference: the bench
creates temp_lib via mkstemp in normalized/ dir; standalone probe did too...

Actually wait: bench run_output_characteristics loops vgs_list [10,12,14,16,18,20]
- Vgs=10 is FIRST. It failed. Standalone with same everything passed?! Retry
the exact bench call in isolation:"""
import sys
sys.path.insert(0, '.')
from pathlib import Path
from config.yaml_io import load_input
from parser.netlist import resolve_pin_connections
from simulation.testbenches.static_sweep import StaticSweepTestbench

cfg = load_input('cpm2-1200')
cond = cfg['conditions']
lib = Path(cfg['library']).resolve()
sub = cfg['subcircuit']
roles = resolve_pin_connections(lib, sub,
    pin_order_semantic=cfg['pins'].get('pin_order_semantic'))
print("ROLES:", roles)
static = StaticSweepTestbench()
oc = static.run_output_characteristics(lib, sub,
    vgs_list=cond['Vgs_sweep'], vds_max=cond['Vds_sweep_max'],
    vds_points=cond['Vds_points'], pin_roles=roles)
print("BENCH curve1 OK:", len(oc), "curves")
