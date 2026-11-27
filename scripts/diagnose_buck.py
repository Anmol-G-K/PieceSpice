"""Root-cause the buck-bench divergence by inspection and bisection."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

from simulation.testbenches.buck_converter import (build_converter_netlist,
                                                   load_converter_config,
                                                   resolve_converter_model)
from engine.ir.parse import read_deck
from engine.solver.transient import solve_transient


def netlist(vendor: str = "epc2216", tier: str = "tier4"):
    cfg = load_converter_config()
    m = resolve_converter_model(cfg, vendor, tier)
    text, sched = build_converter_netlist(m, config=cfg)
    path = Path("reports/models/_buck_probe.cir")
    path.write_text(text, encoding="ascii")
    return path, read_deck(path)


def show(parsed) -> None:
    circ = parsed.circuit
    print(f"elements {len(circ.elements)}  nodes {len(circ.node_index.names)}")
    for e in circ.elements:
        print(f"  {str(e.kind):10s} {e.name:10s} {e.nodes}")


def run(circ, end_time: float, max_step: float, label: str):
    r = solve_transient(circ, engine="compiled", end_time=end_time,
                        max_step=max_step)
    t = np.asarray(r.traces["time"])
    worst = 0.0
    for name, y in r.traces.items():
        if name.startswith(("V(", "I(")):
            y = np.asarray(y)
            if not np.all(np.isfinite(y)):
                print(f"  {label}: NON-FINITE {name} at t={t[0]:g}")
                return
            worst = max(worst, float(np.max(np.abs(y))))
    print(f"  {label}: {len(t)} pts, worst |V|/|I| = {worst:.4g}")


if __name__ == "__main__":
    path, parsed = netlist()
    show(parsed)
    circ = parsed.circuit
    for step in (50e-9, 20e-9, 10e-9, 2e-9):
        run(circ, 5e-6, step, f"2 cycles  step {step*1e9:5.1f} ns")
