"""Run the golden-vs-ROM runtime benchmark for all configured vendors.

The circuit is the clamped-inductive DPT topology already validated in
notebook 03; the gate drive is extended to an N-pulse train so the long
case is genuinely longer without changing the stress per pulse.

Usage: uv run python scripts/run_speedup_benchmark.py
Writes reports/comparison/speedup_benchmark.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parser.netlist import resolve_pin_connections
from simulation.testbenches.speedup_bench import run_inductive_benchmark

VENDORS = ("irfs7730pbf", "csd16412q5a", "epc2216", "cpm2-1200")
CASES = {
    "short_forced": {"n_pulses": 2, "max_step": 2e-9},
    "short_adaptive": {"n_pulses": 2, "max_step": None},
    "long_adaptive": {"n_pulses": 20, "max_step": None},
    "long_100ns": {"n_pulses": 20, "max_step": 100e-9},
}
REPEATS = 3
TIMEOUT_S = 90.0

# EPC's vendor GaN model stalls when LTspice grows the timestep past
# ~100-200 ns; it completes with a 100 ns cap (see long_100ns). CPM2 has
# no skip: earlier failures came from a wrong subcircuit name in a probe,
# not from the model.
SKIP_GOLDEN = {
    ("epc2216", "long_adaptive"):
        "vendor stalls under adaptive step; use long_100ns",
}


def main() -> None:
    out = {"schema": "piecespice.speedup/1.0",
           "cases": CASES, "repeats": REPEATS, "vendors": {}}
    for vendor in VENDORS:
        cfg = yaml.safe_load((ROOT / "config" / "inputs" /
                              f"{vendor}.yaml").read_text(encoding="utf-8"))
        d = cfg["dpt"]
        roles = resolve_pin_connections(
            ROOT / cfg["library"], cfg["subcircuit"],
            cfg["pins"]["pin_order_semantic"])
        common = dict(vbus=d["vbus"], vgs_on=d["vgs_on"],
                      i_load=d["i_load_target"], l_load=d["l_load"],
                      pulse_width=d["t2_width"],
                      first_pulse_width=d["t1_width"],
                      pulse_period=d["t2_on"] - d["t1_on"],
                      edge_time=d["edge_time"],
                      repeats=REPEATS, timeout_s=TIMEOUT_S)
        rom_lib = ROOT / "reports" / "models" / f"{vendor}_PIECESPICE_T2.lib"
        out["vendors"][vendor] = {}
        for case, params in CASES.items():
            entry = {}
            for label, lib, sub, pins in (
                ("golden", ROOT / cfg["library"], cfg["subcircuit"],
                 roles),
                ("rom", rom_lib, rom_lib.stem, ["drain", "gate", "source"]),
            ):
                if label == "golden" and (vendor, case) in SKIP_GOLDEN:
                    entry[label] = {"error": SKIP_GOLDEN[(vendor, case)]}
                    continue
                try:
                    entry[label] = run_inductive_benchmark(
                        lib, sub, pins, **common, **params)
                except Exception as exc:  # never fabricate a failed timing
                    entry[label] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
            g, r = entry.get("golden", {}), entry.get("rom", {})
            for key, short in (("wall_clock_s", "wall"),
                               ("solver_elapsed_s", "solver")):
                if g.get(key) and r.get(key):
                    entry[f"{short}_speedup"] = g[key] / r[key]
            out["vendors"][vendor][case] = entry
            print(f"{vendor:14s} {case:15s} "
                  f"wall={entry.get('wall_speedup')} "
                  f"solver={entry.get('solver_speedup')}", flush=True)

    path = ROOT / "reports" / "comparison" / "speedup_benchmark.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
