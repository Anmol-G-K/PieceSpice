"""Run the golden-vs-ROM runtime benchmark for all configured vendors.

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

from simulation.testbenches.speedup_bench import run_switching_benchmark  # noqa: E402

VENDORS = ("irfs7730pbf", "csd16412q5a", "epc2216", "cpm2-1200")
CASES = {
    "short_forced": {"cycles": 3, "max_step": 2e-9},
    "short_adaptive": {"cycles": 3, "max_step": None},
    "long_adaptive": {"cycles": 20, "max_step": None},
}
REPEATS = 3


def main() -> None:
    out = {"schema": "piecespice.speedup/1.0", "cases": CASES,
           "repeats": REPEATS, "vendors": {}}
    for vendor in VENDORS:
        cfg = yaml.safe_load((ROOT / "config" / "inputs" /
                              f"{vendor}.yaml").read_text(encoding="utf-8"))
        d = cfg["dpt"]
        pins_golden = cfg["pins"]["pin_order"]
        pins_rom = ["drain", "gate", "source"]
        rom_lib = ROOT / "reports" / "models" / f"{vendor}_PIECESPICE_T2.lib"
        out["vendors"][vendor] = {}
        for case, params in CASES.items():
            common = dict(vbus=d["vbus"], vgs_on=d["vgs_on"],
                          i_load=d["i_load_target"], repeats=REPEATS,
                          **params)
            entry = {}
            for label, lib, sub, pins in (
                ("golden", ROOT / cfg["library"], cfg["subcircuit"], pins_golden),
                ("rom", rom_lib, rom_lib.stem, pins_rom),
            ):
                try:
                    entry[label] = run_switching_benchmark(lib, sub, pins,
                                                           **common)
                except Exception as exc:  # no fabricated timing on failure
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
