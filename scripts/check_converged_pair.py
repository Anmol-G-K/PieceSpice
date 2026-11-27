"""Refine BOTH arms and see what the engine error really is.

Every earlier comparison had our solver and the reference sampled at different
densities on a 2-3 ns edge, which measures sampling phase rather than the
engine.  Here the reference deck's ``.tran`` ceiling is divided by the same
factor as the solver's, so both arms are converged by the same amount.
"""
from __future__ import annotations

import sys
import time

from engine.harness.comparison import (_compare, _conditions, _load_config,
                                       _ltspice_arm, _rom_subcircuit,
                                       _solver_arm, ROOT)
from engine.rom import solver_rom


def run(vendor: str, divs=(1, 4, 20), basis: str = "t4ann") -> None:
    cfg = _load_config(vendor)
    base = _conditions(cfg)
    lib = ROOT / "reports" / "models" / f"{vendor}_PIECESPICE_T4_ann.lib"
    name, pins = _rom_subcircuit(lib)
    rom = solver_rom(vendor, basis)
    print(f"\n{vendor}")
    for div in divs:
        cond = dict(base)
        cond["max_step"] = base["max_step"] / div
        t0 = time.perf_counter()
        b = _ltspice_arm("ref", lib, name, pins, cond, 1)
        wall = time.perf_counter() - t0
        t0 = time.perf_counter()
        c = _solver_arm(rom, cond, 1)
        spent = time.perf_counter() - t0
        comp = _compare(b, c)
        vds = comp["vds"]["rel_rmse_pct"]
        idd = comp["id"]["rel_rmse_pct"]
        print(f"  /{div:<3d} step {cond['max_step']*1e12:7.0f} ps  "
              f"pts ref {b.points:6d} solver {c.steps:6d}  "
              f"engine Vds {vds:5.2f} %  Id {idd:5.2f} %   "
              f"ltspice {wall:5.2f} s  solver {spent:6.3f} s")


if __name__ == "__main__":
    for v in sys.argv[1:] or ["epc2216", "csd16412q5a", "cpm2-1200",
                              "irfs7730pbf"]:
        run(v)
