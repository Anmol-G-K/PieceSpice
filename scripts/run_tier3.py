"""Build and validate Tier-3 TPWL terminal models for the configured vendors.

Usage:
    uv run python scripts/run_tier3.py                 # all four vendors
    uv run python scripts/run_tier3.py irfs7730pbf     # one vendor
"""
from __future__ import annotations
import math

import sys

from pipeline.tier3 import run_vendor

VENDORS = ("irfs7730pbf", "csd16412q5a", "epc2216", "cpm2-1200")
SHOW = ("tr_ns", "tf_ns", "e_on_uj", "e_off_uj")


def _fmt(v, pct=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:+.1f}%" if pct else f"{v:.4g}"


def main() -> None:
    names = tuple(sys.argv[1:]) or VENDORS
    for vendor in names:
        record = run_vendor(vendor)
        algo = record["algorithm"]
        ff = record["function_fit"]
        dpt = record["dpt_run"]
        print(f"{vendor:14s} cells={algo['cells']:2d} "
              f"static={_fmt(ff['static_curve_rmse_rel'])} "
              f"static_emitted={_fmt(ff['emitted_static_curve_rmse_rel'])} "
              f"DPT={'complete' if dpt['completed'] else 'FAILED'} "
              f"points={dpt['trace_points']}/{dpt['golden_trace_points']} "
              f"iterations={_fmt(dpt['iterations'])}", flush=True)
        wf = dpt.get("waveform", {})
        if wf:
            print(f"    waveform Vds={_fmt(wf.get('vds_rel_rmse_pct'))}% "
                  f"Id={_fmt(wf.get('id_rel_rmse_pct'))}% "
                  f"compare_points={wf.get('compare_points')}", flush=True)
        for m in SHOW:
            row = record["dpt_comparison"][m]
            t3, t2 = row["tier3"], row.get("tier2")
            t3_txt = (f"{t3:9.4g} ({_fmt(row['tier3_rel_err_pct'], True)})"
                      if t3 is not None else "N/A")
            t2_txt = (f"{t2:9.4g} "
                      f"({_fmt(row.get('tier2_rel_err_pct'), True)})"
                      if t2 is not None else "N/A")
            print(f"    {m:9s} golden={_fmt(row['golden']):>9s} "
                  f"tier3={t3_txt}  tier2={t2_txt}", flush=True)
        rt = record["runtime"]
        g, t3 = rt["golden"], rt["tier3"]
        print(f"    runtime golden: solver={_fmt(g['solver_elapsed_s'])}s "
              f"wall={_fmt(g['wall_clock_s'])}s "
              f"iterations={_fmt(g['iterations'])} "
              f"points={_fmt(g['trace_points'])}", flush=True)
        print(f"    runtime tier3:  solver={_fmt(t3['solver_elapsed_s'])}s "
              f"wall={_fmt(t3['wall_clock_s'])}s "
              f"iterations={_fmt(t3['iterations'])} "
              f"points={_fmt(t3['trace_points'])} "
              f"complete={rt['tier3_bench_completed']}", flush=True)
        print(f"    solver_speedup={_fmt(rt['solver_speedup'])} "
              f"(N/A unless both benches completed)", flush=True)


if __name__ == "__main__":
    main()
