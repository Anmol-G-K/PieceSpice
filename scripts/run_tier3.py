"""Build and validate Tier-3 TPWL terminal models for the configured vendors.

Usage:
    uv run python scripts/run_tier3.py                 # all four vendors
    uv run python scripts/run_tier3.py irfs7730pbf     # one vendor
"""
from __future__ import annotations

import sys

from tier3.pipeline import run_vendor

VENDORS = ("irfs7730pbf", "csd16412q5a", "epc2216", "cpm2-1200")
SHOW = ("tr_ns", "tf_ns", "e_on_uj", "e_off_uj")


def main() -> None:
    names = tuple(sys.argv[1:]) or VENDORS
    for vendor in names:
        record = run_vendor(vendor)
        algo = record["algorithm"]
        ff = record["function_fit"]
        st = ff.get("static_curve_rmse_rel")
        st_txt = f"{st:.3g}" if st is not None else "N/A"
        print(f"{vendor:14s} cells={algo['cells']:2d} static={st_txt}"
              f" fit_id={ff['heldout_id_rmse_rel']:.3g}", flush=True)
        for m in SHOW:
            row = record["dpt_comparison"][m]
            t3, t2 = row["tier3"], row.get("tier2")
            t3_txt = (f"{t3:9.4g} ({row['tier3_rel_err_pct']:+7.1f}%)"
                      if t3 is not None else "N/A")
            t2_txt = (f"{t2:9.4g} "
                      f"({row.get('tier2_rel_err_pct', float('nan')):+7.1f}%)"
                      if t2 is not None else "N/A")
            print(f"    {m:9s} golden={row['golden']:9.4g} "
                  f"tier3={t3_txt}  tier2={t2_txt}", flush=True)


if __name__ == "__main__":
    main()
