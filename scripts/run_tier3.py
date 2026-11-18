"""Build and validate Tier-3 TPWL/POD-DEIM models for the configured vendors.

Usage:
    uv run python scripts/run_tier3.py                 # all four vendors
    uv run python scripts/run_tier3.py irfs7730pbf     # one vendor
"""
from __future__ import annotations

import sys

from tier3.pipeline import run_vendor

VENDORS = ("irfs7730pbf", "csd16412q5a", "epc2216", "cpm2-1200")


def main() -> None:
    names = tuple(sys.argv[1:]) or VENDORS
    for vendor in names:
        record = run_vendor(vendor)
        py = record["python_validation"]
        sp = record["spice_validation"]
        py_rmse = py.get("rmse_mean", py.get("reason", "N/A"))
        if sp.get("evaluable"):
            detail = (f"spice_id_rel={sp['rmse_id_rel']:.4g} "
                      f"spice_ig_rel={sp['rmse_ig_rel']:.4g}")
        else:
            detail = f"spice=N/A ({sp.get('error', 'failed')[:60]})"
        print(f"{vendor:14s} r={record['algorithm']['pod_rank']:2d} "
              f"s={record['algorithm']['linearization_points']:2d} "
              f"py_rmse={py_rmse} {detail}", flush=True)


if __name__ == "__main__":
    main()
