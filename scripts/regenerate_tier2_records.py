"""Regenerate the Tier-2 comparison JSONs from the current emitted lib.

Writes, for every configured vendor:

* ``reports/comparison/<vendor>/full_sweep_comparison.json``
* ``reports/comparison/<vendor>/endpoint_comparison.json``

Every number is measured by a live LTspice DC sweep of the emitted Tier-2
library against the extracted golden Curve-1 family.  No value is copied from
an older artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

from config.yaml_io import load_input
from metrics.compare import compare_curve
from pipeline.fitting import (
    compare_tier2_vs_golden,
    compare_tier2_vs_golden_full,
)
from pipeline.registry import ROOT, VENDORS, lib_path


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def regenerate(vendor: str, *, n_points: int = 41) -> dict:
    lib = lib_path(vendor, 2)
    if not lib.exists():
        raise FileNotFoundError(
            f"{vendor}: Tier-2 lib missing at {lib}; run fit_pwl_model first")
    rows = compare_tier2_vs_golden_full(
        vendor, lib_path=lib, n_points=n_points)
    full = []
    for vgs, vds, golden, candidate in rows:
        comparison = compare_curve(
            vds, golden, candidate, name=f"Vgs = {vgs:g} V")
        full.append({
            "curve": comparison.name,
            "points": comparison.n_points,
            "rmse": comparison.rmse,
            "rel_rmse_pct": comparison.rel_rmse_pct,
            "peak_rmse_pct": comparison.peak_rmse_pct,
            "max_err": comparison.max_abs_error,
            "max_rel_pct": comparison.max_rel_error_pct,
            "mape_pct": comparison.mape_pct,
            "r_squared": comparison.r_squared,
            "pass": comparison.passed,
        })
    endpoint = []
    endpoint_rows, _worst = compare_tier2_vs_golden(
        vendor, lib_path=lib)
    for vgs, golden, tier2, rel in endpoint_rows:
        endpoint.append({
            "vgs_v": float(vgs),
            "golden_a": float(golden),
            "tier2_a": float(tier2),
            "rel_err_pct": float(rel),
        })
    base = ROOT / "reports" / "comparison" / vendor
    _write(base / "full_sweep_comparison.json", full)
    _write(base / "endpoint_comparison.json", endpoint)
    return {
        "vendor": vendor,
        "lib": str(lib.relative_to(ROOT)),
        "curves": len(full),
        "endpoint_rows": len(endpoint),
        "full_sweep_comparison": str(
            (base / "full_sweep_comparison.json").relative_to(ROOT)),
        "endpoint_comparison": str(
            (base / "endpoint_comparison.json").relative_to(ROOT)),
    }


def main() -> None:
    cfg = load_input(VENDORS[0])
    n_points = int(cfg["conditions"].get("Vds_points", 41))
    for vendor in VENDORS:
        print(json.dumps(regenerate(vendor, n_points=n_points), sort_keys=True))


if __name__ == "__main__":
    main()
