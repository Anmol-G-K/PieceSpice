"""Load the extracted Curve-1 output family for TPWL channel fitting.

The DPT trajectory drives only two Vgs levels (0 and vgs_on), so a model
fitted on it alone cannot represent the Vgs family — the project's headline
metric.  These are the same extracted rows the Tier-2 channel fit uses,
loaded from config/outputs/<vendor>/curve1_output_characteristics.yaml.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_output_family(vendor: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (vgs_levels, vds_grid, Id[level, point]) from Curve 1."""
    path = ROOT / "config" / "outputs" / vendor / \
        "curve1_output_characteristics.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    cond = data["conditions_used"]
    levels = np.asarray(cond["Vgs_sweep"], dtype=float)
    grid = np.linspace(0.0, float(cond["Vds_sweep_max"]),
                       int(cond["Vds_points"]))
    curves = np.array([
        np.asarray(data["extracted"][f"curve_id_vgs_{g:g}_v"], dtype=float)
        for g in levels])
    if curves.shape != (len(levels), len(grid)):
        raise RuntimeError(f"{vendor}: Curve-1 shape mismatch")
    return levels, grid, curves


def static_samples(vendor: str):
    """Quasi-static training points (operating coords, Id, Vgs level)."""
    levels, grid, curves = load_output_family(vendor)
    vgs = np.repeat(levels[:, None], len(grid), axis=1).ravel()
    vds = np.tile(grid, len(levels))
    u = np.vstack([vgs, vds])
    return u, curves.ravel(), np.repeat(levels, len(grid))
