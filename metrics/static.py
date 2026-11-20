"""Static curve comparison metrics (L_inf, L_2 RMSE, relative error)."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel


class StaticErrorMetrics(BaseModel):
    max_abs_error: float
    rmse: float
    relative_rmse_pct: float
    peak_rmse_pct: float = 0.0       # RMSE / max|golden|; stable for near-zero curves
    r_squared: float | None


def compute_curve_error(y_golden: np.ndarray, y_candidate: np.ndarray) -> StaticErrorMetrics:
    """Compute mathematical error norms between golden SPICE and candidate model."""
    diff = y_golden - y_candidate
    max_err = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # Relative RMSE normalized by peak-to-peak span
    span = np.max(y_golden) - np.min(y_golden)
    rel_rmse = float((rmse / max(1e-9, span)) * 100.0)
    peak = float(np.max(np.abs(y_golden)))
    peak_rmse = float((rmse / max(1e-9, peak)) * 100.0)

    # R^2: never floor ss_tot with an absolute epsilon.  A 1e-9 floor is
    # orders of magnitude larger than the variance of a near-zero golden
    # curve (~1e-13 A^2) and turns a true R^2 of 0.88 into 0.999983.
    # A flat golden curve has no variance to explain: R^2 is undefined (None).
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((y_golden - np.mean(y_golden)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else None

    return StaticErrorMetrics(
        max_abs_error=max_err,
        rmse=rmse,
        relative_rmse_pct=rel_rmse,
        peak_rmse_pct=peak_rmse,
        r_squared=r2,
    )
