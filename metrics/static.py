"""Static curve comparison metrics (L_inf, L_2 RMSE, relative error)."""

from typing import Dict
import numpy as np
from pydantic import BaseModel


class StaticErrorMetrics(BaseModel):
    max_abs_error: float
    rmse: float
    relative_rmse_pct: float
    peak_rmse_pct: float = 0.0       # RMSE / max|golden|; stable for near-zero curves
    r_squared: float


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

    # R² correlation coefficient
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((y_golden - np.mean(y_golden)) ** 2)
    r2 = float(1.0 - (ss_res / max(1e-9, ss_tot)))

    return StaticErrorMetrics(
        max_abs_error=max_err,
        rmse=rmse,
        relative_rmse_pct=rel_rmse,
        peak_rmse_pct=peak_rmse,
        r_squared=r2,
    )
