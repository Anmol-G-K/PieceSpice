"""Optimal continuous piecewise linear fitting using pwlf."""

from typing import Tuple, List
import numpy as np
import pwlf


def fit_piecewise_linear(
    x_data: np.ndarray,
    y_data: np.ndarray,
    n_segments: int = 4
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit optimal continuous piecewise linear segments to (x, y) data.

    Returns:
        breakpoints_x: Coordinates of breakpoints
        predicted_y: Evaluated piecewise model on original x_data
        rmse: Root-mean-square error of the fit
    """
    model = pwlf.PiecewiseLinFit(x_data, y_data)
    breakpoints = model.fit(n_segments)
    y_pred = model.predict(x_data)
    rmse = float(np.sqrt(np.mean((y_data - y_pred) ** 2)))
    return np.array(breakpoints), np.array(y_pred), rmse
