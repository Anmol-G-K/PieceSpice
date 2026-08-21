"""Evaluation metrics for static curves, switching waveforms, and simulation speed."""

from piecespice.metrics.static import compute_curve_error
from piecespice.metrics.switching import extract_switching_times
from piecespice.metrics.loss import compute_switching_energy

__all__ = [
    "compute_curve_error",
    "extract_switching_times",
    "compute_switching_energy",
]
