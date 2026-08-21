"""Evaluation metrics for static curves, switching waveforms, and simulation speed."""

from metrics.static import compute_curve_error
from metrics.switching import extract_switching_times
from metrics.loss import compute_switching_energy

__all__ = [
    "compute_curve_error",
    "extract_switching_times",
    "compute_switching_energy",
]
