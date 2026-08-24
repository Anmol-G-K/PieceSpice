"""Tests for approximation.pwlf_fit (M3.1)."""

import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

from approximation.pwlf_fit import (
    fit_adaptive, fit_piecewise_linear, fit_piecewise_linear_tuple,
    PWLFitResult,
)


def _piecewise_signal():
    """3-segment ground truth: slope 2, then flat, then slope -1."""
    x = np.linspace(0.0, 10.0, 301)
    y = np.piecewise(
        x,
        [x < 4.0, (x >= 4.0) & (x < 7.0)],
        [lambda t: 2.0 * t, 8.0, lambda t: 8.0 - (t - 7.0)])
    return x, y


def test_exact_k_recovers_truth():
    x, y = _piecewise_signal()
    r = fit_piecewise_linear(x, y, n_segments=3)
    assert r.n_segments == 3
    assert r.rmse < 1e-6
    # breakpoints near the true knees 4 and 7
    interior = r.breakpoints_x[1:-1]
    assert abs(interior[0] - 4.0) < 0.2
    assert abs(interior[1] - 7.0) < 0.2


def test_adaptive_selects_minimal_k():
    x, y = _piecewise_signal()
    r = fit_adaptive(x, y, rel_tolerance=0.001)
    assert r.rmse <= 0.001 * (y.max() - y.min())
    assert r.converged
    # should not need more than the structural minimum (+slack)
    assert r.n_segments <= 4


def test_adaptive_respects_cap_and_reports_non_convergence():
    # Smooth curved signal: PWL can only approximate it, so an absurdly
    # tight tolerance must stay unreachable within the segment cap.
    x = np.linspace(0.0, 10.0, 401)
    y = 8.0 + 3.0 * np.sin(1.7 * x) + 0.2 * x
    r = fit_adaptive(x, y, rel_tolerance=1e-12, max_segments=5)
    assert not r.converged
    assert "not reachable" in " ".join(r.notes)
    assert r.n_segments <= 5


def test_evaluate_matches_predicted():
    x, y = _piecewise_signal()
    r = fit_piecewise_linear(x, y, n_segments=3)
    probe = np.array([1.0, 4.5, 8.5])
    assert np.allclose(r.evaluate(probe), r.predicted_y[
        np.searchsorted(x, probe).clip(0, len(x) - 1)], atol=1e-6)


def test_single_segment_fallback():
    x = np.linspace(0, 1, 50)
    y = 3.0 * x + 1.0          # perfectly linear -> K=1 exact
    r = fit_piecewise_linear(x, y, n_segments=1)
    assert r.rmse < 1e-9


def test_nan_and_unsorted_input_sanitized():
    x = np.array([5.0, np.nan, 1.0, 3.0, 2.0, 9.0])
    y = np.array([10.0, 5.0, 2.0, 6.0, 4.0, np.inf])
    r = fit_adaptive(x, y, rel_tolerance=0.01)
    assert np.all(np.diff(r.breakpoints_x) > 0), "breakpoints must be sorted"
    assert np.isfinite(r.rmse)


def test_legacy_tuple_api():
    x, y = _piecewise_signal()
    breaks, pred, rmse = fit_piecewise_linear_tuple(x, y, 3)
    assert isinstance(rmse, float)
    assert len(breaks) == 4      # K+1 breakpoints
    assert len(pred) == len(x)


def test_too_few_samples_raises():
    with pytest.raises(ValueError):
        fit_piecewise_linear(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "--no-header"]))
