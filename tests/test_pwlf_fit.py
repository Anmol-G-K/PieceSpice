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
    # pwlf uses differential evolution: knees land within ~2e-5 of truth but
    # the refit/chaining residual is ~1e-5 absolute, so a fixed 1e-6 bound
    # fails intermittently under full-suite load. Compare to signal span.
    assert r.rmse < 1e-4 * float(np.ptp(y))
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


def test_adaptive_explores_past_first_improvement():
    """Bug #1 regression: early-exit must fire on STAGNATION, not on
    improvement. A sine signal keeps improving up to K~6; the old buggy
    check exited at K=3 with 7x worse RMSE."""
    x = np.linspace(0, 10, 200)
    y = 8 + 3 * np.sin(1.7 * x) + 0.2 * x
    r = fit_adaptive(x, y, rel_tolerance=1e-12, max_segments=10)
    # The old bug returned K=3 / rmse=1.68; correct behavior reaches deep K.
    assert r.n_segments >= 5
    assert r.rmse < 0.3


def _worst_break_jump(res):
    b = res.breakpoints_x
    interior = b[1:-1]
    if len(interior) == 0:
        return 0.0
    return max(
        abs(res.slopes[i + 1] * bi + res.intercepts[i + 1]
            - (res.slopes[i] * bi + res.intercepts[i]))
        for i, bi in enumerate(interior))


def test_starved_segment_merge_keeps_continuity():
    """Bug #2 regression: a segment with <2 samples used to be filled with
    a parallel line through the lone sample -> C0 discontinuity (45-unit
    jump in the audit case). Merged/connector repair must keep the model
    continuous to float precision."""
    x = np.array([0., 1., 5.])
    y = np.array([0., 1., 50.])
    r = fit_piecewise_linear.__globals__['_result_from_breaks'](
        x, y, np.array([0., 3., 6.]))
    assert _worst_break_jump(r) < 1e-9


def test_interior_starved_segment_merge_keeps_continuity():
    x = np.array([0., 1., 2., 6., 7., 8.])
    y = np.array([0., 1., 2., 30., 31., 32.])
    r = fit_piecewise_linear.__globals__['_result_from_breaks'](
        x, y, np.array([0., 4., 5., 9.]))
    assert _worst_break_jump(r) < 1e-9


def test_random_break_placements_stay_continuous():
    rng = np.random.default_rng(42)
    for _ in range(20):
        xs = np.sort(rng.uniform(0, 10, 40))
        ys = np.cumsum(rng.normal(0, 1, 40))
        brk = np.concatenate([
            [xs[0]], np.sort(rng.uniform(xs[2], xs[-2], 4)), [xs[-1]]])
        r = fit_piecewise_linear.__globals__['_result_from_breaks'](xs, ys, brk)
        assert _worst_break_jump(r) < 1e-9
