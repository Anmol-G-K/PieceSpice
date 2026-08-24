"""Tests for approximation.rdp_compress (M3.2)."""

import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

from approximation.rdp_compress import (
    compress_rdp, rdp_epsilon_for_max_error,
)


def _knee_curve(n: int = 2001):
    """Diode-like exponential knee: dense data with one sharp feature."""
    x = np.linspace(0.0, 1.0, n)
    y = 1e-9 * np.exp(12.0 * x)
    return x, y


def test_perfectly_linear_collapses_to_endpoints():
    x = np.linspace(0, 10, 1000)
    y = 3 * x + 2
    xc, yc = compress_rdp(x, y, epsilon=1e-6)
    assert len(xc) == 2
    assert yc[0] == pytest.approx(y[0])
    assert yc[-1] == pytest.approx(y[-1])


def test_sharp_knee_is_retained():
    x = np.linspace(0, 1, 301)
    y = np.where(x < 0.5, 0.0, (x - 0.5) * 10.0)   # sharp corner at 0.5
    xc, yc = compress_rdp(x, y, epsilon=1e-3)
    # a point very near the corner must survive
    assert np.min(np.abs(xc - 0.5)) < 0.02


def test_endpoints_always_kept():
    x, y = _knee_curve()
    xc, yc = compress_rdp(x, y, epsilon=0.01)
    assert xc[0] == x[0] and xc[-1] == x[-1]
    assert yc[0] == y[0] and yc[-1] == y[-1]


def test_relative_epsilon_scales_across_units():
    # Uniform rescale (both axes, same factor) must give identical
    # compression with relative epsilon - that is the "unit change" case.
    x1 = np.linspace(0, 10, 501)
    y1 = np.sin(3 * x1)
    scale = 1e-6                   # volts -> microvolts, amps -> microamps
    x2, y2 = x1 * scale, y1 * scale

    xc1, _ = compress_rdp(x1, y1, epsilon=0.01, relative=True)
    xc2, _ = compress_rdp(x2, y2, epsilon=0.01, relative=True)
    assert len(xc1) == len(xc2)


def test_iterative_core_handles_large_arrays():
    n = 120_000                    # would blow the recursion limit if recursive
    x = np.linspace(0, 1, n)
    y = np.sin(50 * x)
    xc, yc = compress_rdp(x, y, epsilon=1e-4)
    assert 2 <= len(xc) < n
    assert np.all(np.diff(xc) > 0), "output x strictly increasing"


def test_nan_dropped_and_unsorted_input_sorted():
    x = np.array([3.0, np.nan, 1.0, 2.0])
    y = np.array([30.0, 99.0, 10.0, 20.0])
    xc, yc = compress_rdp(x, y, epsilon=1e-6)
    assert np.all(np.diff(xc) > 0)
    assert not np.any(np.isnan(xc))


def test_epsilon_binary_search_hits_budget():
    x, y = _knee_curve(1001)
    eps = rdp_epsilon_for_max_error(x, y, max_error=1e-6)
    xc, yc = compress_rdp(x, y, epsilon=eps)
    interp = np.interp(x, xc, yc)
    assert np.max(np.abs(interp - y)) <= 1e-6 * 1.001   # tiny slack
    # and it should be reasonably compressed, not just endpoints
    assert 3 <= len(xc) <= 200


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        compress_rdp(np.arange(5), np.arange(6))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "--no-header"]))


def test_rdp_rejects_negative_epsilon():
    with pytest.raises(ValueError):
        compress_rdp(np.linspace(0, 1, 50), np.linspace(0, 1, 50),
                     epsilon=-1e-3)


def test_rdp_rejects_nan_epsilon():
    with pytest.raises(ValueError):
        compress_rdp(np.arange(10.0), np.arange(10.0), epsilon=float("nan"))
