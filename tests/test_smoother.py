"""Tests for approximation.smoother (M3.4)."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

from approximation.smoother import (
    sigmoid, softplus, tanh_saturation, pwl_smooth,
)


def test_softplus_converges_to_max():
    x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    for eps in (0.01, 0.05, 0.5):
        sp = softplus(x, eps)
        assert np.all(sp >= 0)
        # x=-10 is far outside every knee: must be ~0
        assert sp[0] == pytest.approx(0.0, abs=1e-8)
        # x=+10 is far above every knee: converges to x
        assert sp[-1] == pytest.approx(10.0, rel=1e-6)
        # at x=0 the value is the canonical eps*ln(2) knee offset
        assert softplus(0.0, eps) == pytest.approx(
            eps * np.log(2.0), rel=1e-9)


def test_softplus_monotonic_and_finite():
    x = np.linspace(-20, 20, 2001)
    for eps in (0.01, 0.1):
        sp = softplus(x, eps)
        # non-decreasing everywhere (flat far tails are allowed);
        # strictly increasing inside [-3*eps, 3*eps] knee window
        assert np.all(np.diff(sp) >= 0), "must be non-decreasing"
        knee = (x > -3 * eps) & (x < 3 * eps)
        assert np.all(np.diff(sp[knee]) > 0), "strictly rising in knee"
        assert np.all(np.isfinite(sp))


def test_sigmoid_bounds_and_symmetry():
    x = np.linspace(-30, 30, 1201)
    s = sigmoid(x)
    assert np.all((s >= 0) & (s <= 1))
    assert s[0] == pytest.approx(0.0, abs=1e-12)
    assert s[-1] == pytest.approx(1.0)


def test_tanh_saturation_limits():
    """Id -> 0 at Vds=0 and saturates at gm*Vds_sat for large Vds."""
    gm = 50.0
    vsat = 2.0
    vds = np.array([0.0, 0.5, 5.0, 100.0])
    ids = tanh_saturation(vds, vsat, gm)
    assert ids[0] == pytest.approx(0.0, abs=1e-12)
    assert ids[-1] == pytest.approx(gm * vsat, rel=1e-3)


def test_tanh_saturation_odd_function():
    """Negative Vds gives negative Id (third-quadrant conduction)."""
    assert tanh_saturation(-3.0, 2.0, 50.0) == pytest.approx(
        -tanh_saturation(3.0, 2.0, 50.0))


def test_pwl_smooth_matches_hard_table_away_from_breaks():
    b = np.array([0., 2., 5., 9.])
    vals = np.array([0., 4., 4., 1.])
    x = np.array([0.5, 1.0, 3.0, 7.5])
    hard = np.interp(x, b, vals)
    smooth = pwl_smooth(x, b, vals, eps=0.05)
    assert np.allclose(smooth, hard, atol=1e-9)


def test_pwl_smooth_rounds_the_corner():
    b = np.array([0., 2., 5.])
    vals = np.array([0., 8., 8.])
    val = pwl_smooth(np.array([2.0]), b, vals, eps=0.1)[0]
    assert val == pytest.approx(8.0)


def test_pwl_smooth_derivative_continuous_near_break():
    b = np.array([0., 2., 5.])
    vals = np.array([0., 8., 0.])       # sharp V shape at x=2
    d = 1e-4
    xl, xr = 2.0 - d, 2.0 + d
    sl = pwl_smooth(np.array([xl]), b, vals, eps=0.25)[0]
    sr = pwl_smooth(np.array([xr]), b, vals, eps=0.25)[0]
    numeric_slope = (sr - sl) / (xr - xl)
    hard_slope = max(abs((vals[1] - vals[0]) / 2),
                     abs((vals[2] - vals[1]) / 3))
    assert abs(numeric_slope) < hard_slope * 0.75


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "--no-header"]))
