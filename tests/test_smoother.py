"""Tests for approximation.smoother (M3.4)."""
import math
import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

from approximation.smoother import (
    sigmoid, softplus, tanh_saturation, pwl_smooth,
    hyp_expr, hyp_value, delta_ramp_chain, eps_from_grid,
    smooth_relu_product, emit_qoss_expr, emit_transfer_ramp,
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


def test_pwl_smooth_warns_when_eps_exceeds_segment(caplog):
    import logging
    b = np.array([0., 1., 2.])
    vals = np.array([0., 1., 2.])
    with caplog.at_level(logging.WARNING, logger="approximation.smoother"):
        pwl_smooth(np.array([0.5]), b, vals, eps=2.0)
    assert any("overshoot" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# T0/T1 C1 shared emitter contract (unified spec 4.0/4.1; C1 study P1/P3/P7)
# ---------------------------------------------------------------------------

def _eval_hinge_chain(expr: str, term_value: float) -> float:
    """Evaluate a one-terminal emitted ramp string at a numeric value."""
    substituted = expr.replace("V(x)", f"({term_value!r})")
    return float(eval(substituted, {"sqrt": math.sqrt, "abs": abs}))


def test_hyp_expr_exact_ascii_and_eps_zero():
    """One sqrt-based ASCII hinge; eps=0 is the exact legacy C0 template."""
    text = hyp_expr("V(di,ss)", 0.0)
    assert text == ("0.5*((V(di,ss))+sqrt((V(di,ss))*(V(di,ss))"
                    "+0.000000e+00))")
    assert text.isascii()
    assert "sqrt(" in text and "abs(" not in text
    # eps^2 is formatted at repository precision (%.6e), never pre-rounded.
    assert "2.500000e-01" in hyp_expr("V(di,ss)", 0.5)
    # -0.0 is normalised to +0.0 so the legacy string is unchanged.
    assert hyp_expr("V(di,ss)", -0.0) == text


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"),
                                 -float("inf")])
def test_hyp_expr_rejects_invalid_eps(bad):
    with pytest.raises(ValueError):
        hyp_expr("V(di,ss)", bad)


def test_hyp_value_matches_expr_formula_and_hard_relu_at_zero():
    x = np.array([-2.0, -0.1, 0.0, 0.1, 2.0])
    assert np.allclose(hyp_value(x, 0.0), np.maximum(x, 0.0),
                       rtol=0.0, atol=1e-15)
    for eps in (0.1, 0.5):
        expected = 0.5 * (x + np.sqrt(x * x + eps * eps))
        assert np.allclose(hyp_value(x, eps), expected,
                           rtol=0.0, atol=1e-15)
    # Unshifted hinge at x=0 is eps/2 > 0 (R8); anchoring removes it.
    assert hyp_value(0.0, 0.5) == pytest.approx(0.25)
    assert hyp_value(-1.0, 0.0) == 0.0


def test_eps_from_grid_data_derived_rule():
    """P7: min(median positive spacing, half the minimum spacing)."""
    grid = [0.0, 1.0, 2.0, 5.0, 10.0]   # spacings 1, 1, 3, 5 -> 0.5
    d = np.diff(grid)
    assert eps_from_grid(grid) == pytest.approx(
        min(float(np.median(d)), 0.5 * float(np.min(d))))
    assert eps_from_grid(grid) == pytest.approx(0.5)
    assert eps_from_grid([0.0, 1.0]) == pytest.approx(0.5)
    assert eps_from_grid([0.0, 0.0, 1.0]) == pytest.approx(0.5)
    assert eps_from_grid([1.0]) == 0.0     # no spacing -> legacy C0
    assert eps_from_grid([]) == 0.0
    with pytest.raises(ValueError):
        eps_from_grid([0.0, float("nan")])
    with pytest.raises(ValueError):
        eps_from_grid([[0.0, 1.0]])


def test_delta_ramp_chain_eps_zero_is_legacy_abs_chain():
    xs, ys = [0.0, 1.0, 3.0, 6.0], [0.0, 2.0, 5.0, 9.0]
    text = delta_ramp_chain("V(x)", xs, ys, 0.0)
    assert text.isascii()
    assert "abs(" in text and "sqrt(" not in text
    assert text.startswith("0.000000e+00")
    # Chords 2, 1.5, 4/3 -> deltas 2, -0.5, -1/6 at breaks 0, 1, 3.
    assert (" + 2.000000e+00*((V(x)-0.000000e+00)"
            "+abs((V(x))-0.000000e+00))/2") in text
    assert (" + -5.000000e-01*((V(x)-1.000000e+00)"
            "+abs((V(x))-1.000000e+00))/2") in text


def test_delta_ramp_chain_eps_zero_rebuilds_table_without_accumulation():
    """Delta slopes rebuild the exact PWL table (audit: absolute slopes)."""
    xs, ys = [0.0, 1.0, 3.0, 6.0], [0.0, 2.0, 5.0, 9.0]
    text = delta_ramp_chain("V(x)", xs, ys, 0.0)
    for x, y in zip(xs, ys):
        assert _eval_hinge_chain(text, x) == pytest.approx(y, rel=1e-5)
    # A straight 5-point line needs one hinge, not five accumulating slopes.
    linear = delta_ramp_chain("V(x)", [0., 1., 2., 3., 4.],
                              [0., 2., 4., 6., 8.], 0.0)
    assert linear.count("abs(") == 1
    assert linear.count(" + ") == 1


def test_delta_ramp_chain_eps_positive_anchors_q_at_zero():
    """Shifted hinge subtracts the V=0 value: no injected charge (R8)."""
    xs, ys = [0.0, 1.0, 3.0, 6.0], [0.0, 2.0, 5.0, 9.0]
    text = delta_ramp_chain("V(x)", xs, ys, 0.5)
    assert text.isascii()
    assert "sqrt(" in text and "abs(" not in text
    assert text != delta_ramp_chain("V(x)", xs, ys, 0.0)
    # Residual is only %.6e coefficient rounding, not an eps/2 offset.
    assert abs(_eval_hinge_chain(text, 0.0)) < 1e-6 * max(ys)
    grid = np.linspace(xs[0], xs[-1], 25)
    values = [_eval_hinge_chain(text, float(v)) for v in grid]
    assert np.all(np.diff(values) >= -1e-6 * max(ys))


def test_delta_ramp_chain_calls_shared_hyp_expr(monkeypatch):
    """R11 drift guard: the emitter owns no fourth inline hyp copy."""
    import approximation.smoother as smoother

    monkeypatch.setattr(smoother, "hyp_expr",
                        lambda x, eps=0.0: f"SHARED_HYP({x})")
    text = delta_ramp_chain("V(x)", [0.0, 1.0], [0.0, 2.0], 0.5)
    assert "SHARED_HYP(" in text
    assert "sqrt(" not in text


def test_emit_wrappers_use_semantic_terms_and_shared_chain():
    xs, ys = [0.0, 1.0, 3.0], [0.0, 1.0, 3.0]
    assert emit_qoss_expr(xs, ys, 0.0) == \
        delta_ramp_chain("V(di,ss)", xs, ys, 0.0)
    assert emit_transfer_ramp(xs, ys, 0.5) == \
        delta_ramp_chain("V(gi,ss)", xs, ys, 0.5)
    assert emit_qoss_expr(xs, ys, 0.5).isascii()
    assert emit_transfer_ramp(xs, ys, 0.0).isascii()


def test_smooth_relu_product_matches_legacy_pwl_and_anchors_zero():
    xs, ys = [0.0, 1.0, 3.0, 6.0], [0.0, 2.0, 5.0, 9.0]
    x = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 6.0])
    assert np.allclose(smooth_relu_product(x, xs, ys, 0.0),
                       np.interp(x, xs, ys), rtol=0.0, atol=1e-12)
    assert smooth_relu_product(0.0, xs, ys, 0.5) == pytest.approx(
        0.0, abs=1e-15)
    values = smooth_relu_product(np.linspace(0.0, 6.0, 101), xs, ys, 0.5)
    assert np.all(np.diff(values) >= -1e-12)
    with pytest.raises(ValueError):
        smooth_relu_product(0.0, xs, ys, -1.0)


# ---------------------------------------------------------------------------
# Non-zero baseline (channel tanh knee) anchor: eps A/B continuity (Finding 1)
# ---------------------------------------------------------------------------

def test_delta_ramp_chain_eps_zero_bit_identity_nonzero_q0():
    """The C0 template is byte-for-byte unchanged for a non-zero baseline."""
    xs, ys = [0.0, 1.0, 2.0], [5.0, 7.0, 9.0]   # q0 = 5, one slope of 2
    text = delta_ramp_chain("V(x)", xs, ys, 0.0)
    assert text.isascii() and "sqrt(" not in text
    assert text == ("5.000000e+00 + 2.000000e+00*((V(x)-0.000000e+00)"
                    "+abs((V(x))-0.000000e+00))/2")
    assert np.allclose(smooth_relu_product(np.asarray(xs), xs, ys, 0.0),
                       ys, rtol=0.0, atol=1e-15)


def test_delta_ramp_chain_nonzero_q0_preserved_and_eps_continuous():
    """eps>0 keeps q0 (knee width); eps -> 0+ converges to the C0 chain."""
    xs, ys = [0.0, 1.0, 2.0], [5.0, 7.0, 9.0]
    text = delta_ramp_chain("V(x)", xs, ys, 0.1)
    tiny = delta_ramp_chain("V(x)", xs, ys, 1e-6)
    assert _eval_hinge_chain(text, 0.0) == pytest.approx(5.0, abs=1e-6)
    assert _eval_hinge_chain(tiny, 0.0) == pytest.approx(5.0, abs=1e-6)
    assert smooth_relu_product(0.0, xs, ys, 0.1) == pytest.approx(
        5.0, abs=1e-12)
    for x, y in zip((0.0, 0.5, 1.0, 1.5, 2.0), (5.0, 6.0, 7.0, 8.0, 9.0)):
        assert _eval_hinge_chain(tiny, x) == pytest.approx(y, abs=1e-4)
        assert smooth_relu_product(x, xs, ys, 1e-6) == pytest.approx(
            y, abs=1e-4)
        # The numeric mirror and the emitter share the same q0 anchor.
        assert _eval_hinge_chain(text, x) == pytest.approx(
            smooth_relu_product(x, xs, ys, 0.1), abs=1e-5)


def test_nonzero_q0_chain_stays_monotone():
    """A baseline offset does not break monotone-table monotonicity."""
    xs, ys = [0.0, 1.0, 3.0, 6.0], [5.0, 7.0, 10.0, 16.0]
    text = delta_ramp_chain("V(x)", xs, ys, 0.25)
    grid = np.linspace(0.0, 6.0, 49)
    emitted = np.array([_eval_hinge_chain(text, float(x)) for x in grid])
    numeric = smooth_relu_product(grid, xs, ys, 0.25)
    assert emitted[0] == pytest.approx(5.0, abs=1e-6)
    assert numeric[0] == pytest.approx(5.0, abs=1e-12)
    assert np.all(np.diff(emitted) >= -1e-6)
    assert np.all(np.diff(numeric) >= -1e-12)
    assert np.allclose(emitted, numeric, rtol=0.0, atol=1e-5)
