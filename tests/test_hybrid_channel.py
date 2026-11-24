"""Offline checks for the standalone hinge_c hybrid channel emitter.

No LTspice, no benchmarks: these tests evaluate the emitted ASCII VALUE
expression itself on the supplied rows and verify the reported error.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.hybrid_channel import (
    _delta_slopes,
    _fmt,
    evaluate_hybrid_channel,
    fit_hybrid_channel,
)


def _synthetic_family():
    """Nonuniform Vds grid, nonlinear Vgs dependence, one sub-threshold row."""
    vds = np.array([0.0, 0.02, 0.06, 0.13, 0.21, 0.35, 0.55, 0.8,
                    1.2, 1.9, 2.7, 3.6, 4.8])
    vgs = np.array([0.0, 1.5, 2.5, 4.0, 6.0])
    amp = 2.0 * np.maximum(vgs - 1.0, 0.0) ** 1.3
    vsat = 0.4 + 0.1 * vgs
    id_rows = np.array([
        np.where(g > 1.0,
                 a * vds / (vds + s) + 0.05 * a * (vds / vds[-1]) ** 2,
                 0.0)
        for g, a, s in zip(vgs, amp, vsat)])
    return vgs, vds, id_rows


def test_emitted_expression_reproduces_rows_with_delta_slopes():
    vgs, vds, rows = _synthetic_family()
    fit = fit_hybrid_channel(vgs, vds, rows, rel_tolerance=0.005)

    pred = evaluate_hybrid_channel(fit, vgs[:, None],
                                   np.broadcast_to(vds, rows.shape))
    assert pred.shape == rows.shape
    assert np.all(np.isfinite(pred))
    max_err = float(np.max(np.abs(pred - rows)))
    assert max_err == pytest.approx(fit.max_abs_fit_error_a, rel=1e-9, abs=1e-12)
    assert fit.meta["achieved_rel_rmse"] <= 0.005 * (1 + 1e-9)

    assert fit.expression.isascii()
    assert "sqrt(" in fit.expression and "abs(" in fit.expression
    assert "^2" in fit.expression
    assert "V(gi,ss)" in fit.expression and "V(di,ss)" in fit.expression
    assert 0 < fit.meta["n_knees"] < len(vds)  # chosen by scan, not hardcoded

    spacing = float(np.median(np.diff(vds)))
    assert fit.smoothing_width_v == pytest.approx(spacing)
    assert fit.meta["smoothing_width_derivation"]["method"] == \
        "median_positive_vds_spacing"

    # Every emitted Vgs ramp coefficient is a *delta* slope, never the
    # accumulated absolute slope that diverged 21x in the repository audit.
    assert _delta_slopes([0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0]) == \
        pytest.approx([2.0, 0.0, 0.0])
    coeff_rows = np.asarray(fit.meta["vgs_coefficients"])
    for col in range(coeff_rows.shape[1]):
        y = coeff_rows[:, col]
        for delta in _delta_slopes(fit.vgs_knots, y):
            if delta != 0.0:
                assert _fmt(delta) in fit.expression

    # Lower Vds endpoint is pinned exactly; conduction direction is positive.
    assert np.allclose(evaluate_hybrid_channel(fit, vgs, vds[0]),
                       rows[:, 0], atol=1e-8 * float(np.ptp(rows)))
    assert float(evaluate_hybrid_channel(fit, vgs[-1], vds[-1])) > 0.0


@pytest.mark.parametrize("vendor", ["csd16412q5a", "cpm2-1200"])
def test_real_curve1_families_fit_within_repository_tolerance(vendor):
    from approximation.static_curves import load_output_family

    vgs, vds, rows = load_output_family(vendor)
    fit = fit_hybrid_channel(vgs, vds, rows)

    pred = evaluate_hybrid_channel(fit, vgs[:, None],
                                   np.broadcast_to(vds, rows.shape))
    assert np.all(np.isfinite(pred))
    assert fit.max_abs_fit_error_a > 0.0  # measured from the emitted string
    assert fit.meta["achieved_rel_rmse"] <= 0.005 * (1 + 1e-9)
    assert np.max(np.abs(pred - rows)) == pytest.approx(
        fit.max_abs_fit_error_a, rel=1e-9, abs=1e-12)
    assert len(fit.vds_knees) == len(vgs)
    assert all(knees == fit.vds_knees[0] for knees in fit.vds_knees)
    assert np.allclose(evaluate_hybrid_channel(fit, vgs, vds[0]),
                       rows[:, 0], atol=1e-9)
    assert "known_limitation" in fit.meta  # DPT regression is reported, not hidden


def test_custom_nodes_are_emitted_and_evaluated():
    vgs, vds, rows = _synthetic_family()
    fit = fit_hybrid_channel(vgs, vds, rows,
                             gate_node="V(gate_ref,0)",
                             drain_node="V(drain_ref,0)")
    assert "V(gate_ref,0)" in fit.expression
    assert "V(drain_ref,0)" in fit.expression
    pred = evaluate_hybrid_channel(fit, vgs[:, None],
                                   np.broadcast_to(vds, rows.shape))
    assert np.all(np.isfinite(pred))
    assert fit.meta["achieved_rel_rmse"] <= 0.005 * (1 + 1e-9)


def test_invalid_inputs_and_unfittable_family_raise():
    vgs, vds, rows = _synthetic_family()
    with pytest.raises(ValueError, match="id_rows shape"):
        fit_hybrid_channel(vgs, vds, rows[:-1])
    with pytest.raises(ValueError, match="strictly increasing"):
        fit_hybrid_channel(vgs, np.array([0.0, 0.2, 0.1, 0.4]),
                           rows[:, :4])
    with pytest.raises(ValueError, match="duplicates"):
        fit_hybrid_channel(np.array([0.0, 1.0, 1.0, 2.0, 3.0]), vds, rows)
    with pytest.raises(ValueError, match="ASCII"):
        fit_hybrid_channel(vgs, vds, rows, gate_node="V(g;halt)")
    # Zero tolerance is unreachable: the emitter must raise, not approximate.
    with pytest.raises(RuntimeError, match="refusing to emit"):
        fit_hybrid_channel(vgs, vds, rows, rel_tolerance=0.0)
