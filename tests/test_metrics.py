"""Unit tests for metrics and error calculation."""

import matplotlib
matplotlib.use("Agg")  # Select noninteractive backend before any pyplot import.

import numpy as np
import pytest
from matplotlib import pyplot as plt
from metrics.static import compute_curve_error
from metrics.loss import compute_switching_energy
from metrics.compare import (
    STATIC_SCORE_PROVENANCE, ComparisonReport, CurveComparison,
    build_report, plot_residuals, recorded_static_summary, report_to_rows)


def test_curve_error():
    y1 = np.array([1.0, 2.0, 3.0, 4.0])
    y2 = np.array([1.0, 2.1, 2.9, 4.0])
    err = compute_curve_error(y1, y2)
    assert err.max_abs_error <= 0.11
    assert err.rmse < 0.1
    assert err.r_squared > 0.99


def test_switching_energy():
    t = np.linspace(0, 10e-6, 1000)
    v = np.full_like(t, 10.0)
    i = np.full_like(t, 2.0)
    # P = 20 W for 10 us -> E = 200 uJ
    energy_uj = compute_switching_energy(t, v, i, 0, 10e-6)
    assert np.isclose(energy_uj, 200.0, rtol=1e-2)


def test_recorded_static_summary_uses_canonical_peak_mean():
    g1 = np.array([0.0, 1.0, 2.0, 3.0])
    s1 = np.array([0.0, 1.0, 2.0, 4.0])
    g2 = np.array([0.0, 10.0, 20.0])
    s2 = np.array([0.0, 12.0, 20.0])
    summary = recorded_static_summary([(g1, s1), (g2, s2)])

    expected = [
        compute_curve_error(g, s).peak_rmse_pct for g, s in ((g1, s1), (g2, s2))
    ]
    assert np.allclose(summary["per_curve_peak_rel_rmse_pct"], expected)
    assert np.isclose(summary["full_sweep_rel_rmse_pct"], float(np.mean(expected)))
    assert np.isclose(summary["peak_normalised_rmse_pct"], float(np.median(expected)))
    assert np.isclose(summary["worst_endpoint_rel_error_pct"],
                      100.0 * abs(4.0 - 3.0) / 3.0)
    assert summary["n_curves"] == 2


def test_recorded_static_summary_gates_mean_not_median():
    # Per-curve peak errors 10/20/60 %: mean 30 %, median 20 %.  The bar is
    # the arithmetic mean; the median must remain a diagnostic only.
    curves = []
    for delta in (20.0, 40.0, 120.0):
        g = np.array([0.0, 0.0, 100.0, 100.0])
        s = g.copy()
        s[0] = delta
        curves.append((g, s))
    summary = recorded_static_summary(curves)

    assert np.allclose(summary["per_curve_peak_rel_rmse_pct"], [10.0, 20.0, 60.0])
    assert np.isclose(summary["full_sweep_rel_rmse_pct"], 30.0)
    assert not np.isclose(summary["full_sweep_rel_rmse_pct"], 20.0)
    assert np.isclose(summary["peak_normalised_rmse_pct"], 20.0)


def test_recorded_static_summary_carries_canonical_provenance():
    summary = recorded_static_summary([
        (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.1, 2.0]))
    ])

    assert summary["provenance"] == STATIC_SCORE_PROVENANCE
    assert summary["gated_estimator"] == STATIC_SCORE_PROVENANCE["gated_estimator"]
    assert STATIC_SCORE_PROVENANCE["kind"] == "measured"
    assert "full_sweep_comparison.json" in STATIC_SCORE_PROVENANCE["source"]
    assert STATIC_SCORE_PROVENANCE["diagnostics_not_gated"] == [
        "span_normalised_rmse_pct", "peak_normalised_rmse_pct",
    ]
    # Records get a copy: a consumer cannot poison the exported constant.
    summary["provenance"]["kind"] = "mutated"
    assert STATIC_SCORE_PROVENANCE["kind"] == "measured"


def test_recorded_static_summary_skips_empty_and_handles_near_zero():
    empty = recorded_static_summary([
        (np.array([]), np.array([1.0])),
        (np.array([1.0]), np.array([])),
    ])
    for key in ("full_sweep_rel_rmse_pct", "span_normalised_rmse_pct",
                "peak_normalised_rmse_pct", "worst_endpoint_rel_error_pct"):
        assert empty[key] is None
    assert empty["n_curves"] == 0
    assert empty["per_curve_peak_rel_rmse_pct"] == []

    # A near-zero golden curve (~2e-7 A) must score against compute_curve_error
    # without a divide-by-zero blow-up.
    g = np.array([0.0, 2e-7])
    s = np.array([0.0, 4e-7])
    near = recorded_static_summary([(g, s)])
    assert near["n_curves"] == 1
    assert np.isclose(near["per_curve_peak_rel_rmse_pct"][0],
                      compute_curve_error(g, s).peak_rmse_pct)
    assert np.isfinite(near["full_sweep_rel_rmse_pct"])
    assert near["full_sweep_rel_rmse_pct"] > 0.0


def test_r_squared_is_true_on_near_zero_curve():
    # IRFS Vgs=3 V regime: golden variance ~1e-13 A^2.  The old absolute
    # 1e-9 floor inflated a true R^2 of 0.880748 to ~0.99999.
    scale = 1e-7
    golden = np.array([0.0, 1.0, 2.0, 3.0, 4.0]) * scale
    amp = scale * np.sqrt(10.0 * (1.0 - 0.880748) / 4.0)
    candidate = golden + np.array([1.0, -1.0, 0.0, 1.0, -1.0]) * amp

    err = compute_curve_error(golden, candidate)

    assert err.r_squared == pytest.approx(0.880748, abs=1e-6)
    assert err.r_squared < 0.99
    # Span/peak RMSE behaviour is unchanged: both normalise by ~4e-7.
    assert err.relative_rmse_pct == pytest.approx(err.peak_rmse_pct)


def test_r_squared_is_none_for_flat_golden():
    err = compute_curve_error(np.full(4, 5e-7), np.full(4, 6e-7))
    assert err.r_squared is None


def test_report_rows_keep_near_zero_absolute_errors():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    golden = np.array([0.0, 1e-7, 2e-7, 3e-7])
    candidate = golden + np.array([0.0, 1e-8, -2e-8, 3e-8])
    row = report_to_rows(build_report([("near_zero", x, golden, candidate)]))[0]

    assert row["rmse"] > 0.0
    assert row["max_err"] > 0.0
    assert row["rel_rmse_pct"] > 0.0
    assert row["r_squared"] is not None

    flat = report_to_rows(build_report([
        ("flat", x, np.full(4, 5e-7), np.full(4, 6e-7))]))[0]
    assert flat["r_squared"] is None
    assert flat["rmse"] > 0.0


def test_summary_line_reports_worst_curve_when_non_empty():
    report = ComparisonReport(target_rel_rmse_pct=5.0, curves=[
        CurveComparison(
            name="good", n_points=4, rmse=0.0, max_abs_error=0.0,
            rel_rmse_pct=1.0, max_rel_error_pct=1.0, mape_pct=1.0,
            r_squared=None, passed=True),
        CurveComparison(
            name="worst", n_points=4, rmse=0.0, max_abs_error=0.0,
            rel_rmse_pct=12.34, max_rel_error_pct=12.34, mape_pct=12.34,
            r_squared=None, passed=False),
    ])
    assert report.summary_line() == (
        "1/2 curves within 5% rel-RMSE target; worst = worst (12.34%)")


def test_plot_residuals_empty_pairs_defines_ylabel():
    fig = plot_residuals([])
    try:
        ylabel = fig.axes[0].get_ylabel()
        assert ylabel == "Residual (cand - golden) [% of span]"
    finally:
        plt.close(fig)
