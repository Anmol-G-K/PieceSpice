"""Unit tests for metrics and error calculation."""

import numpy as np
from metrics.static import compute_curve_error
from metrics.loss import compute_switching_energy
from metrics.compare import STATIC_SCORE_PROVENANCE, recorded_static_summary


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
