"""Stage-6 harness arithmetic: alignment, normalisation, shape correlation.

The three-arm comparison is only as good as its bookkeeping.  A mis-aligned
time grid or a span-normalised RMSE on a tiny signal silently turns a good
engine into a bad one, so the arithmetic is tested directly with synthetic arms
instead of only through a live LTspice run (which is far too slow for the suite
and would not isolate a maths error anyway).
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.harness.comparison import Arm, _edge_mask, _metric_delta, _waveform_difference


def _arm(t: np.ndarray, v_ds: np.ndarray, **kwargs) -> Arm:
    defaults = dict(
        name="arm", time=t, v_ds=v_ds, i_d=np.ones_like(t),
        v_gs=np.zeros_like(t), wall_s=1.0, elapsed_s=1.0, points=t.size,
    )
    defaults.update(kwargs)
    return Arm(**defaults)


def test_identical_arms_score_zero_error_and_unit_correlation() -> None:
    t = np.linspace(0.0, 1.0, 101)
    ref = _arm(t, np.sin(t))
    same = _arm(t, np.sin(t), name="same")
    result = _waveform_difference(ref, same, "v_ds")
    assert result["rel_rmse_pct"] == pytest.approx(0.0, abs=1e-12)
    assert result["pearson_r"] == pytest.approx(1.0)


def test_a_gain_error_and_a_phase_shift_score_differently() -> None:
    # Same RMSE, different shape: the correlation is what separates "scaled"
    # from "moved", and the engine comparison needs both numbers.
    t = np.linspace(0.0, 1.0, 201)
    ref = _arm(t, np.sin(2 * np.pi * t))
    scaled = _arm(t, 1.2 * np.sin(2 * np.pi * t), name="scaled")
    shifted = _arm(t, np.sin(2 * np.pi * (t + 0.1)), name="shifted")
    a = _waveform_difference(ref, scaled, "v_ds")
    b = _waveform_difference(ref, shifted, "v_ds")
    assert a["pearson_r"] == pytest.approx(1.0, abs=1e-9)
    assert b["pearson_r"] < 0.9


def test_error_is_normalised_by_the_reference_peak() -> None:
    # A tiny-span trace would otherwise report a huge relative error.
    t = np.linspace(0.0, 1.0, 51)
    ref = _arm(t, 1e-6 * np.ones_like(t))
    test = _arm(t, 2e-6 * np.ones_like(t), name="test")
    result = _waveform_difference(ref, test, "v_ds")
    assert result["rel_rmse_pct"] == pytest.approx(100.0, rel=1e-6)


def test_arms_are_sampled_onto_the_shared_span_only() -> None:
    ref_t = np.linspace(0.0, 1.0, 11)
    test_t = np.linspace(0.5, 1.5, 11)
    ref = _arm(ref_t, np.zeros_like(ref_t))
    test = _arm(test_t, np.ones_like(test_t), name="test")
    result = _waveform_difference(ref, test, "v_ds")
    # Reference peak is 0 here, so the guard keeps the divisor positive.
    assert result["points"] == 6
    assert result["max_abs"] == pytest.approx(1.0)


def test_edge_mask_is_centred_on_the_gate_transitions() -> None:
    t = np.linspace(0.0, 1e-6, 1001)
    v_gs = np.where(t < 5e-7, 0.0, 5.0)
    arm = _arm(t, np.zeros_like(t), v_gs=v_gs)
    mask = _edge_mask(arm, half_width=50e-9)
    assert mask.any()
    assert mask.sum() == pytest.approx(2 * 50, abs=30)   # ~100 ns window
    assert not mask[0] and not mask[-1]


def test_edge_fraction_separates_timing_from_fidelity() -> None:
    t = np.linspace(0.0, 1e-6, 1001)
    v_gs = np.where(t < 5e-7, 0.0, 5.0)
    ref = _arm(t, np.zeros_like(t), v_gs=v_gs)
    # An error only at the end of the trace is not an edge error.
    late = np.zeros_like(t)
    late[-20:] = 1.0
    result = _waveform_difference(ref, _arm(t, late, name="late"), "v_ds")
    assert result["error_in_edges_fraction"] == pytest.approx(0.0, abs=1e-9)


def test_metric_delta_reports_relative_and_absolute() -> None:
    ref = {"tr_ns": 10.0, "e_on_uj": 2.0, "i_peak_a": None}
    other = {"tr_ns": 12.0, "e_on_uj": 1.0, "i_peak_a": 5.0}
    delta = _metric_delta(ref, other)
    assert delta["tr_ns"]["delta"] == pytest.approx(2.0)
    assert delta["tr_ns"]["rel_pct"] == pytest.approx(20.0)
    assert delta["e_on_uj"]["rel_pct"] == pytest.approx(-50.0)
    assert delta["i_peak_a"]["delta"] is None
