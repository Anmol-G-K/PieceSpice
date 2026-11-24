"""Offline tests for the ANN channel/charge contracts added in this pass.

No LTspice, no benchmarks.  Every fit is synthetic and every assertion is
structural or analytic; no fidelity or speed number is fabricated here.
"""
from __future__ import annotations

import numpy as np
import pytest

from approximation import ann_charge, ann_channel
from approximation.ann_charge import fit_charge_qv
from approximation.ann_channel import fit_channel_iv
from common.types import CurveData, PWLModelSpec
from models.ann_netlist_builder import (
    EXP_CAP,
    _fmt,
    build_ann_subcircuit,
    emit_ann_channel,
)
from models.netlist_builder import build_tier2_subcircuit

V_TH = 1.0


def _families(amplitudes=(2.0, 4.0, 7.0, 10.0, 13.0), n_vds: int = 9):
    vds = np.linspace(0.0, 5.0, n_vds)
    out = []
    for amp, vgs in zip(amplitudes, (2.0, 3.5, 5.0, 7.0, 10.0)):
        i_d = amp * (1.0 - np.exp(-vds / 1.1))
        out.append(CurveData(
            name=f"curve1_vgs_{vgs:g}", x_name="Vds", x_unit="V",
            y_name="Id", y_unit="A", x_data=vds.tolist(),
            y_data=i_d.tolist(), family_param="Vgs",
            family_value=vgs,
            metadata={"source": f"curve1_output_characteristics:{vgs:g}"}))
    return out


def _charge_curves():
    v = np.linspace(0.0, 5.0, 13)
    cap = (1.0e-9 * (1.0 + 0.6 * np.exp(-v / 1.2)
                     + 0.15 * np.exp(-((v - 2.5) / 0.25) ** 2)))
    c4 = CurveData(
        name="curve4_capacitances_coss", x_name="Vds", x_unit="V",
        y_name="Coss_eq", y_unit="F", x_data=v.tolist(),
        y_data=cap.tolist(),
        metadata={
            "ciss_f": 4.0e-9, "coss_f": 1.0e-9, "crss_f": 1.2e-10,
            "cap_ref_vds": 1.0, "activation": "isru", "neurons": 3,
            "source": "synthetic:curve4_capacitances",
        })
    xg = np.linspace(0.0, 10.0, 15)
    qg = 2.0e-9 * (1.0 - np.exp(-xg / 2.0))
    c5 = CurveData(
        name="curve5_gate_charge", x_name="Vgs", x_unit="V",
        y_name="Qg", y_unit="C", x_data=xg.tolist(),
        y_data=qg.tolist(), metadata={"source": "synthetic:curve5"})
    return c4, c5, v, cap


def _spec() -> PWLModelSpec:
    vds = [0.0, 0.1, 0.4, 1.0, 2.0, 3.5, 5.0]
    vgs = [2.0, 3.5, 5.0, 7.0, 10.0]
    rows = [[5.0 * max(g - V_TH, 0.0) * float(np.tanh(d / 0.7))
             for d in vds] for g in vgs]
    return PWLModelSpec(
        device_name="ann_channel_test",
        breakpoints_vds=vds, breakpoints_id=rows, vgs_values=vgs,
        rds_on=0.02, v_th=V_TH,
        body_diode_vsd=[0.5, 0.8], body_diode_isd=[0.1, 1.0],
        q_oss_vs_v=[0.0, 5.0], q_oss_charge_c=[0.0, 1e-9])


@pytest.fixture(scope="module")
def charge_fit():
    c4, c5, _, _ = _charge_curves()
    return fit_charge_qv(c4, c5)


# ---------------------------------------------------------------------------
# fitted charge emission (not measured resampling)
# ---------------------------------------------------------------------------
def test_charge_tables_are_the_fitted_net(charge_fit):
    c4, _, x_meas, cap = _charge_curves()
    q_meas = np.concatenate((
        [0.0], np.cumsum(0.5 * (cap[1:] + cap[:-1]) * np.diff(x_meas))))
    for name in ("oss", "iss", "rss"):
        table = charge_fit.tables[name]
        v = np.asarray(table["v"], float)
        q = np.asarray(table["q"], float)
        coef = charge_fit.coefficients[name]
        q0 = charge_fit.diagnostics[name]["q0_c"]
        raw = ann_charge._net_charge_q(coef, charge_fit.activation, q0, v)
        expected = np.maximum.accumulate(raw - raw[0])
        expected[0] = 0.0
        assert np.allclose(q, expected, rtol=1e-12, atol=1e-18)
        # The emitted nodes are the fitted net, not a resample of the
        # measured charge (the two grids differ).  Only ``oss`` shares the
        # Curve-4 voltage axis; ``iss``/``rss`` have their own sources.
        if name == "oss":
            resampled = np.interp(v, x_meas, q_meas)
            assert np.max(np.abs(q - resampled)) > 0.0
        assert float(q[0]) == 0.0
        assert np.all(np.isfinite(q))
        assert np.all(np.diff(q) >= 0.0)
        diag = charge_fit.diagnostics[name]
        assert diag["emission"] == "fitted_net_monotone_anchored"
        assert diag["q_zero_anchored"] is True
        assert diag["monotone"] is True
        assert np.isfinite(diag["fit_vs_measured_rmse"])
        assert diag["fit_vs_measured_rmse"] >= 0.0
    assert charge_fit.provenance["emission"] == \
        "fitted_net_monotone_anchored"
    assert 8 <= charge_fit.diagnostics["oss"]["table_points"] <= 32
    assert charge_fit.diagnostics["oss"]["max_cap_error_pct"] is not None
    assert charge_fit.diagnostics["oss"]["max_cap_error_pct"] >= 0.0


def test_charge_fit_fails_closed_on_non_finite_fitted_net(
        monkeypatch):
    c4, c5, _, _ = _charge_curves()

    def _bad(coef, activation, q0, v):
        return np.full(np.shape(v), np.nan)

    monkeypatch.setattr(ann_charge, "_net_charge_q", _bad)
    with pytest.raises(RuntimeError, match="non-finite"):
        fit_charge_qv(c4, c5)


def test_charge_elements_remain_ascii_and_use_fitted_tables(charge_fit):
    channel = fit_channel_iv(_families(), V_TH, 2.0, neurons=3,
                             guard_width=0.1)
    spec = _spec()
    net = build_ann_subcircuit(spec, channel, charge_fit)
    assert net.isascii()
    for element in ("C_QOSS ", "C_ISS ", "C_RSS "):
        lines = [line for line in net.splitlines()
                 if line.startswith(element)]
        assert len(lines) == 1 and "Q = {" in lines[0]


# ---------------------------------------------------------------------------
# held-out validation
# ---------------------------------------------------------------------------
def test_no_holdout_reports_not_measured_never_zero():
    fit = fit_channel_iv(_families(), V_TH, 2.0, neurons=3,
                         guard_width=0.1)
    for key in ("holdout_id_rms_rel", "holdout_gm_rms_rel",
                "holdout_gds_rms_rel", "holdout_n"):
        assert fit.diagnostics[key] == "not_measured"
    assert fit.diagnostics["holdout_selection"] is None
    assert fit.diagnostics["temperature_center"] is None
    assert fit.diagnostics["temperature_scale"] is None


def test_holdout_by_vgs_family_is_fit_on_the_rest():
    fit = fit_channel_iv(_families(), V_TH, 2.0, neurons=3,
                         guard_width=0.1, holdout_fraction=0.4)
    sel = fit.diagnostics["holdout_selection"]
    assert sel["method"] == "vgs_family"
    assert sel["n_holdout"] == fit.diagnostics["holdout_n"] > 0
    assert fit.diagnostics["train_samples"] + sel["n_holdout"] == 5 * 8
    for key in ("holdout_id_rms_rel", "holdout_gm_rms_rel",
                "holdout_gds_rms_rel"):
        value = fit.diagnostics[key]
        assert isinstance(value, float) and np.isfinite(value) and value >= 0


def test_holdout_mask_selects_whole_family():
    fams = _families()
    fit = fit_channel_iv(fams, V_TH, 2.0, neurons=3, guard_width=0.1,
                         holdout_mask=np.concatenate([
                             np.full(9, v == 10.0) for v in (2.0, 3.5, 5.0,
                                                             7.0, 10.0)]))
    assert fit.diagnostics["holdout_selection"]["method"] == "caller_mask"
    assert fit.diagnostics["holdout_n"] == 8    # Vds = 0 is excluded


# ---------------------------------------------------------------------------
# monotone penalty (soft; never claimed as a hard guarantee)
# ---------------------------------------------------------------------------
def test_monotone_penalty_reduces_negative_gm(monkeypatch):
    fams = _families(amplitudes=(10.0, 10.0, 7.0, 5.0, 8.0))
    monkeypatch.setattr(ann_channel, "_MONOTONE_WEIGHT", 0.0)
    loose = fit_channel_iv(fams, V_TH, 2.0, neurons=3, guard_width=0.1)
    monkeypatch.setattr(ann_channel, "_MONOTONE_WEIGHT", 100.0)
    tight = fit_channel_iv(fams, V_TH, 2.0, neurons=3, guard_width=0.1)
    g = np.linspace(2.0, 10.0, 41)
    d = np.full_like(g, 2.5)
    _, gm_loose, _ = loose.evaluate(g, d)
    _, gm_tight, _ = tight.evaluate(g, d)
    n_loose = int(np.count_nonzero(gm_loose < 0.0))
    n_tight = int(np.count_nonzero(gm_tight < 0.0))
    assert n_loose > 0
    assert n_tight < n_loose
    assert tight.diagnostics["monotone_penalty"]["hard_guarantee"] is False
    assert tight.diagnostics["monotone_penalty"]["weight"] == 100.0


# ---------------------------------------------------------------------------
# bounded emitter and optional temperature
# ---------------------------------------------------------------------------
def test_emitted_exponent_is_bounded_and_ascii():
    fit = fit_channel_iv(_families(), V_TH, 2.0, neurons=3,
                         guard_width=0.1)
    expr = emit_ann_channel(fit)
    assert expr.isascii()
    assert "min(exp(" in expr
    assert f",{_fmt(EXP_CAP)})" in expr
    assert "hypot" not in expr
    assert fit.provenance["exp_cap"] == EXP_CAP
    assert "LTspice" in fit.provenance["exp_cap_rule"]


def test_temperature_absent_keeps_two_input_form():
    fit = fit_channel_iv(_families(), V_TH, 2.0, neurons=3,
                         guard_width=0.1)
    assert all(len(row) == 2 for row in fit.coefficients["w1"])
    assert "temperature_center" not in fit.coefficients
    assert fit.provenance["temperature"] is None
    assert "temperature_value" not in fit.coefficients


def test_temperature_present_adds_third_column_and_emitter_term():
    two = fit_channel_iv(_families(), V_TH, 2.0, neurons=3,
                         guard_width=0.1)
    three = fit_channel_iv(_families(), V_TH, 2.0, neurons=3,
                           guard_width=0.1, temperature_c=25.0)
    assert all(len(row) == 3 for row in three.coefficients["w1"])
    assert three.coefficients["temperature_center"] == 25.0
    assert three.coefficients["temperature_scale"] == 1.0
    assert three.provenance["temperature"]["n_inputs"] == 3
    assert three.diagnostics["temperature_center"] == 25.0
    e2 = emit_ann_channel(two)
    e3 = emit_ann_channel(three)
    c = three.coefficients
    nt = (f"((({_fmt(c['temperature_value'])})-"
          f"({_fmt(c['temperature_center'])}))"
          f"/{_fmt(c['temperature_scale'])})")
    assert nt in e3
    assert nt not in e2
    assert e3.isascii()


def test_channel_emitter_keeps_closed_c0_charge_lines():
    charge_c4, charge_c5, _, _ = _charge_curves()
    charge = fit_charge_qv(charge_c4, charge_c5)
    channel = fit_channel_iv(_families(), V_TH, 2.0, neurons=3,
                             guard_width=0.1)
    spec = _spec()
    base = build_tier2_subcircuit(spec)
    net = build_ann_subcircuit(spec, channel, charge)
    assert net.isascii()
    # The closed Tier-2 body/parasitic and charge-element names survive; only
    # the Q expression is the ANN table.
    for token in ("D_BODY", "RG ", "RD ", "RS ", "C_QOSS ", "C_ISS ",
                  "C_RSS "):
        assert token in base and token in net


def test_fit_channel_iv_monotone_guarantees_positive_gds():
    """positive_conductance ANN has zero at Vds=0 and hard gds > 0."""
    from approximation.ann_channel import fit_channel_iv_monotone
    fit = fit_channel_iv_monotone(_families(), V_TH, 2.0, neurons=4, max_nfev=200)
    assert fit.coefficients["model_form"] == "positive_conductance"
    assert fit.diagnostics["gds_negative_fraction"] == 0.0
    vg = np.linspace(0.0, 10.0, 20)
    vd = np.linspace(0.0, 20.0, 20)
    VG, VD = np.meshgrid(vg, vd)
    id_pred, _, gds = fit.evaluate(VG.ravel(), VD.ravel())
    assert np.all(gds >= 0.0)
    id_zero, _, _ = fit.evaluate(vg, np.zeros_like(vg))
    assert np.allclose(id_zero, 0.0)
    expr = emit_ann_channel(fit)
    assert expr.isascii() and "ln(1+exp" in expr


# ---------------------------------------------------------------------------
# measured Curve-7 cutoff calibration (positive_conductance only)
# ---------------------------------------------------------------------------
def _positive_api_fit(**kwargs):
    """Positive-conductance fit through the public dispatch (deterministic)."""
    return fit_channel_iv(_families(), V_TH, 2.0, guard_width=0.1,
                          model_form="positive_conductance", **kwargs)


def _synthetic_curve7(fit, alpha_true, vth_true):
    """Diode-connected Curve-7 transfer generated from *fit*'s static base."""
    vgs = np.linspace(0.0, 6.0, 601)
    base = ann_channel._positive_base_current(fit, vgs, vgs)
    cut = 0.5 * (1.0 + np.tanh(alpha_true * (vgs - vth_true)))
    id7 = cut * base
    base_th = float(ann_channel._positive_base_current(
        fit, np.array([vth_true]), np.array([vth_true]))[0])
    return vgs, id7, 0.5 * base_th


def test_curve7_cutoff_recovers_synthetic_ground_truth():
    fit = _positive_api_fit()
    alpha_true, vth_true = 6.0, 1.45
    vgs7, id7, id_th = _synthetic_curve7(fit, alpha_true, vth_true)
    static_before = {
        key: fit.coefficients[key] for key in (
            "sat", "knee", "gout", "scale_i", "scale_v", "scale_g",
            "vknee_min", "vgs_center", "vgs_scale", "guard_width",
            "model_form")
    }
    diag = ann_channel.calibrate_cutoff_curve7(
        fit, vgs7, id7, id_th_target=id_th)

    assert diag["status"] == "measured"
    assert diag["id_th_target_source"].startswith("curve7_threshold_voltage")
    assert diag["points_used"] >= 3
    assert diag["log_rms_after"] < diag["log_rms_before"]
    # Noiseless synthetic data generated by this exact base: the measured
    # cutoff is recoverable, with no magic constants beyond the ground truth.
    assert diag["log_rms_after"] < 1e-6
    assert fit.coefficients["alpha"] == pytest.approx(alpha_true, abs=1e-3)
    assert fit.coefficients["vth_eff"] == pytest.approx(vth_true, abs=1e-3)
    for key, value in static_before.items():
        assert fit.coefficients[key] == value
    assert fit.v_th == pytest.approx(V_TH)


def test_curve7_cutoff_preserves_zero_current_gds_and_emission():
    fit = _positive_api_fit()
    coeffs_before = {k: v for k, v in fit.coefficients.items()
                     if k not in ("alpha", "vth_eff")}
    vgs7, id7, id_th = _synthetic_curve7(fit, 5.0, 1.5)
    ann_channel.calibrate_cutoff_curve7(fit, vgs7, id7, id_th_target=id_th)

    vg = np.linspace(0.0, 10.0, 20)
    vd = np.linspace(0.0, 20.0, 20)
    VG, VD = np.meshgrid(vg, vd)
    id_pred, gm, gds = fit.evaluate(VG.ravel(), VD.ravel())
    assert np.all(np.isfinite(id_pred))
    assert np.all(gds >= 0.0)                 # G1 hard gds>0 semantics
    id_zero, _, _ = fit.evaluate(vg, np.zeros_like(vg))
    assert np.allclose(id_zero, 0.0)          # Vds=0 zero-current semantics
    # static coefficients/scales are byte-identical: only cutoff moved.
    for key, value in coeffs_before.items():
        assert fit.coefficients[key] == value
    expr = emit_ann_channel(fit)
    assert _fmt(fit.coefficients["alpha"]) in expr
    assert _fmt(fit.coefficients["vth_eff"]) in expr
    assert expr.isascii() and "tanh(" in expr


def test_curve7_cutoff_fails_closed_on_missing_or_malformed_data():
    fit = _positive_api_fit()
    vgs7, id7, id_th = _synthetic_curve7(fit, 5.0, 1.5)
    with pytest.raises(ValueError, match="empty"):
        ann_channel.calibrate_cutoff_curve7(fit, [], [], id_th_target=id_th)
    bad = id7.copy()
    bad[10] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        ann_channel.calibrate_cutoff_curve7(fit, vgs7, bad, id_th_target=id_th)
    # Genuine decrease (well outside the measured float-noise band).
    non_mono = id7.copy()
    non_mono[200] = non_mono[199] * 0.5
    with pytest.raises(ValueError, match="non-monotone"):
        ann_channel.calibrate_cutoff_curve7(fit, vgs7, non_mono,
                                           id_th_target=id_th)
    # Only the zero anchor falls below the threshold -> fewer than 3 points.
    with pytest.raises(ValueError, match="subthreshold"):
        ann_channel.calibrate_cutoff_curve7(fit, vgs7, id7,
                                           id_th_target=1e-30)
    with pytest.raises(ValueError, match="positive_conductance"):
        legacy = fit_channel_iv(_families(), V_TH, 2.0, neurons=3,
                                guard_width=0.1)
        ann_channel.calibrate_cutoff_curve7(legacy, vgs7, id7,
                                           id_th_target=id_th)


def test_curve7_subthreshold_fraction_is_data_derived_when_target_absent():
    fit = _positive_api_fit()
    vgs7, id7, _ = _synthetic_curve7(fit, 5.0, 1.5)
    diag = ann_channel.calibrate_cutoff_curve7(fit, vgs7, id7,
                                               id_th_target=None)
    assert "measured Curve-7 peak" in diag["id_th_target_source"]
    assert diag["id_th_target_a"] == pytest.approx(0.01 * float(id7.max()))


def test_fit_channel_iv_forwards_curve7_only_in_positive_path():
    base = _positive_api_fit()
    vgs7, id7, id_th = _synthetic_curve7(base, 6.0, 1.45)
    pos = _positive_api_fit(curve7_vgs=vgs7, curve7_id=id7,
                            curve7_id_th_target=id_th)
    assert pos.provenance["curve7_cutoff"]["status"] == "measured"
    assert pos.diagnostics["curve7_cutoff"]["status"] == "measured"
    assert pos.coefficients["alpha"] == pytest.approx(6.0, abs=1e-3)
    assert pos.coefficients["vth_eff"] == pytest.approx(1.45, abs=1e-3)

    # The legacy exponential path must not consume Curve-7 (even malformed).
    legacy = fit_channel_iv(_families(), V_TH, 2.0, neurons=3,
                            guard_width=0.1, model_form="legacy_exp",
                            curve7_vgs=[0.0, np.nan],
                            curve7_id=[0.0, np.nan],
                            curve7_id_th_target=np.nan)
    assert "curve7_cutoff" not in legacy.provenance
    assert "model_form" not in legacy.coefficients


def test_curve7_gate_uses_measured_static_coverage():
    from pipeline.ann_fit import _curve7_cutoff_gate

    fams = _families()                      # lowest Curve-1 family Vgs = 2.0
    ok = _curve7_cutoff_gate(fams, 1.0)     # 2.0 >= 1.0 -> applicable
    assert ok["applicable"] is True
    assert ok["status"] == "applicable_static_coverage_at_or_above_vth"
    assert ok["min_curve1_vgs_v"] == pytest.approx(2.0)
    assert ok["v_th_v"] == pytest.approx(1.0)

    blocked = _curve7_cutoff_gate(fams, 2.5)   # 2.0 < 2.5 -> keep joint cutoff
    assert blocked["applicable"] is False
    assert blocked["status"] == \
        "not_applicable_no_subthreshold_static_coverage"
    assert blocked["min_curve1_vgs_v"] == pytest.approx(2.0)
    assert blocked["v_th_v"] == pytest.approx(2.5)
    # Boundary is inclusive: a family exactly at Vth has no sub-threshold
    # coverage and is safe to calibrate.
    assert _curve7_cutoff_gate(fams, 2.0)["applicable"] is True


def test_ann_fit_records_gate_status_for_both_sides(monkeypatch):
    import copy
    from pipeline import ann_fit as af

    fams = _families()                    # min family Vgs = 2.0
    template = _positive_api_fit()        # real positive fit, reused as stub
    state = {"vth": 1.0, "kwargs": None}
    stub_charge = type("_StubCharge", (), {
        "provenance": {"derivative_terms": [], "emission": "stub"},
        "diagnostics": {},
    })()

    def fake_extract(vendor, bench):
        if bench == "curve7_threshold_voltage":
            return {
                "vth_v": state["vth"],
                "curve_vgs": [0.0, 1.0, 2.0],
                "curve_id": [0.0, 1e-4, 1e-2],
                "Id_th_target": 1e-4,
            }
        raise FileNotFoundError(bench)

    monkeypatch.setattr(af, "load_input",
                        lambda vendor: {"device": vendor, "conditions": {}})
    monkeypatch.setattr(af, "_channel_families",
                        lambda vendor, cond: (fams, 5.0))
    monkeypatch.setattr(af, "_static_guard_width", lambda f: (0.1, {}))
    monkeypatch.setattr(af, "_v_sat", lambda f: 2.0)
    monkeypatch.setattr(af, "_charge_curves",
                        lambda vendor, cond: (None, None, {}))
    monkeypatch.setattr(af, "_extract", fake_extract)
    monkeypatch.setattr(af, "fit_charge_qv", lambda *a, **k: stub_charge)

    def fake_fit_channel(inputs, v_th, v_sat, **kwargs):
        state["kwargs"] = dict(kwargs)
        return copy.deepcopy(template)

    monkeypatch.setattr(af, "fit_channel_iv", fake_fit_channel)

    # Applicable side: Curve-1 starts above Vth -> Curve-7 is forwarded.
    state["vth"] = 1.0
    res_ok = af.ann_fit_for_vendor("synthetic-ok", include_dpt_points=False)
    assert "curve7_vgs" in state["kwargs"]
    assert "curve7_cutoff" not in res_ok["channel_fit"].provenance

    # Blocked side: Curve-1 extends below Vth -> calibration is skipped and
    # the rejection is recorded in the provenance/record (never "measured").
    state["vth"] = 2.5
    res_no = af.ann_fit_for_vendor("synthetic-no", include_dpt_points=False)
    assert "curve7_vgs" not in state["kwargs"]
    gate = res_no["channel_fit"].provenance["curve7_cutoff"]
    assert gate["status"] == \
        "not_applicable_no_subthreshold_static_coverage"
    assert gate["min_curve1_vgs_v"] == pytest.approx(2.0)
    assert gate["v_th_v"] == pytest.approx(2.5)
    assert res_no["provenance"]["curve7_cutoff"]["status"] == gate["status"]
    assert res_no["record"]["curve7_cutoff"]["status"] == gate["status"]
