"""Offline tests for the hybrid C0+ANN residual basis.

No LTspice, no benchmarks, no repository writes.  The only fitted model is a
small synthetic Curve-1 family whose golden current is C0 plus a known signed
residual; every assertion checks a structural or analytic property of the
emitted hybrid basis, not a fabricated fidelity number.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from approximation.ann_residual import fit_channel_residual
from approximation.smoother import smooth_relu_product
from common.types import CurveData, PWLModelSpec
from models.ann_netlist_builder import (
    build_hybrid_ann_subcircuit,
    emit_ann_residual,
)
from models.netlist_builder import (
    _channel_ramps,
    build_tier2_subcircuit,
    evaluate_tier2_channel,
)

V_TH = 1.0


def _spec() -> PWLModelSpec:
    vds = np.array([0.0, 0.1, 0.4, 1.0, 2.0, 3.5, 5.0])
    vgs = np.array([2.0, 3.5, 5.0, 7.0, 10.0])
    rows = np.array([
        [18.0 * max(g - V_TH, 0.0) ** 1.1 * float(np.tanh(d / 0.7))
         for d in vds] for g in vgs])
    return PWLModelSpec(
        device_name="hybrid_test",
        breakpoints_vds=vds.tolist(), breakpoints_id=rows.tolist(),
        vgs_values=vgs.tolist(), rds_on=0.02, v_th=V_TH,
        body_diode_vsd=[0.5, 0.8], body_diode_isd=[0.1, 1.0],
        q_oss_vs_v=[0.0, 5.0], q_oss_charge_c=[0.0, 1e-9])


def _families(spec: PWLModelSpec) -> list[CurveData]:
    vds = np.asarray(spec.breakpoints_vds)
    g = np.asarray(spec.vgs_values)[:, None]
    c0 = evaluate_tier2_channel(spec, g, vds[None, :])
    # Known signed residual: negative at low Vgs, positive at high Vgs.
    golden = c0 + 0.03 * vds[None, :] * (g - 5.0)
    return [CurveData(
        name=f"curve1_vgs_{v:g}", x_name="Vds", x_unit="V",
        y_name="Id", y_unit="A", x_data=vds.tolist(), y_data=row.tolist(),
        family_param="Vgs", family_value=float(v),
        metadata={"source": "curve1_output_characteristics:curve_id_vgs_"})
        for v, row in zip(spec.vgs_values, golden)]


@pytest.fixture(scope="module")
def fitted():
    spec = _spec()
    fit = fit_channel_residual(_families(spec), spec, v_th=V_TH,
                               neurons=4, guard_width=0.05,
                               guard_width_provenance={
                                   "value": 0.05,
                                   "dpt_points_excluded": True})
    return spec, fit


# ---------------------------------------------------------------------------
# numeric C0 mirror
# ---------------------------------------------------------------------------
def test_tier2_evaluator_mirrors_closed_channel():
    spec = _spec()
    g = np.array([1.0, 3.0, 10.0])
    d = np.array([0.0, 0.5, 5.0])
    out = evaluate_tier2_channel(spec, g, d)
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))
    assert float(out[0]) == 0.0                    # Vds=0 closed anchor
    grid = np.array([g[0], g[2]])
    assert evaluate_tier2_channel(spec, grid[:, None], d[None, :]).shape == (2, 3)
    c1 = evaluate_tier2_channel(spec, g, d, eps=0.01)
    assert np.all(np.isfinite(c1))
    assert float(c1[0]) == 0.0
    assert "tanh(V(di,ss)" in build_tier2_subcircuit(spec)


def test_tier2_saturation_ramp_uses_gate_axis():
    """The C0 mirror's knee table is Vgs-indexed, exactly as emitted G_CH is.

    ``_spec()`` has the same fitted knee in every Vgs family, so it cannot
    expose an axis swap.  This fixture gives each family a distinct tanh
    knee and uses deliberately different gate and drain arrays: with the old
    Vds-indexed saturation ramp these currents do not match.
    """
    vds = np.array([0.0, 0.1, 0.4, 1.0, 2.0, 3.5, 5.0])
    vgs = np.array([2.0, 3.5, 5.0, 7.0, 10.0])
    knees = [0.4, 0.8, 1.5, 2.5, 3.5]   # distinct per-curve tanh widths
    rows = np.array([
        [12.0 * (g - V_TH) ** 1.1 * float(np.tanh(d / knee))
         for d in vds] for g, knee in zip(vgs, knees)])
    spec = PWLModelSpec(
        device_name="axis_test",
        breakpoints_vds=vds.tolist(), breakpoints_id=rows.tolist(),
        vgs_values=vgs.tolist(), rds_on=0.02, v_th=V_TH,
        body_diode_vsd=[0.5, 0.8], body_diode_isd=[0.1, 1.0],
        q_oss_vs_v=[0.0, 5.0], q_oss_charge_c=[0.0, 1e-9])

    xs, iv, sv = _channel_ramps(spec)
    g = np.array([1.5, 3.0, 5.5, 9.0])
    d = np.array([0.4, 1.2, 2.8, 6.5])
    amp = np.asarray(smooth_relu_product(g, xs, iv, 0.0), dtype=float)
    sat = np.asarray(smooth_relu_product(g, xs, sv, 0.0), dtype=float)
    expected = amp * np.tanh(d / sat)

    assert np.allclose(evaluate_tier2_channel(spec, g, d), expected,
                       rtol=1e-12, atol=1e-12)

    # At fixed Vgs the knee does not move when Vds changes; only the tanh
    # ratio may change.  This is the gate-axis law emitted by G_CH.
    d2 = np.array([5.0, 0.1, 4.0, 0.9])
    got2 = evaluate_tier2_channel(spec, g, d2)
    assert np.allclose(got2 / expected,
                       np.tanh(d2 / sat) / np.tanh(d / sat),
                       rtol=1e-12, atol=1e-12)

    # Guard the fixture itself: a Vds-indexed knee must give a different
    # prediction, so reintroducing the old axis swap fails the assertions.
    wrong_sat = np.asarray(smooth_relu_product(d, xs, sv, 0.0), dtype=float)
    assert not np.allclose(amp * np.tanh(d / wrong_sat), expected,
                           rtol=1e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# residual fit physics
# ---------------------------------------------------------------------------
def test_residual_zero_at_vds_zero_and_cutoff_below_vth(fitted):
    _, fit = fitted
    g = np.linspace(0.0, 12.0, 25)
    assert np.all(fit.evaluate(g, np.zeros_like(g)) == 0.0)
    d = np.linspace(0.1, 5.0, 9)
    below = np.max(np.abs(fit.evaluate(np.full_like(d, V_TH - 0.5), d)))
    above = np.max(np.abs(fit.evaluate(np.full_like(d, V_TH + 0.5), d)))
    assert below < 1e-3 * max(above, 1e-30)
    assert fit.diagnostics["zero_vds_max_abs_a"] == 0.0
    assert fit.provenance["guard_width"]["dpt_points_excluded"] is True


def test_signed_residual_capability(fitted):
    spec, fit = fitted
    g = np.asarray(spec.vgs_values)
    d = float(spec.breakpoints_vds[-1])
    pred = fit.evaluate(g, np.full_like(g, d))
    assert np.any(pred < 0.0) and np.any(pred > 0.0)
    assert fit.diagnostics["negative_target_samples"] > 0
    assert fit.diagnostics["positive_target_samples"] > 0
    assert fit.diagnostics["residual_rel_rmse"] < 0.5


def test_analytic_first_and_second_derivatives(fitted):
    _, fit = fitted
    g = np.array([2.0, 5.0, 9.0])
    d = np.array([0.3, 1.5, 4.0])
    val, dg, dd, dgg, ddd, dgd = fit.evaluate_with_derivatives(g, d)
    assert np.all(np.isfinite(np.column_stack([val, dg, dd, dgg, ddd, dgd])))
    h = 1e-5
    fd_g = (fit.evaluate(g + h, d) - fit.evaluate(g - h, d)) / (2 * h)
    fd_d = (fit.evaluate(g, d + h) - fit.evaluate(g, d - h)) / (2 * h)
    assert np.allclose(dg, fd_g, atol=2e-5)
    assert np.allclose(dd, fd_d, atol=2e-5)
    max_err = max(fit.diagnostics["analytic_vs_finite_difference"]
                  ["max_abs_diff"].values())
    assert max_err < 1e-4


def test_dpt_samples_counted_without_collapsing_guard(fitted):
    from approximation.ann_channel import DptPoints

    spec, _ = fitted
    dense = 5.2 + np.linspace(0.0, 1e-6, 12)
    dpt = DptPoints(np.full(12, 11.0), dense, np.full(12, 42.0),
                    {"source": "recorded_dpt", "status": "ok"})
    fit = fit_channel_residual(_families(spec) + [dpt], spec, v_th=V_TH,
                               neurons=3, guard_width=0.05)
    assert fit.diagnostics["dpt_samples"] == 12
    # The Vds=0 anchor is excluded by construction: the residual is
    # identically zero there, so it is not counted as a fitted sample.
    assert fit.diagnostics["curve1_samples"] == 5 * (len(spec.breakpoints_vds) - 1)
    assert fit.coefficients["guard_width"] == 0.05


# ---------------------------------------------------------------------------
# emitter
# ---------------------------------------------------------------------------
def test_residual_expression_is_ascii_and_signed_form(fitted):
    _, fit = fitted
    expr = emit_ann_residual(fit)
    assert expr.isascii()
    assert "V(di,ss)" in expr and "V(gi,ss)" in expr
    assert "tanh(" in expr and "hypot" not in expr
    assert "-" in expr  # signed weights, not positive-only exp() transform


def test_hybrid_netlist_preserves_c0_adds_residual_and_no_nodes(fitted):
    spec, fit = fitted
    name = "HYBRID_C0_TEST"
    base = build_tier2_subcircuit(spec, name=name)
    net = build_hybrid_ann_subcircuit(spec, fit, name=name)
    assert net.isascii()
    base_lines = base.splitlines()
    net_lines = net.splitlines()
    i_sub_b = next(i for i, l in enumerate(base_lines)
                   if l.startswith(".SUBCKT "))
    i_sub_n = next(i for i, l in enumerate(net_lines)
                   if l.startswith(".SUBCKT "))
    i_ch_n = next(i for i, l in enumerate(net_lines)
                  if l.startswith("G_CH "))
    gann = [i for i, l in enumerate(net_lines) if l.startswith("G_ANN ")]
    assert len(gann) == 1
    assert gann[0] == i_ch_n + 3  # blank line, comment, then G_ANN
    # Strip the inserted block: every remaining line must be byte-identical
    # to the closed Tier-2 C0 emitter at the same subckt name.
    assert (net_lines[i_sub_n:i_ch_n + 1] + net_lines[gann[0] + 1:]
            == base_lines[i_sub_b:])
    for element in ("C_QOSS ", "C_ISS ", "C_RSS "):
        assert [line for line in net_lines if line.startswith(element)] == [
            line for line in base_lines if line.startswith(element)]
    assert "ANN Q nets" not in net
    assert "D_BODY" in net and "RG " in net and "RD " in net and "RS " in net
    nodes = {"d", "g", "s", "di", "gi", "ss"}

    def _used(text: str) -> set[str]:
        out: set[str] = set()
        for line in text.splitlines():
            if line.startswith(("*", ".")) or not line.strip():
                continue
            parts = line.split()
            if parts[0].startswith(("G_CH", "G_ANN", "C_QOSS", "C_ISS",
                                    "C_RSS", "C_OSS", "D_BODY", "RG", "RD",
                                    "RS")):
                out.update(parts[1:3])
        return out

    assert _used(base) == _used(net) == nodes


def test_hybrid_name_override_preserves_c0_charge(fitted):
    spec, fit = fitted
    new = "CUSTOM_HYBRID"
    base = build_tier2_subcircuit(spec, name=new)
    net = build_hybrid_ann_subcircuit(spec, fit, name="IGNORED",
                                      subckt_name=new)
    assert ".SUBCKT CUSTOM_HYBRID d g s" in net
    assert ".ENDS CUSTOM_HYBRID" in net
    assert "D_BD_CUSTOM_HYBRID" in net
    assert "ANN Q nets" not in net
    for element in ("C_QOSS ", "C_ISS ", "C_RSS "):
        assert [line for line in net.splitlines()
                if line.startswith(element)] == [
            line for line in base.splitlines() if line.startswith(element)]


def test_hybrid_builder_rejects_missing_or_incorrect_residual_fit(fitted):
    spec, fit = fitted
    for bad in (None, object(), "residual", fit.coefficients):
        with pytest.raises(TypeError,
                           match="residual_fit must be an AnnResidualFit"):
            build_hybrid_ann_subcircuit(spec, bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError,
                       match="residual_fit must be an AnnResidualFit"):
        build_hybrid_ann_subcircuit(spec, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError,
                       match="missing 1 required positional argument"):
        build_hybrid_ann_subcircuit(spec)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# pipeline contract: signed residual current only, C0 charge unchanged
# ---------------------------------------------------------------------------
def test_hybrid_fit_does_not_require_or_train_full_curve4(monkeypatch):
    from pipeline import ann_fit

    real_load = ann_fit.load_output

    def _scalar_only(vendor: str, bench: str):
        doc = copy.deepcopy(real_load(vendor, bench))
        if bench == "curve4_capacitances":
            for key in ann_fit._CURVE4_SWEEP_KEYS:
                doc.get("extracted", {}).pop(key, None)
                doc.pop(key, None)
        return doc

    def _forbidden_charge_fit(*args, **kwargs):
        raise AssertionError("hybrid basis must not train an ANN charge net")

    monkeypatch.setattr(ann_fit, "load_output", _scalar_only)
    monkeypatch.setattr(ann_fit, "fit_charge_qv", _forbidden_charge_fit)
    out = ann_fit.hybrid_ann_fit_for_vendor("csd16412q5a",
                                            include_dpt_points=False)
    note = out["provenance"]["curve4_full_sweep"]
    assert note["status"] == "partial"
    assert note["missing_keys"] == list(ann_fit._CURVE4_SWEEP_KEYS)
    assert out["record"]["curve4_data_richness"] == note


def test_hybrid_pipeline_record_and_provenance(monkeypatch):
    from pipeline import ann_fit

    out = ann_fit.hybrid_ann_fit_for_vendor("csd16412q5a",
                                            include_dpt_points=False,
                                            neurons=4)
    rec = out["record"]
    assert set(out) == {"vendor", "device", "c0_spec", "residual_fit",
                        "record", "provenance", "limitations"}
    assert rec["basis"] == "hybrid"
    assert "no ANN charge fit" in rec["charge_basis"]
    assert "charge_fit" not in rec
    assert not any(k in rec for k in ("charge_sources",
                                      "charge_sample_counts"))
    assert "charge" not in rec["optimizer"]
    assert "charge_fit" not in out["provenance"]
    assert rec["residual_curve1_samples"] > 0
    assert rec["residual_dpt_samples"] == 0
    assert out["residual_fit"].provenance["target"].startswith(
        "I_golden - I_C0")
    assert out["provenance"]["guard_width"]["dpt_points_excluded"] is True
    net = build_hybrid_ann_subcircuit(out["c0_spec"], out["residual_fit"])
    base = build_tier2_subcircuit(out["c0_spec"])
    assert net.isascii() and "G_ANN di ss" in net
    assert "ANN Q nets" not in net
    for element in ("C_QOSS ", "C_ISS ", "C_RSS "):
        assert [line for line in net.splitlines()
                if line.startswith(element)] == [
            line for line in base.splitlines() if line.startswith(element)]

