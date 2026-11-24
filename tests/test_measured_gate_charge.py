"""Offline tests for the measured gate-charge candidate (Curve 5 + Curve 4).

No LTspice is invoked: synthetic curve documents exercise the loader's
fail-closed validation and the shipped ``cpm2-1200`` YAML fixtures exercise
the real split.  The Tier-2 emitter is checked for byte-identical default
output and for the opt-in nonlinear ``Q=`` pair.
"""
from __future__ import annotations

import numpy as np
import pytest

from approximation.charge_model import (
    _MEASURED_Q_MAX_POINTS,
    _MEASURED_Q_REL_TOLERANCE,
    _compact_measured_q,
    load_measured_gate_charge,
    measured_gate_charge_from_curves,
)
from common.types import PWLModelSpec
from models.netlist_builder import build_tier2_subcircuit
from pipeline.unified_rom import (
    MEASURED_GATE_CHARGE_ENV,
    QOSS_COMPACTION_ENV,
    _c0_bit_identity,
    _compact_qoss_table,
    _emit_tier2_c0,
    _measured_gate_charge_opt_in,
    _qoss_compaction_opt_in,
    _spec_measured_gate_charge,
)


def _curve5(vgs, qg_nc, *, v_plateau=2.0, qgd_nc=3.0, qgs_nc=4.0):
    return {
        "device": "synthetic",
        "testbench": "curve5_gate_charge",
        "extracted": {
            "curve_vgs": list(vgs),
            "curve_qg_nc": list(qg_nc),
            "v_plateau_v": v_plateau,
            "qgd_nc": qgd_nc,
            "qgs_nc": qgs_nc,
            "q_tail_nc": qgs_nc + qgd_nc,
            "qg_total_nc": float(qg_nc[-1]),
        },
    }


def _curve4(vds=(0.1, 1.0, 2.0), crss=(1e-9, 0.5e-9, 0.25e-9)):
    return {
        "device": "synthetic",
        "testbench": "curve4_capacitances",
        "extracted": {
            "curve_vds": list(vds),
            "crss_curve_f": list(crss),
        },
    }


# Curve 5 repeats Vgs across the Miller plateau; the last plateau sample has
# accumulated the full qgd_nc, so Qgs there equals qgs_nc.
_VGS = (0.0, 1.0, 2.0, 2.0, 2.0, 3.0, 4.0)
_QG_NC = (0.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0)


def test_measured_split_is_zero_anchored_and_monotone():
    result = measured_gate_charge_from_curves(_curve5(_VGS, _QG_NC), _curve4())

    # Duplicate plateau voltages collapse to one point; Qgs below is Qg and
    # at/above the plateau is Qg - qgd_nc (in coulombs).  The 0.5 % greedy
    # charge compaction keeps the zero anchor, both slope breaks and the
    # endpoint: (0,0), (2 V, 4 nC), (3 V, 5 nC), (4 V, 7 nC).
    np.testing.assert_allclose(result.qgs_v, [0.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(
        result.qgs_c, np.array([0.0, 4.0, 5.0, 7.0]) * 1e-9)
    assert result.qgs_c[0] == 0.0
    assert np.all(np.diff(result.qgs_c) >= 0.0)
    assert result.qgs_c[1] == pytest.approx(4e-9)   # reported qgs_nc

    # Qgd integrates the measured Crss along Vdg = Vds and anchors at 0.
    np.testing.assert_allclose(result.qgd_v, [0.0, 0.1, 1.0, 2.0])
    np.testing.assert_allclose(
        result.qgd_c[0], 0.0)
    np.testing.assert_allclose(
        result.qgd_c[1], 0.0)                      # no invented sub-domain
    assert result.qgd_c[-1] == pytest.approx(
        0.5 * (1.0e-9 + 0.5e-9) * 0.9 + 0.5 * (0.5e-9 + 0.25e-9) * 1.0)
    assert np.all(np.diff(result.qgd_c) >= 0.0)
    assert result.provenance["kind"] == "measured"
    assert result.provenance["qgd_axis"] == "Vdg = Vds (legacy identity)"

    # Compaction retains measured point counts separately and records the
    # achieved charge error; emitted tables stay within the documented bar.
    prov = result.provenance
    assert prov["qgs_points"] == 5                 # full unique measured grid
    assert prov["qgs_emitted_points"] == result.qgs_v.size == 4
    assert prov["qgd_points"] == result.qgd_v.size
    for key in ("qgs_compaction", "qgd_compaction"):
        assert prov[key]["emitted_points"] <= _MEASURED_Q_MAX_POINTS
        assert prov[key]["rel_tolerance_pct"] == pytest.approx(
            100.0 * _MEASURED_Q_REL_TOLERANCE)
        assert prov[key]["max_error_pct"] <= 100.0 * _MEASURED_Q_REL_TOLERANCE


def test_fixture_cpm2_split_consistency():
    result = load_measured_gate_charge("cpm2-1200")
    prov = result.provenance
    assert result.qgs_v[0] == 0.0
    assert result.qgd_v[0] == 0.0
    assert result.qgs_c[0] == 0.0
    assert result.qgd_c[0] == 0.0
    assert np.all(np.diff(result.qgs_c) >= 0.0)
    assert np.all(np.diff(result.qgd_c) >= 0.0)
    # Full measured counts survive in provenance; emission is compact.
    assert prov["qgs_points"] == 399
    assert prov["qgs_emitted_points"] == result.qgs_v.size == 3
    assert prov["qgd_points"] == 26
    assert prov["qgd_emitted_points"] == result.qgd_v.size == 12
    assert prov["qgd_cv_bias_aware"] is True
    assert prov["qgd_vgs_bias_v"] == -5.0
    assert result.qgd_v[1] == pytest.approx(5.1)   # Vdg = Vds - (-5 V)
    assert prov["qgs_compaction"]["max_error_pct"] <= 0.5
    assert prov["qgd_compaction"]["max_error_pct"] <= 0.5
    # The split still evaluates to qgs_nc at the plateau within the 0.5 %
    # charge budget (the compact grid need not contain the plateau sample).
    q_at_plateau = float(np.interp(
        prov["v_plateau_v"], result.qgs_v, result.qgs_c))
    assert abs(q_at_plateau - 17.83965331061183e-9) <= (
        prov["qgs_compaction"]["budget_c"])


@pytest.mark.parametrize("vendor", [
    "cpm2-1200", "irfs7730pbf", "csd16412q5a", "epc2216"])
def test_all_fixtures_compact_within_budget(vendor):
    result = load_measured_gate_charge(vendor)
    prov = result.provenance
    for v, q, key in (
            (result.qgs_v, result.qgs_c, "qgs_compaction"),
            (result.qgd_v, result.qgd_c, "qgd_compaction")):
        assert v.size == q.size <= _MEASURED_Q_MAX_POINTS
        assert v[0] == 0.0 and q[0] == 0.0
        assert np.all(np.diff(v) >= 0.0)
        assert np.all(np.diff(q) >= 0.0)
        assert prov[key]["max_error_pct"] <= 100.0 * (
            _MEASURED_Q_REL_TOLERANCE)
        assert prov[key]["max_error_c"] <= prov[key]["budget_c"]
        assert prov[key]["emitted_points"] == v.size


def test_bias_aware_qgd_uses_largest_nonzero_bias():
    curve4 = _curve4()
    # The legacy primary pair must be ignored when cv_sweeps is present.
    curve4["extracted"]["cv_sweeps"] = [
        {"curve_vds": [0.1, 1.0, 2.0],
         "crss_curve_f": [1e-9, 1e-9, 1e-9], "vgs_bias_v": 0.0},
        {"curve_vds": [0.1, 1.0, 2.0],
         "crss_curve_f": [2e-9, 2e-9, 2e-9], "vgs_bias_v": -2.0},
        {"curve_vds": [0.1, 1.0, 2.0],
         "crss_curve_f": [5e-9, 5e-9, 5e-9], "vgs_bias_v": -5.0},
    ]
    result = measured_gate_charge_from_curves(
        _curve5(_VGS, _QG_NC), curve4)
    prov = result.provenance
    assert prov["qgd_cv_bias_aware"] is True
    assert prov["qgd_vgs_bias_v"] == -5.0
    assert prov["qgd_axis"].startswith("Vdg = Vds - Vgs_bias")
    assert prov["qgd_selection"]["source"].endswith("[vgs_bias_v=-5]")
    # Axis = Vds - (-5 V) = Vds + 5 V; the first measured point is 5.1 V.
    assert prov["qgd_points"] == 4
    assert result.qgd_v[1] == pytest.approx(5.1)
    assert result.qgd_v[-1] == pytest.approx(7.0)
    budget = prov["qgd_compaction"]["budget_c"]
    assert np.interp(6.0, result.qgd_v, result.qgd_c) == pytest.approx(
        4.5e-9, abs=budget)
    assert result.qgd_c[-1] == pytest.approx(9.5e-9, abs=budget)
    assert result.qgd_c[0] == 0.0


def test_legacy_qgd_fallback_without_cv_sweeps():
    result = measured_gate_charge_from_curves(
        _curve5(_VGS, _QG_NC), _curve4())
    prov = result.provenance
    assert prov["qgd_cv_bias_aware"] is False
    assert prov["qgd_vgs_bias_v"] is None
    assert prov["qgd_axis"] == "Vdg = Vds (legacy identity)"
    assert prov["qgd_selection"]["cv_sweeps_count"] == 0
    assert result.qgd_v[1] == pytest.approx(0.1)


def test_zero_bias_only_sweeps_keep_legacy_axis():
    curve4 = _curve4()
    curve4["extracted"]["cv_sweeps"] = [
        {"curve_vds": [0.1, 1.0, 2.0],
         "crss_curve_f": [1e-9, 5e-10, 2.5e-10], "vgs_bias_v": 0.0},
    ]
    prov = measured_gate_charge_from_curves(
        _curve5(_VGS, _QG_NC), curve4).provenance
    assert prov["qgd_cv_bias_aware"] is False
    assert prov["qgd_vgs_bias_v"] is None
    assert prov["qgd_axis"] == "Vdg = Vds (legacy identity)"
    assert prov["qgd_selection"]["cv_sweeps_count"] == 1


def test_irfs_measured_tables_compact_to_expected_counts():
    prov = load_measured_gate_charge("irfs7730pbf").provenance
    assert prov["qgs_points"] == 399
    assert prov["qgs_emitted_points"] == 3
    assert prov["qgd_points"] == 26
    assert prov["qgd_emitted_points"] == 9
    assert prov["qgd_cv_bias_aware"] is True
    assert prov["qgd_axis"] == "Vdg = Vds - Vgs_bias (bias-aware Crss sweep)"


def test_compaction_fails_when_budget_unmet_within_cap():
    v = np.linspace(0.0, 10.0, 101)
    q = (v / 10.0) ** 3
    with pytest.raises(ValueError, match="cannot meet"):
        _compact_measured_q("Qtest", v, q, max_points=2)


def test_missing_curve_key_fails():
    curve5 = _curve5(_VGS, _QG_NC)
    del curve5["extracted"]["curve_qg_nc"]
    with pytest.raises(ValueError, match="curve_qg_nc"):
        measured_gate_charge_from_curves(curve5, _curve4())


def test_non_monotone_measured_charge_fails():
    curve5 = _curve5((0.0, 1.0, 2.0), (0.0, 3.0, 2.0))
    with pytest.raises(ValueError, match="non-decreasing"):
        measured_gate_charge_from_curves(curve5, _curve4())


def test_negative_qgs_split_fails():
    # Last plateau sample is 4.5 nC but qgd_nc is 5 nC -> negative Qgs.
    curve5 = _curve5((0.0, 1.0, 2.0, 2.0, 3.0),
                     (0.0, 2.0, 4.0, 4.5, 6.0), qgd_nc=5.0)
    with pytest.raises(ValueError, match="negative"):
        measured_gate_charge_from_curves(curve5, _curve4())


def test_split_monotonicity_fails():
    # Flat Qg through a large qgd step makes Qgs decrease below the plateau.
    curve5 = _curve5((0.0, 1.0, 2.0, 2.0, 3.0),
                     (0.0, 10.0, 10.0, 10.0, 11.0),
                     qgd_nc=5.0, qgs_nc=5.0)
    with pytest.raises(ValueError, match="derived Qgs"):
        measured_gate_charge_from_curves(curve5, _curve4())


def test_non_finite_measured_data_fails():
    curve4 = _curve4()
    curve4["extracted"]["crss_curve_f"][1] = float("nan")
    with pytest.raises(ValueError, match="NaN/Inf"):
        measured_gate_charge_from_curves(_curve5(_VGS, _QG_NC), curve4)


@pytest.fixture
def charge_spec():
    result = measured_gate_charge_from_curves(_curve5(_VGS, _QG_NC), _curve4())
    return PWLModelSpec(
        device_name="Synthetic",
        breakpoints_vds=[0.0, 1.0, 2.0, 5.0, 10.0],
        breakpoints_id=[[0.0, 2.0, 4.0, 10.0, 20.0],
                        [0.0, 3.0, 6.0, 15.0, 30.0]],
        vgs_values=[5.0, 10.0],
        q_oss_vs_v=[0.0, 25.0, 50.0, 100.0],
        q_oss_charge_c=[0.0, 5e-8, 7e-8, 9e-8],
        q_iss_vs_v=result.qgs_v.tolist(),
        q_iss_charge_c=result.qgs_c.tolist(),
        q_rss_vs_v=result.qgd_v.tolist(),
        q_rss_charge_c=result.qgd_c.tolist(),
        rds_on=0.005,
        v_th=2.5,
        metadata={
            "urom_gate_charge": {
                "enabled": True,
                "source": result.provenance["source"],
                "provenance": result.provenance,
            },
        },
    )


def test_default_emission_keeps_legacy_linear_lines(charge_spec):
    text = build_tier2_subcircuit(charge_spec)
    assert "C_ISS gi ss 1.500000e-09" in text
    assert "C_RSS gi di 5.000000e-11" in text
    assert "C_ISS gi ss Q" not in text
    assert "C_RSS gi di Q" not in text
    assert sum(line.startswith("C_ISS ") for line in text.splitlines()) == 1
    assert sum(line.startswith("C_RSS ") for line in text.splitlines()) == 1


def test_opt_in_emits_one_nonlinear_element_each(charge_spec):
    text = build_tier2_subcircuit(charge_spec, measured_gate_charge=True)
    iss = [line for line in text.splitlines() if line.startswith("C_ISS ")]
    rss = [line for line in text.splitlines() if line.startswith("C_RSS ")]
    assert len(iss) == 1 and len(rss) == 1
    assert iss[0].startswith("C_ISS gi ss Q = {")
    assert "V(gi,ss)" in iss[0]
    assert rss[0].startswith("C_RSS gi di Q = {")
    assert "V(gi,di)" in rss[0]
    # No duplicate linear lines next to the nonlinear pair.
    assert "C_ISS gi ss 1.500000e-09" not in text
    assert "C_RSS gi di 5.000000e-11" not in text
    assert text.count("C_ISS ") == 1 and text.count("C_RSS ") == 1
    assert "* Gate charge: measured nonlinear" in text


def test_opt_in_rejects_non_monotone_tables(charge_spec):
    charge_spec.q_iss_charge_c = [0.0, 4e-9, 2e-9, 7e-9]
    with pytest.raises(ValueError, match="q_iss charge table is not monotone"):
        build_tier2_subcircuit(charge_spec, measured_gate_charge=True)


def test_opt_in_rejects_non_zero_anchored_tables(charge_spec):
    # Offset every sample so the table stays monotone and only the zero
    # anchor is violated; the validator must still reject it.
    charge_spec.q_rss_charge_c = [
        1e-9 + q for q in charge_spec.q_rss_charge_c]
    with pytest.raises(ValueError, match="q_rss is not zero-anchored"):
        build_tier2_subcircuit(charge_spec, measured_gate_charge=True)


def test_emit_tier2_c0_uses_flag_only_with_recorded_opt_in(charge_spec):
    text = _emit_tier2_c0(charge_spec, name="Synthetic_UROM")
    assert "C_ISS gi ss Q = {" in text
    assert "C_RSS gi di Q = {" in text

    charge_spec.metadata.pop("urom_gate_charge")
    text = _emit_tier2_c0(charge_spec, name="Synthetic_UROM")
    assert "C_ISS gi ss 1.500000e-09" in text
    assert "C_ISS gi ss Q" not in text


def test_c0_identity_is_mode_aware(charge_spec):
    assert _spec_measured_gate_charge(charge_spec) is True
    measured_text = _emit_tier2_c0(charge_spec, name="Synthetic_UROM")
    # The A/B follows the recorded mode instead of falsely comparing a
    # measured emission against the linear canonical C0.
    assert _c0_bit_identity(
        charge_spec, measured_text, "Synthetic_UROM") is True
    linear = build_tier2_subcircuit(charge_spec, name="Synthetic_UROM")
    assert linear != measured_text
    assert _c0_bit_identity(
        charge_spec, measured_text, "Synthetic_UROM",
        measured_gate_charge=False) is False

    charge_spec.metadata.pop("urom_gate_charge")
    assert _spec_measured_gate_charge(charge_spec) is False
    linear_text = _emit_tier2_c0(charge_spec, name="Synthetic_UROM")
    assert _c0_bit_identity(
        charge_spec, linear_text, "Synthetic_UROM") is True


def test_environment_opt_in_is_narrow(monkeypatch):
    monkeypatch.delenv(MEASURED_GATE_CHARGE_ENV, raising=False)
    assert _measured_gate_charge_opt_in() is False
    monkeypatch.setenv(MEASURED_GATE_CHARGE_ENV, "0")
    assert _measured_gate_charge_opt_in() is False
    monkeypatch.setenv(MEASURED_GATE_CHARGE_ENV, "1")
    assert _measured_gate_charge_opt_in() is True


def test_qoss_compaction_opt_in_is_narrow(monkeypatch):
    monkeypatch.delenv(QOSS_COMPACTION_ENV, raising=False)
    assert _qoss_compaction_opt_in() is False
    monkeypatch.setenv(QOSS_COMPACTION_ENV, "0")
    assert _qoss_compaction_opt_in() is False
    monkeypatch.setenv(QOSS_COMPACTION_ENV, "true")
    assert _qoss_compaction_opt_in() is True


def _qoss_table(n=25):
    v = np.linspace(0.0, 50.0, n)
    q = 1e-7 * (v / 50.0) ** 1.5
    return v.tolist(), q.tolist()


def _qoss_spec(qoss_v, qoss_q):
    return PWLModelSpec(
        device_name="QossDev",
        breakpoints_vds=[0.0, 1.0, 2.0],
        breakpoints_id=[[0.0, 1.0, 2.0]],
        vgs_values=[1.0],
        q_oss_vs_v=list(qoss_v),
        q_oss_charge_c=list(qoss_q),
        rds_on=0.005,
        v_th=1.0,
    )


def test_qoss_compaction_counts_and_error():
    v, q = _qoss_table()
    cv, cq, meta = _compact_qoss_table("qdev", v, q, enabled=True)
    assert meta is not None and meta["enabled"] is True
    assert meta["opt_in"] == QOSS_COMPACTION_ENV
    assert meta["full_points"] == len(v)
    assert meta["emitted_points"] == len(cv) == len(cq)
    assert len(cv) <= _MEASURED_Q_MAX_POINTS
    assert len(cv) < len(v)
    assert cv[0] == 0.0 and cq[0] == 0.0
    assert cv[-1] == v[-1] and cq[-1] == pytest.approx(q[-1], rel=1e-12)
    assert np.all(np.diff(cq) >= 0.0)
    assert meta["max_error_pct"] <= 100.0 * _MEASURED_Q_REL_TOLERANCE
    assert meta["max_error_c"] <= meta["budget_c"]


def test_qoss_compaction_disabled_is_byte_identity():
    v, q = _qoss_table()
    dv, dq, meta = _compact_qoss_table("qdev", v, q, enabled=False)
    assert meta is None
    assert dv is v and dq is q
    # Feeding the disabled result through the builder is byte-identical to
    # the uncompacted source table.
    full = build_tier2_subcircuit(_qoss_spec(v, q))
    unchanged = build_tier2_subcircuit(_qoss_spec(dv, dq))
    assert full == unchanged
    full_line = next(
        line for line in full.splitlines() if line.startswith("C_QOSS "))
    assert full_line.count("abs(") == len(v) - 1   # one hinge per segment


def test_qoss_compaction_emits_fewer_hinges():
    v, q = _qoss_table()
    cv, cq, meta = _compact_qoss_table("qdev", v, q, enabled=True)
    full = build_tier2_subcircuit(_qoss_spec(v, q))
    compact = build_tier2_subcircuit(_qoss_spec(cv, cq))
    full_line = next(
        line for line in full.splitlines() if line.startswith("C_QOSS "))
    compact_line = next(
        line for line in compact.splitlines() if line.startswith("C_QOSS "))
    assert compact_line.count("abs(") < full_line.count("abs(")
    assert compact_line.count("abs(") <= meta["emitted_points"]
    assert full != compact
