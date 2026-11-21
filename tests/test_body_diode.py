"""Measured body-diode (Curve 6) fit: recovery, acceptance and fail-closed.

Offline only: the synthetic curves are generated from the same SPICE diode
equation the emitter writes, so no simulator or vendor file is needed.
"""

import re

import numpy as np
import pytest

from common.types import PWLModelSpec
from models.netlist_builder import (
    _BODY_DIODE_RMS_MAX_DEC,
    _VT_300K,
    body_diode_current,
    body_diode_log10_rms,
    build_tier2_subcircuit,
    fit_body_diode,
    heuristic_body_diode,
)


def _curve(isat=1e-9, n=1.2, rs=0.01, v_lo=0.05, v_hi=1.5, points=61):
    """Exact forward curve of the emitted diode equation."""
    v = np.linspace(v_lo, v_hi, points)
    return v, body_diode_current(v, isat, n, rs)


def _spec(vsd, isd, *, rds_on=0.02):
    return PWLModelSpec(
        device_name="BodyDiodeDev",
        breakpoints_vds=[0.0, 1.0, 2.0],
        breakpoints_id=[[0.0, 1.0, 2.0]],
        vgs_values=[5.0],
        q_oss_vs_v=[0.0, 5.0],
        q_oss_charge_c=[0.0, 1e-9],
        rds_on=rds_on,
        v_th=2.0,
        body_diode_vsd=list(vsd),
        body_diode_isd=list(isd),
    )


def _model_line(text):
    match = re.search(
        r"^\.MODEL D_BD_(\S+) D\(IS=(\S+) N=(\S+) RS=(\S+)\)$",
        text, re.MULTILINE)
    assert match is not None, "fitted .MODEL D_BD_ line missing"
    return match


# ---------------------------------------------------------------------------
# Implicit equation
# ---------------------------------------------------------------------------
def test_current_solves_the_implicit_diode_equation():
    v = np.array([0.2, 0.45, 0.7, 1.1])
    isat, n, rs = 2e-9, 1.35, 0.02
    i = body_diode_current(v, isat, n, rs)
    assert np.all(i > 0.0)
    rhs = isat * np.expm1((v - i * rs) / (n * _VT_300K))
    assert np.allclose(i, rhs, rtol=1e-10, atol=0.0)


def test_zero_rs_reduces_to_expm1():
    v = np.linspace(0.0, 0.6, 7)
    got = body_diode_current(v, 1e-10, 1.7, 0.0)
    assert np.allclose(got, 1e-10 * np.expm1(v / (1.7 * _VT_300K)),
                       rtol=1e-12)


# ---------------------------------------------------------------------------
# Recovery of known parameters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "isat, n, rs",
    [
        (1e-9, 1.2, 0.01),
        (1e-12, 3.8, 0.008),
        (3.5e-10, 3.2, 0.03),
    ],
)
def test_fit_recovers_synthetic_parameters(isat, n, rs):
    v, i = _curve(isat, n, rs)
    fit = fit_body_diode(v, i, rds_on=max(3 * rs, 1e-3))
    assert fit.isat == pytest.approx(isat, rel=2e-3)
    assert fit.n == pytest.approx(n, rel=2e-3)
    assert fit.rs == pytest.approx(rs, rel=2e-3)
    assert fit.rms_log10 <= 1e-6
    assert fit.rms_log10 <= _BODY_DIODE_RMS_MAX_DEC
    assert 3 <= fit.n_retained <= fit.n_total == len(i)


def test_fit_beats_full_span_heuristic_on_hard_case():
    """RS-dominated high-current end biases the closed heuristic's N/IS."""
    v, i = _curve(isat=1e-9, n=1.15, rs=0.05, v_lo=0.0, points=61)
    heuristic = heuristic_body_diode(v, i, rds_on=0.5)
    fit = fit_body_diode(v, i, rds_on=0.5)

    assert heuristic.rms_log10 > 1.0          # old rule: many decades off
    assert fit.rms_log10 <= _BODY_DIODE_RMS_MAX_DEC
    assert fit.rms_log10 < heuristic.rms_log10
    # The comparison is on the samples the fit actually uses.
    mask = i >= fit.current_fraction * float(i.max())
    on_mask = body_diode_log10_rms(v[mask], i[mask],
                                   heuristic.isat, heuristic.n, heuristic.rs)
    assert fit.rms_log10 < on_mask


def test_fit_accepts_negative_measured_sign_convention():
    """Curve 6 stores Isd with vendor-dependent sign; the magnitude is used."""
    v, i = _curve(2e-10, 1.9, 0.005)
    pos = fit_body_diode(v, i, rds_on=0.02)
    neg = fit_body_diode(v, -i, rds_on=0.02)
    assert neg.isat == pytest.approx(pos.isat, rel=1e-12)
    assert neg.n == pytest.approx(pos.n, rel=1e-12)
    assert neg.rs == pytest.approx(pos.rs, rel=1e-12)


# ---------------------------------------------------------------------------
# Emitted model
# ---------------------------------------------------------------------------
def test_emits_fitted_model_and_provenance():
    v, i = _curve(1e-9, 1.2, 0.01)
    fit = fit_body_diode(v, i, rds_on=0.02)
    text = build_tier2_subcircuit(_spec(v, i))

    match = _model_line(text)
    assert match.group(1) == "BodyDiodeDev_PIECESPICE_T2"
    assert float(match.group(2)) == pytest.approx(fit.isat, rel=1e-5)
    assert float(match.group(3)) == pytest.approx(fit.n, abs=1e-3)
    assert float(match.group(4)) == pytest.approx(fit.rs, rel=1e-5)
    assert "D_BODY ss di D_BD_BodyDiodeDev_PIECESPICE_T2" in text

    assert "body-diode provenance:" in text
    assert "curve6_body_diode.yaml" in text
    assert "I=IS*(exp((V-I*RS)/(N*Vt))-1)" in text
    assert "Vt=0.026 V" in text
    assert f"RMS log10 error={fit.rms_log10:.6e} dec" in text
    text.encode("ascii")


def test_no_fabricated_reverse_recovery_fields():
    v, i = _curve()
    text = build_tier2_subcircuit(_spec(v, i))
    for token in ("Qrr", "trr", "TT=", "CJO", "Cj0"):
        assert token not in text


def test_two_point_curve_keeps_documented_heuristic():
    """Three parameters cannot be identified from two samples."""
    v, i = np.array([0.5, 0.8]), np.array([0.1, 1.0])
    reference = heuristic_body_diode(v, i, rds_on=0.02)
    text = build_tier2_subcircuit(_spec(v, i))

    match = _model_line(text)
    assert float(match.group(2)) == pytest.approx(reference.isat, rel=1e-5)
    assert float(match.group(3)) == pytest.approx(reference.n, abs=1e-3)
    assert float(match.group(4)) == pytest.approx(reference.rs, rel=1e-5)
    assert "underdetermined" in text
    assert "closed 2-point heuristic" in text
    assert "D_BODY ss di D_BD_BodyDiodeDev_PIECESPICE_T2" in text


def test_empty_curve_keeps_ideal_diode_fallback():
    text = build_tier2_subcircuit(_spec([], []))
    assert "* body-diode samples unavailable; ideal diode fallback" in text
    assert ".MODEL D_IDEAL D(IS=1e-12)" in text
    assert "D_BODY ss di D_IDEAL" in text


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------
def test_fit_rejects_missing_short_and_non_finite_data():
    with pytest.raises(ValueError, match="empty"):
        fit_body_diode([], [], rds_on=0.02)
    with pytest.raises(ValueError, match="at least 3"):
        fit_body_diode([0.5], [0.1], rds_on=0.02)
    with pytest.raises(ValueError, match="at least 3"):
        fit_body_diode([0.3, 0.5], [1e-3, 1e-2], rds_on=0.02)
    with pytest.raises(ValueError, match="non-finite"):
        fit_body_diode([0.2, 0.4, 0.6], [1e-3, np.nan, 0.1], rds_on=0.02)
    with pytest.raises(ValueError, match="non-finite"):
        fit_body_diode([0.2, 0.4, 0.6], [1e-3, np.inf, 0.1], rds_on=0.02)
    with pytest.raises(ValueError, match="non-negative forward"):
        fit_body_diode([-0.2, 0.4, 0.6], [1e-3, 1e-2, 0.1], rds_on=0.02)
    with pytest.raises(ValueError, match="length mismatch"):
        fit_body_diode([0.2, 0.4, 0.6], [1e-3, 1e-2], rds_on=0.02)
    with pytest.raises(ValueError, match="finite positive Rds"):
        fit_body_diode(*_curve(), rds_on=0.0)


def test_unfittable_curve_fails_closed():
    """A non-monotone 'curve' cannot meet the 0.20-decade budget."""
    v = np.array([0.2, 0.4, 0.6, 0.8])
    i = np.array([1e-6, 1e-3, 1e-6, 1e-3])
    with pytest.raises(ValueError, match="cannot meet the acceptance budget"):
        fit_body_diode(v, i, rds_on=0.02)
    with pytest.raises(ValueError, match="cannot meet the acceptance budget"):
        build_tier2_subcircuit(_spec(v, i))


def test_emitter_rejects_short_and_corrupt_curves():
    with pytest.raises(ValueError, match="1 sample"):
        build_tier2_subcircuit(_spec([0.5], [0.1]))
    with pytest.raises(ValueError, match="non-finite"):
        build_tier2_subcircuit(
            _spec([0.2, 0.4, 0.6], [1e-3, np.nan, 0.1]))
    with pytest.raises(ValueError, match="length mismatch"):
        build_tier2_subcircuit(
            _spec([0.2, 0.4, 0.6, 0.8], [1e-3, 1e-2, 0.1]))
