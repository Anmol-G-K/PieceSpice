"""Tests for approximation.charge_model (M3.3)."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

from approximation.charge_model import (
    integrate_charge, equivalent_capacitance, pwl_charge_segments,
    synthesize_charge_source,
)


def test_constant_capacitor_exact():
    """C = const -> Q must be exactly C*V and E exactly 0.5*C*V^2."""
    v = np.linspace(0, 100, 101)
    c = np.full_like(v, 1e-9)          # 1 nF flat
    r = integrate_charge(v, c)
    assert np.allclose(r.q, 1e-9 * v, rtol=1e-12)
    assert r.e[-1] == pytest.approx(0.5 * 1e-9 * 100**2, rel=1e-10)


def test_equivalent_caps_constant_case_agree():
    """For a linear C both equivalent capacitances equal C."""
    v = np.linspace(0, 400, 401)
    c = np.full_like(v, 2e-9)
    cq, ce = equivalent_capacitance(v, c, v_ref=400.0)
    assert cq == pytest.approx(2e-9, rel=1e-9)
    assert ce == pytest.approx(2e-9, rel=1e-9)


def test_nonlinear_caps_differ_but_both_physical():
    """Classic MOSFET Coss: high at low V, collapsing at high V.
    Charge-equivalent > energy-equivalent for this shape (plan §2.3)."""
    v = np.linspace(0, 100, 201)
    c = 5e-9 / (1 + v / 5.0)           # monotonic collapse
    cq, ce = equivalent_capacitance(v, c, v_ref=80.0)
    assert 0 < ce < cq                 # energy weighting favors high-V samples
    assert ce < 5e-9 and cq < 5e-9


def test_v_ref_clamped_into_range():
    v = np.linspace(0, 50, 51)
    c = np.full_like(v, 1e-9)
    r = integrate_charge(v, c, v_ref=500.0)   # out of range -> clamped to 50
    assert r.v_ref == pytest.approx(50.0)


def test_negative_capacitance_rejected():
    with pytest.raises(ValueError, match="negative capacitance"):
        integrate_charge(np.array([0., 1., 2.]),
                         np.array([1e-9, -1e-9, 1e-9]))


def test_pwl_table_within_budget():
    v = np.linspace(0, 100, 501)
    c = 5e-9 / (1 + v / 5.0)
    r = integrate_charge(v, c)
    budget = 0.005 * r.q[-1]
    t = pwl_charge_segments(v, c)
    assert t.max_abs_error_c <= budget * 1.01   # tiny numeric slack
    assert t.n_points <= 64


def test_pwl_table_monotonic_q():
    v = np.linspace(0, 100, 301)
    c = 3e-9 / (1 + v / 8.0)
    t = pwl_charge_segments(v, c)
    assert np.all(np.diff(t.q_values) >= 0), "charge must be monotonic"
    assert np.all(np.diff(t.v_breaks) > 0), "breaks strictly increasing"


def test_synthesize_source_contains_element_and_expression():
    v = np.linspace(0, 100, 201)
    c = 5e-9 / (1 + v / 5.0)
    text = synthesize_charge_source("drn", "src", v, c)
    assert "Q={" in text
    assert "Cdrn_src" in text or "CQ_drn" in text
    assert "abs(" in text              # soft-abs ramp form present
    assert text.count("\n") >= 1       # header comment + element line


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "--no-header"]))




def test_integrate_charge_rejects_v_ref_zero():
    v = np.linspace(0, 50, 51)
    c = np.full_like(v, 1e-9)
    with pytest.raises(ValueError, match="v_ref must be > 0"):
        integrate_charge(v, c, v_ref=0.0)
    with pytest.raises(ValueError, match="v_ref must be > 0"):
        integrate_charge(v, c, v_ref=-5.0)
