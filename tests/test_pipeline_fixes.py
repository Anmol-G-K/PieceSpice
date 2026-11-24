"""Regressions for correctness defects found in the Tier-2 pipeline.

No LTspice required: the tests exercise the reference-voltage, metric and
comparison logic that previously produced silently wrong numbers.
"""

import logging

import numpy as np
import pytest

from approximation.charge_model import integrate_charge
from metrics.fitness import (
    _rel_err_pct, compare_switching_fitness, infer_drive_conditions)
from metrics.switching import extract_switching_times
from pipeline.curves import _cap_ref_vds
from simulation.testbenches.simple_switching import SwitchingRun
from approximation.smoother import fit_tanh_knee_ls


def _make_run(t, vds, vgs, vbus, vgs_on):
    return SwitchingRun(time=t, v_ds=vds, i_d=np.ones_like(t), v_gs=vgs,
                        v_bus=vbus, i_load=1.0, r_load=vbus, vgs_on=vgs_on)


def test_cap_reference_is_not_sweep_minimum():
    cond = {"Vds_cap_min": 0.1, "Vds_cap_max": 25.0}
    assert _cap_ref_vds(cond) == pytest.approx(6.325)
    assert _cap_ref_vds({**cond, "Cap_ref_Vds": 7.5}) == 7.5


def test_equivalent_caps_nonzero_at_reference():
    v = np.linspace(0.1, 25.0, 50)
    c = np.full_like(v, 1e-9)
    at_ref = integrate_charge(v, c, v_ref=6.325)
    expected_q = 1e-9 * (6.325 - 0.1) / 6.325   # Q(Vref)/Vref, curve starts at 0.1 V
    expected_e = 1e-9 * (1.0 - (0.1 / 6.325) ** 2)
    assert at_ref.c_eq_charge == pytest.approx(expected_q, rel=1e-6)
    assert at_ref.c_eq_energy == pytest.approx(expected_e, rel=2e-3)


def test_missing_crossings_are_nan_not_zero():
    t = np.linspace(0, 1e-6, 20)
    zero = np.zeros_like(t)
    m = extract_switching_times(t, zero, zero, zero, v_bus=10.0)
    assert np.isnan(m.delay_on_ns) and np.isnan(m.fall_time_ns)
    assert np.isnan(m.delay_off_ns) and np.isnan(m.rise_time_ns)


def test_tanh_knee_ls_recovers_full_curve_width():
    v = np.linspace(0.0, 10.0, 101)
    i = 50.0 * np.tanh(v / 3.0)
    assert fit_tanh_knee_ls(v, i, 50.0) == pytest.approx(3.0, rel=1e-3)

def test_emitted_channel_blocks_below_extracted_vth():
    from common.types import PWLModelSpec
    from models.netlist_builder import build_tier2_subcircuit

    spec = PWLModelSpec(
        device_name="VthDev",
        breakpoints_vds=[0.0, 1.0, 10.0],
        breakpoints_id=[[0.0, 5.0, 10.0], [0.0, 20.0, 40.0]],
        vgs_values=[4.5, 10.0], rds_on=0.01, v_th=2.0)
    line = next(l for l in build_tier2_subcircuit(spec).splitlines()
                if l.startswith("G_CH"))
    expr = line.split("VALUE = {", 1)[1].rsplit("}", 1)[0]
    pwl = expr[1:].split("*tanh")[0].replace("V(gi,ss)", "v")
    id_of_vgs = eval("lambda v: " + pwl, {"abs": abs})
    assert abs(id_of_vgs(1.0)) < 1e-12       # below Vth: blocked
    assert abs(id_of_vgs(2.0)) < 1e-12       # at Vth: still blocked
    assert id_of_vgs(3.0) > 0.0              # above Vth: conducting
    assert id_of_vgs(4.5) > id_of_vgs(3.0)

def test_relative_error_guards_zero_reference():
    assert np.isnan(_rel_err_pct(1.0, 0.0))
    assert np.isnan(_rel_err_pct(float("nan"), 1.0))
    assert _rel_err_pct(1.1, 1.0) == pytest.approx(10.0)


def test_drive_inferred_from_golden_waveform():
    g = {"v_ds": np.array([10.0] * 8 + [0.0] * 2),
         "v_gs": np.array([5.0] * 8 + [0.0] * 2)}
    assert infer_drive_conditions(g) == (10.0, 5.0)


def test_mismatched_drive_is_rejected():
    t = np.linspace(0, 2e-5, 100)
    g = _make_run(t, np.full_like(t, 10.0), np.zeros_like(t), 10.0, 5.0)
    m = _make_run(t, np.full_like(t, 10.0), np.zeros_like(t), 12.0, 5.0)
    with pytest.raises(ValueError, match="different v_bus"):
        compare_switching_fitness(vbus=10.0, vgs_on=5.0,
                                  golden_run=g, tier2_run=m)


def test_truncated_golden_is_not_compared_to_last_tier2_period():
    # 14 us golden (VM-10 truncation) vs 60 us Tier-2: the leading span of
    # the Tier-2 run must be used, not its final 20 us.
    tg = np.linspace(0, 14e-6, 700)
    tt = np.linspace(0, 60e-6, 3000)
    g = _make_run(tg, np.full_like(tg, 10.0), np.zeros_like(tg), 10.0, 10.0)
    m = _make_run(tt, np.full_like(tt, 10.0), np.zeros_like(tt), 10.0, 10.0)
    fit = compare_switching_fitness(vbus=10.0, vgs_on=10.0,
                                    golden_run=g, tier2_run=m)
    assert fit.waveform_rel_rmse_pct == pytest.approx(0.0, abs=1e-9)
    assert fit.evaluable is False        # no turn-off crossing in the golden
    assert fit.verdict.startswith("N/A")


def _stub_curve4(monkeypatch, *, c_eq_charge=1e-11, c_eq_energy=1e-12,
                 cap_ref_at_min=False):
    """Offline Coss capture + charge integration; no LTspice invocation."""
    from types import SimpleNamespace

    import approximation.charge_model as charge_mod
    import pipeline.fitting as fitting
    import simulation.testbenches.capacitance as cap_mod

    v = np.array([0.1, 6.3, 25.0])
    trace = SimpleNamespace(x_data=v, y_data=np.full(3, 1e-11))

    class _FakeTB:
        def __init__(self, **kwargs):
            pass

        def run(self, *args, **kwargs):
            return None, trace, None

    def _fake_integrate(x, y, *, v_ref=None, **kwargs):
        x = np.asarray(x, dtype=float)
        return charge_mod.ChargeModelResult(
            v=x, c=np.asarray(y, dtype=float),
            q=x * 1e-11, e=x ** 2 * 5e-12,
            c_eq_charge=c_eq_charge, c_eq_energy=c_eq_energy,
            v_ref=float(v_ref))

    monkeypatch.setattr(cap_mod, "CapacitanceTestbench", _FakeTB)
    monkeypatch.setattr(charge_mod, "integrate_charge", _fake_integrate)
    if cap_ref_at_min:
        monkeypatch.setattr(
            fitting, "_cap_ref_vds", lambda cond: float(cond["Vds_cap_min"]))
    return fitting


def test_zero_equivalent_caps_use_documented_fallback(monkeypatch, caplog):
    fitting = _stub_curve4(monkeypatch, c_eq_charge=0.0, c_eq_energy=0.0)
    with caplog.at_level(logging.WARNING):
        spec = fitting.build_pwl_spec("csd16412q5a")

    assert spec.metadata["c_iss_lin"] == pytest.approx(4.5e-9)
    assert spec.metadata["c_oss_lin"] == pytest.approx(500e-12)
    assert spec.metadata["c_rss_lin"] == pytest.approx(7.5e-12)
    assert "non-positive" in caplog.text.lower()
    assert "fallback" in caplog.text.lower()

    from models.netlist_builder import build_tier2_subcircuit
    text = build_tier2_subcircuit(spec)
    assert "C_ISS gi ss 0.000000e+00" not in text
    assert "C_RSS gi di 0.000000e+00" not in text


def test_cap_ref_at_sweep_min_uses_documented_fallback(monkeypatch, caplog):
    fitting = _stub_curve4(monkeypatch, cap_ref_at_min=True)
    with caplog.at_level(logging.WARNING):
        spec = fitting.build_pwl_spec("csd16412q5a")

    assert spec.metadata["c_iss_lin"] == pytest.approx(4.5e-9)
    assert spec.metadata["c_rss_lin"] == pytest.approx(7.5e-12)
    assert "cap_ref_vds" in caplog.text.lower()


def test_missing_curve7_fails_closed(monkeypatch):
    fitting = _stub_curve4(monkeypatch)
    real_load = fitting.load_output

    def _load_without_curve7(vendor, tb):
        if tb == "curve7_threshold_voltage":
            raise FileNotFoundError(tb)
        return real_load(vendor, tb)

    monkeypatch.setattr(fitting, "load_output", _load_without_curve7)
    with pytest.raises(ValueError, match="curve7"):
        fitting.build_pwl_spec("csd16412q5a")
