"""Regressions for correctness defects found in the Tier-2 pipeline.

No LTspice required: the tests exercise the reference-voltage, metric and
comparison logic that previously produced silently wrong numbers.
"""

import numpy as np
import pytest

from approximation.charge_model import integrate_charge
from metrics.fitness import (
    _rel_err_pct, compare_switching_fitness, infer_drive_conditions)
from metrics.switching import extract_switching_times
from pipeline.curves import _cap_ref_vds
from simulation.testbenches.simple_switching import SwitchingRun


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
    at_min = integrate_charge(v, c, v_ref=float(v[0]))
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
