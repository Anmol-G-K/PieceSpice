"""Offline tests for the Tier-4 synchronous-buck converter bench.

No LTspice is required: the bench is driven with a synthetic ``SimResult``
through its ``runner`` hook, so these tests prove efficiency extraction and
the completion guard without measuring a simulator.  The library file is a
minimal ideal-switch subcircuit written to ``tmp_path``.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulation.engine import SimResult
from simulation.testbenches.converter import (
    extract_efficiency,
    run_buck_converter,
)

SUB = "IDEAL_T4"
PINS = ["drain", "gate", "source"]
PERIOD = 1.0e-5            # 100 kHz
CYCLES = 20
T_END = CYCLES * PERIOD


def _library(tmp_path):
    path = tmp_path / "ideal_t4.lib"
    path.write_text(
        ".SUBCKT IDEAL_T4 d g s\n"
        "S_SW d s g s SW_T4\n"
        ".MODEL SW_T4 SW(Ron=0.01 Roff=1e12 Vt=2 Vh=0)\n"
        ".ENDS IDEAL_T4\n",
        encoding="ascii")
    return path


def _synthetic_result(*, t_end: float = T_END, vout: float = 5.0,
                      iin: float = -1.0, iout: float = 1.0,
                      iterations: int | None = 1000,
                      include_iin: bool = True,
                      include_load: bool = True,
                      bad_vout: bool = False) -> SimResult:
    time = np.linspace(0.0, t_end, 4001)
    vout_trace = np.full_like(time, vout)
    if bad_vout:
        vout_trace[-1] = float("nan")
    traces = {
        "time": time,
        "V(vout)": vout_trace,
        "I(Vbus)": np.full_like(time, iin),
    }
    if include_load:
        traces["I(Rload)"] = np.full_like(time, iout)
    if not include_iin:
        traces.pop("I(Vbus)")
    log = "" if iterations is None else f"Total iterations: {iterations}\n"
    return SimResult(traces=traces, sweep_var="time", log_text=log)


def _run(tmp_path, result: SimResult, *, duty: float = 0.5, **overrides):
    captured: dict = {}

    def runner(netlist, directive):
        captured["netlist"] = netlist
        captured["directive"] = directive
        return result

    kwargs = dict(
        vbus_v=12.0, duty=duty, f_sw_hz=1.0 / PERIOD,
        l_h=10.0e-6, c_out_f=100.0e-6, r_load_ohm=5.0,
        vgs_on_v=10.0, cycles=CYCLES, runner=runner)
    kwargs.update(overrides)
    out = run_buck_converter(_library(tmp_path), SUB, PINS, **kwargs)
    return out, captured


def test_buck_netlist_uses_two_switches_and_complementary_gates(tmp_path):
    out, captured = _run(tmp_path, _synthetic_result())
    netlist = captured["netlist"]
    assert "Xhs vin ghs sw IDEAL_T4" in netlist
    assert "Xls sw gls 0 IDEAL_T4" in netlist
    assert "Vghs ghs 0 PULSE(" in netlist
    assert "Vgls gls 0 PULSE(" in netlist
    assert "Lout sw vout" in netlist
    assert "Rload vout 0 5" in netlist
    assert out["schedule"]["t_end_s"] == pytest.approx(T_END)
    assert out["schedule"]["dead_time_s"] > 0.0
    assert captured["directive"].end_time == pytest.approx(T_END)


def test_complete_run_extracts_efficiency_from_measured_means(tmp_path):
    out, _ = _run(tmp_path, _synthetic_result())
    assert out["status"] == "measured"
    assert out["completed"] is True
    assert out["vout_v"] == pytest.approx(5.0)
    assert out["iout_a"] == pytest.approx(1.0)
    assert out["iin_a"] == pytest.approx(1.0)
    assert out["pout_w"] == pytest.approx(5.0)
    assert out["pin_w"] == pytest.approx(12.0)
    assert out["efficiency"] == pytest.approx(5.0 / 12.0)
    guard = out["guard"]
    assert guard["completed"] is True
    assert guard["trace_points"] == 4001
    assert guard["iterations"] == 1000
    assert guard["t_last_s"] == pytest.approx(T_END)
    assert guard["t_end_s"] == pytest.approx(T_END)
    assert out["extraction"]["points"] > 0
    assert out["extraction"]["iin_raw_mean_a"] == pytest.approx(-1.0)
    assert "iin = -mean" in out["extraction"]["iin_sign_rule"]
    assert len(out["netlist_sha256"]) == 64
    assert out["library"]["available"] is True
    assert len(out["library"]["sha256"]) == 64


def test_load_trace_absence_falls_back_to_vout_over_rload(tmp_path):
    out, _ = _run(tmp_path, _synthetic_result(include_load=False))
    assert out["status"] == "measured"
    assert out["iout_a"] == pytest.approx(5.0 / 5.0)
    assert "Vout/Rload" in out["extraction"]["iout_source"]


@pytest.mark.parametrize("kwargs", [
    {"t_end": T_END / 2.0},          # truncated waveform
    {"iterations": 0},               # aborted solver
    {"iterations": None},            # no solver-iteration evidence
])
def test_incomplete_runs_return_na_without_fabricated_efficiency(
        tmp_path, kwargs):
    out, _ = _run(tmp_path, _synthetic_result(**kwargs))
    assert out["status"] == "incomplete"
    assert out["completed"] is False
    assert out["efficiency"] is None
    assert out["pout_w"] is None and out["pin_w"] is None
    assert out["vout_v"] is None and out["iout_a"] is None
    assert "incomplete" in out["reason"]
    assert out["guard"]["completed"] is False


def test_missing_required_trace_is_reported_not_zero(tmp_path):
    out, _ = _run(tmp_path, _synthetic_result(include_iin=False))
    assert out["status"] == "missing_trace"
    assert out["completed"] is False
    assert "I(Vbus)" in out["reason"]
    assert out["efficiency"] is None
    assert out["guard"]["iterations"] == 1000


def test_nonfinite_waveform_keeps_efficiency_na(tmp_path):
    out, _ = _run(tmp_path, _synthetic_result(bad_vout=True))
    assert out["completed"] is True
    assert out["status"] == "invalid_measurement"
    assert out["efficiency"] is None
    assert out["pout_w"] is None
    assert out["measurement_status"] == "invalid_input"


def test_invalid_conditions_return_n_a_without_running(tmp_path):
    out, captured = _run(tmp_path, _synthetic_result(), duty=1.5)
    assert out["status"] == "invalid_input"
    assert out["completed"] is False
    assert out["efficiency"] is None
    assert "duty" in out["reason"]
    assert "netlist" not in captured


def test_extract_efficiency_requires_positive_input_power():
    good = extract_efficiency(5.0, 1.0, 1.0, 12.0)
    assert good["status"] == "computed"
    assert good["pout_w"] == pytest.approx(5.0)
    assert good["pin_w"] == pytest.approx(12.0)
    assert good["efficiency"] == pytest.approx(5.0 / 12.0)

    zero = extract_efficiency(5.0, 1.0, 0.0, 12.0)
    assert zero["status"] == "non_positive_input_power"
    assert zero["efficiency"] is None
    assert zero["pout_w"] == pytest.approx(5.0)

    invalid = extract_efficiency(float("nan"), 1.0, 1.0, 12.0)
    assert invalid["status"] == "invalid_input"
    assert invalid["pout_w"] is None and invalid["pin_w"] is None
    assert invalid["efficiency"] is None

    negative = extract_efficiency(5.0, -1.0, 1.0, 12.0)
    assert negative["status"] == "invalid_input"
    assert negative["efficiency"] is None

    no_bus = extract_efficiency(5.0, 1.0, 1.0, 0.0)
    assert no_bus["status"] == "invalid_input"
    assert no_bus["efficiency"] is None


def test_runner_exception_is_reported_as_error(tmp_path):
    def boom(netlist, directive):
        raise RuntimeError("no converter convergence")

    out = run_buck_converter(
        _library(tmp_path), SUB, PINS, vbus_v=12.0, duty=0.5,
        f_sw_hz=1.0 / PERIOD, l_h=10.0e-6, c_out_f=100.0e-6,
        r_load_ohm=5.0, vgs_on_v=10.0, cycles=CYCLES, runner=boom)
    assert out["status"] == "error"
    assert out["completed"] is False
    assert out["efficiency"] is None
    assert "RuntimeError" in out["reason"]
    assert out["guard"]["completed"] is False
