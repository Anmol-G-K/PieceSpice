"""DPT bench current identity: device current is I(Lload) - I(Dclamp)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from simulation.engine import SimResult
from simulation.testbenches import double_pulse as dp


class _FakeEngine:
    def __init__(self, traces, **_kwargs):
        self._traces = traces

    def run_transient(self, _text, _directive):
        return SimResult(traces=self._traces, sweep_var="time",
                         log_text="Total iterations: 10\n")


def _run(monkeypatch, traces):
    monkeypatch.setattr(dp, "SpiceEngine", lambda **_kw: _FakeEngine(traces))
    return dp.run_double_pulse(
        "unused.lib", "TOY", ["drain", "gate", "source"],
        vbus=10.0, vgs_on=5.0, i_load_target=2.0, l_load=1e-6,
        t1_on=1e-6, t1_width=1e-6, t2_on=5e-6, t2_width=1e-6,
        edge_time=1e-9, t_end=10e-6, max_step=1e-9, temperature=25.0)


def test_dpt_uses_device_current_when_clamp_trace_exists(monkeypatch):
    t = np.linspace(0.0, 1e-5, 11)
    run = _run(monkeypatch, {
        "time": t,
        "V(drain)": np.full_like(t, 10.0),
        "V(gate)": np.zeros_like(t),
        "I(Lload)": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                              7.0, 8.0, 9.0, 10.0, 11.0]),
        "I(Dclamp)": np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                               0.5, 0.5, 0.5, 0.5, 0.5]),
    })
    assert run.current_definition == "I(Lload)-I(Dclamp)"
    assert np.allclose(run.i_d, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
                                 6.5, 7.5, 8.5, 9.5, 10.5])


def test_dpt_fallback_is_labelled_and_never_silent(monkeypatch):
    t = np.linspace(0.0, 1e-5, 3)
    run = _run(monkeypatch, {
        "time": t,
        "V(drain)": np.full_like(t, 10.0),
        "V(gate)": np.zeros_like(t),
        "I(Lload)": np.array([-1.0, 2.0, 3.0]),
    })
    assert "fallback" in run.current_definition
    assert np.allclose(run.i_d, [1.0, 2.0, 3.0])


# ---- signed trace: energy integration only ---------------------------------

def test_bench_keeps_the_signed_trace_beside_the_absolute_one(monkeypatch):
    t = np.linspace(0.0, 1e-5, 3)
    run = _run(monkeypatch, {
        "time": t,
        "V(drain)": np.full_like(t, 10.0),
        "V(gate)": np.zeros_like(t),
        "I(Lload)": np.array([2.0, 2.0, 2.0]),
        "I(Dclamp)": np.array([3.0, 0.0, 0.5]),
    })
    assert np.allclose(run.i_d, [1.0, 2.0, 1.5])
    assert np.allclose(run.i_d_signed, [-1.0, 2.0, 1.5])


def _waveform(*, ring: bool = True):
    """Turn-off at full current, then an optional zero-mean current ring."""
    t = np.arange(0.0, 201e-9, 0.25e-9)
    vgs = np.where(t < 40e-9, 10.0,
                   np.where(t < 45e-9, 10.0 * (1.0 - (t - 40e-9) / 5e-9), 0.0))
    vds = np.where(t < 30e-9, 10.0,
                   np.where(t < 31e-9, 10.0 * (1.0 - (t - 30e-9) / 1e-9), 0.1))
    vds = np.where(t < 50e-9, vds,
                   np.where(t < 51e-9,
                            0.1 + (10.0 - 0.1) * (t - 50e-9) / 1e-9, 10.0))
    signed = np.where(t < 50e-9, 5.0,
                      np.where(t < 51e-9, 5.0 * (1.0 - (t - 50e-9) / 1e-9), 0.0))
    if ring:
        signed = np.where(t >= 51e-9,
                          np.sin(2.0 * np.pi * (t - 51e-9) / 4e-9), signed)
    return t, vds, vgs, signed


def _synthetic(*, ring: bool = True) -> dp.DoublePulseRun:
    t, vds, vgs, signed = _waveform(ring=ring)
    return dp.DoublePulseRun(
        time=t, v_ds=vds, i_d=np.abs(signed), v_gs=vgs, v_bus=10.0,
        i_load=5.0, l_load=1e-6, vgs_on=10.0, wall_clock_s=0.0, t2_on=0.0)


def test_energy_uses_the_signed_current_and_timing_keeps_the_magnitude():
    abs_run = _synthetic()
    signed_run = replace(abs_run, i_d_signed=abs_run.i_d.copy())
    signed_run.i_d_signed[abs_run.time >= 51e-9] *= -1.0  # reversed tail
    signed = dp.extract_dpt_metrics(signed_run)
    absolute = dp.extract_dpt_metrics(abs_run)
    # Same magnitude -> identical timing, slew and peak.
    for key in ("td_on_ns", "tr_ns", "td_off_ns", "tf_ns",
                "dv_dt_max_v_per_ns", "di_dt_max_a_per_ns", "i_peak_a"):
        assert np.isclose(signed[key], absolute[key], equal_nan=True), key
    # Reversed tail current is recovered reactance, not dissipation.
    assert signed["e_off_uj"] < absolute["e_off_uj"]
    assert signed["energy_current_definition"].endswith("signed")
    assert absolute["energy_current_definition"].endswith("abs fallback")


def test_abs_fallback_is_unchanged_when_no_signed_trace_exists():
    run = _synthetic(ring=False)
    metrics = dp.extract_dpt_metrics(run)
    # The pre-fix integration path: same numbers, labelled fallback.
    assert metrics["e_off_uj"] > 0.0
    assert "abs fallback" in metrics["energy_current_definition"]
    assert np.allclose(run.i_d, np.abs(run.i_d))


# ---- propagation into the unified-ROM arm ----------------------------------

class _FakeRawResult:
    """Minimal raw solver result carrying the DPT save traces."""

    def __init__(self, traces):
        self.traces = traces

    def get(self, key):
        return self.traces[key]


def test_urom_arm_passes_the_signed_trace_and_keeps_abs_for_waveforms(
        monkeypatch):
    from types import SimpleNamespace

    from pipeline import unified_rom as urom

    t = np.linspace(0.0, 1.0e-5, 3)
    load = np.array([1.0, -2.0, 3.0])
    clamp = np.array([0.5, 0.5, 0.5])
    raw = _FakeRawResult({
        "time": t, "V(drain)": np.array([10.0, 0.0, 10.0]),
        "V(gate)": np.array([0.0, 5.0, 0.0]),
        "I(Lload)": load, "I(Dclamp)": clamp})
    monkeypatch.setattr(
        urom, "_one_transient",
        lambda *a, **k: {"result": raw, "iterations": 7, "points": 3,
                         "t_last": 10.0e-6, "wall_time_s": 0.25,
                         "log_tail": ""})
    captured = {}

    def fake_extract(run):
        captured["run"] = run
        return {"energy_current_definition":
                f"{run.current_definition} signed"}

    monkeypatch.setattr(urom, "extract_dpt_metrics", fake_extract)
    plan = {"dpt_max_step": 1e-9, "dpt_t_end": 10.0e-6, "temperature": 25.0,
            "vbus": 10.0, "i_load": 2.0, "l_load": 1e-6, "vgs_on": 5.0}
    arm = urom._run_dpt_arm(plan, "urom", "", SimpleNamespace(time=t),
                            {"t2_on": 5e-6})
    run = captured["run"]
    assert np.allclose(run.i_d_signed, load - clamp)
    assert np.allclose(run.i_d, np.abs(load - clamp))
    assert run.i_d_signed[1] < 0.0 < run.i_d[1]
    assert run.current_definition == "I(Lload)-I(Dclamp)"
    # G3/waveform identity stays the absolute magnitude.
    assert np.allclose(arm["_raw"]["ids"], np.abs(load - clamp))
    assert arm["energy_current_definition"].endswith("signed")


def test_urom_energy_definition_mismatch_is_reported_not_scored():
    from pipeline import unified_rom as urom

    rows, ok = urom._timing_score_rows(
        {"td_on_ns": 10.0, "tr_ns": 1.0, "td_off_ns": 10.0, "tf_ns": 1.0,
         "e_on_uj": 0.5, "e_off_uj": 0.5},
        {"td_on_ns": 10.0, "tr_ns": 1.0, "td_off_ns": 10.0, "tf_ns": 1.0,
         "e_on_uj": 0.5, "e_off_uj": 0.5})
    assert ok is True
    rows, comparable = urom._apply_energy_definition(
        rows, "I(Lload)-I(Dclamp) signed",
        dp.LEGACY_ABS_ENERGY_DEFINITION)
    assert comparable is False
    assert rows["e_off_uj"]["status"] == "definition_mismatch"
    assert rows["e_off_uj"]["pass"] is False
    assert rows["e_off_uj"]["actual_energy_definition_kind"] == "signed"
    assert rows["e_off_uj"]["bar_energy_definition_kind"] == "abs"
    # Timing rows are absolute-current metrics and stay scored.
    assert rows["td_on_ns"]["status"] == "measured"
    assert rows["td_on_ns"]["pass"] is True


def test_tuning_excludes_energy_when_golden_and_candidate_definitions_differ(
        monkeypatch):
    from pipeline import unified_rom as urom

    emp = {"td_on_ns": 1.0, "tr_ns": 1.0, "td_off_ns": 1.0, "tf_ns": 1.0,
           "e_on_uj": 1.0, "e_off_uj": 1.0, "i_peak_a": 1.0}
    monkeypatch.setattr(urom, "_recorded_golden_emp",
                        lambda vendor: (dict(emp), True, "recorded"))
    monkeypatch.setattr(
        urom, "_recorded_energy_definition",
        lambda *a, **k: urom.LEGACY_ABS_ENERGY_DEFINITION)
    plan = {"vbus": 10.0, "vgs_on": 5.0, "i_load": 2.0, "l_load": 1e-6,
            "temperature": 25.0}
    conditions = urom._tuning_condition_descriptors("v", plan)
    assert conditions
    note = conditions[0]["energy_definition"]
    assert note["comparable"] is False
    assert note["excluded_metrics"] == ["e_on_uj", "e_off_uj"]
    for cond in conditions:
        assert "e_on_uj" not in cond["emp"]
        assert "e_off_uj" not in cond["emp"]

    monkeypatch.setattr(
        urom, "_recorded_energy_definition",
        lambda *a, **k: "I(Lload)-I(Dclamp) signed")
    signed = urom._tuning_condition_descriptors("v", plan)
    assert signed[0]["energy_definition"]["comparable"] is True
    assert signed[0]["energy_definition"]["excluded_metrics"] == []


def test_urom_reads_signed_energy_labels_from_record_mappings(
        tmp_path, monkeypatch):
    import json

    from pipeline import unified_rom as urom

    signed = "I(Lload)-I(Dclamp) signed"
    base = tmp_path / "reports" / "comparison" / "v"
    base.mkdir(parents=True)
    (base / "tier3_validation.json").write_text(json.dumps({
        "energy_current_definition": {"golden": signed, "tier3": signed},
    }), encoding="utf-8")
    (base / "dpt_metrics.json").write_text(json.dumps({
        "energy_current_definition": {"golden": signed, "tier2": signed},
    }), encoding="utf-8")
    monkeypatch.setattr(urom.registry, "ROOT", tmp_path)
    assert urom._recorded_energy_definition(
        "v", "tier3_validation.json") == signed
    assert urom._recorded_energy_definition(
        "v", "dpt_metrics.json", column="tier2_metrics") == signed
    rows, _ = urom._timing_score_rows(
        {"e_on_uj": 1.0, "e_off_uj": 1.0},
        {"e_on_uj": 1.0, "e_off_uj": 1.0})
    _, comparable = urom._apply_energy_definition(rows, signed, signed)
    assert comparable is True
    assert rows["e_off_uj"]["energy_definition_comparison"] == "comparable"


def test_tier3_reads_energy_labels_from_dpt_metrics_mapping(
        tmp_path, monkeypatch):
    import json

    from pipeline import tier3

    signed = "I(Lload)-I(Dclamp) signed"
    base = tmp_path / "reports" / "comparison" / "v"
    base.mkdir(parents=True)
    (base / "dpt_metrics.json").write_text(json.dumps({
        "energy_current_definition": {"golden": signed, "tier2": signed},
    }), encoding="utf-8")
    monkeypatch.setattr(tier3, "ROOT", tmp_path)
    assert tier3._recorded_energy_definitions("v") == {
        "golden": signed, "tier2": signed}
