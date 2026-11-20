"""Offline tests for the Tier-4 build/validation pipeline.

No LTspice bench is executed here. The recorded validation JSONs are checked
as committed artifacts, and the pipeline helpers are exercised with synthetic
failures to prove that aborted or missing runs stay N/A instead of becoming
zero or a speedup.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest

import pipeline.tier4 as tier4
from config.yaml_io import load_input

ROOT = Path(__file__).resolve().parent.parent
VENDORS = ("irfs7730pbf", "csd16412q5a", "epc2216", "cpm2-1200")


def _generated_lib(vendor: str) -> Path:
    """Path to the generated (untracked) Tier-4 ASCII subcircuit."""
    return ROOT / "reports" / "models" / f"{vendor}_PIECESPICE_T4.lib"


def _skip_without_generated_lib(vendor: str) -> None:
    lib = _generated_lib(vendor)
    if not lib.is_file():
        pytest.skip(f"generated Tier-4 lib absent locally: {lib}")


def _skip_without_vendor_lib(vendor: str) -> None:
    vendor_lib = ROOT / load_input(vendor)["library"]
    if not vendor_lib.is_file():
        pytest.skip(f"vendor library not present locally: {vendor_lib}")



def _record(vendor: str) -> dict:
    path = (ROOT / "reports" / "comparison" / vendor /
            "tier4_validation.json")
    assert path.exists(), f"missing validation record for {vendor}"
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_energy_disclosure(row: dict, vendor: str) -> None:
    """Zero error against the LUT's own golden source is not a valid gate."""
    assert row["independent_energy_gate"] is False
    assert row["source_shared"] is True
    assert row["kind"] == "source_self_consistency"
    assert row["shared_source"] == (
        f"reports/comparison/{vendor}/dpt_metrics.json#golden_metrics")
    assert "not an independent" in row["disclosure"]


@pytest.mark.parametrize("vendor", VENDORS)
def test_record_schema_and_retired_waveform_gate(vendor):
    rec = _record(vendor)
    assert rec["generation"]["mode"] in ("offline_recorded", "live")
    assert rec["generation"]["live_benches_executed"] == (
        rec["generation"]["mode"] == "live")
    assert rec["schema"] == "piecespice.tier4/1.0"
    assert rec["vendor"] == vendor
    assert rec["waveform_gate"] == "retired_by_design"
    assert rec["guards"]["waveform_gate"] == "retired_by_design"
    assert rec["emitted_lib"].endswith(f"{vendor}_PIECESPICE_T4.lib")
    assert rec["subcircuit"] == f"{vendor}_PIECESPICE_T4"
    assert rec["pin_roles"] == ["drain", "gate", "source"]


@pytest.mark.parametrize("vendor", VENDORS)
def test_energy_rows_are_finite_and_self_consistent(vendor):
    rec = _record(vendor)
    assert rec["energy_gate_disclosure"] == tier4.ENERGY_GATE_DISCLOSURE
    for metric in ("e_on_uj", "e_off_uj"):
        row = rec["energy_comparison"][metric]
        assert row["status"] == "measured", (vendor, metric, row)
        assert isinstance(row["golden"], float) and math.isfinite(row["golden"])
        assert isinstance(row["tier4"], float) and math.isfinite(row["tier4"])
        assert math.isclose(row["abs_error"],
                            abs(row["tier4"] - row["golden"]), abs_tol=1e-12)
        if row["golden"]:
            assert math.isclose(
                row["rel_error_pct"],
                100.0 * (row["tier4"] - row["golden"]) / abs(row["golden"]),
                abs_tol=1e-9)
        _assert_energy_disclosure(row, vendor)


@pytest.mark.parametrize("vendor", VENDORS)
def test_recorded_speedup_never_fabricates_timing(vendor):
    rec = _record(vendor)
    speed = rec["speedup"]
    # The bench exposes no per-run time vector: t_last is N/A either way.
    assert speed["t_last_s"] is None
    assert speed["measured_trace_endpoint_available"] is False
    assert "no measured trace endpoint" in speed["t_last_source"]
    assert ".tran stop" in speed["end_time_s_source"]
    if vendor == "cpm2-1200":
        assert speed["status"] == "N/A"
        err = speed["error"]
        assert "contract-excluded" in err
        assert "VM-10 vendor turn-off abort" in err
        assert "artifact conflict is disclosed in vendor_scope_notes" in err
        assert "no ratio is derived" in err
        assert speed["solver_speedup"] is None
        assert speed["wall_speedup"] is None
        return
    if speed["status"] == "N/A":
        # An aborted/unmeasured arm is N/A, never zero or a fabricated ratio.
        assert speed["solver_speedup"] is None
        assert speed["wall_speedup"] is None
        assert speed["end_time_s"] is None
        return
    # A completed live run reports positive counts for both arms and a
    # schedule-labeled end time; count similarity is not required.
    assert speed["status"] == "measured"
    for key in ("golden_iterations", "tier4_iterations",
                "golden_points", "tier4_points"):
        assert isinstance(speed[key], float) and speed[key] > 0.0
    assert "count similarity is not required" in speed["completion_basis"]
    assert speed["end_time_s"] == pytest.approx(1.76e-4)
    assert speed["solver_speedup"] > 0.0
    assert speed["wall_speedup"] > 0.0


def test_cpm2_speed_is_na_from_recorded_golden_failure():
    rec = _record("cpm2-1200")
    policy = rec["runtime"]["golden_cap"]
    assert policy["reason"] is None
    speed = rec["speedup"]
    assert speed["case"] == "long_adaptive"
    assert speed["status"] == "N/A"
    assert speed["solver_speedup"] is None
    assert speed["wall_speedup"] is None
    err = speed["error"]
    assert "contract-excluded" in err
    assert "VM-10" in err
    assert "artifact conflict is disclosed in vendor_scope_notes" in err
    assert "no ratio is derived" in err
    notes = rec["vendor_scope_notes"]
    assert notes["defect"] == "VM-10"
    assert "long_adaptive" in notes["artifact_conflict"]
    # The golden DPT energy crossing is still evaluable for CPM2.
    assert rec["energy_comparison"]["e_on_uj"]["status"] == "measured"


@pytest.mark.parametrize("vendor", VENDORS)
def test_emitted_lib_is_ascii_with_ideal_switch_header(vendor):
    _skip_without_generated_lib(vendor)
    lib = _generated_lib(vendor)
    assert lib.exists()
    text = lib.read_text(encoding="ascii")
    assert text.isascii()
    name, roles = tier4._parse_emitted_header(text)
    assert name == f"{vendor}_PIECESPICE_T4"
    assert roles == ["drain", "gate", "source"]
    assert "S_SW d s g s SW_T4" in text
    models = [ln for ln in text.splitlines()
              if ln.lstrip().upper().startswith(".MODEL SW_T4 SW(")]
    assert len(models) == 1
    assert "Roff=" in models[0]
    caps = [ln for ln in text.splitlines()
            if not ln.lstrip().startswith("*")
            and ln.lstrip().upper().startswith("C_QOSS ")]
    assert len(caps) == 1, caps
    tokens = caps[0].split()
    assert tokens[1:3] == ["d", "s"]
    assert float(tokens[3]) > 0.0


@pytest.mark.parametrize("vendor", VENDORS)
def test_loss_tables_are_metadata_not_active_elements(vendor):
    _skip_without_generated_lib(vendor)
    text = _generated_lib(vendor).read_text(encoding="ascii")

    for line in text.splitlines():
        if "e_on_uj" in line or "e_off_uj" in line:
            assert line.lstrip().startswith("*"), line
    active = [ln.lstrip().upper() for ln in text.splitlines()
              if not ln.lstrip().startswith("*")]
    caps = [ln for ln in active if ln.startswith("C")]
    assert len(caps) == 1 and caps[0].startswith("C_QOSS "), caps
    # No voltage-dependent capacitor or body diode in the fast tier.
    assert not any("Q=" in ln for ln in caps)
    assert not any(ln.startswith("D") for ln in active)


def test_speed_policy_derives_epc_cap_from_recorded_artifact():
    speed = tier4._read_json(ROOT / "reports" / "comparison" /
                             "speedup_benchmark.json")
    epc = tier4._speed_policy("epc2216", speed)
    assert epc["case"] == "long_100ns"
    assert epc["golden_max_step"] == 1e-7
    assert epc["cap_reason"] == "vendor stalls under adaptive step; use long_100ns"
    assert epc["golden_failure"] is None
    assert tier4._speed_policy("csd16412q5a", speed)["golden_max_step"] is None


def test_json_safe_maps_nan_to_none():
    value = {"a": float("nan"), "b": [1.0, float("inf"), None], "c": 2.5}
    assert tier4._json_safe(value) == {"a": None, "b": [1.0, None, None],
                                       "c": 2.5}
    with pytest.raises(ValueError):
        json.dumps(value, allow_nan=False)
    json.dumps(tier4._json_safe(value), allow_nan=False)


def test_energy_comparison_reports_nan_crossing_as_na():
    class _FakeLut:
        def energy_at(self, vbus, current):
            assert (vbus, current) == (11.0, 10.0)
            return {"e_on_uj": float("nan"), "e_off_uj": 15.0,
                    "status": "not_evaluable", "source": {"gap": "no crossing"}}

    rows = tier4._energy_comparison(
        _FakeLut(),
        {"golden_metrics": {"e_on_uj": 1.0, "e_off_uj": 15.0}},
        11.0, 10.0)
    assert rows["e_on_uj"]["status"] == "N/A"
    assert rows["e_on_uj"]["rel_error_pct"] is None
    assert rows["e_off_uj"]["status"] == "N/A"  # status gate applies to both
    assert rows["e_off_uj"]["tier4"] == 15.0
    for row in rows.values():
        assert row["independent_energy_gate"] is False
        assert row["source_shared"] is True
        assert row["shared_source"] == "dpt_metrics.json#golden_metrics"


SCHEDULED_END_S = 1.76e-4  # 20 pulses at the pinned config step/width
POLICY = {"golden_max_step": None, "case": "long_100ns"}


def _arm(iterations: float, points: float, **overrides) -> dict:
    """A complete run_inductive_benchmark arm with its schedule proof."""
    arm = {"solver_elapsed_s": 0.3, "iterations": iterations,
           "trace_points": points, "end_time_s": SCHEDULED_END_S,
           "repeats": 3, "n_pulses": 20}
    arm.update(overrides)
    return arm


def _pair_ok(left: dict, right: dict) -> bool:
    return tier4._bench_complete(
        left, right, scheduled_end_s=SCHEDULED_END_S)


def test_scheduled_bench_end_matches_full_20_pulse_window():
    bench_kw = {"n_pulses": 20, "pulse_period": 9e-6,
                "pulse_width": 2e-6, "first_pulse_width": 1.5e-6}
    assert tier4._scheduled_bench_end(bench_kw) == pytest.approx(1.76e-4)
    assert tier4.SPEED_N_PULSES == 20


def test_bench_complete_accepts_20x_count_reduction():
    golden = _arm(120000.0, 60000.0, solver_elapsed_s=6.0)
    fast = _arm(6000.0, 3000.0, solver_elapsed_s=0.3)
    assert golden["iterations"] / fast["iterations"] == 20.0
    assert golden["trace_points"] / fast["trace_points"] == 20.0
    assert _pair_ok(golden, fast) is True
    assert _pair_ok(fast, golden) is True
    row = tier4._speedup_row(golden, fast, True, POLICY)
    assert row["status"] == "measured"
    assert row["solver_speedup"] == pytest.approx(20.0)
    assert row["t_last_s"] is None
    assert "count similarity is not required" in row["completion_basis"]
    assert "lower counts are expected for faster models" in row["completion_basis"]
    assert ("aborted runs are rejected by positive iterations/points and no "
            "error") in row["completion_basis"]


def test_bench_complete_rejects_zero_error_and_aborted_arms():
    good = _arm(12000.0, 3000.0)
    assert _pair_ok(good, good) is True
    for bad in (
        dict(good, iterations=0.0),
        dict(good, trace_points=0.0),
        dict(good, solver_elapsed_s=0.0),
        dict(good, iterations=float("nan")),
        dict(good, error="no raw output / rc=1"),
        dict(good, iterations=0.0, trace_points=0.0,
             error="aborted by timeout"),
        dict(good, end_time_s=SCHEDULED_END_S / 2),
        dict(good, repeats=2),
        {"end_time_s": SCHEDULED_END_S},  # schedule alone is not a run
    ):
        assert _pair_ok(good, bad) is False
        assert _pair_ok(bad, good) is False
    row = tier4._speedup_row(
        good, dict(good, error="aborted by timeout"), False, POLICY)
    assert row["status"] == "N/A"
    assert row["solver_speedup"] is None
    assert row["t_last_s"] is None
    assert row["end_time_s"] is None
    assert row["golden_iterations"] == 12000.0
    assert row["tier4_points"] == 3000.0
    assert ".tran stop" in row["end_time_s_source"]


def test_dpt_guard_rejects_truncated_waveform():
    class _Result:
        log_text = "Total iterations: 10\n"

        def get(self, name):
            assert name == "time"
            return [0.0, 1.0e-6]

    guard = tier4._dpt_guard(
        _Result(), "Total iterations: 10\n", {"t_end": 1.4e-5}, 7000, 0.1)
    assert guard["executed"] is True
    assert guard["completed"] is False
    assert guard["t_last_s"] == 1.0e-6
    assert "incomplete" in guard["error"]


def _seed_repo(tmp_path: Path, vendor: str) -> None:
    shutil.copytree(ROOT / "config", tmp_path / "config")
    out = tmp_path / "reports" / "comparison"
    (out / vendor).mkdir(parents=True)
    shutil.copy(ROOT / "reports" / "comparison" / vendor /
                "dpt_metrics.json", out / vendor / "dpt_metrics.json")
    shutil.copy(ROOT / "reports" / "comparison" / "speedup_benchmark.json",
                out / "speedup_benchmark.json")


def test_run_vendor_offline_builds_full_record(tmp_path, monkeypatch):
    vendor = "epc2216"
    _seed_repo(tmp_path, vendor)
    monkeypatch.setattr(tier4, "ROOT", tmp_path)
    rec = tier4.run_vendor(vendor, live=False)
    assert rec["schema"] == tier4.SCHEMA
    assert rec["emission"]["status"] == "emitted"
    assert rec["guards"]["emission_valid"] is True
    assert rec["guards"]["ascii_emission"] is True
    assert rec["dpt_run"]["executed"] is False
    assert rec["speedup"]["status"] == "N/A"
    assert rec["energy_comparison"]["e_off_uj"]["status"] == "measured"
    out = (tmp_path / "reports" / "comparison" / vendor /
           "tier4_validation.json")
    assert json.loads(out.read_text(encoding="utf-8"))["schema"] == tier4.SCHEMA
    assert (tmp_path / "reports" / "models" /
            f"{vendor}_PIECESPICE_T4.lib").exists()
    completion = rec["speedup"]["completion_basis"]
    assert "count similarity is not required" in completion
    assert "lower counts are expected for faster models" in completion
    assert ("aborted runs are rejected by positive iterations/points and no "
            "error") in completion
    decisions = rec["provenance"]["decisions"]
    assert "speed_count_ratio_fraction" not in (
        decisions["run_constants"]["values"])
    assert "count similarity is not required" in (
        decisions["speed_completion"]["basis"])
    assert not hasattr(tier4, "BENCH_COUNT_RATIO_FRACTION")


def test_run_vendor_never_benches_excluded_golden(tmp_path, monkeypatch):
    vendor = "cpm2-1200"
    _seed_repo(tmp_path, vendor)
    monkeypatch.setattr(tier4, "ROOT", tmp_path)

    def _forbid(*args, **kwargs):
        raise AssertionError("no speed bench may run for an excluded golden")

    monkeypatch.setattr(tier4, "run_inductive_benchmark", _forbid)
    monkeypatch.setattr(tier4, "_run_emitted_dpt", _forbid)
    rec = tier4.run_vendor(vendor, live=True)
    speed = rec["speedup"]
    assert speed["status"] == "N/A"
    assert speed["solver_speedup"] is None
    assert speed["wall_speedup"] is None
    assert rec["dpt_run"]["completed"] is None
    err = speed["error"]
    assert "contract-excluded" in err
    assert "VM-10" in err
    assert "artifact conflict is disclosed in vendor_scope_notes" in err
    assert "no ratio is derived" in err


def test_live_flow_uses_20_pulses_and_matching_step(tmp_path, monkeypatch):
    from approximation.loss_lut import build_lut as real_build_lut

    vendor = "epc2216"
    _skip_without_vendor_lib(vendor)
    _seed_repo(tmp_path, vendor)

    monkeypatch.setattr(tier4, "ROOT", tmp_path)
    monkeypatch.setattr(tier4, "build_lut",
                        lambda v, **kw: real_build_lut(v, simulate=False))
    monkeypatch.setattr(tier4, "resolve_pin_connections",
                        lambda *a, **k: ["drain", "gate", "source"])

    good = {"solver_elapsed_s": 0.4, "wall_clock_s": 1.0,
            "iterations": 16514.0, "trace_points": 4047.0,
            "end_time_s": 1.76e-4, "repeats": 3, "n_pulses": 20}
    calls = []

    def _fake_bench(lib, sub, roles, **kw):
        calls.append((sub, kw))
        return dict(good)

    class _Result:
        log_text = "Total iterations: 100\n"

        def get(self, name):
            assert name == "time"
            return [i * 1.4e-5 / 8000 for i in range(8001)]

    monkeypatch.setattr(tier4, "run_inductive_benchmark", _fake_bench)
    monkeypatch.setattr(tier4, "_run_emitted_dpt",
                        lambda *a, **k: (_Result(), 0.2))

    rec = tier4.run_vendor(vendor, live=True)
    assert len(calls) == 2
    assert all(kw["n_pulses"] == tier4.SPEED_N_PULSES == 20 for _, kw in calls)
    assert all(kw["repeats"] == tier4.SPEED_REPEATS == 3 for _, kw in calls)
    assert all(kw["timeout_s"] == tier4.SPEED_TIMEOUT_S == 90.0 for _, kw in calls)
    assert all(kw["max_step"] == 1e-7 for _, kw in calls)
    assert calls[0][0] == "EPC2216"
    assert calls[1][0] == "epc2216_PIECESPICE_T4"
    assert rec["dpt_run"]["completed"] is True
    assert rec["speedup"]["status"] == "measured"
    assert rec["speedup"]["solver_speedup"] == 1.0
    assert rec["speedup"]["wall_speedup"] == 1.0
    assert rec["speedup"]["end_time_s"] == 1.76e-4
    assert rec["speedup"]["t_last_s"] is None


@pytest.mark.parametrize("status", (
    "measured", "constant_node", "single_node", "interpolated"))
def test_energy_comparison_accepts_every_finite_lut_status(status):
    class _FakeLut:
        source = {"dpt_crossing_evaluable": True}

        def energy_at(self, vbus, current):
            return {"e_on_uj": 1.0, "e_off_uj": 2.0, "status": status,
                    "source": {"node_status": status}}

    rows = tier4._energy_comparison(
        _FakeLut(),
        {"golden_metrics": {"e_on_uj": 1.0, "e_off_uj": 2.0}}, 10.0, 5.0)
    assert rows["e_on_uj"]["status"] == "measured"
    assert rows["e_on_uj"]["abs_error"] == 0.0
    assert rows["e_off_uj"]["source"] == {"node_status": status}


@pytest.mark.parametrize("vendor", VENDORS)
def test_efficiency_gate_has_no_converter_output_power(vendor):
    rec = _record(vendor)
    gate = rec["efficiency_gate"]
    assert rec["efficiency"]["efficiency"] is None
    assert gate["efficiency"] is None
    assert gate["status"] == "output_power_required"
    assert "output power" in gate["reason"].lower()
    assert gate["p_bus_in_w"] == pytest.approx(
        rec["conditions"]["vbus_v"] * rec["conditions"]["i_switch_a"])
    assert "never used as" in gate["p_bus_in_role"]
    assert gate["p_cond_w"] is not None and gate["p_cond_w"] > 0.0
    assert "thermal" in gate["thermal_note"].lower()
    decisions = rec["provenance"]["decisions"]
    assert decisions["efficiency_inputs"]["p_out_w"] is None
    assert decisions["efficiency_inputs"]["p_bus_in_w"] == pytest.approx(
        gate["p_bus_in_w"])


def test_named_constants_and_provenance_disclose_all_decisions():
    rec = _record("csd16412q5a")
    decisions = rec["provenance"]["decisions"]
    constants = decisions["run_constants"]
    assert constants["kind"] == "project_decision"
    values = constants["values"]
    assert values["speed_n_pulses"] == tier4.SPEED_N_PULSES == 20
    assert values["speed_repeats"] == tier4.SPEED_REPEATS == 3
    assert values["speed_timeout_s"] == tier4.SPEED_TIMEOUT_S == 90.0
    assert values["emitted_dpt_timeout_s"] == tier4.EMITTED_DPT_TIMEOUT_S
    assert values["emitted_dpt_t_last_fraction"] == tier4.DPT_T_LAST_FRACTION
    assert values["emitted_dpt_trace_point_fraction"] == tier4.DPT_TRACE_POINT_FRACTION
    assert "DCLAMP" in values["clamp_diode_model"]

    ron = decisions["r_on_drive_voltage"]
    lut_ron = rec["lut"]["source"]["r_on"]
    assert ron["kind"] == "measured"
    assert ron["gate_drive_v"] == lut_ron["gate_drive_v"]
    assert ron["rule"] == lut_ron["gate_drive_rule"]
    assert "NOT dpt.vgs_on" in ron["note"]

    limits = rec["lut"]["data_limitations"]
    assert limits and all(d.get("kind") == "data_limitation" for d in limits)
    text = " ".join(str(d).lower() for d in limits)
    assert "temperature" in text and "node" in text and "offline" in text
    assert decisions["data_limitations"] == limits
    observables = decisions["retired_observables"]
    assert "dv/dt" in observables["inference"] and "ringing" in observables["inference"]
    assert any("miller" in item.lower() for item in observables["documented_by_p_demo"])
    assert "thermal_note" in decisions["efficiency_inputs"]


def test_generation_note_reflects_live_or_offline_mode(tmp_path, monkeypatch):
    from approximation.loss_lut import build_lut as real_build_lut

    vendor = "epc2216"
    _skip_without_vendor_lib(vendor)
    _seed_repo(tmp_path, vendor)

    monkeypatch.setattr(tier4, "ROOT", tmp_path)
    monkeypatch.setattr(tier4, "build_lut",
                        lambda v, **kw: real_build_lut(v, simulate=False))

    def _forbid(*args, **kwargs):
        raise RuntimeError("offline unit test: no simulator")

    monkeypatch.setattr(tier4, "_run_emitted_dpt", _forbid)
    monkeypatch.setattr(tier4, "run_inductive_benchmark", _forbid)
    offline = tier4.run_vendor(vendor, live=False)
    assert offline["generation"]["mode"] == "offline_recorded"
    assert "recorded measured artifacts" in offline["generation"]["note"]
    live = tier4.run_vendor(vendor, live=True)
    assert live["generation"]["mode"] == "live"
    assert "Live generation ran" in live["generation"]["note"]


def test_nonfinite_emitter_values_stay_na_and_never_bench(tmp_path, monkeypatch):
    vendor = "csd16412q5a"
    _seed_repo(tmp_path, vendor)
    monkeypatch.setattr(tier4, "ROOT", tmp_path)

    def _reject(*args, **kwargs):
        raise ValueError("Tier-4 emitter requires a finite vth_v/r_on_ohms")

    def _forbid(*args, **kwargs):
        raise AssertionError("no bench may run without a valid emitted switch")

    monkeypatch.setattr(tier4, "build_tier4_subcircuit", _reject)
    monkeypatch.setattr(tier4, "_run_emitted_dpt", _forbid)
    monkeypatch.setattr(tier4, "run_inductive_benchmark", _forbid)
    rec = tier4.run_vendor(vendor, live=True)
    assert rec["emission"]["status"] == "N/A"
    assert "finite vth_v/r_on_ohms" in rec["emission"]["reason"]
    assert rec["emission"]["finite_device_values"] is False
    assert rec["guards"]["emission_valid"] is False
    assert rec["dpt_run"]["completed"] is None
    assert "emission N/A" in rec["dpt_run"]["reason"]
    assert rec["speedup"]["status"] == "N/A"
    assert "emission N/A" in rec["speedup"]["error"]
    assert not (tmp_path / "reports" / "models" /
                f"{vendor}_PIECESPICE_T4.lib").exists()


def test_emitted_stabilization_reads_c_qoss_and_roff_from_lib():
    text = (
        ".SUBCKT X d g s\n"
        "S_SW d s g s SW_T4\n"
        "C_QOSS d s 4.5301913e-10\n"
        ".MODEL SW_T4 SW(Ron=0.0104433 Roff=1.9089e+11 Vt=1.8823697 Vh=0)\n"
        ".ENDS X\n")
    row = tier4._emitted_stabilization(text)
    assert row["c_qoss_emitted"] is True
    assert row["c_qoss_value_f"] == pytest.approx(4.5301913e-10)
    assert row["c_qoss_line"].startswith("C_QOSS d s ")
    assert row["roff_ohms"] == pytest.approx(1.9089e11)


def test_solver_stabilization_fallback_stays_disclosed():
    class _FakeLut:
        coss_f = float("nan")
        r_off_ohms = float("nan")
        source = {
            "coss_f": {"kind": "missing", "reason": "no curve4 extraction"},
            "r_off_ohms": {
                "kind": "measurement_failed", "value_ohms": float("nan"),
                "fallback": {"source": "project_decision"},
            },
        }

    text = ".MODEL SW_T4 SW(Ron=0.01 Roff=1e+12 Vt=1.5 Vh=0)\n"
    block = tier4._solver_stabilization(_FakeLut(), text, {"status": "emitted"})
    assert block["status"] == "not_applied"
    assert block["c_qoss"]["emitted"] is False
    assert block["c_qoss"]["value_f"] is None
    assert block["r_off"]["value_ohms"] is None
    assert block["r_off"]["source"]["value_ohms"] is None
    assert block["r_off"]["basis"] == "project_decision_fallback"
    assert block["r_off"]["source"]["fallback"]["source"] == "project_decision"
    assert block["r_off"]["emitted_ohms"] == pytest.approx(1e12)
    json.dumps(tier4._json_safe(block), allow_nan=False)


def test_solver_stabilization_provenance_in_offline_record(tmp_path, monkeypatch):
    vendor = "csd16412q5a"
    _seed_repo(tmp_path, vendor)
    monkeypatch.setattr(tier4, "ROOT", tmp_path)
    rec = tier4.run_vendor(vendor, live=False)
    block = rec["solver_stabilization"]
    assert block["status"] == "applied"
    assert block["waveform_gate"] == "retired_by_design"
    assert block["root_cause_reference"] == tier4.SOLVER_STABILIZATION_REF
    assert "convergence" in tier4.SOLVER_STABILIZATION_RATIONALE.lower()

    coss = block["c_qoss"]
    assert coss["emitted"] is True
    coss_src = coss["source"]
    assert coss_src["kind"] == "measured"
    assert coss_src["file"].endswith("curve4_capacitances.yaml")
    assert coss_src["key"] == "extracted.coss_f"
    assert coss_src["ref_voltage_key"] == "extracted.cap_ref_vds"
    assert coss_src["temperature_c"] == pytest.approx(
        rec["conditions"]["temperature_c"])
    assert coss_src["unit"] == "F"
    assert coss["value_f"] == pytest.approx(coss_src["value_f"])
    assert coss["emitted_value_f"] == pytest.approx(coss["value_f"], rel=1e-8)

    roff = block["r_off"]
    roff_src = roff["source"]
    assert roff_src["vgs_v"] == 0.0
    assert roff_src["vbus_v"] == pytest.approx(rec["conditions"]["vbus_v"])
    vendor_file = roff_src["vendor_file"]
    assert vendor_file["file"]
    if vendor_file["available"]:
        assert vendor_file["sha256"]
        assert vendor_file["bytes"] > 0
    else:
        # A CI checkout has no untracked SpiceModels/** vendor input. The
        # offline sweep then records the absence explicitly instead of a hash.
        assert vendor_file["sha256"] is None
        assert vendor_file["bytes"] is None
        assert roff_src["kind"] == "not_measured"
        assert "no live vendor DC sweep" in roff_src["reason"]
    assert roff_src["config_sha256"]
    assert roff_src["fallback"]["source"] == "project_decision"
    # Offline build cannot run the Vgs=0 DC sweep: no fabricated Roff value.
    assert roff["value_ohms"] is None
    assert roff["basis"] == "project_decision_fallback"
    assert roff["emitted_ohms"] is not None
    assert "NOT a voltage-dependent" in block["disclosure"]

    out = (tmp_path / "reports" / "comparison" / vendor /
           "tier4_validation.json")
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["solver_stabilization"]["status"] == "applied"
    json.dumps(written, allow_nan=False)


@pytest.mark.parametrize("vendor", VENDORS)
def test_recorded_solver_stabilization_keeps_disclosures(vendor):
    block = _record(vendor)["solver_stabilization"]
    assert block["status"] == "applied"
    assert block["waveform_gate"] == "retired_by_design"
    assert block["c_qoss"]["emitted"] is True
    assert block["c_qoss"]["source"]["kind"] == "measured"
    assert block["r_off"]["basis"] in ("measured", "project_decision_fallback")
    assert "waveform gate" in block["disclosure"].lower()


def test_every_speedup_row_has_mandatory_method_label():
    """Research G0/gate 1: method is mandatory; Gear stays a separate report."""
    golden, fast = _arm(12000.0, 3000.0), _arm(600.0, 150.0)
    rows = (
        tier4._speedup_row(golden, fast, True, POLICY),
        tier4._speedup_row(golden, dict(fast, error="aborted"), False, POLICY),
    )
    assert tier4.SPEED_METHOD == "LTspice default/trap (both arms)"
    for row in rows:
        assert row["method"] == tier4.SPEED_METHOD
        assert "Gear" not in row["method"]


@pytest.mark.parametrize("vendor", VENDORS)
def test_offline_record_speedup_carries_method_label(tmp_path, monkeypatch,
                                                     vendor):
    _seed_repo(tmp_path, vendor)
    monkeypatch.setattr(tier4, "ROOT", tmp_path)
    rec = tier4.run_vendor(vendor, live=False)
    assert rec["speedup"]["method"] == tier4.SPEED_METHOD
    out = (tmp_path / "reports" / "comparison" / vendor /
           "tier4_validation.json")
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["speedup"]["method"] == "LTspice default/trap (both arms)"


def test_interpolation_wording_is_truthful_and_capability_derived():
    from approximation.loss_lut import (
        build_lut,
        interpolation_provenance_text,
    )
    from models.tier4_ideal_switch_builder import build_tier4_subcircuit

    # A multi-node grid with no full cell must never be called single-node.
    cap = {"status": "grid_interpolation", "finite_nodes": 3,
           "complete_cells": 0, "vbus_points": 2, "current_points": 2}
    text = interpolation_provenance_text(cap)
    assert "3 finite measured node(s)" in text
    assert "no 2x2 interpolation cell has all four corners finite" in text
    assert "off-node queries are NaN" in text
    assert "one finite measured node per vendor" not in text

    for vendor in VENDORS:
        lut = build_lut(vendor, simulate=False)
        basis = next(d["basis"] for d in lut.source["decisions"]
                     if d["name"] == "interpolation")
        assert "one finite measured node per vendor" not in basis
        assert "finite measured node(s)" in basis
        assert "off-node queries are NaN" in basis
        assert lut.grid_provenance() == interpolation_provenance_text(
            lut.interpolation_capability())
        assert lut.interpolation_capability()["complete_cells"] == 0
        emitted = build_tier4_subcircuit(lut, i_load_a=10.0)
        assert "one finite measured node per vendor" not in emitted
        assert "off-node queries are NaN" in emitted
        assert "finite measured node(s)" in emitted


def test_no_template_placeholder_in_stabilization_provenance(tmp_path,
                                                             monkeypatch):
    vendor = "cpm2-1200"
    rationale = tier4.SOLVER_STABILIZATION_RATIONALE
    assert "<curve4.coss_f>" not in rationale
    assert "vendor Curve-4 Coss at cap_ref_Vds" in rationale
    _seed_repo(tmp_path, vendor)
    monkeypatch.setattr(tier4, "ROOT", tmp_path)
    rec = tier4.run_vendor(vendor, live=False)
    assert "<curve4.coss_f>" not in rec["solver_stabilization"]["rationale"]
    out = (tmp_path / "reports" / "comparison" / vendor /
           "tier4_validation.json")
    payload = out.read_text(encoding="utf-8")
    assert "<curve4.coss_f>" not in payload
    assert "vendor Curve-4 Coss at cap_ref_Vds" in payload


def test_vendor_scope_notes_live_at_record_root(tmp_path, monkeypatch):
    vendor = "cpm2-1200"
    assert "record root" in tier4.GOLDEN_SPEED_EXCLUSIONS[vendor]
    _seed_repo(tmp_path, vendor)
    monkeypatch.setattr(tier4, "ROOT", tmp_path)
    rec = tier4.run_vendor(vendor, live=False)
    assert "vendor_scope_notes" in rec
    assert "vendor_scope_notes" not in rec["provenance"]
    notes = rec["vendor_scope_notes"]
    assert notes["defect"] == "VM-10"
    assert "VM-10 vendor turn-off abort" in notes["note"]
    assert "long_adaptive" in notes["artifact_conflict"]
    assert "completed" in notes["artifact_conflict"]
    speed = rec["speedup"]
    assert speed["status"] == "N/A"
    assert speed["solver_speedup"] is None
    out = (tmp_path / "reports" / "comparison" / vendor /
           "tier4_validation.json")
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["vendor_scope_notes"]["defect"] == "VM-10"
    assert "vendor_scope_notes" not in written["provenance"]
