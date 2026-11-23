"""Offline tests for the T4 unified one-ROM registry, record and CLI.

Structural integers/times below are *probes* used to exercise control flow;
they are not PieceSpice measurements.  Every comparison bar and bench constant
is asserted against the pinned values in ``local://unified_rom_spec.md``
(spec sections 1.3, 1.4, 4.4, 4.6 and 6).  No simulator is started: an autouse
guard fails if a live bench runner is reached, so this module is safe with
``PIECESPICE_LIVE=0``.
"""
from __future__ import annotations
import copy
import dataclasses
import hashlib
import inspect
import json
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from cli import main as cli_main
from cli.main import app
from common.acceptance import DEFAULT_PATH as ACCEPTANCE_CONFIG_PATH
from common.acceptance import load_acceptance
from common.types import PWLModelSpec
from pipeline import registry, unified_rom as urom
_REAL_BUILD_EMITTED = urom._build_emitted
_RUN_DPT_ARM_SOURCE = inspect.getsource(urom._run_dpt_arm)
VENDOR = "irfs7730pbf"
_runner = CliRunner()
GATE_NAMES = ("G0", "G1_static", "G2_derivative", "G3_dpt_waveform",
              "G4_timing_energy", "G5_speed", "G6_charge")
SPLIT_VERDICTS = ("fidelity_verdict", "timing_verdict", "speed_verdict",
                  "overall_verdict")


@pytest.fixture(autouse=True)
def forbid_live_execution(monkeypatch):
    """Any accidental bench/simulator call fails the test that made it."""
    def fail(*args, **kwargs):
        raise AssertionError("live execution is forbidden in offline T4 tests")

    for name in ("SpiceEngine", "_build_emitted", "_run_interleaved_speed",
                 "_run_dpt_arm", "_static_bench", "capture_trajectory"):
        if name == "SpiceEngine":
            monkeypatch.setattr(urom.SpiceEngine, "run_transient", fail)
        else:
            monkeypatch.setattr(urom, name, fail)


def _flat(output: str) -> str:
    """Collapse Rich wrapping/padding for stable substring assertions."""
    return " ".join(output.split())


# ---------------------------------------------------------------------------
# synthetic, explicitly non-measured control-flow fixtures
# ---------------------------------------------------------------------------
def _arm(solver: float, wall: float, *, complete: bool = True,
         repeats: int = 3) -> dict:
    # Per-repeat clocks are paired by schedule index.  The multiplicative
    # probe keeps each golden/ROM ratio equal to the nominal point ratio, so
    # a bootstrap CI over the paired ratios is non-degenerate in shape only
    # (never a PieceSpice measurement).
    per_repeat = []
    jitter = (1.0, 1.02, 0.98)
    for i in range(repeats):
        scale = jitter[i % len(jitter)]
        per_repeat.append({
            "iterations": 100, "points": 1000, "t_last": 1.0,
            "solver_time_s": solver * scale,
            "wall_time_s": wall * scale,
        })
    return {
        "iterations": 100, "points": 1000, "t_last": 1.0,
        "solver_time_s": solver, "wall_time_s": wall,
        "solver_min_s": solver * 0.95, "solver_max_s": solver * 1.05,
        "wall_min_s": wall * 0.95, "wall_max_s": wall * 1.05,
        "completion": complete,
        "per_repeat": per_repeat,
    }

def _synthetic_spec(**fields) -> PWLModelSpec:
    """Structurally valid, explicitly NON-measured spec for pure gate tests.

    Channel arrays are synthetic placeholders; charge tables and scalars under
    test are supplied per call.  No device-specific number is inferred from
    these values.
    """
    base = {
        "device_name": VENDOR,
        "breakpoints_vds": [0.0, 1.0, 2.0],
        "breakpoints_id": [[0.0, 0.0, 0.0]],
        "vgs_values": [0.0],
        "rds_on": 1.0,   # synthetic ohm, not a device measurement
        "v_th": 3.0,     # synthetic volts, not a device measurement
    }
    base.update(fields)
    return PWLModelSpec(**base)


def _observations(method: str = "trap/default") -> dict:
    """Complete observations sitting exactly on the pinned spec bars."""
    vds_bar, id_bar = urom.DPT_BARS_PCT[method][VENDOR]
    wf = {"vds_rel_rmse_pct": vds_bar, "id_rel_rmse_pct": id_bar,
          "compare_points": 1000, "compare_grid_step_s": 0.01,
          "status": "measured"}
    obs = {
        "equal_sandbox": {
            "deck_diff_only_dut": True,
            "effective_options_dumped": True,
            "tolerance_overrides": [],
            "libraries_inlined_both_arms": True,
            "temperature_both_arms": True,
        },
        "speed": {
            "repeats": urom.SPEED_REPEATS,
            "n_pulses": urom.SPEED_N_PULSES,
            "golden": _arm(2.0, 2.2),
            "urom": _arm(1.0, 1.1),
        },
        "dpt": {
            "golden": {"completed": True, "waveform": dict(wf)},
            "urom": {"completed": True, "waveform": dict(wf)},
        },
        "static": {
            "full_sweep_rel_rmse_pct": urom.STATIC_BARS_PCT[VENDOR],
            "worst_endpoint_rel_error_pct": urom.ENDPOINT_BAR_PCT,
            "span_normalised_rmse_pct": 1.0,
            "peak_normalised_rmse_pct": 1.0,
        },
        "derivative": {"pass": True, "status": "measured"},
        "timing_energy": {
            "pass": True, "status": "measured", "rows": {},
            "bar_source": urom.GOLDEN_DPT_BAR_SOURCE,
            "tier2_regression": {
                "status": "measured", "rows": {},
                "all_within_band": True, "used_for_verdict": False,
                "diagnostic_only": True,
            },
        },
    }
    obs["clock_spread"] = urom._clock_spread_record(obs["speed"])
    obs["equal_sandbox"]["clock_spread"] = obs["clock_spread"]
    obs["dpt_compare"] = {
        "status": "measured", "guard_ok": True, "window_equal": True,
        "window_s": [0.0, 1.0], "compare_grid_step_s": 0.01,
        "compare_points": 1000, "min_compare_points": 500,
        "endpoints": {"reference": {"t_start": 0.0, "t_end": 1.0},
                      "golden": {"t_start": 0.0, "t_end": 1.0},
                      "urom": {"t_start": 0.0, "t_end": 1.0}},
        "reference": {"completed": True, "points": 1000,
                      "min_points": 500, "reasons": []},
        "reason": "common grid over the equal overlapping window",
    }
    return obs


def _record(vendor: str = VENDOR, method: str = "trap/default") -> dict:
    tables = {"q_oss": True, "q_iss": True, "q_rss": True}
    return {
        "schema": urom.SCHEMA,
        "vendor": vendor,
        "method": method,
        "c1": {
            "status": "measured", "monotone_q": True,
            "monotone_q_by_table": dict(tables),
            "all_charge_monotone": True,
            "q_zero_anchored": True,
            "q_zero_anchored_by_table": {k: True for k in tables},
            "all_charge_zero_anchored": True,
            "applicable_tables": sorted(tables),
            "not_applicable_tables": [],
            "charge_fit_error_pct": 0.0, "q_zero": 0.0, "q_max": 1.0,
            "breakpoint_crossings": {"q_oss": 0, "q_iss": 0, "q_rss": 0},
            "c_ds_scaling_excluded": True,
        },
        "tuning": {"mode": "off", "status": "not_run",
                   "verdict": "not_run"},
        "speed_adaptive_20": {
            "n_pulses": urom.SPEED_N_PULSES, "max_step": None,
            "method_label": method,
            "solver_speedup": 2.0,
            "wall_speedup": 2.0,
            "solver_speedup_ci95": {
                "point": 2.0, "low": 2.0, "high": 2.0, "n": 3,
                "confidence": urom.SPEED_CI_CONFIDENCE,
                "n_resamples": urom.SPEED_CI_RESAMPLES},
            "wall_speedup_ci95": {
                "point": 2.0, "low": 2.0, "high": 2.0, "n": 3,
                "confidence": urom.SPEED_CI_CONFIDENCE,
                "n_resamples": urom.SPEED_CI_RESAMPLES},
            "effect_min_solver_speedup": urom.SPEED_EFFECT_MIN,
            "t_last": {"golden": 1.0, "urom": 1.0},
            "iterations": {"golden": 100, "urom": 100},
            "points": {"golden": 1000, "urom": 1000},
            "global_claim": False, "min_default_gear": None,
        },
    }


def _gates(obs: dict | None = None, rec: dict | None = None) -> dict:
    # Bypass the autouse guard for the pure gate function under test.
    from pipeline.unified_rom import _evaluate_gates
    return _evaluate_gates(rec or _record(), obs or _observations())


def _cli_record() -> dict:
    """CLI display fixture: N/A speed row and one incomplete arm."""
    rec = _record()
    rec["variant"] = registry.UNIFIED_VARIANT
    rec["channel_basis"] = "ann"
    rec["verdict"] = "reject"
    rec["fidelity_verdict"] = "accept"
    rec["timing_verdict"] = "reject"
    rec["speed_verdict"] = "reject"
    rec["overall_verdict"] = "reject"
    rec["gates"] = {name: {"pass": name != "G5_speed", "status": "measured"}
                    for name in GATE_NAMES}
    rec["gates"]["G4_tier2_regression"] = {
        "pass": False, "status": "measured", "diagnostic_only": True}
    rec["gates"]["all_pass"] = False
    rec["gates"]["fidelity_all_pass"] = True
    rec["gates"]["timing_all_pass"] = False
    rec["gates"]["speed_all_pass"] = False
    rec["speed_adaptive_20"]["solver_speedup"] = None
    rec["speed_adaptive_20"]["wall_speedup"] = None
    rec["speed_adaptive_20"]["solver_speedup_ci95"] = None
    rec["speed_adaptive_20"]["wall_speedup_ci95"] = None
    rec["emitted_lib"] = (
        f"reports/models/{VENDOR}_PIECSPICE_UROM.lib")
    rec["runtime"] = {
        "iterations": {"golden": 100, "urom": 100},
        "points": {"golden": 1000, "urom": 1000},
        "t_last": {"golden": 1.0, "urom": 1.0},
        "repeats": urom.SPEED_REPEATS,
        "completion": {"golden": True, "urom": False},
    }
    return rec


# ---------------------------------------------------------------------------
# registry variant + library path
# ---------------------------------------------------------------------------
def test_urom_is_a_registered_tier3_variant_without_changing_tiers():
    assert registry.DEFAULT_VARIANT == "v1"
    assert registry.UNIFIED_VARIANT == "urom"
    assert registry.TIERS == (2, 3, 4)
    for vendor in registry.VENDORS:
        variants = registry.variants_for(vendor)
        assert variants[0] == registry.DEFAULT_VARIANT
        assert "urom" in variants
    assert registry.variants_for("not-a-vendor") == (registry.DEFAULT_VARIANT,)


def test_unified_lib_name_and_lib_path_are_variant_free():
    expected = f"{VENDOR}_PIECSPICE_UROM.lib"
    assert registry.unified_lib_name(VENDOR) == expected
    assert registry.lib_path(VENDOR, 3, "urom") == (
        registry.ROOT / "reports" / "models" / expected)
    # Existing tiers/variants keep their names; pin the registry's own
    # spellings verbatim (T2/UROM and T3/T4 differ in registry.py).
    assert registry.lib_path(VENDOR, 2).name == f"{VENDOR}_PIECESPICE_T2.lib"
    assert registry.lib_path(VENDOR, 3).name == f"{VENDOR}_PIECESPICE_T3.lib"
    assert registry.lib_path(VENDOR, 3, "simplis").name == (
        f"{VENDOR}_PIECESPICE_T3_simplis.lib")
    assert registry.lib_path(VENDOR, 4).name == f"{VENDOR}_PIECESPICE_T4.lib"


def test_unified_validation_path_matches_record_writer():
    assert registry.unified_validation_path(VENDOR) == urom._record_path(VENDOR)
    assert registry.unified_validation_path(VENDOR) == (
        registry.ROOT / "reports" / "comparison" / VENDOR
        / "unified_validation.json")


def test_registry_fit_and_validate_route_urom_to_run_unified(monkeypatch):
    calls: list = []
    record = _cli_record()
    stub = ModuleType("pipeline.unified_rom")

    def run_unified(vendor: str, **kwargs) -> dict:
        calls.append((vendor, kwargs))
        return record

    stub.run_unified = run_unified
    monkeypatch.setitem(sys.modules, "pipeline.unified_rom", stub)
    assert registry.fit(VENDOR, 3, "urom") == Path(record["emitted_lib"])
    assert registry.validate(VENDOR, 3, "urom") is record
    assert calls == [(VENDOR, {"variant": "urom", "emit": "tier2"})] * 2


def test_registry_routes_explicit_ann_basis(monkeypatch):
    calls: list = []
    record = _cli_record()
    stub = ModuleType("pipeline.unified_rom")

    def run_unified(vendor: str, **kwargs) -> dict:
        calls.append((vendor, kwargs))
        return record

    stub.run_unified = run_unified
    monkeypatch.setitem(sys.modules, "pipeline.unified_rom", stub)
    assert registry.fit(VENDOR, 3, "urom", basis="ann") == Path(
        record["emitted_lib"])
    assert registry.validate(VENDOR, 3, "urom", basis="ann") is record
    assert calls == [(VENDOR, {"variant": "urom", "emit": "ann"})] * 2
    for fn in (registry.fit, registry.validate):
        assert inspect.signature(fn).parameters["basis"].default == "tier2"


def test_default_configuration_uses_canonical_paths_and_ann_is_keyed(
        monkeypatch, tmp_path):
    monkeypatch.setattr(urom.registry, "ROOT", tmp_path)
    kw = dict(emit="tier2", method="trap/default", tuning="off",
              c1_variant=None)
    canonical_rec = urom._record_path_for_config(VENDOR, **kw)
    canonical_lib = urom._lib_path_for_config(VENDOR, **kw)
    assert canonical_rec == registry.unified_validation_path(VENDOR)
    assert canonical_lib.name == f"{VENDOR}_PIECSPICE_UROM.lib"
    ann_kw = dict(kw, emit="ann")
    ann_rec = urom._record_path_for_config(VENDOR, **ann_kw)
    ann_lib = urom._lib_path_for_config(VENDOR, **ann_kw)
    assert ann_rec != canonical_rec and ann_lib != canonical_lib
    assert "ann" in ann_rec.name and "ann" in ann_lib.name
    assert urom._is_canonical_config(**kw) is True
    assert urom._is_canonical_config(**ann_kw) is False


# ---------------------------------------------------------------------------
# keyed ANN emission identity (offline, synthetic fits -- no bench, no record)
# ---------------------------------------------------------------------------
def _synthetic_ann_fits():
    """Structurally valid, explicitly NON-measured ANN fits for emitter tests."""
    from approximation.ann_channel import AnnChannelFit
    from approximation.ann_charge import AnnChargeFit

    channel = AnnChannelFit(
        coefficients={
            "w1": [[0.1, 0.01]], "b1": [0.0], "w2": [0.5], "b2": 0.0,
            "vgs_center": 5.0, "vgs_scale": 2.0,
            "vds_center": 5.0, "vds_scale": 5.0,
            "alpha": 1.0, "guard_width": 0.25,
        },
        v_th=3.0, v_sat=0.2, activation="isru", neurons=1,
        loss_weights={}, train_range={}, out_of_range_check="pass",
        diagnostics={},
    )
    tables = {
        name: {"v": [0.0, 1.0, 2.0], "q": [0.0, 1e-9, 3e-9]}
        for name in ("oss", "iss", "rss")
    }
    charge = AnnChargeFit(tables, None, "isru", 1, {}, {})
    return channel, charge


def _synthetic_ann_spec():
    """Spec carrying a body-diode table so the D_BD model suffix is emitted."""
    return _synthetic_spec(
        q_oss_vs_v=[0.0, 1.0, 2.0], q_oss_charge_c=[0.0, 1e-9, 3e-9],
        body_diode_vsd=[0.5, 0.8], body_diode_isd=[1e-6, 1e-3])


def test_ann_builder_name_override_matches_tier2_convention():
    from models.ann_netlist_builder import build_ann_subcircuit

    spec = _synthetic_ann_spec()
    channel, charge = _synthetic_ann_fits()
    canonical = f"{VENDOR}_PIECSPICE_UROM"
    keyed = f"{VENDOR}_PIECSPICE_UROM_ann_trap-default"

    base = build_ann_subcircuit(spec, channel, charge)
    assert f".SUBCKT {canonical} d g s" in base
    assert f".MODEL D_BD_{canonical} " in base
    assert f".ENDS {canonical}" in base

    by_name = build_ann_subcircuit(spec, channel, charge, name=keyed)
    assert by_name == base.replace(canonical, keyed)
    # The keyed name extends the canonical name, so substring checks such as
    # ".ENDS <canonical>" in ".ENDS <keyed>" always give a false result.
    keyed_lines = by_name.splitlines()
    assert f".SUBCKT {keyed} d g s" in keyed_lines
    assert f".SUBCKT {canonical} d g s" not in keyed_lines
    assert any(line.startswith(f".MODEL D_BD_{keyed} ")
               for line in keyed_lines)
    assert not any(line.startswith(f".MODEL D_BD_{canonical} ")
                   for line in keyed_lines)
    assert f".ENDS {keyed}" in keyed_lines
    assert f".ENDS {canonical}" not in keyed_lines

    # subckt_name wins when both overrides are supplied (Tier-2 convention).
    by_subckt = build_ann_subcircuit(
        spec, channel, charge, name="ignored", subckt_name=keyed)
    assert by_subckt == by_name


def test_keyed_ann_emission_identity_roundtrip_without_clobber(
        monkeypatch, tmp_path):
    from pipeline import ann_fit as ann_fit_mod

    channel, charge = _synthetic_ann_fits()
    spec = _synthetic_ann_spec()
    ann_record = {"channel_fit": channel, "charge_fit": charge,
                  "synthetic": True}

    monkeypatch.setattr(urom.registry, "ROOT", tmp_path)
    monkeypatch.setattr(urom, "_pwl_spec", lambda vendor: spec)
    monkeypatch.setattr(urom, "_theta_bounds", lambda s, v: {})
    monkeypatch.setattr(urom, "_normalize_unified_metadata",
                        lambda meta, vendor: None)
    monkeypatch.setattr(ann_fit_mod, "ann_fit_for_vendor",
                        lambda vendor: ann_record)

    canonical_kw = dict(emit="tier2", method="trap/default", tuning="off",
                        c1_variant=None)
    canonical_lib = urom._lib_path_for_config(VENDOR, **canonical_kw)
    canonical_rec = urom._record_path_for_config(VENDOR, **canonical_kw)
    assert canonical_lib.name == f"{VENDOR}_PIECSPICE_UROM.lib"
    assert canonical_rec.name == "unified_validation.json"
    sentinel = {"sentinel": "canonical-c0-untouched"}
    canonical_rec.parent.mkdir(parents=True, exist_ok=True)
    canonical_rec.write_text(json.dumps(sentinel), encoding="utf-8")
    before = canonical_rec.read_bytes()

    plan = {
        "method_label": "trap/default",
        "tuning_conditions": [
            {"name": "synthetic", "completed": False, "condition": {}}],
    }
    lib, _spec, recorded_ann, tuning = _REAL_BUILD_EMITTED(
        VENDOR, emit="ann", tuning="off", plan=plan)

    key = urom._config_key(
        emit="ann", method="trap/default", tuning="off", c1_variant=None)
    assert key == "ann_trap-default"
    assert lib.name == f"{VENDOR}_PIECSPICE_UROM_{key}.lib"
    assert lib != canonical_lib
    text = lib.read_text(encoding="ascii")
    assert urom._subckt_name_text(text) == lib.stem
    assert f".MODEL D_BD_{lib.stem} " in text
    assert f".ENDS {lib.stem}" in text
    assert urom._identity_header_hint(text) == (
        "ann", "trap/default", "off", "C0")
    assert recorded_ann is ann_record
    assert tuning["verdict"] == "not_run"

    # A second keyed emission must not write the canonical paths either.
    lib2, *_ = _REAL_BUILD_EMITTED(
        VENDOR, emit="ann", tuning="off", plan=plan)
    assert lib2 == lib
    assert not canonical_lib.exists()
    assert canonical_rec.read_bytes() == before
    assert json.loads(canonical_rec.read_text(encoding="utf-8")) == sentinel

    # Identity helper round-trips on the emitted keyed ANN artifact.
    record = {
        "emit": "ann", "channel_basis": "ann", "tuning_mode": "off",
        "c1_variant": None, "method": "trap/default",
        "record_path": str(urom._record_path_for_config(
            VENDOR, emit="ann", method="trap/default", tuning="off",
            c1_variant=None)),
        # Relative emitted_lib resolves against urom.ROOT, not the patched
        # registry.ROOT; this synthetic record uses an absolute path.
        "emitted_lib": str(lib),
        "emitted_lib_sha256": hashlib.sha256(lib.read_bytes()).hexdigest(),
    }
    selected = dict(emit="ann", method="trap/default", tuning="off",
                    c1_variant=None)
    ok = urom._verify_artifact_identity(record, **selected)
    assert ok["status"] == "match" and ok["match"] is True
    assert ok["actual_sha256"] == record["emitted_lib_sha256"]

    lib.write_text(text + "* tampered\n", encoding="ascii")
    bad = urom._verify_artifact_identity(record, **selected)
    assert bad["status"] == "mismatch"
    assert any("sha256" in reason for reason in bad["reasons"])

# ---------------------------------------------------------------------------
# run_unified signature / argument validation (no bench entered)
# ---------------------------------------------------------------------------
def test_run_unified_signature_is_keyword_only_with_documented_defaults():
    sig = inspect.signature(urom.run_unified)
    assert list(sig.parameters) == [
        "vendor", "variant", "method", "emit", "c1_variant", "tuning",
        "repeats"]
    params = sig.parameters
    assert params["vendor"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("variant", "method", "emit", "c1_variant", "tuning",
                 "repeats"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["variant"].default == registry.UNIFIED_VARIANT
    assert params["method"].default == "trap/default"
    assert params["emit"].default == "tier2"  # canonical C0, not ANN candidate
    assert params["c1_variant"].default is None
    assert params["tuning"].default == "off"
    assert params["repeats"].default == urom.SPEED_REPEATS == 3
    assert urom.METHODS == ("trap/default", "gear-both")
    assert urom.EMIT_MODES == ("ann", "hybrid", "tier2")
    assert urom.TUNING_MODES == ("on", "off")


def test_schema_and_provenance_pin_spec_invariants():
    assert urom.SCHEMA == "piecespice.unified/1.0"
    for key in ("static_bars_pct", "dpt_bars_pct", "timing_energy_bars",
                "speed_n_pulses", "speed_repeats", "endpoint_bar_pct",
                "charge_budget_pct"):
        assert key in urom.PROVENANCE
    assert urom.PROVENANCE["external_sources"]["status"] == (
        "conditional_until_url_and_section_pinned")
    assert urom.STATIC_BARS_PCT == {
        "irfs7730pbf": 7.0683, "csd16412q5a": 8.4935,
        "epc2216": 2.0394, "cpm2-1200": 3.4287}


def test_record_provenance_hashes_environment_and_inputs(monkeypatch,
                                                        tmp_path):
    from common import provenance as prov

    assert urom.environment_metadata is prov.environment_metadata
    assert urom.file_hashes is prov.file_hashes
    present = tmp_path / "present.yaml"
    present.write_text("x: 1\n", encoding="utf-8")
    hashes = prov.file_hashes({
        "present": present,
        "missing": tmp_path / "missing.yaml",
    })
    assert len(hashes["present"]) == 64
    assert hashes["missing"] is None

    source = inspect.getsource(urom.run_unified)
    assert '"environment": environment_metadata()' in source
    assert "file_hashes(input_paths)" in source
    assert '"input_hashes": input_hashes' in source
    assert '"record_inputs": list(input_paths)' in source
    # The required input set is named explicitly, including every curve.
    assert 'f"config/inputs/{vendor}.yaml"' in source
    assert '"vendor_library:' in source
    assert 'glob("curve*.yaml")' in source
    assert '"emitted_lib": lib' in source
    for key in ("record_inputs", "speed_effect_min", "speed_speedup_ci",
                "derivative_sign_gate", "charge_monotonicity",
                "tier2_regression_gate"):
        assert key in urom.PROVENANCE


def test_record_contract_declares_current_definition_and_speed_ci():
    source = inspect.getsource(urom.run_unified)
    assert '"current_definition": compare.get("current_definition")' in source
    assert '"current_definition_by_arm"' in source
    assert 'record["speed_adaptive_20"]' not in source  # built via _speed_row
    row_source = inspect.getsource(urom._speed_row)
    assert '"solver_speedup_ci95"' in row_source
    assert '"wall_speedup_ci95"' in row_source
    assert "ACCEPTANCE_POLICY.speed_min_solver_speedup" in row_source


def test_dpt_bars_provenance_names_the_tier2_control_row():
    prov = urom.PROVENANCE["dpt_bars_pct"]
    src = prov["source"]
    assert "tier3_validation_simplis.json#dpt_run.waveform" in src
    assert "vds_rel_rmse_pct" in src and "id_rel_rmse_pct" in src
    assert "channel_candidate_selected=tier2" in src
    assert "method label" in src and "EPC 100 ns cap on both arms" in src
    assert prov["waveform_columns"] == "vds_rel_rmse_pct, id_rel_rmse_pct"
    assert prov["channel_candidate"] == (
        "algorithm.channel_candidate_selected=tier2")
    assert prov["kind"] == "measured"


def test_bench_plan_pins_pulses_repeats_method_and_epc_cap():
    for vendor in (VENDOR, "epc2216"):
        plan = urom._bench_plan(vendor, "trap/default", 3)
        assert plan["method_label"] == "trap/default"
        assert plan["options_lines"] == []
        assert plan["repeats"] == 3
        assert plan["speed"]["n_pulses"] == 20
        assert plan["speed"]["repeats"] == 3
        assert plan["speed"]["max_step"] == urom.ADAPTIVE_MAX_STEP.get(vendor)
    assert urom.ADAPTIVE_MAX_STEP["epc2216"] == 1.0e-7
    gear = urom._bench_plan(VENDOR, "gear-both", 3)
    assert gear["options_lines"] == [".options method=gear"]
    assert gear["speed"]["n_pulses"] == 20


@pytest.mark.parametrize(("overrides", "needle"), [
    ({"vendor": "not-a-vendor"}, "unknown vendor"),
    ({"variant": "v1"}, "unsupported unified variant"),
    ({"method": "gear"}, "unknown method"),
    ({"emit": "c0"}, "unknown emit mode"),
    ({"tuning": "maybe"}, "unknown tuning mode"),
    ({"c1_variant": "c1"}, "unknown C1 variant"),
    ({"emit": "ann", "c1_variant": "tier2-c1"}, "requires emit"),
    ({"repeats": 2}, "repeats must be at least 3"),
])
def test_run_unified_rejects_bad_options_before_any_bench(overrides, needle):
    kwargs = {"vendor": VENDOR, **overrides}
    with pytest.raises(ValueError, match=needle):
        urom.run_unified(**kwargs)


# ---------------------------------------------------------------------------
# compare_unified reads recorded JSON only
# ---------------------------------------------------------------------------
def test_compare_unified_reads_record_without_recomputing(monkeypatch, tmp_path):
    path = tmp_path / "unified_validation.json"
    payload = _cli_record()
    payload["schema"] = urom.SCHEMA
    payload["sentinel_read_from_disk"] = "recorded-only"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(urom, "_record_path", lambda vendor: path)
    def recompute(*args, **kwargs):
        raise AssertionError("compare_unified recomputed a number")

    monkeypatch.setattr(urom, "run_unified", recompute)

    data = urom.compare_unified(VENDOR)
    assert data["present"] is True
    assert data["sentinel_read_from_disk"] == "recorded-only"
    assert data["artifact_identity"]["status"] == "unverifiable"


def test_lib_identity_header_names_basis_method_tuning_and_roundtrips():
    header = urom._identity_header(
        emit="ann", method="gear-both", tuning="on", c1_variant=None)
    assert header.isascii()
    assert "basis=ann" in header and "method=gear-both" in header
    assert "tuning=on" in header and "c1_variant=C0" in header
    assert urom._identity_header_hint(header) == (
        "ann", "gear-both", "on", "C0")
    assert urom._strip_identity_header(header + "BODY\n") == "BODY\n"


def test_artifact_identity_detects_basis_mismatch_and_disk_tampering():
    header = urom._identity_header(
        emit="tier2", method="trap/default", tuning="off", c1_variant=None)
    lib_text = header + ".SUBCKT X d g s\n.ENDS\n"
    record = {
        "emit": "tier2", "channel_basis": "tier2", "tuning_mode": "off",
        "c1_variant": None, "method": "trap/default",
    }
    # A real temp lib is needed because identity verification hashes on disk.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        lib = Path(tmp) / "urom.lib"
        record["emitted_lib"] = str(lib)
        lib.write_text(lib_text, encoding="ascii")
        # Hash the bytes actually written: write_text may translate newlines.
        record["emitted_lib_sha256"] = hashlib.sha256(
            lib.read_bytes()).hexdigest()
        ok = urom._verify_artifact_identity(
            record, emit="tier2", method="trap/default", tuning="off",
            c1_variant=None)
        assert ok["status"] == "match" and ok["match"] is True
        wrong = urom._verify_artifact_identity(
            record, emit="ann", method="trap/default", tuning="off",
            c1_variant=None)
        assert wrong["status"] == "mismatch" and wrong["match"] is False
        assert any("emit" in reason for reason in wrong["reasons"])
        lib.write_text(lib_text + "* tampered\n", encoding="ascii")
        tampered = urom._verify_artifact_identity(
            record, emit="tier2", method="trap/default", tuning="off",
            c1_variant=None)
        assert tampered["status"] == "mismatch"
        assert any("sha256" in reason for reason in tampered["reasons"])


def test_compare_unified_reads_selected_keyed_record_and_flags_bad_lib(
        monkeypatch, tmp_path):
    rec_dir = tmp_path / "reports" / "comparison" / VENDOR
    rec_dir.mkdir(parents=True)
    lib = (tmp_path / "reports" / "models"
           / f"{VENDOR}_PIECSPICE_UROM_ann_trap-default.lib")
    lib.parent.mkdir(parents=True, exist_ok=True)
    header = urom._identity_header(
        emit="ann", method="trap/default", tuning="off", c1_variant=None)
    lib.write_text(header + ".SUBCKT X d g s\n.ENDS\n", encoding="ascii")
    payload = _cli_record()
    payload.update({
        "emit": "ann", "channel_basis": "ann", "tuning_mode": "off",
        "c1_variant": None, "method": "trap/default",
        "emitted_lib": str(lib),
        "emitted_lib_sha256": hashlib.sha256(lib.read_bytes()).hexdigest(),
    })
    (rec_dir / "unified_ann_trap-default.json").write_text(
        json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(urom.registry, "ROOT", tmp_path)
    selected = urom.compare_unified(VENDOR, emit="ann")
    assert selected["present"] is True
    assert selected["artifact_identity"]["status"] == "match"
    # The canonical C0 record was not touched by selecting ANN.
    assert urom.compare_unified(VENDOR)["present"] is False
    lib.write_text(header + ".SUBCKT X d g s\n.ENDS\n* tamper\n",
                   encoding="ascii")
    bad = urom.compare_unified(VENDOR, emit="ann")
    assert bad["artifact_identity"]["status"] == "mismatch"


def test_compare_unified_missing_record_is_explicitly_absent(monkeypatch,
                                                             tmp_path):
    monkeypatch.setattr(
        urom, "_record_path", lambda vendor: tmp_path / "absent.json")
    assert urom.compare_unified(VENDOR) == {
        "present": False, "schema": urom.SCHEMA, "vendor": VENDOR}


def test_compare_unified_rejects_unknown_vendor():
    with pytest.raises(ValueError, match="unknown vendor"):
        urom.compare_unified("not-a-vendor")


# ---------------------------------------------------------------------------
# equal-sandbox deck identity (text only, no transient)
# ---------------------------------------------------------------------------
def _deck(include: str = "/tmp/golden.lib", inst: str = "XGOLD",
          *, tran: str = ".tran 0 1u 0 2n", options: str = ".options method=trap",
          extra: str = "") -> str:
    return "\n".join([
        f".include {include}",
        ".param Vbus=13.5",
        ".temp 25",
        tran,
        options,
        ".measure tran tr param=1n",
        extra,
        f"{inst} drain gate source 0",
        ".end",
    ]) + "\n"


def test_deck_identity_accepts_only_dut_include_and_instance_differences():
    golden = _deck()
    rom = _deck(include="/tmp/urom.lib", inst="XUROM")
    result = urom._deck_identity(golden, rom, [".options method=trap"])
    assert result["deck_diff_only_dut"] is True
    assert result["effective_options_dumped"] is True
    assert result["effective_options"] == {
        "golden": [".options method=trap"],
        "rom": [".options method=trap"],
        "declared": [".options method=trap"],
    }
    assert result["tolerance_overrides"] == []
    assert result["libraries_inlined_both_arms"] is True


@pytest.mark.parametrize("rom_kwargs", [
    {"tran": ".tran 0 2u 0 2n"},
    {"options": ""},
    {"extra": ".param Vbus=20"},
    {"extra": ".temp 100"},
    {"extra": ".measure tran tf param=2n"},
    {"extra": "Xaux drain gate source 0"},
])
def test_deck_identity_rejects_any_non_dut_difference(rom_kwargs):
    golden = _deck()
    rom = _deck(include="/tmp/urom.lib", inst="XUROM", **rom_kwargs)
    assert urom._deck_identity(golden, rom, [])["deck_diff_only_dut"] is False


@pytest.mark.parametrize("card", ["reltol", "abstol", "vntol", "trtol", "gmin"])
def test_deck_identity_flags_tolerance_override_on_either_arm(card):
    override = f".options {card}=1e-5"
    golden = _deck()
    rom = _deck(include="/tmp/urom.lib", inst="XUROM", options=override)
    result = urom._deck_identity(golden, rom, [])
    assert any(card in item for item in result["tolerance_overrides"])
    # A mismatch in solver options is itself a deck difference.
    assert result["deck_diff_only_dut"] is False
    both = urom._deck_identity(rom, rom.replace("XUROM", "XGOLD"), [])
    assert both["deck_diff_only_dut"] is True
    assert any(card in item for item in both["tolerance_overrides"])


@pytest.mark.parametrize("builder", ["_dpt_deck", "_speed_deck"])
@pytest.mark.parametrize("method", urom.METHODS)
def test_generated_decks_are_identical_except_dut(builder, method):
    plan = urom._bench_plan(VENDOR, method, 3)
    pins = ["drain", "gate", "source"]
    golden = {"library_path": "/tmp/golden.lib", "subcircuit_name": "GOLD",
              "pin_order": pins}
    rom = {"library_path": "/tmp/urom.lib", "subcircuit_name": "UROM",
           "pin_order": pins}
    decks = {"golden": getattr(urom, builder)(plan, golden),
             "urom": getattr(urom, builder)(plan, rom)}
    result = urom._deck_identity(decks["golden"], decks["urom"],
                                 plan["options_lines"])
    assert result["deck_diff_only_dut"] is True
    assert result["tolerance_overrides"] == []
    for deck in decks.values():
        if method == "gear-both":
            assert ".options method=gear" in deck
        else:
            assert ".options" not in deck


# ---------------------------------------------------------------------------
# CPM2 4-pin thermal port: golden keeps Tj, emitted UROM stays 3-terminal
# ---------------------------------------------------------------------------
def _write_pin_lib(path, header):
    path.write_text(f"* offline fixture\n.SUBCKT {header}\nR1 d s 1\n.ENDS\n",
                    encoding="ascii")


def test_cpm2_arm_specs_resolve_rom_pins_from_emitted_header(monkeypatch,
                                                             tmp_path):
    golden_lib = tmp_path / "cpm2.lib"
    rom_lib = tmp_path / "urom.lib"
    _write_pin_lib(golden_lib, "CPM2 d g s Tj")
    _write_pin_lib(rom_lib, "UROM d g s")
    cfg = {
        "library": str(golden_lib),
        "subcircuit": "CPM2",
        "pins": {
            "pin_order": ["drain", "gate", "source"],
            "pin_order_semantic": {
                "drain": "d", "gate": "g", "source": "s", "tj": "Tj"},
        },
    }
    monkeypatch.setattr(urom, "load_input", lambda vendor: cfg)
    golden, rom = urom._arm_specs("cpm2-1200", rom_lib)
    assert golden["pin_order"] == ["drain", "gate", "source", "tj"]
    assert rom["pin_order"] == ["drain", "gate", "source"]
    assert "tj" not in rom["pin_order"]


def test_rom_pin_order_handles_numeric_and_gate_first_semantics(tmp_path):
    rom_lib = tmp_path / "urom.lib"
    _write_pin_lib(rom_lib, "UROM d g s")
    numeric = {"drain": "1", "gate": "2", "source": "3"}
    gate_first = {"drain": "drainin", "gate": "gatein",
                  "source": "sourcein"}
    for semantic in (numeric, gate_first):
        assert urom._rom_pin_order(rom_lib, "UROM", semantic) == [
            "drain", "gate", "source"]


@pytest.mark.parametrize("builder", ["_dpt_deck", "_speed_deck"])
def test_cpm2_four_pin_golden_vs_three_pin_rom_decks(builder):
    plan = urom._bench_plan("cpm2-1200", "trap/default", 3)
    golden = {"library_path": "/tmp/cpm2.lib", "subcircuit_name": "GOLD",
              "pin_order": ["drain", "gate", "source", "tj"]}
    rom = {"library_path": "/tmp/urom.lib", "subcircuit_name": "UROM",
           "pin_order": ["drain", "gate", "source"]}
    decks = {"golden": getattr(urom, builder)(plan, golden),
             "rom": getattr(urom, builder)(plan, rom)}
    gd, rd = decks["golden"], decks["rom"]
    temp = f"{plan['temperature']:g}"
    assert f"Vtjsrc_dut tj_dut 0 DC {temp}" in gd
    assert "Vtjsrc_dut" not in rd
    assert "Xdut drain gate 0 tj_dut GOLD" in gd
    assert "Xdut drain gate 0 UROM" in rd
    assert f".temp {temp}" in gd and f".temp {temp}" in rd
    result = urom._deck_identity(gd, rd, plan["options_lines"])
    assert result["deck_diff_only_dut"] is True
    assert result["temperature_both_arms"] == plan["temperature"]


def test_cpm2_deck_identity_still_rejects_non_dut_change_with_tj():
    plan = urom._bench_plan("cpm2-1200", "trap/default", 3)
    golden = {"library_path": "/tmp/cpm2.lib", "subcircuit_name": "GOLD",
              "pin_order": ["drain", "gate", "source", "tj"]}
    rom = {"library_path": "/tmp/urom.lib", "subcircuit_name": "UROM",
           "pin_order": ["drain", "gate", "source"]}
    gd = urom._dpt_deck(plan, golden)
    rd = urom._dpt_deck(plan, rom).replace(
        f".temp {plan['temperature']:g}", ".temp 100")
    result = urom._deck_identity(gd, rd, plan["options_lines"])
    assert result["deck_diff_only_dut"] is False
    assert result["temperature_both_arms"] is None


# ---------------------------------------------------------------------------
# G0-G6: fabricated / missing counts must never pass
# ---------------------------------------------------------------------------
def test_baseline_gates_pass_only_on_pinned_measured_bars():
    gates = _gates()
    assert all(gates[name]["pass"] is True for name in GATE_NAMES)
    assert gates["all_pass"] is True
    assert gates["fidelity_all_pass"] is True
    assert gates["timing_all_pass"] is True
    assert gates["speed_all_pass"] is True
    assert urom.FIDELITY_GATES == ("G0", "G1_static", "G2_derivative",
                                   "G3_dpt_waveform", "G6_charge")
    assert urom.TIMING_GATES == ("G4_timing_energy",)
    assert urom.SPEED_GATES == ("G5_speed",)
    g1 = gates["G1_static"]["detail"]
    assert g1["bar_pct"] == urom.STATIC_BARS_PCT[VENDOR]
    assert g1["band_pp"] == urom.G1_BAND_PP == 0.05
    assert g1["endpoint_bar_pct"] == urom.ENDPOINT_BAR_PCT == 4.13
    g3 = gates["G3_dpt_waveform"]["detail"]["golden"]
    assert g3["bars_vds_id_pct"] == urom.DPT_BARS_PCT["trap/default"][VENDOR]
    g5 = gates["G5_speed"]["detail"]
    assert g5["method_label"] == "trap/default"
    assert g5["global_claim"] is False
    assert g5["min_default_gear"] is None
    assert g5["effect_min"] == urom.SPEED_EFFECT_MIN == 1.05
    assert g5["solver_effect_ok"] is True
    assert g5["solver_ci_low_above_one"] is True
    assert gates["G6_charge"]["detail"]["budget_pct"] == 0.5
    verdicts = urom._verdicts(gates)
    assert set(verdicts) == set(SPLIT_VERDICTS)
    assert all(verdicts[key] == "accept" for key in SPLIT_VERDICTS)


@pytest.mark.parametrize("mutate", [
    lambda o: o["equal_sandbox"].update(deck_diff_only_dut=False),
    lambda o: o["equal_sandbox"].update(effective_options_dumped=False),
    lambda o: o["equal_sandbox"].update(tolerance_overrides=["reltol=1e-5"]),
    lambda o: o["equal_sandbox"].update(libraries_inlined_both_arms=False),
    lambda o: o["equal_sandbox"].update(temperature_both_arms=False),
    lambda o: o["speed"].update(repeats=2),
    lambda o: o["speed"]["golden"].update(completion=False),
    lambda o: o["speed"]["urom"].update(completion=False),
])
def test_g0_rejects_equal_sandbox_or_repeat_violations(mutate):
    obs = _observations()
    mutate(obs)
    gates = _gates(obs)
    assert gates["G0"]["pass"] is False
    assert gates["G0"]["detail"] == obs["equal_sandbox"]
    assert gates["all_pass"] is False


@pytest.mark.parametrize("mutate", [
    lambda o: o["static"].update(full_sweep_rel_rmse_pct=None),
    lambda o: o["static"].update(worst_endpoint_rel_error_pct=None),
    lambda o: o["static"].update(
        full_sweep_rel_rmse_pct=urom.STATIC_BARS_PCT[VENDOR] + 1.0),
    lambda o: o["static"].update(
        worst_endpoint_rel_error_pct=urom.ENDPOINT_BAR_PCT + 1.0),
])
def test_g1_rejects_missing_or_regressed_static_numbers(mutate):
    obs = _observations()
    mutate(obs)
    gates = _gates(obs)
    assert gates["G1_static"]["pass"] is False
    if obs["static"]["full_sweep_rel_rmse_pct"] is None:
        assert gates["G1_static"]["status"] == "not_measured"
        assert gates["G1_static"]["detail"]["rel_rmse_pct"] is None
    assert gates["all_pass"] is False


def test_g1_does_not_accept_static_rmse_alone_as_a_pass():
    obs = _observations()
    obs["static"]["full_sweep_rel_rmse_pct"] = 0.01  # structural probe only
    obs["derivative"] = {"pass": False, "status": "derivative_regressed",
                         "detail": {}}
    gates = _gates(obs)
    assert gates["G2_derivative"]["pass"] is False
    assert gates["all_pass"] is False


@pytest.mark.parametrize("derivative", [
    None,
    {"pass": False, "status": "value_only_fit"},
])
def test_g2_rejects_missing_or_failed_derivative_check(derivative):
    obs = _observations()
    obs["derivative"] = derivative
    gates = _gates(obs)
    assert gates["G2_derivative"]["pass"] is False
    assert gates["all_pass"] is False


@pytest.mark.parametrize(("mutate", "status"), [
    (lambda o: o["dpt"]["golden"].update(completed=False), "infeasible"),
    (lambda o: o["dpt"]["urom"].update(completed=False), "infeasible"),
    (lambda o: o["dpt"]["urom"]["waveform"].update(vds_rel_rmse_pct=None),
     "measured"),
    (lambda o: o["dpt"]["urom"]["waveform"].update(id_rel_rmse_pct=None),
     "measured"),
    (lambda o: o["dpt"]["urom"]["waveform"].update(
        vds_rel_rmse_pct=urom.DPT_BARS_PCT["trap/default"][VENDOR][0] + 1.0),
     "measured"),
])
def test_g3_rejects_truncated_golden_and_missing_waveform_numbers(mutate,
                                                                  status):
    obs = _observations()
    mutate(obs)
    gates = _gates(obs)
    g3 = gates["G3_dpt_waveform"]
    assert g3["pass"] is False
    assert g3["status"] == status
    assert gates["all_pass"] is False


def test_g3_uses_the_declared_method_bar_not_one_global_number():
    gear_vds_bar = urom.DPT_BARS_PCT["gear-both"][VENDOR][0]
    obs = _observations("trap/default")
    for arm in ("golden", "urom"):
        obs["dpt"][arm]["waveform"]["vds_rel_rmse_pct"] = gear_vds_bar
    trap = _gates(obs, _record(method="trap/default"))["G3_dpt_waveform"]
    assert trap["pass"] is False
    assert trap["detail"]["golden"]["bars_vds_id_pct"] == (
        urom.DPT_BARS_PCT["trap/default"][VENDOR])

    obs_gear = _observations("gear-both")
    gear = _gates(obs_gear, _record(method="gear-both"))["G3_dpt_waveform"]
    assert gear["pass"] is True
    assert gear["detail"]["golden"]["bars_vds_id_pct"] == (
        urom.DPT_BARS_PCT["gear-both"][VENDOR])


@pytest.mark.parametrize("timing", [
    None,
    {"pass": False, "status": "not_measured", "reason": "no crossing"},
    {"pass": False, "status": "measured",
     "rows": {"tr_ns": {"status": "not_measured", "value": None, "bar": 1.0}}},
])
def test_g4_rejects_missing_crossings_without_substituting_zero(timing):
    obs = _observations()
    obs["timing_energy"] = timing
    gates = _gates(obs)
    assert gates["G4_timing_energy"]["pass"] is False
    if timing is not None and "rows" in timing:
        row = gates["G4_timing_energy"]["detail"]["rows"]["tr_ns"]
        assert row["value"] is None and row["status"] == "not_measured"
    assert gates["all_pass"] is False


@pytest.mark.parametrize("key", [
    "n_pulses", "method_label", "t_last", "iterations", "points"])
def test_g5_rejects_fabricated_speed_row_without_anti_fake_counts(key):
    rec = _record()
    rec["speed_adaptive_20"].pop(key)
    gates = _gates(_observations(), rec)
    g5 = gates["G5_speed"]
    assert g5["pass"] is False
    assert g5["status"] == "not_measured"
    assert g5["detail"]["scoped_row_valid"] is False
    assert gates["all_pass"] is False


def test_g5_detail_reports_nullable_max_step_without_changing_pass():
    rec = _record()
    gates = _gates(rec=rec)
    assert gates["G5_speed"]["detail"]["max_step"] is None
    assert gates["G5_speed"]["pass"] is True

    rec["speed_adaptive_20"]["max_step"] = 1.0e-7
    capped = _gates(rec=rec)
    assert capped["G5_speed"]["detail"]["max_step"] == 1.0e-7
    assert capped["G5_speed"]["pass"] is True
    assert capped["all_pass"] is True


def test_g5_rejects_a_speed_row_from_aborted_arms():
    obs = _observations()
    obs["speed"]["urom"].update(completion=False)
    gates = _gates(obs)
    assert gates["G5_speed"]["pass"] is False
    assert gates["G5_speed"]["detail"]["completion"]["urom"] is False
    assert gates["all_pass"] is False


def test_speed_row_withholds_ratios_until_completion_counts_exist():
    plan = {"method_label": "trap/default",
            "speed": {"n_pulses": urom.SPEED_N_PULSES, "max_step": None}}
    speed = {"golden": _arm(2.0, 2.2), "urom": _arm(1.0, 1.1)}
    row = urom._speed_row(plan, speed)
    assert row["solver_speedup"] == pytest.approx(2.0)  # structural probe
    assert row["wall_speedup"] == pytest.approx(2.0)
    assert row["n_pulses"] == 20
    assert row["max_step"] is None
    assert row["method_label"] == "trap/default"
    assert row["t_last"]["golden"] == row["t_last"]["urom"] == 1.0
    assert row["iterations"]["golden"] == row["iterations"]["urom"] == 100
    assert row["points"]["golden"] == row["points"]["urom"] == 1000
    assert row["global_claim"] is False and row["min_default_gear"] is None
    assert row["effect_min_solver_speedup"] == urom.SPEED_EFFECT_MIN == 1.05
    solver_ci = row["solver_speedup_ci95"]
    assert solver_ci["point"] == pytest.approx(2.0)
    assert solver_ci["low"] > 1.0
    assert solver_ci["n"] == 3
    assert solver_ci["confidence"] == urom.SPEED_CI_CONFIDENCE == 0.95
    assert solver_ci["n_resamples"] == urom.SPEED_CI_RESAMPLES
    assert row["net_solver_speedup"] == pytest.approx(
        (2.0 - urom.LTSPICE_INTERNAL_FLOOR_S)
        / (1.0 - urom.LTSPICE_INTERNAL_FLOOR_S))
    assert row["net_solver_speedup_ci95"]["point"] == pytest.approx(
        row["net_solver_speedup"])
    assert row["net_solver_speedup_ci95"]["low"] > 1.0
    assert row["ltspice_internal_floor_s"] == pytest.approx(
        urom.LTSPICE_INTERNAL_FLOOR_S)
    assert row["wall_speedup_ci95"]["point"] == pytest.approx(2.0)
    assert "aborts are failures" in row["completion_basis"]
    assert "3 interleaved repeats" in row["completion_basis"]
    assert "0.999" in row["completion_basis"]

    speed["golden"]["completion"] = False
    missing = urom._speed_row(plan, speed)
    assert missing["solver_speedup"] is None
    assert missing["wall_speedup"] is None
    assert missing["net_solver_speedup"] is None
    assert missing["solver_speedup_ci95"] is None
    assert missing["wall_speedup_ci95"] is None
    assert missing["net_solver_speedup_ci95"] is None
    assert missing["method_label"] == "trap/default"
    assert missing["n_pulses"] == 20
    assert missing["t_last"]["urom"] == 1.0


def _speed_plan(method: str = "trap/default") -> dict:
    return {"method_label": method,
            "speed": {"n_pulses": urom.SPEED_N_PULSES, "max_step": None}}


def _speed_observations(golden: dict, rom: dict) -> dict:
    obs = _observations()
    obs["speed"] = {"repeats": urom.SPEED_REPEATS,
                    "n_pulses": urom.SPEED_N_PULSES,
                    "golden": golden, "urom": rom}
    obs["clock_spread"] = urom._clock_spread_record(obs["speed"])
    obs["equal_sandbox"]["clock_spread"] = obs["clock_spread"]
    return obs


def test_paired_speedup_ci_pairs_repeats_and_fails_closed():
    speed = {"golden": _arm(2.0, 2.2), "urom": _arm(1.0, 1.1)}
    ci = urom._paired_speedup_ci(speed, "solver_time_s")
    assert ci["point"] == pytest.approx(2.0)
    assert ci["low"] == pytest.approx(2.0)
    assert ci["n"] == 3
    assert ci["confidence"] == 0.95
    slower = {"golden": _arm(1.0, 1.1), "urom": _arm(2.0, 2.2)}
    slower_ci = urom._paired_speedup_ci(slower, "solver_time_s")
    assert slower_ci["point"] == pytest.approx(0.5)
    assert slower_ci["high"] < 1.0

    unpaired = {"golden": _arm(2.0, 2.2),
                "urom": _arm(1.0, 1.1, repeats=2)}
    assert urom._paired_speedup_ci(unpaired, "solver_time_s") is None
    no_rows = {"golden": _arm(2.0, 2.2), "urom": _arm(1.0, 1.1)}
    no_rows["urom"]["per_repeat"] = []
    assert urom._paired_speedup_ci(no_rows, "solver_time_s") is None


def test_epc_solver_ratio_0_6457_never_passes_the_speed_gate():
    speed = {"golden": _arm(0.6457, 0.8), "urom": _arm(1.0, 1.2)}
    obs = _speed_observations(speed["golden"], speed["urom"])
    rec = _record()
    rec["speed_adaptive_20"] = urom._speed_row(_speed_plan(), speed)
    gates = _gates(obs, rec)
    g5 = gates["G5_speed"]
    assert g5["pass"] is False
    assert g5["detail"]["solver_speedup"] == pytest.approx(0.6457)
    assert g5["detail"]["solver_effect_ok"] is False
    assert gates["speed_all_pass"] is False
    verdicts = urom._verdicts(gates)
    assert verdicts["speed_verdict"] == "reject"
    assert verdicts["overall_verdict"] == "reject"
    # A slower model must never pass a speed gate.
    assert g5["detail"]["solver_speedup_ci95"]["high"] < 1.0


def test_g5_requires_the_effect_minimum_not_just_a_ratio_above_one():
    speed = {"golden": _arm(1.02, 1.2), "urom": _arm(1.0, 1.1)}
    obs = _speed_observations(speed["golden"], speed["urom"])
    rec = _record()
    rec["speed_adaptive_20"] = urom._speed_row(_speed_plan(), speed)
    gates = _gates(obs, rec)
    g5 = gates["G5_speed"]
    assert g5["detail"]["solver_speedup"] == pytest.approx(1.02)
    assert g5["detail"]["effect_min"] == 1.05
    assert g5["detail"]["solver_effect_ok"] is False
    assert g5["pass"] is False
    assert urom._verdicts(gates)["speed_verdict"] == "reject"


def test_g5_requires_a_measured_paired_ci_even_with_a_fast_point_ratio():
    speed = {"golden": _arm(2.0, 2.2), "urom": _arm(1.0, 1.1)}
    obs = _speed_observations(speed["golden"], speed["urom"])
    # Break the pairing: the point ratio is still 2.0, the CI is unmeasured.
    obs["speed"]["urom"]["per_repeat"] = (
        obs["speed"]["urom"]["per_repeat"][:2])
    rec = _record()
    rec["speed_adaptive_20"] = urom._speed_row(_speed_plan(), obs["speed"])
    gates = _gates(obs, rec)
    g5 = gates["G5_speed"]
    assert g5["detail"]["solver_speedup"] == pytest.approx(2.0)
    assert g5["detail"]["solver_speedup_ci95"] is None
    assert g5["detail"]["solver_ci_low_above_one"] is False
    assert g5["status"] == "not_measured"
    assert g5["pass"] is False
    assert urom._verdicts(gates)["speed_verdict"] == "reject"


def test_split_verdicts_isolate_timing_from_fidelity_and_speed():
    obs = _observations()
    obs["timing_energy"]["pass"] = False
    gates = _gates(obs)
    assert gates["timing_all_pass"] is False
    assert gates["fidelity_all_pass"] is True
    assert gates["speed_all_pass"] is True
    verdicts = urom._verdicts(gates)
    assert verdicts["timing_verdict"] == "reject"
    assert verdicts["fidelity_verdict"] == "accept"
    assert verdicts["speed_verdict"] == "accept"
    assert verdicts["overall_verdict"] == "reject"
    assert gates["all_pass"] is False

    obs = _observations()
    # A completing but slower arm isolates the speed verdict: the sandbox
    # completion guard (G0) stays satisfied, only the speed effect fails.
    obs["speed"]["golden"] = _arm(0.6457, 0.8)
    obs["speed"]["urom"] = _arm(1.0, 1.2)
    obs["clock_spread"] = urom._clock_spread_record(obs["speed"])
    obs["equal_sandbox"]["clock_spread"] = obs["clock_spread"]
    gates = _gates(obs)
    verdicts = urom._verdicts(gates)
    assert gates["G0"]["pass"] is True
    assert verdicts["fidelity_verdict"] == "accept"
    assert verdicts["timing_verdict"] == "accept"
    assert verdicts["speed_verdict"] == "reject"
    assert verdicts["overall_verdict"] == "reject"


def test_verdict_alias_is_the_overall_verdict_contract():
    src = inspect.getsource(urom.run_unified)
    assert 'record.update(_verdicts(record["gates"]))' in src
    assert 'record["verdict"] = record["overall_verdict"]' in src
    for field in ("fidelity_verdict", "timing_verdict", "speed_verdict",
                  "overall_verdict"):
        assert f'"{field}"' in inspect.getsource(urom._verdicts)


def test_runtime_record_serializes_wall_min_max_alongside_solver():
    def arm(solver, wall, s_min, s_max, w_min, w_max):
        return {**_arm(solver, wall),
                "solver_min_s": s_min, "solver_max_s": s_max,
                "wall_min_s": w_min, "wall_max_s": w_max}

    speed = {
        "golden": arm(2.0, 2.2, 1.9, 2.1, 2.1, 2.3),
        "urom": arm(1.0, 1.1, 0.9, 1.1, 1.0, 1.2),
        "repeats": 3,
    }
    runtime = urom._runtime_record(speed)
    assert runtime["min"]["solver"] == {"golden": 1.9, "urom": 0.9}
    assert runtime["min"]["wall"] == {"golden": 2.1, "urom": 1.0}
    assert runtime["max"]["solver"] == {"golden": 2.1, "urom": 1.1}
    assert runtime["max"]["wall"] == {"golden": 2.3, "urom": 1.2}
    assert runtime["median"]["wall"] == {"golden": 2.2, "urom": 1.1}
    assert runtime["repeats"] == 3

    # Unmeasured wall spread stays None, never substituted with zero.
    speed["golden"]["wall_min_s"] = None
    speed["golden"]["wall_max_s"] = None
    unmeasured = urom._runtime_record(speed)
    assert unmeasured["min"]["wall"]["golden"] is None
    assert unmeasured["max"]["wall"]["golden"] is None


@pytest.mark.parametrize("mutate", [
    lambda c: c["monotone_q_by_table"].update(q_oss=False),
    lambda c: c["monotone_q_by_table"].update(q_iss=False),
    lambda c: c["monotone_q_by_table"].update(q_rss=False),
    lambda c: c.update(all_charge_monotone=False),
    lambda c: c["q_zero_anchored_by_table"].update(q_iss=False),
    lambda c: c.update(all_charge_zero_anchored=False),
    lambda c: c.update(monotone_q_by_table={}),
    lambda c: c.pop("monotone_q_by_table"),
    lambda c: c.update(charge_fit_error_pct=None),
    lambda c: c.update(charge_fit_error_pct=urom.CHARGE_BUDGET_PCT + 1.0),
    lambda c: c.update(c_ds_scaling_excluded=False),
    lambda c: c.update(status="not_measured"),
    lambda c: c.update(breakpoint_crossings={}),
    lambda c: c.update(breakpoint_crossings={"q_oss": None}),
])
def test_g6_rejects_charge_offset_budget_and_cds_scaling(mutate):
    rec = _record()
    mutate(rec["c1"])
    gates = _gates(_observations(), rec)
    assert gates["G6_charge"]["pass"] is False
    assert gates["all_pass"] is False


def test_g6_requires_every_applicable_charge_table_to_be_monotone():
    rec = _record()
    rec["c1"]["monotone_q_by_table"]["q_rss"] = False
    rec["c1"]["all_charge_monotone"] = False
    gates = _gates(_observations(), rec)
    g6 = gates["G6_charge"]
    assert g6["pass"] is False
    detail = g6["detail"]
    assert detail["monotone_q_by_table"]["q_rss"] is False
    assert detail["all_charge_monotone"] is False
    assert detail["applicable_tables"] == ["q_iss", "q_oss", "q_rss"]


def test_g6_fails_closed_when_no_charge_table_is_measured():
    rec = _record()
    rec["c1"].update({
        "monotone_q_by_table": {"q_oss": None, "q_iss": None,
                                "q_rss": None},
        "all_charge_monotone": False,
        "applicable_tables": [],
    })
    gates = _gates(_observations(), rec)
    assert gates["G6_charge"]["pass"] is False
    assert gates["G6_charge"]["detail"]["applicable_tables"] == []


@pytest.mark.parametrize("verdict", ["reject", "infeasible"])
def test_g6_rejects_a_failed_tuning_row(verdict):
    rec = _record()
    rec["tuning"] = {"mode": "on", "verdict": verdict}
    gates = _gates(_observations(), rec)
    assert gates["G6_charge"]["pass"] is False
    assert gates["G6_charge"]["detail"]["tuning_verdict"] == verdict


# ---------------------------------------------------------------------------
# N/A honesty: never turn missing into a zero or a pass
# ---------------------------------------------------------------------------
def test_json_safe_maps_nonfinite_to_none_not_zero():
    value = {"nan": float("nan"), "inf": float("inf"),
             "ninf": -float("inf"), "zero": 0.0, "finite": 2.5,
             "nested": [None, float("nan"), 0.0]}
    assert urom._json_safe(value) == {
        "nan": None, "inf": None, "ninf": None, "zero": 0.0, "finite": 2.5,
        "nested": [None, None, 0.0]}
    json.dumps(urom._json_safe(value), allow_nan=False)


def test_num_and_median_keep_missing_as_none():
    for missing in (None, "N/A", float("nan"), float("inf")):
        assert urom._num(missing) is None
    assert urom._num(0.0) == 0.0
    assert urom._num("2.5") == 2.5
    assert urom._median([]) is None
    assert urom._median([None, float("nan")]) is None
    assert urom._median([None, float("inf"), 1.0, 3.0]) == 2.0
    assert math.isfinite(urom._median([1.0, 3.0]))


def test_cli_renderers_never_invent_a_number_for_missing_data():
    for missing in (None, float("nan")):
        assert cli_main._num(missing) == "N/A"
    assert cli_main._num(0.0) == "0"
    assert cli_main._yn(None) == "N/A"
    assert cli_main._yn(True) == "yes"
    assert cli_main._yn(False) == "no"



def test_num_renders_float_counts_with_d_format_without_raising():
    for count in (0.0, 20.0, 12718.0, 12492.0):
        assert cli_main._num(count, "d") == str(int(count))
    assert cli_main._num(20, "d") == "20"
    for missing in (None, float("nan")):
        assert cli_main._num(missing, "d") == "N/A"
    # A non-integral value must fall back to a numeric format, not raise.
    assert cli_main._num(3.5, "d") == "3.5"
    # The default '.4g' rendering is untouched.
    assert cli_main._num(12718.0) == "1.272e+04"
    assert cli_main._num(0.0) == "0"


def test_cli_tier3_summary_renders_float_iteration_count(monkeypatch):
    record = {
        "vendor": VENDOR,
        "dpt_run": {
            "completed": True,
            "iterations": 21594.0,
            "trace_points": 7128,
            "golden_trace_points": 7114,
        },
        "emitted_lib": f"reports/models/{VENDOR}_PIECESPICE_T3.lib",
    }
    monkeypatch.setattr(registry, "validate", lambda *a, **k: record)
    result = _runner.invoke(app, ["validate", VENDOR, "--tier", "3"])
    assert result.exit_code == 0, result.output
    text = _flat(result.output)
    assert "Tier-3 DPT complete" in text
    assert "trace_points=7128/7114" in text
    assert "iterations=21594" in text
    assert "iterations=21594.0" not in text


def test_cli_tier4_summary_renders_float_iteration_and_point_counts(monkeypatch):
    record = {
        "vendor": VENDOR,
        "speedup": {
            "status": "measured",
            "solver_speedup": 2.5,
            "wall_speedup": 1.75,
            "golden_iterations": 131558.0,
            "tier4_iterations": 15306.0,
            "golden_points": 44050.0,
            "tier4_points": 3205.0,
            "t_last_s": 0.000176,
        },
    }
    monkeypatch.setattr(registry, "validate", lambda *a, **k: record)
    result = _runner.invoke(app, ["validate", VENDOR, "--tier", "4"])
    assert result.exit_code == 0, result.output
    text = _flat(result.output)
    assert "iters=131558/15306" in text
    assert "points=44050/3205" in text
    assert "iters=131558.0" not in text


# ---------------------------------------------------------------------------
# CLI help and commands, all with stubbed registry calls
# ---------------------------------------------------------------------------
def test_cli_help_lists_commands_and_urom_variant():
    top = _runner.invoke(app, ["--help"])
    assert top.exit_code == 0, top.output
    flat = _flat(top.output)
    for command in ("parse", "sweep", "curves", "fit", "validate", "compare"):
        assert command in flat
    for command in ("fit", "validate"):
        # Rich wraps option text differently across terminals/Python versions.
        # Assert the callback signature/defaults plus the source declaration
        # of the two Typer options; the stubbed command tests below exercise
        # the actual routing.
        cmd = next(c for c in app.registered_commands
                   if getattr(c.callback, "__name__", None) == command)
        sig = inspect.signature(cmd.callback)
        assert "variant" in sig.parameters
        assert "tier" in sig.parameters
        src = inspect.getsource(cmd.callback)
        assert ' "--variant"' in src and ' "--tier"' in src
        assert "urom" in src


def test_cli_fit_and_validate_urom_print_gate_record_without_bench(monkeypatch):
    calls: list = []
    record = _cli_record()

    def validate(vendor, tier, variant, basis="tier2"):
        calls.append((vendor, tier, variant, basis))
        return record

    monkeypatch.setattr(registry, "validate", validate)
    monkeypatch.setattr(
        registry, "fit",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("urom fit must use validate, not fit")))

    for command in ("fit", "validate"):
        result = _runner.invoke(
            app, [command, VENDOR, "--tier", "3", "--variant", "urom"])
        assert result.exit_code == 0, result.output
        text = _flat(result.output)
        assert "Unified one-ROM" in text
        assert "variant=urom basis=ann method=trap/default" in text
        assert "all_pass=no" in text and "verdict=reject" in text
        for tenant in ("overall_verdict=reject", "fidelity_verdict=accept",
                       "timing_verdict=reject", "speed_verdict=reject",
                       "fidelity_all_pass=yes", "timing_all_pass=no",
                       "speed_all_pass=no", "G4_tier2_regression",
                       "(diagnostic only)"):
            assert tenant in text
        for name in GATE_NAMES:
            assert name in text
        assert "solver=N/Ax" in text and "wall=N/Ax" in text
        assert "solver_ci95=N/A" in text and "wall_ci95=N/A" in text
        assert "effect_min=1.05" in text
        assert "complete=no" in text
        assert "Tier-3 DPT" not in text
    assert calls == [(VENDOR, 3, "urom", "tier2")] * 2


def test_cli_urom_default_basis_is_canonical_tier2(monkeypatch):
    calls: list = []
    record = _cli_record()
    record["emit"] = "tier2"
    record["channel_basis"] = "tier2"

    def validate(vendor, tier, variant, basis="tier2"):
        calls.append((vendor, tier, variant, basis))
        return record

    monkeypatch.setattr(registry, "validate", validate)
    result = _runner.invoke(app, ["validate", VENDOR, "--tier", "3",
                                  "--variant", "urom"])
    assert result.exit_code == 0, result.output
    text = _flat(result.output)
    assert calls == [(VENDOR, 3, "urom", "tier2")]
    assert "variant=urom basis=tier2 method=trap/default" in text
    assert "tuning=off c1_variant=C0" in text


def test_cli_urom_explicit_ann_basis_is_displayed_and_passed(monkeypatch):
    calls: list = []
    record = _cli_record()
    record["emit"] = "ann"
    record["tuning_mode"] = "off"

    def validate(vendor, tier, variant, basis="tier2"):
        calls.append((vendor, tier, variant, basis))
        return record

    monkeypatch.setattr(registry, "validate", validate)
    result = _runner.invoke(
        app, ["validate", VENDOR, "--tier", "3", "--variant", "urom",
              "--basis", "ann"])
    assert result.exit_code == 0, result.output
    assert calls == [(VENDOR, 3, "urom", "ann")]
    assert "basis=ann" in _flat(result.output)


def test_cli_compare_prints_recorded_unified_summary_without_recomputing(
        monkeypatch, tmp_path):
    path = tmp_path / "unified_validation.json"
    path.write_text(json.dumps(_cli_record()), encoding="utf-8")
    monkeypatch.setattr(urom, "_record_path", lambda vendor: path)
    monkeypatch.setattr(registry, "compare_rows", lambda vendor: [])
    monkeypatch.setattr(registry, "tier4_rows", lambda vendor: [])

    result = _runner.invoke(app, ["compare", VENDOR])
    assert result.exit_code == 0, result.output
    text = _flat(result.output)
    assert "Unified one-ROM (recorded only; never recomputed)" in text
    assert "method=trap/default" in text
    assert "n_pulses=20" in text
    assert "repeats=3" in text
    assert "complete=no" in text
    assert "No recorded unified validation" not in text


def test_cli_compare_completes_from_recorded_json_float_counts(monkeypatch):
    # The checked-in record stores iterations/points as floats (medians);
    # the CLI must render them as integers rather than raising on 'd'.
    path = registry.unified_validation_path(VENDOR)
    recorded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(recorded["runtime"]["iterations"]["golden"], float)
    assert isinstance(recorded["runtime"]["points"]["golden"], float)
    monkeypatch.setattr(urom, "_record_path", lambda vendor: path)
    monkeypatch.setattr(registry, "compare_rows", lambda vendor: [])
    monkeypatch.setattr(registry, "tier4_rows", lambda vendor: [])

    result = _runner.invoke(app, ["compare", VENDOR])
    assert result.exit_code == 0, result.output
    text = _flat(result.output)
    expected_iters = int(recorded["runtime"]["iterations"]["golden"])
    expected_points = int(recorded["runtime"]["points"]["golden"])
    assert f"iterations={expected_iters}" in text
    assert f"points={expected_points}" in text
    assert "n_pulses=20" in text
    assert "repeats=3" in text


def test_python_m_cli_main_compare_completes_from_recorded_fixture():
    # End-to-end check of the documented invocation (`python -m cli.main`).
    path = registry.unified_validation_path(VENDOR)
    recorded = json.loads(path.read_text(encoding="utf-8"))
    expected_iters = int(recorded["runtime"]["iterations"]["golden"])
    expected_points = int(recorded["runtime"]["points"]["golden"])
    completed = subprocess.run(
        [sys.executable, "-m", "cli.main", "compare", VENDOR],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ValueError" not in completed.stderr
    text = _flat(completed.stdout)
    assert "Unified one-ROM (recorded only" in text
    assert f"iterations={expected_iters}" in text
    assert f"points={expected_points}" in text


def test_cli_compare_without_unified_record_reports_absence(monkeypatch,
                                                            tmp_path):
    monkeypatch.setattr(
        urom, "_record_path", lambda vendor: tmp_path / "absent.json")
    monkeypatch.setattr(
        registry, "compare_rows",
        lambda vendor: [("tr_ns", 1.0, 1.0, 1.0)])
    monkeypatch.setattr(registry, "tier4_rows", lambda vendor: [])

    result = _runner.invoke(app, ["compare", VENDOR])
    assert result.exit_code == 0, result.output
    text = _flat(result.output)
    assert "No recorded unified validation" in text
    assert "Unified one-ROM (recorded only" not in text




# ---------------------------------------------------------------------------
# hardened production contracts (offline, no bench)
# ---------------------------------------------------------------------------
def test_theta_bounds_derive_o_gd_from_gate_charge_not_qoss():
    spec = _synthetic_spec(
        metadata={"c_rss_lin": 2.0e-11, "Vds_cap_max": 25.0},
        q_oss_vs_v=[0.0, 10.0], q_oss_charge_c=[0.0, 1.0e-6],
    )
    bounds = urom._theta_bounds(spec)
    assert bounds["O_GD"] == pytest.approx((0.0, 5.0e-10))
    assert bounds["A_GD"] == (0.5, 2.0)


def test_bench_plan_tuning_conditions_carry_cached_golden_emp():
    plan = urom._bench_plan(VENDOR, "trap/default", 3)
    conditions = plan["tuning_conditions"]
    assert len(conditions) == 2
    assert [c["held_out"] for c in conditions] == [False, True]
    for condition in conditions:
        assert condition["emp"]
        assert all(math.isfinite(v) for v in condition["emp"].values())
        assert isinstance(condition["completed"], bool)
        assert condition["condition"]["vbus"] == plan["vbus"]
        # The live simulator is attached only in _build_emitted.
        assert "simulate" not in condition


def test_deck_identity_takes_temperature_from_the_deck():
    golden = _deck()
    result = urom._deck_identity(golden, _deck(inst="XUROM"), [],
                                 expected_temperature=25.0)
    assert result["temperature_both_arms"] == 25.0
    mismatch = urom._deck_identity(
        golden, _deck(inst="XUROM", extra=".temp 100"), [])
    assert mismatch["deck_diff_only_dut"] is False
    assert mismatch["temperature_both_arms"] is None


def test_derivative_checks_fail_closed_for_both_emit_modes():
    rows = [
        (2.0, [0.0, 1.0], 0.0, [0.0, 1.0e-3]),
        (5.0, [0.0, 1.0], 0.0, [0.0, 5.0e-3]),
    ]
    spec = _synthetic_spec(
        v_th=3.0,
        q_oss_vs_v=[0.0, 1.0], q_oss_charge_c=[0.0, 1.0e-12],
    )
    ann = urom._derivative_checks(rows, spec, {}, "ann")
    assert ann["pass"] is False
    assert ann["out_of_range_check"] is None
    tier2 = urom._derivative_checks(
        rows, spec, {"record": {"out_of_range_check": "pass"}}, "tier2")
    assert tier2["pass"] is False
    assert tier2["out_of_range_check"] == "not_applicable"
    # The stale "no static sweep rows" reason must not survive a measured
    # branch: these rows are accepted and gm/gds are computed.
    assert tier2["reason"] != "no static sweep rows"
    assert ann["reason"] != "no static sweep rows"
    hybrid = urom._derivative_checks(rows, spec, {}, "hybrid")
    assert hybrid["pass"] is False
    assert "hybrid_residual_probe_missing" in hybrid["reason"]
    assert hybrid["hybrid_residual_probe"]["status"] == "not_measured"
    assert hybrid["hybrid_residual_probe"]["label"] == (
        urom.HYBRID_RESIDUAL_PROBE_LABEL)


def test_charge_checks_compute_residual_and_crossing_counts():
    spec = _synthetic_spec(
        q_oss_vs_v=[0.0, 1.0, 2.0],
        q_oss_charge_c=[0.0, 1.0e-12, 3.0e-12],
        q_iss_vs_v=[0.0, 1.0, 2.0],
        q_iss_charge_c=[0.0, 2.0e-12, 4.0e-12],
        q_rss_vs_v=[0.0, 1.0, 2.0],
        q_rss_charge_c=[0.0, 1.0e-12, 2.0e-12],
    )
    checks = urom._charge_checks(spec, {"q_oss": 0.0})
    assert checks["status"] == "measured"
    assert checks["charge_fit_error_pct"] == pytest.approx(0.0, abs=1e-9)
    assert checks["c0_bit_identical_at_eps0"] is None
    assert all(isinstance(v, int) for v in
               checks["breakpoint_crossings"].values())
    rec = _record()
    rec["c1"] = checks
    assert _gates(_observations(), rec)["G6_charge"]["pass"] is True


def test_timing_band_rejects_a_tiny_never_turn_on_delay():
    band = urom._timing_band(20.0)
    assert band == pytest.approx((19.0, 21.0))
    assert 1.0 < band[0]
    assert urom._timing_band(None) is None


def test_timing_band_uses_an_absolute_floor_for_near_zero_bars():
    band = urom._timing_band(0.0)
    assert band == (-urom.TIMING_BAND_ABS, urom.TIMING_BAND_ABS)
    assert urom.TIMING_BAND_ABS == 1e-12
    # A zero bar is not auto-passed: a clearly non-zero value fails.
    assert not (band[0] <= 1e-9 <= band[1])
    tiny = urom._timing_band(1e-13)
    half = max(1e-13 * urom.TIMING_BAND_REL, urom.TIMING_BAND_ABS)
    assert tiny == pytest.approx((1e-13 - half, 1e-13 + half))


def _write_dpt_metrics(monkeypatch, tmp_path, payload):
    folder = tmp_path / "reports" / "comparison" / VENDOR
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "dpt_metrics.json").write_text(
        json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(urom.registry, "ROOT", tmp_path)


def _write_tier3_validation(monkeypatch, tmp_path, golden: dict,
                            *, names=("golden", "tier3", "tier2")):
    folder = tmp_path / "reports" / "comparison" / VENDOR
    folder.mkdir(parents=True, exist_ok=True)
    comparison = {metric: {name: value for name in names}
                  for metric, value in golden.items()}
    (folder / "tier3_validation.json").write_text(
        json.dumps({"dpt_comparison": comparison}), encoding="utf-8")
    monkeypatch.setattr(urom.registry, "ROOT", tmp_path)


def test_g4_golden_bars_come_from_tier3_validation_not_tier2(monkeypatch,
                                                              tmp_path):
    golden = {"td_on_ns": 11.0, "tr_ns": 10.0, "td_off_ns": 13.0,
              "tf_ns": 12.0, "e_on_uj": 14.0, "e_off_uj": 15.0}
    dpt_golden = {k: 100.0 for k in golden}   # synthetic, not measured
    tier2 = {k: 20.0 for k in golden}
    _write_dpt_metrics(monkeypatch, tmp_path,
                       {"golden_metrics": dpt_golden, "tier2_metrics": tier2})
    _write_tier3_validation(monkeypatch, tmp_path, golden)
    # The independent G4 bar is the device-current golden comparison; the
    # pipeline's own dpt_metrics golden and Tier-2 tables never move it.
    assert urom._timing_bars(VENDOR) == golden
    assert urom._tier2_regression_bars(VENDOR) == tier2
    assert urom.GOLDEN_DPT_BAR_SOURCE.endswith(
        "tier3_validation.json#dpt_comparison.<metric>.golden")
    assert urom.TIER2_REGRESSION_BAR_SOURCE.endswith(
        "dpt_metrics.json#tier2_metrics")
    source = urom._timing_bars_source()
    assert "tier3_validation.json#dpt_comparison.<metric>.golden" in source
    regression_source = urom._tier2_regression_source()
    assert "#tier2_metrics" in regression_source
    assert "diagnostic only" in regression_source
    prov = urom.PROVENANCE["timing_energy_bars"]
    assert "tier3_validation.json" in prov["source"]
    assert "dpt_comparison.<metric>.golden" in prov["golden_column"]
    assert prov["tier2_regression_role"].startswith("diagnostic only")
    assert "tier3_validation.json" in urom.PROVENANCE["timing_band"]["source"]
    assert urom.PROVENANCE["tier2_regression_gate"]["source"] == (
        urom.TIER2_REGRESSION_BAR_SOURCE)


def test_g4_scores_the_golden_bar_even_when_tier2_regresses(monkeypatch,
                                                             tmp_path):
    golden = {k: 10.0 for k in urom._TIMING_METRICS}
    tier2 = {k: 20.0 for k in urom._TIMING_METRICS}
    _write_dpt_metrics(monkeypatch, tmp_path,
                       {"golden_metrics": golden, "tier2_metrics": tier2})
    _write_tier3_validation(monkeypatch, tmp_path, golden)
    actual = {k: 10.0 for k in golden}
    rows, ok = urom._timing_score_rows(actual, urom._timing_bars(VENDOR))
    assert ok is True
    assert rows["td_on_ns"]["bar"] == 10.0
    regression_rows, regression_ok = urom._timing_score_rows(
        actual, urom._tier2_regression_bars(VENDOR))
    assert regression_ok is False
    assert regression_rows["td_on_ns"]["bar"] == 20.0

    obs = _observations()
    obs["timing_energy"]["pass"] = ok
    obs["timing_energy"]["rows"] = rows
    obs["timing_energy"]["tier2_regression"] = {
        "status": "measured", "rows": regression_rows,
        "all_within_band": regression_ok, "used_for_verdict": False,
        "diagnostic_only": True,
    }
    gates = _gates(obs)
    assert gates["G4_timing_energy"]["pass"] is True
    diagnostic = gates["G4_tier2_regression"]
    assert diagnostic["pass"] is False
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["scored"] is False
    assert diagnostic["detail"]["used_for_verdict"] is False
    assert gates["timing_all_pass"] is True
    assert urom._verdicts(gates)["timing_verdict"] == "accept"
    # The diagnostic may fail without touching the overall verdict.
    assert gates["all_pass"] is True
    assert urom._verdicts(gates)["overall_verdict"] == "accept"


# ---------------------------------------------------------------------------
# set A: canonical C0 fallback, producer provenance, channel ramp mirror
# ---------------------------------------------------------------------------
def _channel_spec() -> PWLModelSpec:
    """Synthetic conducting channel + anchored tables (not measured)."""
    return _synthetic_spec(
        breakpoints_vds=[0.0, 2.0, 10.0],
        breakpoints_id=[[0.0, 4.0, 8.0], [0.0, 8.0, 16.0]],
        vgs_values=[3.0, 6.0],
        v_th=1.5,
        q_oss_vs_v=[0.0, 5.0, 10.0],
        q_oss_charge_c=[0.0, 1.0e-12, 3.0e-12],
        q_iss_vs_v=[0.0, 5.0, 10.0],
        q_iss_charge_c=[0.0, 2.0e-12, 4.0e-12],
        q_rss_vs_v=[0.0, 5.0, 10.0],
        q_rss_charge_c=[0.0, 1.0e-12, 2.0e-12],
    )


def _conducting_rows():
    """Synthetic emitted-sweep rows with a clean Vds=0 anchor (not measured)."""
    return [
        (3.0, [0.0, 2.0, 10.0], 0.0, [0.0, 4.0, 8.0]),
        (6.0, [0.0, 2.0, 10.0], 0.0, [0.0, 8.0, 16.0]),
    ]


def test_tier2_g2_passes_from_emitted_sweep_and_cutoff_anchor():
    spec = _channel_spec()  # Vth=1.5 lies inside the drive grid: C0 emits it
    checks = urom._derivative_checks(_conducting_rows(), spec, {}, "tier2")
    assert checks["status"] == "measured"
    assert checks["pass"] is True
    assert checks["out_of_range_check"] == "not_applicable"
    assert checks["vds_zero_anchor"] is True
    assert checks["charge_derivative_non_negative"] is True
    assert checks["vth_zero_anchor"] is True
    assert checks["cutoff_probe_vgs"] == pytest.approx(1.5)
    assert checks["reason"] != "no static sweep rows"
    assert checks["activation"] is None  # pass cannot depend on an ANN record
    assert checks["gm_min"] >= 0.0 and checks["gds_min"] >= 0.0
    assert checks["gm_negative_fraction"] == pytest.approx(0.0)
    assert checks["gds_negative_fraction"] == pytest.approx(0.0)
    sign_gate = checks["derivative_sign_gate"]
    assert sign_gate["pass"] is True
    assert sign_gate["max_negative_fraction"] == (
        urom.DERIVATIVE_NEGATIVE_FRACTION_MAX) == 0.01


def test_g2_rejects_a_clearly_negative_gm_sweep():
    # Synthetic rows whose drain current falls as Vgs rises: gm < 0 everywhere.
    rows = [
        (3.0, [0.0, 2.0, 10.0], 0.0, [0.0, 8.0, 4.0]),
        (6.0, [0.0, 2.0, 10.0], 0.0, [0.0, 4.0, 2.0]),
    ]
    checks = urom._derivative_checks(rows, _channel_spec(), {}, "tier2")
    assert checks["pass"] is False
    assert checks["gm_min"] < 0.0
    assert checks["gm_negative_fraction"] > (
        urom.DERIVATIVE_NEGATIVE_FRACTION_MAX)
    assert "gm_negative_fraction" in checks["reason"]
    assert checks["derivative_sign_gate"]["pass"] is False
    # The measured fractions stay in the result as diagnostics.
    assert 0.0 <= checks["gm_negative_fraction"] <= 1.0


def test_g2_sign_fractions_fail_closed_when_not_measured():
    rows = [(3.0, [0.0, 2.0, 10.0], 0.0, [np.nan, np.nan, np.nan])]
    checks = urom._derivative_checks(rows, _channel_spec(), {}, "tier2")
    assert checks["pass"] is False
    assert checks["gm_negative_fraction"] is None
    assert checks["gds_negative_fraction"] is None
    assert checks["derivative_sign_gate"]["status"] == "not_measured"
    assert "derivative_sign_fractions_not_measured" in checks["reason"]


def test_tier2_g2_probes_vgs_zero_when_builder_emits_no_vth_anchor():
    spec = _channel_spec()
    spec.v_th = 12.0  # above every drive: the builder emits only Vgs=0
    checks = urom._derivative_checks(_conducting_rows(), spec, {}, "tier2")
    assert checks["cutoff_probe_vgs"] == pytest.approx(0.0)
    assert checks["vth_zero_anchor"] is True
    assert checks["pass"] is True


def test_tier2_g2_cutoff_probe_uses_emitted_width_not_derived_c1_width():
    spec = _channel_spec()
    # Replicate the C0 emitter metadata: derived C1 widths exist as
    # provenance, but the emitted width is empty (eps=0).
    spec.metadata["urom_c1_eps_by_table"] = {"transfer_vgs": 0.05}
    spec.metadata["urom_emitted_eps_by_table"] = {}
    probe = urom._emitted_cutoff_anchor(spec)
    assert probe is not None
    assert probe[1] == pytest.approx(0.0, abs=1e-12)
    checks = urom._derivative_checks(_conducting_rows(), spec, {}, "tier2")
    assert checks["vth_zero_anchor"] is True
    assert checks["pass"] is True


def test_tier2_g2_never_passes_without_measured_evidence():
    spec = _channel_spec()
    spec.q_oss_charge_c = [1.0e-12, 0.0, 3.0e-12]  # non-monotone Q
    checks = urom._derivative_checks(_conducting_rows(), spec, {}, "tier2")
    assert checks["pass"] is False
    assert checks["out_of_range_check"] == "not_applicable"
    assert checks["charge_derivative_non_negative"] is False
    assert "negative_charge_derivative" in checks["reason"]

    broken = _channel_spec()
    broken.breakpoints_id = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]  # no cutoff
    none_checks = urom._derivative_checks(
        _conducting_rows(), broken, {}, "tier2")
    assert none_checks["pass"] is False
    assert none_checks["vth_zero_anchor"] is None
    assert "gate_cutoff_anchor" in none_checks["reason"]

    no_rows = urom._derivative_checks([], _channel_spec(), {}, "tier2")
    assert no_rows["pass"] is False
    assert no_rows["status"] == "not_measured"
    assert no_rows["reason"] == "no static sweep rows"


def test_ann_g2_still_requires_its_own_cutoff_and_out_of_range_evidence():
    record = {"record": {"out_of_range_check": "pass",
                         "activation": "tanh", "neurons": 8}}
    checks = urom._derivative_checks(_conducting_rows(), _channel_spec(),
                                     record, "ann")
    # No swept sub-Vth row exists, so the explicit ANN Vth guard is
    # unmeasured; an out-of-range pass alone must not fabricate a G2 pass.
    assert checks["vth_zero_anchor"] is None
    assert checks["pass"] is False
    assert checks["out_of_range_check"] == "pass"


def test_tier2_fallback_emits_canonical_c0_modulo_name():
    from models.netlist_builder import build_tier2_subcircuit

    spec = _channel_spec()
    name = f"{VENDOR}_PIECSPICE_UROM"
    emitted = urom._emit_tier2_c0(spec, name=name)
    assert emitted == build_tier2_subcircuit(spec, c1_eps=None, name=name)
    canonical_name = f"{VENDOR}_PIECESPICE_T2"
    assert emitted.replace(name, canonical_name) == build_tier2_subcircuit(spec)
    g_ch = next(line for line in emitted.splitlines()
                if line.startswith("G_CH "))
    assert "sqrt(" not in g_ch and "abs(" in g_ch
    assert urom._c0_bit_identity(spec, emitted, name) is True
    assert urom._c0_bit_identity(spec, emitted + "\n", name) is False


def test_c1_tier2_emission_remains_an_explicit_variant():
    spec = _channel_spec()
    widths = urom._eps_by_table(spec)
    assert widths["transfer_vgs"] > 0.0 and widths["q_oss"] > 0.0
    assert urom.C1_TIER2_VARIANT == "tier2-c1"
    c1 = urom._emit_tier2_per_table(spec, widths, name="urom")
    c0 = urom._emit_tier2_c0(spec, name="urom")
    assert c1 != c0
    assert any("sqrt(" in line for line in c1.splitlines())


@pytest.mark.parametrize("variant,expect_c0", [
    (None, True),
    (urom.C1_TIER2_VARIANT, False),
])
def test_build_emitted_tier2_uses_c0_unless_explicitly_labelled(
        monkeypatch, tmp_path, variant, expect_c0):
    from models.netlist_builder import build_tier2_subcircuit

    spec = _channel_spec()
    spec.metadata.update({
        "r_gate": 1.5, "r_source": 1.0e-4, "r_drain": 1.0e-4,
        "Vds_cap_max": 10.0, "c_iss_lin": 1.0e-12,
        "c_oss_lin": 1.0e-12, "c_rss_lin": 1.0e-12,
        "urom_provenance": {"gate_charge_tables": {
            "source": "synthetic curve4", "kind": "source_derived"}},
    })
    lib_path = tmp_path / f"{VENDOR}_PIECSPICE_UROM.lib"
    monkeypatch.setattr(urom, "_pwl_spec", lambda vendor: spec)
    monkeypatch.setattr(urom.registry, "lib_path",
                        lambda *args, **kwargs: lib_path)
    plan = {"tuning_conditions": [{}], "method_label": "trap/default"}
    lib, out, _, _ = _REAL_BUILD_EMITTED(
        VENDOR, emit="tier2", tuning="off", plan=plan, c1_variant=variant)
    emitted = lib.read_text(encoding="ascii")
    elements = urom._strip_identity_header(emitted)
    canonical = build_tier2_subcircuit(out, c1_eps=None, name=lib.stem)
    assert urom._identity_header_hint(emitted) == (
        "tier2", "trap/default", "off", variant or "C0")
    if expect_c0:
        assert elements == canonical
        assert out.metadata["urom_c1_eps"] == 0.0
        assert urom._emitted_eps_by_table(out) == {}
        assert urom._c0_bit_identity(out, emitted, lib.stem)
    else:
        assert elements != canonical
        assert out.metadata["urom_c1_eps"] > 0.0
        assert urom._emitted_eps_by_table(out)
    assert out.metadata["urom_c1_variant"] == variant
    assert out.metadata["urom_c1_eps_by_table"]
    provenance = out.metadata["urom_provenance"]
    assert provenance["gate_charge_tables"]["source"] == "synthetic curve4"
    assert provenance["static_bars_pct"] == urom.PROVENANCE["static_bars_pct"]


def test_build_emitted_rejects_unlabelled_c1_variant():
    with pytest.raises(ValueError, match="unknown C1 variant"):
        _REAL_BUILD_EMITTED(
            VENDOR, emit="tier2", tuning="off",
            plan={"tuning_conditions": [{}], "method_label": "trap/default"},
            c1_variant="c1")


def test_provenance_merge_keeps_producer_and_base_entries():
    producer = {"gate_charge_tables": {"kind": "source_derived"},
                "static_bars_pct": "producer"}
    merged = urom._merge_provenance(producer)
    assert merged["gate_charge_tables"] == producer["gate_charge_tables"]
    assert merged["static_bars_pct"] == "producer"
    assert merged["dpt_bars_pct"] == urom.PROVENANCE["dpt_bars_pct"]
    assert urom._merge_provenance(None) == urom.PROVENANCE


def test_ramp_numeric_preserves_nonzero_baseline_at_positive_eps():
    xs = [0.0, 1.0, 2.0]
    ys = [8.0, 3.0, 3.5]
    query = np.asarray(xs, dtype=float)
    c0 = urom._ramp_numeric(xs, ys, 0.0, query)
    c1 = urom._ramp_numeric(xs, ys, 0.1, query)
    assert c0[0] == pytest.approx(ys[0])
    assert c1[0] == pytest.approx(ys[0], abs=1e-9)
    assert c0[-1] == pytest.approx(ys[-1])
    assert math.isfinite(float(c1[-1]))


def test_charge_checks_mirror_channel_saturation_and_catch_broken_anchor(
        monkeypatch):
    spec = _channel_spec()
    spec.metadata["urom_channel_basis"] = "tier2"
    xs, sv = urom._channel_sat_ramp(spec)
    assert xs[0] == 0.0
    assert sv[0] == sv[1] and sv[0] > 0.0
    clean = urom._charge_checks(spec, {}, emitted_eps_by_table={})
    row = clean["per_table"]["channel_saturation"]
    assert row["status"] == "measured"
    assert row["eps"] == 0.0
    assert row["charge_fit_error_pct"] < 1e-6
    assert clean["emitted_eps_by_table"] == {}
    rec = _record()
    rec["c1"] = clean
    assert _gates(_observations(), rec)["G6_charge"]["pass"] is True

    original = urom._ramp_numeric

    def dropped_baseline(x, y, eps, query):
        out = np.array(original(x, y, eps, query), dtype=float, copy=True)
        if eps > 0.0:
            out -= float(y[0])
        return out

    monkeypatch.setattr(urom, "_ramp_numeric", dropped_baseline)
    bad = urom._charge_checks(spec, {"transfer_vgs": 0.05},
                              emitted_eps_by_table={"transfer_vgs": 0.05})
    broken_row = bad["per_table"]["channel_saturation"]
    assert broken_row["charge_fit_error_pct"] > 50.0
    rec["c1"] = bad
    assert _gates(_observations(), rec)["G6_charge"]["pass"] is False
    spec.metadata["urom_channel_basis"] = "ann"
    assert "channel_saturation" not in urom._charge_checks(
        spec, {})["per_table"]


def test_charge_checks_emit_monotonicity_for_every_applicable_table():
    checks = urom._charge_checks(_channel_spec(), {})
    assert checks["monotone_q_by_table"] == {
        "q_oss": True, "q_iss": True, "q_rss": True}
    assert checks["all_charge_monotone"] is True
    assert checks["q_zero_anchored_by_table"] == {
        "q_oss": True, "q_iss": True, "q_rss": True}
    assert checks["all_charge_zero_anchored"] is True
    assert checks["applicable_tables"] == ["q_iss", "q_oss", "q_rss"]
    assert checks["not_applicable_tables"] == []


def test_charge_checks_mark_nonmonotone_tables_as_failures():
    spec = _channel_spec()
    spec.q_rss_charge_c = [0.0, 2.0e-12, 1.0e-12]  # synthetic regression
    checks = urom._charge_checks(spec, {})
    assert checks["monotone_q_by_table"]["q_rss"] is False
    assert checks["all_charge_monotone"] is False
    rec = _record()
    rec["c1"] = checks
    gates = _gates(_observations(), rec)
    assert gates["G6_charge"]["pass"] is False
    assert gates["G6_charge"]["detail"]["monotone_q_by_table"]["q_rss"] is False


def test_charge_checks_mark_empty_tables_not_applicable_not_a_pass():
    spec = _synthetic_spec(
        breakpoints_vds=[0.0, 2.0, 10.0],
        breakpoints_id=[[0.0, 4.0, 8.0]],
        vgs_values=[3.0],
        v_th=1.5,
        q_oss_vs_v=[0.0, 5.0, 10.0],
        q_oss_charge_c=[0.0, 1.0e-12, 3.0e-12],
        q_iss_vs_v=[], q_iss_charge_c=[],
        q_rss_vs_v=[], q_rss_charge_c=[],
    )
    checks = urom._charge_checks(spec, {})
    assert checks["monotone_q_by_table"]["q_iss"] is None
    assert checks["monotone_q_by_table"]["q_rss"] is None
    assert checks["per_table"]["q_iss"]["status"] == "not_applicable"
    assert checks["per_table"]["q_rss"]["status"] == "not_applicable"
    assert "q_iss" in checks["not_applicable_tables"]
    assert checks["applicable_tables"] == ["q_oss"]
    assert checks["all_charge_monotone"] is True
    # An empty table never becomes a monotone pass, and the overall status
    # fails closed because a table the emitter may write has no source data.
    assert checks["monotone_q_by_table"]["q_iss"] is not True
    assert checks["status"] == "not_measured"


def test_charge_zero_anchor_applies_to_every_applicable_table():
    spec = _channel_spec()
    spec.q_iss_charge_c = [1.0e-12, 2.0e-12, 4.0e-12]  # synthetic offset
    checks = urom._charge_checks(spec, {})
    assert checks["q_zero_anchored_by_table"]["q_iss"] is False
    assert checks["all_charge_zero_anchored"] is False
    rec = _record()
    rec["c1"] = checks
    assert _gates(_observations(), rec)["G6_charge"]["pass"] is False


def test_g6_applicable_tables_match_the_ann_three_table_contract():
    spec = _channel_spec()
    spec.metadata["urom_channel_basis"] = "ann"
    checks = urom._charge_checks(spec, {})
    assert checks["applicable_tables"] == ["q_iss", "q_oss", "q_rss"]
    assert checks["all_charge_monotone"] is True
    assert checks["all_charge_zero_anchored"] is True


def test_pwl_spec_keeps_fallback_when_curve3_has_no_positive_rds(monkeypatch):
    from pipeline import fitting

    cfg = {
        "device": VENDOR,
        "conditions": {"Vds_sweep_max": 10.0, "Vds_cap_min": 0.0,
                       "Vds_cap_max": 10.0, "Cap_points": 5,
                       "F_meas": 1.0e6},
        "library": "does-not-exist.lib",
        "subcircuit": "x",
        "pins": {},
    }
    c3 = {"curve_id": [0.0, 1.0], "curve_rds_mohm": [0.0, -2.0]}
    curves = {"curve1_output_characteristics": {
        "extracted": {"curve_id_vgs_3_v": [0.0, 1.0, 2.0]}},
        "curve3_rds_on": {"extracted": c3}}
    monkeypatch.setattr(urom, "load_input", lambda vendor: cfg)
    monkeypatch.setattr(urom, "load_output",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            FileNotFoundError()))
    monkeypatch.setattr(urom, "_cap_ref_vds",
                        lambda cond: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(fitting, "_curve",
                        lambda vendor, table: curves[table])
    spec = urom._pwl_spec(VENDOR)
    assert spec.rds_on == 0.005
    assert math.isfinite(spec.rds_on)
    c3["curve_rds_mohm"] = [4.0, 6.0]
    assert urom._pwl_spec(VENDOR).rds_on == pytest.approx(0.005)

def _synthetic_traj(n: int, t_end: float = 14.0e-6,
                    *, start: float = 0.0):
    from types import SimpleNamespace

    t = np.linspace(start, t_end, n)
    vds = np.linspace(0.0, 12.0, n)
    ids = np.linspace(0.0, 13.0, n)
    result = SimpleNamespace(
        log_text="Total iteration count: 100",
        get=lambda key: {"time": t, "I(Lload)": ids}.get(key))
    return SimpleNamespace(
        time=t, input=np.vstack([np.zeros_like(t), vds]),
        output=np.vstack([ids, np.zeros_like(t)]), result=result)


def _synthetic_arm(n: int = 7101, t_end: float = 14.0e-6,
                   current_definition: str = urom.DPT_CURRENT_WITH_DCLAMP
                   ) -> dict:
    t = np.linspace(0.0, t_end, n)
    return {"time": t, "vds": np.linspace(0.0, 12.0, n),
            "ids": np.linspace(0.0, 13.0, n),
            "current_definition": current_definition}


def test_device_current_uses_the_difference_identity_and_explicit_fallback():
    class _FakeResult:
        def __init__(self, traces):
            self.traces = traces

        def get(self, key):
            return self.traces[key]

    traces = {"I(Lload)": np.array([1.0, 2.0, 3.0]),
              "I(Dclamp)": np.array([0.5, 0.5, 0.5])}
    current, definition = urom._dpt_device_current(_FakeResult(traces))
    assert definition == urom.DPT_CURRENT_WITH_DCLAMP
    assert np.allclose(current, [0.5, 1.5, 2.5])

    fallback = {"I(Lload)": np.array([-1.0, 2.0])}
    current, definition = urom._dpt_device_current(_FakeResult(fallback))
    assert definition == urom.DPT_CURRENT_FALLBACK
    assert "fallback" in definition
    assert np.allclose(current, [1.0, 2.0])

    source = _RUN_DPT_ARM_SOURCE
    assert '"current_definition": current_definition' in source
    assert '"ids": device_current' in source


def test_h2_reference_current_uses_the_same_device_identity(monkeypatch):
    import simulation.testbenches.dpt_trajectory as dptmod

    monkeypatch.setattr(dptmod, "_log_iterations", lambda _text: 100)
    plan = {"dpt_t_end": 14.0e-6, "dpt_max_step": 2.0e-9}
    source = inspect.getsource(urom._dpt_common_compare)
    assert "traj.output[0]" in source
    assert 'traj.result.get("I(Lload)")' not in source
    result = urom._dpt_common_compare(
        {"golden": _synthetic_arm(), "urom": _synthetic_arm()},
        _synthetic_traj(7101), plan)
    assert result["status"] == "measured"
    assert result["current_definition"] == urom.DPT_CURRENT_WITH_DCLAMP
    assert result["reference_current_definition"] == (
        urom.DPT_CURRENT_WITH_DCLAMP)
    assert result["current_identity_match"] is True
    assert result["current_definition_by_arm"] == {
        "golden": urom.DPT_CURRENT_WITH_DCLAMP,
        "urom": urom.DPT_CURRENT_WITH_DCLAMP}
    for label in ("golden", "urom"):
        assert result["arms"][label]["current_definition"] == (
            urom.DPT_CURRENT_WITH_DCLAMP)


def test_h2_current_identity_mismatch_fails_closed(monkeypatch):
    import simulation.testbenches.dpt_trajectory as dptmod

    monkeypatch.setattr(dptmod, "_log_iterations", lambda _text: 100)
    plan = {"dpt_t_end": 14.0e-6, "dpt_max_step": 2.0e-9}
    raw = {"golden": _synthetic_arm(),
           "urom": _synthetic_arm(
               current_definition=urom.DPT_CURRENT_FALLBACK)}
    result = urom._dpt_common_compare(raw, _synthetic_traj(7101), plan)
    assert result["status"] == "not_measured"
    assert result["current_identity_match"] is False
    assert "device current definition" in result["reason"]
    assert result["arms"] == {}
    assert result["guard_ok"] is False


def test_h2_reference_trajectory_fails_closed_on_short_capture(monkeypatch):
    import simulation.testbenches.dpt_trajectory as dptmod

    monkeypatch.setattr(dptmod, "_log_iterations", lambda _text: 100)
    short = dptmod.trajectory_completion(
        np.linspace(0.0, 14.0e-6, 7), "log", t_end=14.0e-6, max_step=2.0e-9)
    assert short["completed"] is False
    assert "insufficient_trace_points" in short["reasons"]
    assert short["min_points"] == 3501
    full = dptmod.trajectory_completion(
        np.linspace(0.0, 14.0e-6, 7101), "log", t_end=14.0e-6,
        max_step=2.0e-9)
    assert full["completed"] is True
    late = dptmod.trajectory_completion(
        np.linspace(0.0, 10.0e-6, 7101), "log", t_end=14.0e-6,
        max_step=2.0e-9)
    assert "t_last_below_guard" in late["reasons"]


def test_h2_epc_two_vs_seven_artifact_cannot_produce_a_g3_pass(monkeypatch):
    import simulation.testbenches.dpt_trajectory as dptmod

    monkeypatch.setattr(dptmod, "_log_iterations", lambda _text: 100)
    plan = {"dpt_t_end": 14.0e-6, "dpt_max_step": 2.0e-9}
    raw = {"golden": _synthetic_arm(), "urom": _synthetic_arm(2, 1.4e-5)}
    artifact = urom._dpt_common_compare(raw, _synthetic_traj(7), plan)
    assert artifact["status"] == "not_measured"
    assert artifact["reference"]["completed"] is False
    assert not artifact.get("guard_ok")
    assert artifact["arms"] == {}

    obs = _observations()
    obs["dpt_compare"] = artifact
    obs["dpt"]["urom"]["waveform"]["compare_points"] = 2
    gates = _gates(obs)
    assert gates["G3_dpt_waveform"]["pass"] is False
    assert gates["G3_dpt_waveform"]["status"] == "not_measured"
    assert gates["G3_dpt_waveform"]["detail"]["window_equal"] is False


def test_h2_common_grid_recorded_window_and_point_count(monkeypatch):
    import simulation.testbenches.dpt_trajectory as dptmod

    monkeypatch.setattr(dptmod, "_log_iterations", lambda _text: 100)
    plan = {"dpt_t_end": 14.0e-6, "dpt_max_step": 2.0e-9}
    raw = {"golden": _synthetic_arm(), "urom": _synthetic_arm()}
    result = urom._dpt_common_compare(raw, _synthetic_traj(7101), plan)
    assert result["status"] == "measured" and result["guard_ok"] is True
    assert result["window_equal"] is True
    assert result["compare_grid_step_s"] <= 2.0e-9 * 1.001
    assert result["compare_grid_step_s"] == pytest.approx(
        (result["window_s"][1] - result["window_s"][0])
        / (result["compare_points"] - 1))
    assert result["min_compare_points"] == 3501
    assert result["compare_points"] >= result["min_compare_points"]
    assert set(result["endpoints"]) == {"reference", "golden", "urom"}
    for label in ("golden", "urom"):
        assert result["arms"][label]["compare_points"] == result[
            "compare_points"]
        assert result["arms"][label]["compare_grid_step_s"] == result[
            "compare_grid_step_s"]
        assert result["arms"][label]["vds_rel_rmse_pct"] is not None
    assert "control" in result["arms"]["golden"]["role"]

    raw["urom"] = _synthetic_arm(7101, 1.0e-5)
    unequal = urom._dpt_common_compare(raw, _synthetic_traj(7101), plan)
    assert unequal["window_equal"] is False
    assert unequal["status"] == "not_measured"
    assert not unequal.get("guard_ok")


def test_h3_tuning_off_is_not_run_with_cached_completion_flags():
    conditions = [{"completed": True}, {"completed": False}]
    bounds = {"O_GD": (0.0, 1.0)}
    off = urom._new_tuning_record("off", conditions, bounds)
    assert off["status"] == off["verdict"] == "not_run"
    assert off["theta"] == {"A_GD": None, "O_GD": None,
                            "A_GS": None, "A_RGI": None}
    assert off["golden_completion_flags"] == [True, False]
    assert "not run" in off["reason"]
    assert off["verdict"] != "infeasible"
    on = urom._new_tuning_record("on", conditions, bounds)
    assert on["status"] == "pending" and on["verdict"] == "reject"
    assert on["golden_completion_flags"] == [False, False]

    rec = _record()
    gates = _gates(rec=rec)
    assert gates["G6_charge"]["detail"]["tuning_verdict"] == "not_run"
    assert gates["G6_charge"]["detail"]["tuning_status"] == "not_run"
    assert gates["G6_charge"]["pass"] is True


def test_h4_ann_gate_probes_the_emitted_channel_expression():
    from approximation.ann_channel import AnnChannelFit
    from models.ann_netlist_builder import emit_ann_channel

    coef = {"w1": [[0.0, 0.0]], "b1": [0.0], "w2": [0.0], "b2": 0.0,
            "vgs_center": 3.0, "vgs_scale": 3.0,
            "vds_center": 5.0, "vds_scale": 5.0,
            "alpha": 10.0, "guard_width": 0.1}
    fit = AnnChannelFit(coefficients=coef, v_th=1.0, v_sat=2.0,
                        activation="isru", neurons=1, loss_weights={},
                        train_range={}, out_of_range_check="pass",
                        diagnostics={})
    expr = emit_ann_channel(fit)
    lib = f"* UROM\nG_CH di ss VALUE = {{({expr})}}\n"
    spec = _channel_spec()
    probe = urom._ann_channel_artifact_probe(
        lib, {"channel_fit": fit}, spec)
    assert probe["status"] == "measured"
    emitted = urom._ann_channel_from_lib(lib)
    assert probe["expression_sha256"] == hashlib.sha256(
        emitted.encode("ascii")).hexdigest()
    assert probe["cutoff_vgs"] == pytest.approx(0.7)
    assert probe["cutoff_ok"] and probe["guard_ok"] and probe["zero_ok"]
    assert probe["tolerance_frac"] == urom.ZERO_CURRENT_FRAC
    missing = urom._ann_channel_artifact_probe(
        "* no channel line\n", {"channel_fit": fit}, spec)
    assert missing["status"] == "not_measured"


def test_h4_ann_g2_requires_probe_and_reports_sign_fractions():
    ann = {"record": {"out_of_range_check": "pass", "activation": "isru"}}
    no_probe = urom._derivative_checks(
        _conducting_rows(), _channel_spec(), ann, "ann")
    assert no_probe["pass"] is False
    assert "ann_channel_probe_missing" in no_probe["reason"]
    good_probe = {"status": "measured", "cutoff_ok": True, "guard_ok": True,
                  "zero_ok": True, "cutoff_vgs": 0.7,
                  "cutoff_i": 1.0e-6, "guard_i": 2.0e-6,
                  "vgs_zero_i": 0.0, "i_high": 1.0,
                  "expression_sha256": "abc"}
    checked = urom._derivative_checks(
        _conducting_rows(), _channel_spec(), ann, "ann",
        ann_probe=good_probe)
    assert checked["pass"] is True
    assert checked["ann_channel_probe"] == good_probe
    assert 0.0 <= checked["gm_nonnegative_fraction"] <= 1.0
    assert 0.0 <= checked["gds_negative_fraction"] <= 1.0
    assert checked["gm_negative_fraction"] == pytest.approx(0.0)


def _hybrid_probe_fit():
    """Simple signed residual fit; the expression under test is the emitted one."""
    from approximation.ann_residual import AnnResidualFit

    return AnnResidualFit(
        coefficients={"w1": [[0.2, 0.05]], "b1": [0.0],
                      "w2": [-1.0e-3], "b2": 0.0},
        v_th=1.5, activation="isru", neurons=1, alpha=10.0)


def test_hybrid_g2_probes_emitted_gann_zero_cutoff_and_signed_on_state():
    from models.ann_netlist_builder import emit_ann_residual

    fit = _hybrid_probe_fit()
    expr = emit_ann_residual(fit)
    lib = f"* UROM\nG_ANN di ss VALUE = {{{expr}}}\n"
    spec = _channel_spec()
    probe = urom._hybrid_residual_artifact_probe(
        lib, {"residual_fit": fit}, spec)
    assert probe["status"] == "measured"
    assert probe["label"] == urom.HYBRID_RESIDUAL_PROBE_LABEL
    assert probe["expression"] == expr
    assert probe["expression_sha256"] == hashlib.sha256(
        expr.encode("ascii")).hexdigest()
    assert probe["vds_zero_i"] == pytest.approx(0.0, abs=1e-15)
    assert probe["zero_vds_ok"] is True
    assert probe["cutoff_vgs"] < fit.v_th
    assert probe["cutoff_ok"] is True
    assert probe["on_state_finite"] is True
    assert probe["i_on"] < 0.0  # signed residual, not a positive-only model
    checked = urom._derivative_checks(
        _conducting_rows(), spec, {}, "hybrid", hybrid_probe=probe)
    assert checked["pass"] is True
    assert checked["hybrid_residual_probe"] == probe
    assert "emitted G_ANN residual probe measured" in checked["reason"]


def test_hybrid_g2_rejects_tampered_gann_zero_cutoff_and_unevaluable_expr():
    fit = _hybrid_probe_fit()
    spec = _channel_spec()
    no_vds = ("G_ANN di ss VALUE = "
              "{((V(gi,ss)-1.5)*0.001)}\n")
    probe = urom._hybrid_residual_artifact_probe(
        no_vds, {"residual_fit": fit}, spec)
    assert probe["status"] == "measured"
    assert probe["zero_vds_ok"] is False
    assert probe["cutoff_ok"] is False
    checked = urom._derivative_checks(
        _conducting_rows(), spec, {}, "hybrid", hybrid_probe=probe)
    assert checked["pass"] is False
    assert checked["hybrid_residual_probe"]["status"] == "measured"
    assert "hybrid_residual_zero_vds" in checked["reason"]
    assert "hybrid_residual_cutoff" in checked["reason"]

    no_gate = "G_ANN di ss VALUE = {((V(di,ss))*(-1e-3))}\n"
    probe = urom._hybrid_residual_artifact_probe(
        no_gate, {"residual_fit": fit}, spec)
    assert probe["status"] == "measured"
    assert probe["cutoff_ok"] is False
    checked = urom._derivative_checks(
        _conducting_rows(), spec, {}, "hybrid", hybrid_probe=probe)
    assert checked["pass"] is False
    assert "hybrid_residual_cutoff" in checked["reason"]

    unevaluable = "G_ANN di ss VALUE = {(1/(V(di,ss)-V(di,ss)))}\n"
    probe = urom._hybrid_residual_artifact_probe(
        unevaluable, {"residual_fit": fit}, spec)
    assert probe["status"] == "not_measured"
    checked = urom._derivative_checks(
        _conducting_rows(), spec, {}, "hybrid", hybrid_probe=probe)
    assert checked["pass"] is False
    assert checked["hybrid_residual_probe"]["status"] == "not_measured"
    assert checked["hybrid_residual_probe"]["reason"] == probe["reason"]


def test_hybrid_g2_never_passes_without_emitted_gann_probe():
    fit = _hybrid_probe_fit()
    spec = _channel_spec()
    missing = urom._hybrid_residual_artifact_probe(
        "* no G_ANN line\n", {"residual_fit": fit}, spec)
    assert missing["status"] == "not_measured"
    assert missing["expression"] is None
    assert missing["reason"]
    for probe in (None, missing):
        checked = urom._derivative_checks(
            _conducting_rows(), spec, {}, "hybrid", hybrid_probe=probe)
        assert checked["pass"] is False
        assert "hybrid_residual_probe_missing" in checked["reason"]
        detail = checked["hybrid_residual_probe"]
        assert detail["label"] == urom.HYBRID_RESIDUAL_PROBE_LABEL
    src = inspect.getsource(urom.run_unified)
    assert "_hybrid_residual_artifact_probe(" in src
    assert "hybrid_probe=hybrid_probe" in src


def test_h5_tier2_regression_is_a_diagnostic_gate_never_a_verdict(
        monkeypatch, tmp_path):
    golden = {k: 10.0 for k in urom._TIMING_METRICS}
    tier2 = {k: 20.0 for k in urom._TIMING_METRICS}
    _write_dpt_metrics(monkeypatch, tmp_path,
                       {"golden_metrics": golden, "tier2_metrics": tier2})
    _write_tier3_validation(monkeypatch, tmp_path, golden)
    assert urom._timing_bars(VENDOR) == golden
    regression_rows, regression_ok = urom._timing_score_rows(
        {k: 10.0 for k in golden}, urom._tier2_regression_bars(VENDOR))
    assert regression_ok is False
    assert regression_rows["td_on_ns"]["pass"] is False

    obs = _observations()
    obs["timing_energy"]["tier2_regression"] = {
        "status": "measured", "rows": regression_rows,
        "all_within_band": False, "used_for_verdict": False,
        "diagnostic_only": True,
    }
    gates = _gates(obs)
    assert gates["G4_timing_energy"]["pass"] is True
    assert gates["G4_tier2_regression"]["pass"] is False
    assert gates["G4_tier2_regression"]["diagnostic_only"] is True
    diag = gates["G4_tier2_regression"]["detail"]
    assert diag["used_for_verdict"] is False
    assert diag["all_within_band"] is False
    assert gates["timing_all_pass"] is True
    assert gates["all_pass"] is True


def test_h6_charge_checks_mirror_emitted_ann_widths_for_all_tables():
    spec = _channel_spec()
    widths = {"q_oss": 0.02, "transfer_vgs": 0.02}
    spec.metadata["urom_emitted_eps_by_table"] = widths
    spec.metadata["urom_channel_basis"] = "ann"
    checks = urom._charge_checks(spec, widths)
    assert checks["emitted_eps_by_element"] == {
        "q_oss": 0.02, "q_iss": 0.02, "q_rss": 0.02}
    for name in ("q_oss", "q_iss", "q_rss"):
        row = checks["per_table"][name]
        assert row["eps"] == pytest.approx(0.02)
        assert row["charge_fit_error_pct"] is not None
        assert row["q_zero_residual_c"] is not None
        assert row["q_vmax_residual_c"] is not None
    assert checks["charge_fit_error_pct"] <= urom.CHARGE_BUDGET_PCT


def test_h6_artifact_q_residuals_read_the_emitted_expressions():
    from approximation.smoother import delta_ramp_chain

    spec = _channel_spec()
    eps = 0.02
    terms = {"q_oss": "V(di,ss)", "q_iss": "V(gi,ss)",
             "q_rss": "V(gi,di)"}
    element = {"q_oss": "C_QOSS", "q_iss": "C_ISS", "q_rss": "C_RSS"}
    tables = {"q_oss": (spec.q_oss_vs_v, spec.q_oss_charge_c),
              "q_iss": (spec.q_iss_vs_v, spec.q_iss_charge_c),
              "q_rss": (spec.q_rss_vs_v, spec.q_rss_charge_c)}
    nodes = {"q_oss": "di ss", "q_iss": "gi ss", "q_rss": "gi di"}
    lines = []
    for name, (vv, qq) in tables.items():
        expr = delta_ramp_chain(terms[name], vv, qq, eps)
        lines.append(
            f"{element[name]} {nodes[name]} Q = {{{expr}}}")
    artifact = urom._emitted_charge_residuals(
        "\n".join(lines), spec, {k: eps for k in terms})
    assert artifact["status"] == "measured"
    assert artifact["mirror_match_all"] is True
    assert set(artifact["applicable_elements"]) == set(terms)
    for name in terms:
        row = artifact["elements"][name]
        assert abs(row["q_zero_residual_c"]) < 1e-18
        assert abs(row["q_vmax_residual_pct"]) < urom.CHARGE_BUDGET_PCT

    c0_expr = delta_ramp_chain(terms["q_oss"], *tables["q_oss"], 0.0)
    c0 = f"C_QOSS di ss Q = {{{c0_expr}}}"
    c0 += "\nC_ISS gi ss 1.500000e-09\nC_RSS gi di 5.000000e-11"
    c0_artifact = urom._emitted_charge_residuals(
        c0, spec, {"q_oss": 0.0, "q_iss": 0.0, "q_rss": 0.0})
    assert c0_artifact["applicable_elements"] == ["q_oss"]
    assert c0_artifact["status"] == "measured"


def test_h7_dpt_high_vi_provenance_never_claims_empty_indices():
    failed = {"provenance": {"dpt_high_vi": {
        "source": "dpt_trajectory", "status": "no_high_v_samples",
        "reason": "no DPT points with Vds > 100 V", "sample_indices": [],
        "n_points": 0, "sample_selection": None}}}
    record = urom._dpt_high_vi_provenance(VENDOR, failed, "ann")
    assert record["source"] == "not_used (fallback: Curve 1 only)"
    assert record["status"] == "no_high_v_samples"
    assert record["reason"] == "no DPT points with Vds > 100 V"
    assert record["n_points"] == 0 and record["dpt_informed"] is False
    ok = {"provenance": {"dpt_high_vi": {
        "source": "dpt_trajectory", "status": "ok",
        "sample_indices": [4, 8, 12], "n_points": 3,
        "sample_selection": "Vds > 100 V, stride 2", "reason": None}}}
    used = urom._dpt_high_vi_provenance(VENDOR, ok, "ann")
    assert used["source"] == "dpt_trajectory"
    assert used["sample_indices"] == [4, 8, 12]
    assert used["sample_selection"] == "Vds > 100 V, stride 2"
    assert used["dpt_informed"] is True
    tier2 = urom._dpt_high_vi_provenance(VENDOR, ok, "tier2")
    assert tier2["dpt_informed"] is False
    assert tier2["source"] == "not_applicable"


def test_h8_clock_spread_is_required_on_both_clocks_and_arms():
    speed = {"golden": _arm(2.0, 2.2), "urom": _arm(1.0, 1.1)}
    clocks = urom._clock_spread_record(speed)
    assert clocks["complete"] is True
    for group in ("median", "min", "max"):
        for clock in ("solver", "wall"):
            assert set(clocks[group][clock]) == {"golden", "urom"}
            assert all(v is not None for v in clocks[group][clock].values())
    speed["urom"].pop("wall_min_s")
    missing = urom._clock_spread_record(speed)
    assert missing["complete"] is False
    assert "min.wall.urom" in missing["missing"]

    obs = _observations()
    obs["clock_spread"] = missing
    obs["equal_sandbox"]["clock_spread"] = missing
    gates = _gates(obs)
    assert gates["G0"]["pass"] is False
    assert gates["G5_speed"]["pass"] is False
    detail = gates["G5_speed"]["detail"]
    assert detail["clock_spread_complete"] is False
    assert detail["clock_spread"]["median"]["wall"]["golden"] is not None
    assert detail["clock_spread"]["min"]["solver"]["urom"] is not None


# ---------------------------------------------------------------------------
# hybrid UROM basis integration (offline: no bench, no live simulation)
# ---------------------------------------------------------------------------
def test_hybrid_basis_is_explicit_and_canonical_c0_default_is_unchanged():
    assert set(urom.EMIT_MODES) == {"ann", "hybrid", "tier2"}
    assert "hybrid" in registry.UROM_BASES
    assert registry.UROM_BASES[0] == "tier2"
    # One independent golden G4 bar table for every basis: no basis-specific
    # self-comparison column exists any more.
    assert not hasattr(urom, "TIMING_BAR_COLUMN")
    assert urom.GOLDEN_DPT_BAR_SOURCE.endswith(
        "tier3_validation.json#dpt_comparison.<metric>.golden")
    sig = inspect.signature(urom.run_unified)
    assert sig.parameters["emit"].default == "tier2"
    for fn in (registry.fit, registry.validate):
        assert inspect.signature(fn).parameters["basis"].default == "tier2"
    assert urom._is_canonical_config(
        emit="tier2", method="trap/default", tuning="off",
        c1_variant=None) is True
    assert urom._is_canonical_config(
        emit="hybrid", method="trap/default", tuning="off",
        c1_variant=None) is False
    with pytest.raises(ValueError, match="requires emit"):
        urom.run_unified(VENDOR, emit="hybrid", c1_variant="tier2-c1")


def test_registry_routes_explicit_hybrid_basis(monkeypatch):
    calls: list = []
    stub = ModuleType("pipeline.unified_rom")

    def run_unified(vendor: str, **kwargs) -> dict:
        calls.append((vendor, kwargs))
        return {"emitted_lib": "reports/models/stub.lib"}

    stub.run_unified = run_unified
    monkeypatch.setitem(sys.modules, "pipeline.unified_rom", stub)
    assert registry.fit(VENDOR, 3, "urom", basis="hybrid") == Path(
        "reports/models/stub.lib")
    assert registry.validate(VENDOR, 3, "urom", basis="hybrid")["emitted_lib"] \
        == "reports/models/stub.lib"
    assert calls == [(VENDOR, {"variant": "urom", "emit": "hybrid"})] * 2
    with pytest.raises(ValueError, match="unknown UROM basis"):
        registry.fit(VENDOR, 3, "urom", basis="c0")


def test_hybrid_keyed_emission_never_clobbers_canonical_or_ann(
        monkeypatch, tmp_path):
    from models.netlist_builder import build_tier2_subcircuit
    from pipeline import ann_fit as ann_fit_mod

    from approximation.ann_residual import AnnResidualFit

    channel, charge = _synthetic_ann_fits()
    residual = AnnResidualFit(
        coefficients={"w1": [[0.1, 0.01]], "b1": [0.0],
                      "w2": [0.5], "b2": 0.0},
        v_th=3.0, activation="isru", neurons=1,
    )
    # Minimal hybrid fit payload: residual current only, no ANN charge net.
    fit_payload = {
        "residual_fit": residual,
        "record": {"basis": "hybrid"},
    }
    monkeypatch.setattr(ann_fit_mod, "hybrid_ann_fit_for_vendor",
                        lambda vendor: fit_payload, raising=False)

    monkeypatch.setattr(urom.registry, "ROOT", tmp_path)
    monkeypatch.setattr(urom, "_pwl_spec",
                        lambda vendor: _synthetic_ann_spec())
    monkeypatch.setattr(urom, "_theta_bounds", lambda s, v: {})
    monkeypatch.setattr(urom, "_normalize_unified_metadata",
                        lambda meta, vendor: None)

    kw = dict(method="trap/default", tuning="off", c1_variant=None)
    canonical_rec = urom._record_path_for_config(VENDOR, emit="tier2", **kw)
    ann_rec = urom._record_path_for_config(VENDOR, emit="ann", **kw)
    hybrid_rec = urom._record_path_for_config(VENDOR, emit="hybrid", **kw)
    canonical_lib = urom._lib_path_for_config(VENDOR, emit="tier2", **kw)
    ann_lib = urom._lib_path_for_config(VENDOR, emit="ann", **kw)
    hybrid_lib = urom._lib_path_for_config(VENDOR, emit="hybrid", **kw)
    assert canonical_rec == registry.unified_validation_path(VENDOR)
    assert canonical_lib.name == f"{VENDOR}_PIECSPICE_UROM.lib"
    assert ann_rec.name == "unified_ann_trap-default.json"
    assert hybrid_rec.name == "unified_hybrid_trap-default.json"
    assert hybrid_lib.name == (
        f"{VENDOR}_PIECSPICE_UROM_hybrid_trap-default.lib")
    assert len({canonical_rec, ann_rec, hybrid_rec}) == 3
    assert len({canonical_lib, ann_lib, hybrid_lib}) == 3
    assert urom._config_key(emit="hybrid", **kw) == "hybrid_trap-default"

    sentinels = (canonical_rec, ann_rec, canonical_lib, ann_lib)
    for path in sentinels:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("sentinel", encoding="ascii")
    before = {path: path.read_bytes() for path in sentinels}

    plan = {
        "method_label": "trap/default",
        "tuning_conditions": [
            {"name": "synthetic", "completed": False, "condition": {}}],
    }
    lib, _spec, recorded, tuning = _REAL_BUILD_EMITTED(
        VENDOR, emit="hybrid", tuning="off", plan=plan)
    assert lib == hybrid_lib
    assert recorded is fit_payload
    assert set(recorded) == {"residual_fit", "record"}
    assert tuning["verdict"] == "not_run"
    text = lib.read_text(encoding="ascii")
    assert urom._subckt_name_text(text) == lib.stem
    assert f".ENDS {lib.stem}" in text
    assert urom._identity_header_hint(text) == (
        "hybrid", "trap/default", "off", "C0")
    # The real production builder ran: strip only the UROM identity comment,
    # then A/B every emitted line against closed C0 at the same subckt name.
    emitted_lines = urom._strip_identity_header(text).splitlines()
    base = build_tier2_subcircuit(_spec, name=lib.stem).splitlines()
    i_sub_b = next(i for i, line in enumerate(base)
                   if line.startswith(".SUBCKT "))
    i_sub_e = next(i for i, line in enumerate(emitted_lines)
                   if line.startswith(".SUBCKT "))
    i_ch_e = next(i for i, line in enumerate(emitted_lines)
                  if line.startswith("G_CH "))
    gann = [i for i, line in enumerate(emitted_lines)
            if line.startswith("G_ANN ")]
    assert len(gann) == 1 and gann[0] == i_ch_e + 3  # blank, comment, G_ANN
    assert (emitted_lines[i_sub_e:i_ch_e + 1]
            + emitted_lines[gann[0] + 1:] == base[i_sub_b:])
    assert "ANN Q nets" not in text
    assert urom._emitted_eps_by_table(_spec) == {}

    lib2, *_ = _REAL_BUILD_EMITTED(
        VENDOR, emit="hybrid", tuning="off", plan=plan)
    assert lib2 == lib
    for path in sentinels:
        assert path.read_bytes() == before[path]

    record = {
        "emit": "hybrid", "channel_basis": "hybrid", "tuning_mode": "off",
        "c1_variant": None, "method": "trap/default",
        "record_path": str(hybrid_rec), "emitted_lib": str(lib),
        "emitted_lib_sha256": hashlib.sha256(lib.read_bytes()).hexdigest(),
    }
    identity = urom._verify_artifact_identity(
        record, emit="hybrid", method="trap/default", tuning="off",
        c1_variant=None)
    assert identity["status"] == "match" and identity["match"] is True
    lib.write_text(text + "* tampered\n", encoding="ascii")
    tampered = urom._verify_artifact_identity(
        record, emit="hybrid", method="trap/default", tuning="off",
        c1_variant=None)
    assert tampered["status"] == "mismatch"


def test_dpt_bars_are_device_current_requirements():
    trap = urom.DPT_BARS_PCT["trap/default"]
    assert trap == dict(urom.ACCEPTANCE_POLICY.dpt_bars_pct["trap/default"])
    assert trap == {
        "irfs7730pbf": (10.0, 5.0), "csd16412q5a": (5.583, 5.0),
        "epc2216": (4.894, 5.0), "cpm2-1200": (3.0, 5.0)}
    assert urom.DPT_BARS_PCT["gear-both"] == {
        "irfs7730pbf": (12.771, 5.0), "csd16412q5a": (5.467, 5.0),
        "epc2216": (5.075, 5.0), "cpm2-1200": (4.122, 5.0)}
    assert urom.STATIC_BARS_PCT == {
        "irfs7730pbf": 7.0683, "csd16412q5a": 8.4935,
        "epc2216": 2.0394, "cpm2-1200": 3.4287}

    obs = _observations()
    obs["dpt"]["urom"]["waveform"]["id_rel_rmse_pct"] = 5.0
    assert _gates(obs)["G3_dpt_waveform"]["pass"] is True
    obs["dpt"]["urom"]["waveform"]["id_rel_rmse_pct"] = 5.1
    assert _gates(obs)["G3_dpt_waveform"]["pass"] is False

    cpm2 = _record(vendor="cpm2-1200")
    obs_cpm2 = _observations()
    for arm in ("golden", "urom"):
        obs_cpm2["dpt"][arm]["waveform"].update(
            vds_rel_rmse_pct=3.0, id_rel_rmse_pct=5.0)
    assert _gates(obs_cpm2, cpm2)["G3_dpt_waveform"]["pass"] is True
    for arm in ("golden", "urom"):
        obs_cpm2["dpt"][arm]["waveform"]["vds_rel_rmse_pct"] = 3.1
    assert _gates(obs_cpm2, cpm2)["G3_dpt_waveform"]["pass"] is False


def test_canonical_c0_irfs_measurement_passes_new_bar_but_fails_old(monkeypatch):
    """Measured canonical C0 (9.999276, 0.575138) pct against the real gate."""
    measured_vds, measured_id = 9.999276, 0.575138
    obs = _observations()
    for arm in ("golden", "urom"):
        obs["dpt"][arm]["waveform"].update(
            vds_rel_rmse_pct=measured_vds, id_rel_rmse_pct=measured_id)

    g3 = _gates(obs)["G3_dpt_waveform"]
    assert g3["pass"] is True
    assert g3["status"] == "measured"
    assert g3["detail"]["urom"]["bars_vds_id_pct"] == (10.0, 5.0)
    assert g3["detail"]["urom"]["vds_rel_rmse_pct"] == measured_vds
    assert g3["detail"]["urom"]["id_rel_rmse_pct"] == measured_id

    copied_dpt = {method: dict(bars) for method, bars in
                  urom.ACCEPTANCE_POLICY.dpt_bars_pct.items()}
    copied_dpt["trap/default"][VENDOR] = (8.529, 0.55)
    old_policy = dataclasses.replace(urom.ACCEPTANCE_POLICY,
                                     dpt_bars_pct=copied_dpt)
    monkeypatch.setattr(urom, "ACCEPTANCE_POLICY", old_policy)
    old = _gates(obs)["G3_dpt_waveform"]
    assert old["pass"] is False
    assert old["detail"]["urom"]["bars_vds_id_pct"] == (8.529, 0.55)


def test_hybrid_timing_bars_come_from_the_independent_golden_dpt(
        monkeypatch, tmp_path):
    golden = {k: 10.0 for k in urom._TIMING_METRICS}
    tier2 = {k: 20.0 for k in urom._TIMING_METRICS}
    _write_dpt_metrics(monkeypatch, tmp_path,
                       {"golden_metrics": golden, "tier2_metrics": tier2})
    _write_tier3_validation(monkeypatch, tmp_path, golden)
    # Basis-independent: hybrid/ann/tier2 all score the same golden DPT table.
    assert urom._timing_bars(VENDOR) == golden
    assert urom._tier2_regression_bars(VENDOR) == tier2
    source = urom._timing_bars_source()
    assert "tier3_validation.json#dpt_comparison.<metric>.golden" in source
    assert "independent" in source
    assert urom.PROVENANCE["timing_energy_bars"]["golden_column"] == (
        "dpt_comparison.<metric>.golden")
    source_text = inspect.getsource(urom.run_unified)
    assert '"timing_bar_column": "dpt_comparison.<metric>.golden"' in (
        source_text)
    assert '"timing_bar_source": _timing_bars_source()' in source_text
    assert '"tier2_regression": {' in source_text
    assert '"used_for_verdict": False' in source_text


def test_cli_explicit_hybrid_basis_is_displayed_and_passed(monkeypatch):
    calls: list = []
    record = _cli_record()
    record["emit"] = "hybrid"
    record["channel_basis"] = "hybrid"

    def validate(vendor, tier, variant, basis="tier2"):
        calls.append((vendor, tier, variant, basis))
        return record

    monkeypatch.setattr(registry, "validate", validate)
    result = _runner.invoke(
        app, ["validate", VENDOR, "--tier", "3", "--variant", "urom",
              "--basis", "hybrid"])
    assert result.exit_code == 0, result.output
    assert calls == [(VENDOR, 3, "urom", "hybrid")]
    assert "basis=hybrid" in _flat(result.output)


def _overridden_policy(tmp_path):
    raw = yaml.safe_load(ACCEPTANCE_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["static"]["mean_peak_rel_rmse_pct_max"][VENDOR]["value"] = 5.0
    raw["static"]["worst_endpoint_pct"]["value"] = 1.0
    raw["dpt"]["trap/default"][VENDOR]["value"] = [1.0, 0.01]
    raw["timing"]["band_rel"]["value"] = 0.5
    raw["charge"]["budget_pct"]["value"] = 0.1
    raw["speed"]["min_solver_speedup"]["value"] = 3.0
    path = tmp_path / "acceptance.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="ascii")
    return load_acceptance(path)


def test_config_override_moves_gate_bars_without_code_edits(monkeypatch,
                                                            tmp_path):
    monkeypatch.setattr(urom, "ACCEPTANCE_POLICY", _overridden_policy(tmp_path))
    gates = _gates(_observations())
    g1 = gates["G1_static"]
    assert g1["detail"]["bar_pct"] == 5.0
    assert g1["detail"]["band_pp"] == urom.G1_BAND_PP  # alias unchanged
    assert g1["detail"]["endpoint_bar_pct"] == 1.0
    assert g1["pass"] is False
    g3 = gates["G3_dpt_waveform"]
    assert g3["detail"]["golden"]["bars_vds_id_pct"] == (1.0, 0.01)
    assert g3["pass"] is False
    assert gates["G6_charge"]["detail"]["budget_pct"] == 0.1
    assert gates["G5_speed"]["detail"]["effect_min"] == 3.0
    assert gates["G5_speed"]["pass"] is False
    assert urom._timing_band(20.0) == pytest.approx((10.0, 30.0))


def test_urom_record_carries_policy_manifest_and_verification(monkeypatch,
                                                              tmp_path):
    """A fully stubbed run checks record assembly without any simulator."""
    monkeypatch.setattr(urom, "ROOT", tmp_path)
    monkeypatch.setattr(registry, "ROOT", tmp_path)
    config_dir = tmp_path / "config" / "inputs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / f"{VENDOR}.yaml"
    config_path.write_text("x: 1\n", encoding="ascii")
    vendor_lib = tmp_path / f"{VENDOR}.spi"
    vendor_lib.write_text("* vendor input\n", encoding="ascii")
    curve_dir = tmp_path / "config" / "outputs" / VENDOR
    curve_dir.mkdir(parents=True)
    curve_path = curve_dir / "curve1.yaml"
    curve_path.write_text("points: [0, 1]\n", encoding="ascii")
    lib = tmp_path / "reports" / "models" / f"{VENDOR}_PIECSPICE_UROM.lib"
    lib.parent.mkdir(parents=True)
    lib.write_text(".SUBCKT X d g s\n.ENDS\n", encoding="ascii")
    monkeypatch.setattr(urom, "load_input", lambda vendor: {
        "library": str(vendor_lib), "dpt": {}, "pins": {}})

    spec = _synthetic_spec()
    spec.metadata["urom_provenance"] = {"source": "synthetic probe"}
    tuning_record = {"mode": "off", "status": "not_run",
                     "verdict": "not_run", "conditions": [], "bounds": {}}
    plan = {
        "method_label": "trap/default", "options_lines": [],
        "temperature": 25.0, "dpt_max_step": 2.0e-9, "repeats": 3,
        "speed": {"n_pulses": urom.SPEED_N_PULSES, "max_step": None,
                  "repeats": 3, "end_time_s": 1.0, "timeout_s": 90.0},
    }
    monkeypatch.setattr(urom, "_bench_plan", lambda *a, **k: plan)
    monkeypatch.setattr(urom, "_build_emitted",
                        lambda *a, **k: (lib, spec, {}, tuning_record))
    monkeypatch.setattr(urom, "_arm_specs", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(urom, "_dpt_deck", lambda *a, **k: "dpt deck")
    monkeypatch.setattr(urom, "_speed_deck", lambda *a, **k: "speed deck")
    monkeypatch.setattr(urom, "_deck_identity", lambda *a, **k: {
        "deck_diff_only_dut": True, "effective_options_dumped": True,
        "tolerance_overrides": [], "libraries_inlined_both_arms": True,
        "temperature_both_arms": True, "effective_options": {}})
    speed = {"golden": _arm(2.0, 2.2), "urom": _arm(1.0, 1.1), "repeats": 3}
    speed["urom"]["log_tail"] = None
    monkeypatch.setattr(urom, "_run_interleaved_speed",
                        lambda *a, **k: speed)
    monkeypatch.setattr(urom, "capture_trajectory", lambda *a, **k: object())
    waveform = _observations()["dpt"]["golden"]["waveform"]

    def fake_dpt_arm(*args, **kwargs):
        return {"completed": True, "waveform": dict(waveform),
                "metrics": {k: 10.0 for k in urom._TIMING_METRICS},
                "_raw": {}, "log_tail": None}

    monkeypatch.setattr(urom, "_run_dpt_arm", fake_dpt_arm)
    compare = dict(_observations()["dpt_compare"])
    compare.update({"current_definition": "synthetic probe",
                    "current_definition_by_arm": {}, "arms": {}})
    monkeypatch.setattr(urom, "_dpt_common_compare", lambda *a, **k: compare)
    monkeypatch.setattr(urom, "_static_bench",
                        lambda *a, **k: (dict(_observations()["static"]), []))
    monkeypatch.setattr(urom, "_derivative_checks",
                        lambda *a, **k: {"pass": True, "status": "measured"})
    monkeypatch.setattr(urom, "_c0_bit_identity", lambda *a, **k: True)
    bars = {k: 10.0 for k in urom._TIMING_METRICS}
    monkeypatch.setattr(urom, "_timing_bars", lambda vendor: bars)
    monkeypatch.setattr(urom, "_tier2_regression_bars", lambda vendor: bars)

    record = urom.run_unified(VENDOR)
    policy = record["acceptance_policy"]
    assert policy["schema"] == "piecespice.acceptance/1.0"
    assert Path(policy["path"]).is_file()
    assert len(policy["sha256"]) == 64
    assert policy["sources"]["static.worst_endpoint_pct"] == (
        "recorded_baseline")
    assert policy["values"]["speed_min_solver_speedup"] == (
        urom.SPEED_EFFECT_MIN)

    manifest = record["artifact_manifest"]
    assert set(manifest["inputs"]) == set(record["record_inputs"])
    assert manifest["lib"]["sha256"] == hashlib.sha256(
        lib.read_bytes()).hexdigest()
    assert manifest["lib"]["bytes"] == len(lib.read_bytes())
    assert record["input_hashes"]["emitted_lib"] == manifest["lib"]["sha256"]
    assert record["environment"]
    assert record["artifact_manifest_verified"] == {
        "ok": True, "mismatches": []}
    assert not any("record" in label for label in manifest["inputs"])

    written = json.loads((tmp_path / record["record_path"]).read_text(
        encoding="ascii"))
    assert written["acceptance_policy"] == policy
    assert written["artifact_manifest"] == manifest
    assert written["artifact_manifest_verified"] == {
        "ok": True, "mismatches": []}
