"""Offline tests for the T4 unified one-ROM registry, record and CLI.

Structural integers/times below are *probes* used to exercise control flow;
they are not PieceSpice measurements.  Every comparison bar and bench constant
is asserted against the pinned values in ``local://unified_rom_spec.md``
(spec sections 1.3, 1.4, 4.4, 4.6 and 6).  No simulator is started: an autouse
guard fails if a live bench runner is reached, so this module is safe with
``PIECESPICE_LIVE=0``.
"""
from __future__ import annotations

import inspect
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from typer.testing import CliRunner

from cli import main as cli_main
from cli.main import app
from common.types import PWLModelSpec
from pipeline import registry, unified_rom as urom
_REAL_BUILD_EMITTED = urom._build_emitted
VENDOR = "irfs7730pbf"
_runner = CliRunner()
GATE_NAMES = ("G0", "G1_static", "G2_derivative", "G3_dpt_waveform",
              "G4_timing_energy", "G5_speed", "G6_charge")


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
def _arm(solver: float, wall: float, *, complete: bool = True) -> dict:
    return {
        "iterations": 100, "points": 1000, "t_last": 1.0,
        "solver_time_s": solver, "wall_time_s": wall,
        "completion": complete,
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
          "compare_points": 1000}
    return {
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
        "timing_energy": {"pass": True, "status": "measured", "rows": {}},
    }


def _record(vendor: str = VENDOR, method: str = "trap/default") -> dict:
    return {
        "schema": urom.SCHEMA,
        "vendor": vendor,
        "method": method,
        "c1": {
            "status": "measured", "monotone_q": True,
            "q_zero_anchored": True,
            "charge_fit_error_pct": 0.0, "q_zero": 0.0, "q_max": 1.0,
            "breakpoint_crossings": {"q_oss": 0, "q_iss": 0, "q_rss": 0},
            "c_ds_scaling_excluded": True,
        },
        "tuning": {"mode": "off", "verdict": "accept"},
        "speed_adaptive_20": {
            "n_pulses": urom.SPEED_N_PULSES, "max_step": None,
            "method_label": method,
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
    rec["gates"] = {name: {"pass": name != "G5_speed", "status": "measured"}
                    for name in GATE_NAMES}
    rec["gates"]["all_pass"] = False
    rec["speed_adaptive_20"]["solver_speedup"] = None
    rec["speed_adaptive_20"]["wall_speedup"] = None
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
    assert calls == [(VENDOR, {"variant": "urom"})] * 2


# ---------------------------------------------------------------------------
# run_unified signature / argument validation (no bench entered)
# ---------------------------------------------------------------------------
def test_run_unified_signature_is_keyword_only_with_documented_defaults():
    sig = inspect.signature(urom.run_unified)
    assert list(sig.parameters) == [
        "vendor", "variant", "method", "emit", "tuning", "repeats"]
    params = sig.parameters
    assert params["vendor"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("variant", "method", "emit", "tuning", "repeats"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["variant"].default == registry.UNIFIED_VARIANT
    assert params["method"].default == "trap/default"
    assert params["emit"].default == "ann"
    assert params["tuning"].default == "off"
    assert params["repeats"].default == urom.SPEED_REPEATS == 3
    assert urom.METHODS == ("trap/default", "gear-both")
    assert urom.EMIT_MODES == ("ann", "tier2")
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
    ({"emit": "hybrid"}, "unknown emit mode"),
    ({"tuning": "maybe"}, "unknown tuning mode"),
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
    assert data == {**payload, "present": True}
    assert data["sentinel_read_from_disk"] == "recorded-only"


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
    assert gates["G6_charge"]["detail"]["budget_pct"] == 0.5


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
    assert "aborts are failures" in row["completion_basis"]
    assert "3 interleaved repeats" in row["completion_basis"]
    assert "0.999" in row["completion_basis"]

    speed["golden"]["completion"] = False
    missing = urom._speed_row(plan, speed)
    assert missing["solver_speedup"] is None
    assert missing["wall_speedup"] is None
    assert missing["method_label"] == "trap/default"
    assert missing["n_pulses"] == 20
    assert missing["t_last"]["urom"] == 1.0


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
    lambda c: c.update(q_zero_anchored=False),
    lambda c: c.update(monotone_q=False),
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
        helped = _runner.invoke(app, [command, "--help"])
        assert helped.exit_code == 0, helped.output
        text = _flat(helped.output)
        assert "--variant" in text
        assert "urom" in text
        assert "--tier" in text


def test_cli_fit_and_validate_urom_print_gate_record_without_bench(monkeypatch):
    calls: list = []
    record = _cli_record()

    def validate(vendor, tier, variant):
        calls.append((vendor, tier, variant))
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
        assert f"variant=urom basis=ann method=trap/default" in text
        assert "all_pass=no" in text and "verdict=reject" in text
        for name in GATE_NAMES:
            assert name in text
        assert "solver=N/Ax" in text and "wall=N/Ax" in text
        assert "complete=no" in text
        assert "Tier-3 DPT" not in text
    assert calls == [(VENDOR, 3, "urom")] * 2


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


def _write_dpt_metrics(monkeypatch, tmp_path, payload):
    folder = tmp_path / "reports" / "comparison" / VENDOR
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "dpt_metrics.json").write_text(
        json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(urom.registry, "ROOT", tmp_path)


def test_timing_bars_select_column_by_emit_mode(monkeypatch, tmp_path):
    golden = {"td_on_ns": 11.0, "tr_ns": 10.0, "td_off_ns": 13.0,
              "tf_ns": 12.0, "e_on_uj": 14.0, "e_off_uj": 15.0}
    tier2 = {k: v + 1.0 for k, v in golden.items()}  # synthetic, not measured
    _write_dpt_metrics(monkeypatch, tmp_path,
                       {"golden_metrics": golden, "tier2_metrics": tier2})
    assert urom._timing_bars(VENDOR, emit="ann") == golden
    assert urom._timing_bars(VENDOR, emit="tier2") == tier2
    assert urom.TIMING_BAR_COLUMN == {
        "ann": "golden_metrics", "tier2": "tier2_metrics"}


def test_g4_bar_source_names_the_column_used_by_each_emit():
    ann_src = urom._timing_bars_source("ann")
    tier2_src = urom._timing_bars_source("tier2")
    assert "#golden_metrics" in ann_src
    assert "#tier2_metrics" in tier2_src
    assert "fallback decision" in tier2_src
    prov = urom.PROVENANCE["timing_energy_bars"]
    assert "golden_metrics" in prov["source"]
    assert "tier2_metrics" in prov["source"]
    assert "tier2_metrics" in urom.PROVENANCE["timing_band"]["source"]


def test_tier2_g4_passes_against_its_own_recorded_table(monkeypatch, tmp_path):
    # Structural probes, not PieceSpice measurements: the C0 row must be
    # scored against tier2_metrics, never the golden column.
    golden = {k: 100.0 for k in ("td_on_ns", "tr_ns", "td_off_ns", "tf_ns",
                                  "e_on_uj", "e_off_uj")}
    tier2 = {k: 20.0 for k in golden}
    _write_dpt_metrics(monkeypatch, tmp_path,
                       {"golden_metrics": golden, "tier2_metrics": tier2})
    bars = urom._timing_bars(VENDOR, emit="tier2")
    actual = {"td_on_ns": 20.0, "tr_ns": 20.0, "td_off_ns": 20.0,
              "tf_ns": 20.0, "e_on_uj": 20.0, "e_off_uj": 20.0}
    assert all(urom._timing_band(bars[k])[0] <= actual[k]
               <= urom._timing_band(bars[k])[1] for k in actual)
    # The same actual row would fail hard against the golden bar.
    golden_bars = urom._timing_bars(VENDOR, emit="ann")
    assert not all(urom._timing_band(golden_bars[k])[0] <= actual[k]
                   <= urom._timing_band(golden_bars[k])[1] for k in actual)


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
    canonical = build_tier2_subcircuit(out, c1_eps=None, name=lib.stem)
    if expect_c0:
        assert emitted == canonical
        assert out.metadata["urom_c1_eps"] == 0.0
        assert urom._emitted_eps_by_table(out) == {}
        assert urom._c0_bit_identity(out, emitted, lib.stem)
    else:
        assert emitted != canonical
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
