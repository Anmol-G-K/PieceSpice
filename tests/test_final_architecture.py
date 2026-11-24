"""Offline tests for the independent final-architecture verifier.

The verifier is deliberately separate from the pipeline: it reads only the
recorded comparison JSON and applies the user's final gates.  These tests use
synthetic pass/fail records to pin the 7.0 % timing/energy boundary, the
1.50x solver-speed boundary, the recorded G3 bars, and fail-closed behaviour
for missing data.  One test runs the harness over the committed records.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "verify_final_architecture.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_final_architecture", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vfa = _load_verifier()
VENDORS = vfa.VENDORS
KEYS = vfa.TIMING_ENERGY_ROWS
VDS_BAR = 3.0
ID_BAR = 5.0


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _record_path(root: Path, tier: str, vendor: str) -> Path:
    return (root / "reports" / "comparison" / vendor
            / vfa.RECORD_FILES[tier])


def _tier4(vendor: str) -> dict:
    return {
        "schema": "piecespice.tier4/1.0",
        "vendor": vendor,
        "waveform_gate": "retired_by_design",
        "energy_comparison": {
            metric: {
                "status": "measured",
                "golden": 1.0,
                "tier4": 1.0,
                "rel_error_pct": 0.0,
            }
            for metric in vfa.TIER4_ROWS
        },
        "speedup": {
            "status": "measured",
            "solver_speedup": 2.0,
            "wall_speedup": 1.8,
        },
        "guards": {"speed_completion": True},
    }


def _unified(vendor: str) -> dict:
    return {
        "schema": "piecespice.unified/1.0",
        "vendor": vendor,
        "method": "trap/default",
        "gates": {
            "G3_dpt_waveform": {
                "detail": {
                    "urom": {
                        "status": "measured",
                        "vds_rel_rmse_pct": 1.0,
                        "id_rel_rmse_pct": 1.0,
                        "bars_vds_id_pct": [VDS_BAR, ID_BAR],
                    }
                }
            },
            "G4_timing_energy": {
                "detail": {
                    "rows": {
                        metric: {
                            "status": "measured",
                            "value": 1.0,
                            "bar": 1.0,
                        }
                        for metric in KEYS
                    }
                }
            },
            "G5_speed": {
                "detail": {
                    "solver_speedup": 2.0,
                    "wall_speedup": 1.8,
                    "completion": {"golden": True, "urom": True},
                    "scoped_row_valid": True,
                }
            },
        },
    }


def _simplis(vendor: str) -> dict:
    return {
        "schema": "piecespice.tier3/2.0",
        "vendor": vendor,
        "variant": "simplis",
        "dpt_comparison": {
            metric: {
                "golden": 1.0,
                "tier3": 1.0,
                "tier3_rel_err_pct": 0.0,
            }
            for metric in KEYS
        },
        "dpt_run": {
            "waveform": {
                "vds_rel_rmse_pct": 1.0,
                "id_rel_rmse_pct": 1.0,
            }
        },
        "speed_adaptive_20": {
            "solver_speedup": 2.0,
            "wall_speedup": 1.8,
            "golden": {"completed": True},
            "tier3_simplis": {"completed": True},
        },
    }


def _write_tree(root: Path) -> None:
    for vendor in VENDORS:
        _write(_record_path(root, "Tier-4", vendor), _tier4(vendor))
        _write(_record_path(root, "UROM", vendor), _unified(vendor))
        _write(_record_path(root, "ANN", vendor), _unified(vendor))
        _write(_record_path(root, "simplis", vendor), _simplis(vendor))


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    _write_tree(tmp_path)
    return tmp_path


def _result(results, tier: str, vendor: str):
    return next(item for item in results
                if item.tier == tier and item.vendor == vendor)


def _row(result, metric: str):
    return next(item for item in result.rows if item.metric == metric)


def test_synthetic_pass_records_cover_all_tiers_and_vendors(tree, capsys):
    results = vfa.evaluate_records(tree)
    assert len(results) == len(vfa.TIERS) * len(VENDORS)
    assert {item.tier for item in results} == set(vfa.TIERS)
    assert {item.vendor for item in results} == set(VENDORS)
    assert all(item.status == "PASS" for item in results), [
        (item.tier, item.vendor, item.failures)
        for item in results if item.status != "PASS"
    ]

    assert vfa.main(["--root", str(tree)]) == 0
    output = capsys.readouterr().out
    assert "ALL GATES PASS" in output
    assert "Tier-4" in output and "UROM" in output
    assert "ANN" in output and "simplis" in output
    assert "PER-ROW TIMING/ENERGY MARGINS" in output


def test_timing_energy_gate_is_inclusive_at_seven_percent(tree):
    path = _record_path(tree, "UROM", "cpm2-1200")
    record = _read(path)
    row = record["gates"]["G4_timing_energy"]["detail"]["rows"]["td_on_ns"]
    row["bar"] = 10.0
    row["value"] = 10.7  # exactly 7.0 %
    _write(path, record)

    result = _result(vfa.evaluate_records(tree), "UROM", "cpm2-1200")
    measured = _row(result, "td_on_ns")
    assert measured.passed
    assert measured.error_pct == pytest.approx(7.0)
    assert measured.margin_pp == pytest.approx(0.0, abs=1e-9)
    assert result.status == "PASS"

    row["value"] = 10.701  # 7.01 %
    _write(path, record)
    result = _result(vfa.evaluate_records(tree), "UROM", "cpm2-1200")
    measured = _row(result, "td_on_ns")
    assert not measured.passed
    assert measured.margin_pp == pytest.approx(-0.01, abs=1e-6)
    assert result.status == "FAIL"
    assert any("td_on_ns" in message and "7.01%" in message
               and "margin -0.01 pp" in message
               for message in result.failures)


def test_simplis_timing_energy_uses_golden_pair(tree):
    path = _record_path(tree, "simplis", "cpm2-1200")
    record = _read(path)
    record["dpt_comparison"]["e_off_uj"] = {
        "golden": 10.0,
        "tier3": 10.7,
    }
    _write(path, record)

    result = _result(vfa.evaluate_records(tree), "simplis", "cpm2-1200")
    measured = _row(result, "e_off_uj")
    assert measured.passed
    assert measured.margin_pp == pytest.approx(0.0, abs=1e-9)

    record["dpt_comparison"]["e_off_uj"]["tier3"] = 10.701
    _write(path, record)
    result = _result(vfa.evaluate_records(tree), "simplis", "cpm2-1200")
    assert result.status == "FAIL"
    assert any("e_off_uj" in message and "7.01%" in message
               for message in result.failures)


def _set_speed(record: dict, tier: str, solver, wall) -> None:
    if tier == "Tier-4":
        record["speedup"]["solver_speedup"] = solver
        record["speedup"]["wall_speedup"] = wall
    elif tier == "simplis":
        record["speed_adaptive_20"]["solver_speedup"] = solver
        record["speed_adaptive_20"]["wall_speedup"] = wall
    else:
        detail = record["gates"]["G5_speed"]["detail"]
        detail["solver_speedup"] = solver
        detail["wall_speedup"] = wall


@pytest.mark.parametrize("tier", vfa.TIERS)
def test_solver_speed_gate_is_one_and_a_half_never_wall(tree, tier):
    path = _record_path(tree, tier, "cpm2-1200")
    record = _read(path)

    _set_speed(record, tier, 1.5, 0.1)
    _write(path, record)
    result = _result(vfa.evaluate_records(tree), tier, "cpm2-1200")
    assert result.speed.passed
    assert result.status == "PASS"

    _set_speed(record, tier, 1.4999, 3.0)
    _write(path, record)
    result = _result(vfa.evaluate_records(tree), tier, "cpm2-1200")
    assert not result.speed.passed
    assert result.status == "FAIL"
    assert any("1.4999x" in message and "1.50x" in message
               for message in result.failures)

    # A large wall speedup cannot rescue a failing solver speedup.
    _set_speed(record, tier, 1.0, 9.0)
    _write(path, record)
    result = _result(vfa.evaluate_records(tree), tier, "cpm2-1200")
    assert result.status == "FAIL"
    assert any("wall 9.0000x" in message
               and "not substituted" in message
               for message in result.failures)

    _set_speed(record, tier, None, 4.0)
    _write(path, record)
    result = _result(vfa.evaluate_records(tree), tier, "cpm2-1200")
    assert result.status == "FAIL"
    assert any("solver_speedup missing" in message
               and "not substituted" in message
               for message in result.failures)


def test_g3_uses_the_recorded_bars_without_inventing_a_bar(tree):
    path = _record_path(tree, "UROM", "cpm2-1200")
    record = _read(path)
    arm = record["gates"]["G3_dpt_waveform"]["detail"]["urom"]
    arm["vds_rel_rmse_pct"] = 2.999
    arm["id_rel_rmse_pct"] = 4.999
    _write(path, record)
    result = _result(vfa.evaluate_records(tree), "UROM", "cpm2-1200")
    assert result.g3.gate_ok
    assert result.status == "PASS"

    arm["bars_vds_id_pct"] = [2.0, ID_BAR]
    _write(path, record)
    result = _result(vfa.evaluate_records(tree), "UROM", "cpm2-1200")
    assert result.status == "FAIL"
    assert any("G3 Vds: 2.999% > 2.000%" in message
               for message in result.failures)

    del arm["bars_vds_id_pct"]
    _write(path, record)
    result = _result(vfa.evaluate_records(tree), "UROM", "cpm2-1200")
    assert result.status == "FAIL"
    assert any("no recorded Vds/Id bar" in message
               for message in result.failures)


def test_simplis_g3_uses_same_vendor_recorded_urom_bars(tree):
    path = _record_path(tree, "simplis", "cpm2-1200")
    record = _read(path)
    record["dpt_run"]["waveform"]["vds_rel_rmse_pct"] = 2.9
    _write(path, record)
    result = _result(vfa.evaluate_records(tree), "simplis", "cpm2-1200")
    assert result.g3.bar_source.startswith("same-vendor")
    assert result.g3.gate_ok

    canonical_path = _record_path(tree, "UROM", "cpm2-1200")
    canonical = _read(canonical_path)
    canonical["gates"]["G3_dpt_waveform"]["detail"]["urom"][
        "bars_vds_id_pct"] = [2.5, ID_BAR]
    _write(canonical_path, canonical)

    result = _result(vfa.evaluate_records(tree), "simplis", "cpm2-1200")
    assert result.status == "FAIL"
    assert any("2.900% > 2.500%" in message
               for message in result.failures)

    del canonical["gates"]["G3_dpt_waveform"]["detail"]["urom"][
        "bars_vds_id_pct"]
    _write(canonical_path, canonical)
    result = _result(vfa.evaluate_records(tree), "simplis", "cpm2-1200")
    assert result.status == "FAIL"
    assert any("no recorded Vds/Id bar in simplis or UROM record"
               in message for message in result.failures)


def test_missing_timing_row_fails_closed(tree):
    path = _record_path(tree, "UROM", "irfs7730pbf")
    record = _read(path)
    del record["gates"]["G4_timing_energy"]["detail"]["rows"]["tr_ns"]
    _write(path, record)

    result = _result(vfa.evaluate_records(tree), "UROM", "irfs7730pbf")
    assert result.status == "FAIL"
    assert any("tr_ns" in message and "not measured" in message
               for message in result.failures)


def test_missing_speed_field_fails_closed_no_wall_substitution(tree):
    path = _record_path(tree, "ANN", "epc2216")
    record = _read(path)
    del record["gates"]["G5_speed"]["detail"]["solver_speedup"]
    record["gates"]["G5_speed"]["detail"]["wall_speedup"] = 7.0
    _write(path, record)

    result = _result(vfa.evaluate_records(tree), "ANN", "epc2216")
    assert result.status == "FAIL"
    assert any("solver_speedup missing" in message
               and "wall 7.0000x" in message
               and "not substituted" in message
               for message in result.failures)


def test_missing_record_and_malformed_json_fail_closed(tree):
    deleted = _record_path(tree, "Tier-4", "epc2216")
    deleted.unlink()
    result = _result(vfa.evaluate_records(tree), "Tier-4", "epc2216")
    assert result.status == "FAIL"
    assert any("missing record" in message for message in result.failures)

    malformed = _record_path(tree, "simplis", "irfs7730pbf")
    malformed.write_text("{not valid json", encoding="utf-8")
    result = _result(vfa.evaluate_records(tree), "simplis", "irfs7730pbf")
    assert result.status == "FAIL"
    assert any("unreadable record" in message for message in result.failures)


def test_missing_waveform_metrics_fail_closed(tree):
    path = _record_path(tree, "simplis", "epc2216")
    record = _read(path)
    record["dpt_run"].pop("waveform")
    _write(path, record)

    result = _result(vfa.evaluate_records(tree), "simplis", "epc2216")
    assert result.status == "FAIL"
    assert any("dpt_run.waveform absent" in message
               for message in result.failures)


def test_current_records_run_without_crashing_and_report_failures():
    results = vfa.evaluate_records(ROOT)
    assert len(results) == len(vfa.TIERS) * len(VENDORS)
    for result in results:
        assert (result.status == "PASS") == (not result.failures)

    # The committed Tier-4 TI record is a real passing row.
    assert _result(results, "Tier-4", "csd16412q5a").status == "PASS"

    expected_exit = 0 if all(item.status == "PASS"
                             for item in results) else 1
    assert vfa.main(["--root", str(ROOT)]) == expected_exit
    if expected_exit:
        assert any(item.status == "FAIL" for item in results)
        report = vfa.format_report(results)
        assert "FAILURES (" in report
        # Each computed failure is present verbatim in the report.
        for failure in (failure for item in results
                        for failure in item.failures[:1]):
            assert failure in report

    # A failing timing row carries the independently computed error and
    # margin in the report (the synthetic tests pin the failure formatting).
    record = _read(_record_path(ROOT, "UROM", "cpm2-1200"))
    row = record["gates"]["G4_timing_energy"]["detail"]["rows"]["tf_ns"]
    error = abs(row["value"] - row["bar"]) / abs(row["bar"]) * 100.0
    if error > vfa.GATE_PCT:
        result = _result(results, "UROM", "cpm2-1200")
        assert any("tf_ns" in message
                   and f"{error:.2f}%" in message
                   and f"{vfa.GATE_PCT - error:+.2f} pp" in message
                   for message in result.failures)
