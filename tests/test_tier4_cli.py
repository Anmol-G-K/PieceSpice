"""Offline wiring tests for Tier-4 registry, CLI and recorded compare rows.

All pipeline calls are replaced with an in-memory record: no LTspice, no
network and no vendor model is touched.  The numeric fixture below is
synthetic plumbing data, not a new PieceSpice measurement.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from typer.testing import CliRunner

from cli.main import app
from pipeline import registry

VENDOR = "irfs7730pbf"
_runner = CliRunner()

# Verbatim shared-source disclosure required on every Tier-4 energy row.
T4_DISCLOSURE = (
    "LUT reproduces its own golden DPT crossing; "
    "not an independent converter validation")


def _flat(output: str) -> str:
    """Collapse Rich wrapping/padding so long disclosure text is searchable."""
    return " ".join(output.split())


def _record() -> dict:
    """Synthetic pipeline record with the pinned Tier-4 key names."""
    return {
        "schema": "piecespice.tier4/1.0",
        "vendor": VENDOR,
        "waveform_gate": "retired_by_design",
        "emitted_lib": f"reports/models/{VENDOR}_PIECESPICE_T4.lib",
        "energy_comparison": {
            "e_on_uj": {"golden": 51.0, "tier4": 51.0, "abs_error": 0.0,
                        "rel_error_pct": 0.0, "independent_energy_gate": False,
                        "source_shared": True,
                        "kind": "source_self_consistency",
                        "shared_source": "reports/comparison/irfs7730pbf/"
                                         "dpt_metrics.json#golden_metrics",
                        "disclosure": T4_DISCLOSURE},
            "e_off_uj": {"golden": 1.3, "tier4": 1.3, "abs_error": 0.0,
                         "rel_error_pct": 0.0, "independent_energy_gate": False,
                         "source_shared": True,
                         "kind": "source_self_consistency",
                         "shared_source": "reports/comparison/irfs7730pbf/"
                                          "dpt_metrics.json#golden_metrics",
                         "disclosure": T4_DISCLOSURE},
        },
        "energy_gate_disclosure": (
            T4_DISCLOSURE + " (no independent converter-energy gate)"),
        "speedup": {
            "solver_speedup": 5.299,
            "wall_speedup": 3.069,
            "golden_iterations": 131558,
            "tier4_iterations": 12073,
            "golden_points": 44050,
            "tier4_points": 2949,
            "t_last_s": 0.000176,
            "status": "complete",
        },
    }


def _record_missing_golden() -> dict:
    """CPM2-shaped record: golden crossings and speed ratios never measured."""
    record = _record()
    record["vendor"] = "cpm2-1200"
    record["emitted_lib"] = "reports/models/cpm2-1200_PIECESPICE_T4.lib"
    record["energy_comparison"]["e_on_uj"] = {
        "golden": None, "tier4": 51.0,
        "abs_error": None, "rel_error_pct": None,
    }
    record["speedup"].update({
        "solver_speedup": None,
        "wall_speedup": None,
        "status": "N/A",
        "golden_iterations": None,
        "golden_points": None,
    })
    return record


def _install_fake_pipeline(monkeypatch, record: dict) -> list:
    """Import hook for ``pipeline.tier4.run_vendor`` with call recording."""
    calls: list = []
    module = types.ModuleType("pipeline.tier4")

    def run_vendor(vendor: str, **kwargs) -> dict:
        calls.append((vendor, kwargs))
        return record

    module.run_vendor = run_vendor
    monkeypatch.setitem(sys.modules, "pipeline.tier4", module)
    return calls


def _write_reports(monkeypatch, tmp_path: Path, record: dict | None) -> None:
    """Point the registry at a temporary reports tree with recorded JSON."""
    monkeypatch.setattr(registry, "ROOT", tmp_path)
    base = tmp_path / "reports" / "comparison" / VENDOR
    base.mkdir(parents=True)
    (base / "dpt_metrics.json").write_text(
        json.dumps({"golden_metrics": {"e_on_uj": 51.0}, "tier2_metrics": {}}),
        encoding="utf-8",
    )
    if record is not None:
        (base / "tier4_validation.json").write_text(
            json.dumps(record), encoding="utf-8")


def test_tier4_registered_with_variant_free_library_name():
    assert 4 in registry.TIERS
    assert registry.tier4_lib_name(VENDOR) == f"{VENDOR}_PIECESPICE_T4.lib"
    assert registry.lib_path(VENDOR, 4) == (
        registry.ROOT / "reports" / "models" /
        f"{VENDOR}_PIECESPICE_T4.lib")
    # Tier-2/3 names and variants are untouched.
    assert registry.lib_path(VENDOR, 2).name == f"{VENDOR}_PIECESPICE_T2.lib"
    assert registry.lib_path(VENDOR, 3).name == f"{VENDOR}_PIECESPICE_T3.lib"
    assert registry.lib_path(VENDOR, 3, "simplis").name == (
        f"{VENDOR}_PIECESPICE_T3_simplis.lib")
    assert registry.variants_for(VENDOR) == (
        registry.DEFAULT_VARIANT, "simplis", registry.UNIFIED_VARIANT)


def test_registry_fit_and_validate_call_tier4_without_variant(monkeypatch):
    record = _record()
    calls = _install_fake_pipeline(monkeypatch, record)
    assert registry.fit(VENDOR, 4) == Path(record["emitted_lib"])
    assert registry.validate(VENDOR, 4) is record
    assert calls == [(VENDOR, {}), (VENDOR, {})]


def test_cli_fit_and_validate_tier4_print_energy_and_speed(monkeypatch):
    record = _record()
    calls = _install_fake_pipeline(monkeypatch, record)

    for command in ("fit", "validate"):
        result = _runner.invoke(app, [command, VENDOR, "--tier", "4"])
        assert result.exit_code == 0, result.output
        out = result.output
        assert "Tier-4 ideal-switch + loss-LUT" in out
        assert "DPT waveform gate retired by design" in _flat(out)
        for metric in ("e_on_uj", "e_off_uj"):
            row = record["energy_comparison"][metric]
            assert metric in out
            assert f"{row['golden']:g}" in out
            assert f"{row['tier4']:g}" in out
            assert f"{row['rel_error_pct']:g}" in out
        assert "speed status=complete" in out
        assert "5.299x" in out and "3.069x" in out
        for count in ("131558", "12073", "44050", "2949", "0.000176"):
            assert count in out
        assert record["emitted_lib"] in out
        # No Tier-3 DPT waveform claim may leak into a Tier-4 summary.
        assert "rel-RMSE" not in out and "trace_points=" not in out
        assert T4_DISCLOSURE in _flat(out)
        assert "LUT-vs-recorded-golden-DPT" in _flat(out)
        assert "no converter bench built" in _flat(out)
        assert "on a converter bench" not in _flat(out)
        assert "no independent converter-energy gate" in _flat(out)
    assert len(calls) == 2


def test_cli_tier4_ignores_tier3_variant(monkeypatch):
    calls = _install_fake_pipeline(monkeypatch, _record())

    result = _runner.invoke(
        app, ["fit", VENDOR, "--tier", "4", "--variant", "simplis"])
    assert result.exit_code == 0, result.output
    assert calls == [(VENDOR, {})]


def test_cli_compare_prints_tier4_energy_and_speed_rows(monkeypatch, tmp_path):
    _write_reports(monkeypatch, tmp_path, _record())

    result = _runner.invoke(app, ["compare", VENDOR])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Golden vs Tier-2 vs Tier-3" in out  # tier 2/3 table preserved
    assert "Tier-4 energy & speed (waveform gate retired)" in out
    for label in ("e_on_uj", "e_off_uj", "solver_speedup", "wall_speedup"):
        assert label in out
    assert "status=complete" in out
    for count in ("131558", "12073", "44050", "2949", "0.000176"):
        assert count in out
    assert T4_DISCLOSURE in _flat(out)
    assert "no converter bench built" in _flat(out)
    assert "on a converter bench" not in _flat(out)
    assert "no independent converter-energy gate" in _flat(out)


def test_compare_rows_render_missing_golden_as_na(monkeypatch, tmp_path):
    _write_reports(monkeypatch, tmp_path, _record_missing_golden())

    rows = {label: (golden, tier4, detail)
            for label, golden, tier4, detail in registry.tier4_rows(VENDOR)}
    assert rows["e_on_uj"][0] is None
    assert rows["e_on_uj"][2].startswith("abs=N/A uJ; rel=N/A%; ")
    assert T4_DISCLOSURE in rows["e_on_uj"][2]
    assert rows["solver_speedup"][1] is None
    assert "status=N/A" in rows["solver_speedup"][2]
    assert "iters N/A/12073" in rows["solver_speedup"][2]

    result = _runner.invoke(app, ["compare", VENDOR])
    assert result.exit_code == 0, result.output
    assert "status=N/A" in result.output
    assert "N/A" in result.output
    assert "fast" not in result.output.lower()
    assert T4_DISCLOSURE in _flat(result.output)
    assert "on a converter bench" not in _flat(result.output)


def test_compare_without_tier4_report_keeps_tier2_3_table(monkeypatch, tmp_path):
    _write_reports(monkeypatch, tmp_path, None)

    result = _runner.invoke(app, ["compare", VENDOR])
    assert result.exit_code == 0, result.output
    assert "Golden vs Tier-2 vs Tier-3" in result.output
    assert "No recorded Tier-4 energy/speed" in result.output
    assert T4_DISCLOSURE in _flat(result.output)
    assert "no converter bench built" in _flat(result.output)
    assert "on a converter bench" not in _flat(result.output)


def test_registry_tier4_energy_rows_carry_shared_source_disclosure(
    monkeypatch, tmp_path
):
    _write_reports(monkeypatch, tmp_path, _record())
    rows = {label: detail for label, _, _, detail in registry.tier4_rows(VENDOR)}
    for label in ("e_on_uj", "e_off_uj"):
        assert "abs=0.0 uJ; rel=0.0%" in rows[label]
        assert "[source_self_consistency]" in rows[label]
        assert T4_DISCLOSURE in rows[label]


def test_legacy_tier4_records_still_show_disclosure(monkeypatch, tmp_path):
    record = _record()
    for row in record["energy_comparison"].values():
        for key in ("disclosure", "kind", "independent_energy_gate",
                    "source_shared", "shared_source"):
            row.pop(key, None)
    record.pop("energy_gate_disclosure", None)
    _write_reports(monkeypatch, tmp_path, record)
    rows = {label: detail for label, _, _, detail in registry.tier4_rows(VENDOR)}
    assert all(T4_DISCLOSURE in rows[label]
               for label in ("e_on_uj", "e_off_uj"))
    _install_fake_pipeline(monkeypatch, record)
    result = _runner.invoke(app, ["validate", VENDOR, "--tier", "4"])
    assert result.exit_code == 0, result.output
    assert T4_DISCLOSURE in _flat(result.output)
    assert "no converter bench built" in _flat(result.output)


def test_registry_speed_row_labels_schedule_time_without_measured_t_last(
    monkeypatch, tmp_path
):
    record = _record()
    record["speedup"].pop("t_last_s", None)
    record["speedup"]["end_time_s"] = 0.000176
    _write_reports(monkeypatch, tmp_path, record)
    rows = {label: detail for label, _, _, detail in registry.tier4_rows(VENDOR)}
    assert ".tran stop schedule=0.000176 s" in rows["solver_speedup"]
    assert "t_last=" not in rows["solver_speedup"]
