"""Tests for the versioned, fail-closed acceptance policy."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from common.acceptance import (
    DEFAULT_PATH,
    DEFAULT_SCHEMA,
    AcceptancePolicy,
    load_acceptance,
)
from common.provenance import sha256_bytes


def _base_raw() -> dict:
    return yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))


def _write_policy(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "acceptance.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=False),
                    encoding="ascii")
    return path


def test_shipped_policy_reproduces_current_module_constants():
    from pipeline import unified_rom as urom

    policy = load_acceptance()
    assert policy.schema == DEFAULT_SCHEMA
    assert policy.path == DEFAULT_PATH
    assert policy.static_bars_pct == urom.STATIC_BARS_PCT
    assert policy.dpt_bars_pct == urom.DPT_BARS_PCT
    assert policy.worst_endpoint_pct == urom.ENDPOINT_BAR_PCT
    assert policy.g1_band_pp == urom.G1_BAND_PP
    assert policy.timing_band_rel == urom.TIMING_BAND_REL
    assert policy.timing_band_abs == urom.TIMING_BAND_ABS
    assert policy.charge_budget_pct == urom.CHARGE_BUDGET_PCT
    assert policy.speed_min_solver_speedup == urom.SPEED_EFFECT_MIN
    assert policy.speed_confidence == urom.SPEED_CI_CONFIDENCE
    assert policy.speed_n_resamples == urom.SPEED_CI_RESAMPLES


def test_policy_record_carries_path_schema_sources_and_sha256():
    policy = load_acceptance()
    assert len(policy.sha256) == 64
    assert policy.sha256 == sha256_bytes(DEFAULT_PATH.read_bytes())
    record = policy.as_record()
    assert record["path"] == str(DEFAULT_PATH)
    assert record["schema"] == DEFAULT_SCHEMA
    assert record["sha256"] == policy.sha256
    assert record["sources"]["static.worst_endpoint_pct"] == (
        "recorded_baseline")
    assert record["sources"]["speed.min_solver_speedup"] == (
        "project_decision")
    assert record["values"]["dpt_bars_pct"]["trap/default"]["irfs7730pbf"] == (
        10.0, 0.60)
    assert "application requirement" in record["override_semantics"]
    json.dumps(record, allow_nan=False)


def test_missing_policy_file_uses_documented_fallback(tmp_path):
    policy = load_acceptance(tmp_path / "absent.yaml")
    assert policy.path is None
    assert policy.sha256 is None
    assert policy.schema == DEFAULT_SCHEMA
    assert policy.static_bars_pct["irfs7730pbf"] == 7.0683
    assert policy.speed_n_resamples == 2000
    assert isinstance(policy, AcceptancePolicy)


def test_config_override_changes_bars_without_code_edits(tmp_path):
    raw = _base_raw()
    raw["static"]["mean_peak_rel_rmse_pct_max"]["irfs7730pbf"]["value"] = 9.5
    raw["dpt"]["trap/default"]["irfs7730pbf"]["value"] = [12.0, 0.7]
    raw["timing"]["band_rel"]["value"] = 0.1
    raw["charge"]["budget_pct"]["value"] = 0.25
    raw["speed"]["min_solver_speedup"]["value"] = 1.5
    raw["speed"]["confidence"]["value"] = 0.9
    raw["speed"]["n_resamples"]["value"] = 500
    path = _write_policy(tmp_path, raw)
    policy = load_acceptance(path)

    assert policy.static_bars_pct["irfs7730pbf"] == 9.5
    assert policy.dpt_bars_pct["trap/default"]["irfs7730pbf"] == (12.0, 0.7)
    assert policy.timing_band_rel == 0.1
    assert policy.charge_budget_pct == 0.25
    assert policy.speed_min_solver_speedup == 1.5
    assert policy.speed_confidence == 0.9
    assert policy.speed_n_resamples == 500
    # Overriding a value does not lose its provenance or the file identity.
    assert policy.sources["static.mean_peak_rel_rmse_pct_max.irfs7730pbf"] == (
        "recorded_baseline")
    assert policy.sources["speed.min_solver_speedup"] == "project_decision"
    assert policy.path == path
    assert policy.sha256 == sha256_bytes(path.read_bytes())


def _mutations():
    def drop_static(raw):
        del raw["static"]

    def drop_vendor(raw):
        del raw["static"]["mean_peak_rel_rmse_pct_max"]["epc2216"]

    def drop_method(raw):
        del raw["dpt"]["gear-both"]

    def drop_source(raw):
        del raw["charge"]["budget_pct"]["source"]

    def bad_schema(raw):
        raw["schema"] = "piecespice.acceptance/9.9"

    def drop_semantics(raw):
        del raw["override_semantics"]

    def zero_bar(raw):
        raw["static"]["worst_endpoint_pct"]["value"] = 0.0

    def negative_bar(raw):
        raw["dpt"]["trap/default"]["irfs7730pbf"]["value"][1] = -0.1

    def nonfinite_bar(raw):
        raw["static"]["mean_peak_rel_rmse_pct_max"]["csd16412q5a"][
            "value"] = float("nan")

    def bad_band_rel(raw):
        raw["timing"]["band_rel"]["value"] = 1.0

    def bad_confidence(raw):
        raw["speed"]["confidence"]["value"] = 0.0

    def bad_speed_min(raw):
        raw["speed"]["min_solver_speedup"]["value"] = 1.0

    def bad_resamples(raw):
        raw["speed"]["n_resamples"]["value"] = 0

    def bad_pair(raw):
        raw["dpt"]["gear-both"]["cpm2-1200"]["value"] = [4.122]

    def bad_source_label(raw):
        raw["speed"]["confidence"]["source"] = "guessed"

    def missing_value(raw):
        del raw["timing"]["band_abs"]["value"]

    def root_not_mapping(raw):
        return []

    return [
        drop_static, drop_vendor, drop_method, drop_source, bad_schema,
        drop_semantics, zero_bar, negative_bar, nonfinite_bar, bad_band_rel,
        bad_confidence, bad_speed_min, bad_resamples, bad_pair,
        bad_source_label, missing_value, root_not_mapping,
    ]


@pytest.mark.parametrize("mutate", _mutations(),
                         ids=[fn.__name__ for fn in _mutations()])
def test_malformed_policy_fails_closed(tmp_path, mutate):
    payload = copy.deepcopy(_base_raw())
    replacement = mutate(payload)
    if replacement is not None:
        payload = replacement
    path = _write_policy(tmp_path, payload)
    with pytest.raises(ValueError):
        load_acceptance(path)


def test_syntactically_invalid_yaml_fails_closed(tmp_path):
    path = tmp_path / "acceptance.yaml"
    path.write_text("schema: [\n", encoding="ascii")
    with pytest.raises(ValueError, match="invalid YAML"):
        load_acceptance(path)


def test_policy_is_immutable_to_callers():
    policy = load_acceptance()
    with pytest.raises(Exception):
        policy.charge_budget_pct = 99.0
