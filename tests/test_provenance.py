"""Tests for shared provenance and benchmark-uncertainty helpers."""
from __future__ import annotations

import json

import pytest

from common.provenance import (
    MANIFEST_SCHEMA,
    artifact_manifest,
    bootstrap_ci,
    environment_metadata,
    file_hashes,
    sha256_file,
    sha256_text,
    verify_artifact_manifest,
)


def test_sha256_text_matches_known_vector():
    assert sha256_text("abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    assert sha256_text("abc") == sha256_text("abc")


def test_file_hashes_marks_missing_as_none_not_empty(tmp_path):
    present = tmp_path / "input.txt"
    present.write_text("payload", encoding="ascii")
    hashes = file_hashes({"present": present, "missing": tmp_path / "nope"})
    assert hashes["present"] == sha256_file(present)
    assert hashes["missing"] is None
    # JSON-safe without inventing values
    json.dumps(hashes, allow_nan=False)


def test_environment_metadata_is_json_safe_without_ltspice():
    meta = environment_metadata(ltspice_exe=None)
    assert meta["python_version"]
    assert meta["platform"]
    assert meta["ltspice_version"] is None
    json.dumps(meta, allow_nan=False)


def test_bootstrap_ci_empty_and_single_sample_are_explicit():
    assert bootstrap_ci([]) == {
        "point": None, "low": None, "high": None, "n": 0,
        "confidence": 0.95, "n_resamples": 2000,
    }
    one = bootstrap_ci([2.5])
    assert one["point"] == one["low"] == one["high"] == 2.5
    assert one["n"] == 1


def test_bootstrap_ci_is_deterministic_and_brackets_the_median():
    values = [1.0, 2.0, 3.0, 4.0, 10.0]
    first = bootstrap_ci(values, n_resamples=500, seed=7)
    second = bootstrap_ci(values, n_resamples=500, seed=7)
    assert first == second
    assert first["point"] == pytest.approx(3.0)
    assert first["low"] <= first["point"] <= first["high"]
    assert first["n"] == 5


def _manifest_fixture(tmp_path):
    lib = tmp_path / "model.lib"
    lib.write_text(".SUBCKT X d g s\n.ENDS\n", encoding="ascii")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("x: 1\n", encoding="ascii")
    curve = tmp_path / "curve.yaml"
    curve.write_text("points: [0, 1]\n", encoding="ascii")
    return lib, {"config": cfg, "curve": curve}


def test_artifact_manifest_roundtrip_and_extra(tmp_path):
    lib, inputs = _manifest_fixture(tmp_path)
    manifest = artifact_manifest(lib, inputs, extra={"basis": "tier2"})
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["lib"]["sha256"] == sha256_file(lib)
    assert manifest["lib"]["bytes"] == len(lib.read_bytes())
    assert set(manifest["inputs"]) == {"config", "curve"}
    for label, path in inputs.items():
        assert manifest["inputs"][label]["sha256"] == sha256_file(path)
        assert manifest["inputs"][label]["bytes"] == len(path.read_bytes())
    assert manifest["extra"] == {"basis": "tier2"}
    json.dumps(manifest, allow_nan=False)
    assert verify_artifact_manifest(manifest) == {
        "ok": True, "mismatches": []}


def test_artifact_manifest_detects_tampered_hash_and_size(tmp_path):
    lib, inputs = _manifest_fixture(tmp_path)
    manifest = artifact_manifest(lib, inputs)
    inputs["curve"].write_text("points: [0, 1, 2, 3]\n", encoding="ascii")
    checked = verify_artifact_manifest(manifest)
    assert checked["ok"] is False
    assert "inputs.curve" in checked["mismatches"]
    assert "lib" not in checked["mismatches"]
    lib.write_text(lib.read_text(encoding="ascii") + "* tamper\n",
                   encoding="ascii")
    checked = verify_artifact_manifest(manifest)
    assert checked["ok"] is False
    assert "lib" in checked["mismatches"]


def test_artifact_manifest_missing_file_is_none_and_not_ok(tmp_path):
    lib, inputs = _manifest_fixture(tmp_path)
    inputs["missing"] = tmp_path / "absent.yaml"
    manifest = artifact_manifest(lib, inputs)
    assert manifest["inputs"]["missing"] is None
    checked = verify_artifact_manifest(manifest)
    assert checked["ok"] is False
    assert "inputs.missing" in checked["mismatches"]
    json.dumps(manifest, allow_nan=False)


def test_verify_artifact_manifest_rejects_bad_schema(tmp_path):
    lib, inputs = _manifest_fixture(tmp_path)
    manifest = artifact_manifest(lib, inputs)
    manifest["schema"] = "not-a-manifest"
    assert verify_artifact_manifest(manifest) == {
        "ok": False, "mismatches": ["schema"]}
    assert verify_artifact_manifest(None)["ok"] is False
