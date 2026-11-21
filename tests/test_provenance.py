"""Tests for shared provenance and benchmark-uncertainty helpers."""
from __future__ import annotations

import json

import pytest

from common.provenance import (
    bootstrap_ci,
    environment_metadata,
    file_hashes,
    sha256_file,
    sha256_text,
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
