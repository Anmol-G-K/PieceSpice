"""Reproducibility helpers shared by validation records.

Three narrow responsibilities:

* SHA-256 hashes of files, text and bytes (input provenance);
* best-effort environment metadata (Python, OS, CPU, LTspice executable);
* a deterministic bootstrap confidence interval for benchmark medians.

Nothing here runs a simulation, invents a measurement, or converts a missing
value into zero.  Missing files and empty samples return ``None``.
"""
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "MANIFEST_SCHEMA",
    "artifact_manifest",
    "bootstrap_ci",
    "environment_metadata",
    "file_hashes",
    "sha256_bytes",
    "sha256_file",
    "sha256_text",
    "verify_artifact_manifest",
]


def sha256_bytes(data: bytes) -> str:
    """Lower-case hex SHA-256 of raw bytes."""
    return hashlib.sha256(bytes(data)).hexdigest()


def sha256_text(text: str) -> str:
    """Lower-case hex SHA-256 of UTF-8 text."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    """Lower-case hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hashes(
    paths: Mapping[str, str | Path] | Iterable[str | Path],
) -> dict[str, str | None]:
    """Hash named inputs; a missing/unreadable path maps to ``None``.

    A mapping keeps its keys as the record labels.  An iterable uses the path
    string itself as the label.  ``None`` means "not hashed", never a fake
    hash and never a silent empty string.
    """
    if isinstance(paths, Mapping):
        items = [(str(label), path) for label, path in paths.items()]
    else:
        items = [(str(path), path) for path in paths]
    out: dict[str, str | None] = {}
    for label, raw in items:
        try:
            out[label] = sha256_file(raw)
        except (OSError, ValueError):
            out[label] = None
    return out


def _ltspice_version(exe: str | None) -> str | None:
    if not exe:
        return None
    path = Path(exe)
    if not path.exists():
        return None
    try:
        completed = subprocess.run(
            [str(path), "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    text = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr)
        if part and part.strip()
    )
    return text[:500] if text else None


def environment_metadata(ltspice_exe: str | None = None) -> dict[str, Any]:
    """Best-effort, JSON-safe environment identity for a benchmark record.

    The function never raises because a simulator path is absent.  A missing
    LTspice executable is recorded as ``None`` and the caller can decide
    whether that is acceptable for the claim being made.
    """
    exe = ltspice_exe or os.environ.get("LTSPICE_EXE")
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "ltspice_exe": str(exe) if exe else None,
        "ltspice_version": _ltspice_version(exe),
    }


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    statistic: Any = np.median,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Deterministic percentile bootstrap interval for one statistic.

    ``values`` is filtered to finite samples.  With fewer than two samples the
    interval collapses to the point estimate (and is reported as such); an
    empty sample returns ``None`` for every field.  No value is imputed.
    """
    raw = np.asarray(list(values) if not isinstance(values, np.ndarray) else values,
                     dtype=float).ravel()
    finite = raw[np.isfinite(raw)]
    n = int(finite.size)
    if n == 0:
        return {"point": None, "low": None, "high": None, "n": 0,
                "confidence": float(confidence), "n_resamples": int(n_resamples)}
    point = float(statistic(finite))
    if n == 1:
        low = high = point
    else:
        if int(n_resamples) < 1:
            raise ValueError("n_resamples must be >= 1")
        if not 0.0 < float(confidence) < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        rng = np.random.default_rng(int(seed))
        draws = rng.integers(0, n, size=(int(n_resamples), n))
        stats = np.asarray([statistic(finite[idx]) for idx in draws], dtype=float)
        stats = stats[np.isfinite(stats)]
        if stats.size == 0:
            low = high = point
        else:
            alpha = (1.0 - float(confidence)) / 2.0
            low = float(np.quantile(stats, alpha))
            high = float(np.quantile(stats, 1.0 - alpha))
    return {
        "point": point,
        "low": low,
        "high": high,
        "n": n,
        "confidence": float(confidence),
        "n_resamples": int(n_resamples),
    }


MANIFEST_SCHEMA = "piecespice.artifact_manifest/1.0"


def _file_entry(path: str | Path | None) -> dict[str, Any] | None:
    """Hash and size one file in a single pass; missing/unreadable -> None."""
    if path is None:
        return None
    try:
        target = Path(path)
        digest = hashlib.sha256()
        size = 0
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except (OSError, ValueError, TypeError):
        return None
    return {"path": str(target), "sha256": digest.hexdigest(), "bytes": size}


def artifact_manifest(
    lib_path: str | Path,
    inputs: Mapping[str, str | Path],
    extra: Any = None,
) -> dict[str, Any]:
    """Describe the emitted lib and every named input by hash and size.

    A missing file maps to ``None`` in the manifest, never a fabricated hash.
    ``extra`` is opaque caller metadata and is not used by verification.
    """
    if not isinstance(inputs, Mapping):
        raise TypeError("inputs must be a mapping of label to path")
    return {
        "schema": MANIFEST_SCHEMA,
        "lib": _file_entry(lib_path),
        "inputs": {str(label): _file_entry(path)
                   for label, path in inputs.items()},
        "extra": extra,
    }


def _entry_matches(entry: Any) -> bool:
    if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
        return False
    actual = _file_entry(entry["path"])
    return (actual is not None
            and actual["sha256"] == entry.get("sha256")
            and actual["bytes"] == entry.get("bytes"))


def verify_artifact_manifest(manifest: Any) -> dict[str, Any]:
    """Re-hash a manifest and report every mismatch; fail closed on absence.

    Returns ``{"ok": bool, "mismatches": [...]}``.  A missing file, a changed
    hash/size, a broken entry or an unrecognised schema is a mismatch.  The
    record itself is never hashed here.
    """
    if not isinstance(manifest, Mapping) or manifest.get("schema") != MANIFEST_SCHEMA:
        return {"ok": False, "mismatches": ["schema"]}
    mismatches: list[str] = []
    if not _entry_matches(manifest.get("lib")):
        mismatches.append("lib")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        mismatches.append("inputs")
    else:
        for label, entry in inputs.items():
            if not _entry_matches(entry):
                mismatches.append(f"inputs.{label}")
    return {"ok": not mismatches, "mismatches": mismatches}
