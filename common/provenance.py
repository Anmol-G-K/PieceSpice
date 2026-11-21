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
    "bootstrap_ci",
    "environment_metadata",
    "file_hashes",
    "sha256_bytes",
    "sha256_file",
    "sha256_text",
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
