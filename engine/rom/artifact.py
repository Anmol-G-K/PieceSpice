"""On-disk form of the solver ROM: versioned, self-describing, hash-checked.

The artifact is the runtime input, which is why it carries its schema version
and a payload hash.  A solver run must be able to say *which* table it stepped,
exactly as the UROM records carry ``emitted_lib_sha256`` -- a number without its
artifact identity cannot be re-derived, and in this project that is the
difference between a measurement and a claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from engine.rom.spec import SCHEMA, SolverRom


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    """Deterministic JSON: sorted keys, no whitespace, ASCII."""
    body = {k: v for k, v in payload.items() if k != "sha256"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("ascii")


def payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def dump(rom: SolverRom, path: str | Path) -> Path:
    """Write *rom* to *path* with its schema version and payload hash."""
    payload = rom.to_dict()
    payload["sha256"] = payload_sha256(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="ascii")
    return target


def load(path: str | Path, *, verify: bool = True) -> SolverRom:
    """Read a solver ROM, optionally checking its payload hash."""
    raw = Path(path).read_text(encoding="utf-8")
    payload = json.loads(raw)
    if payload.get("schema") != SCHEMA:
        raise ValueError(
            f"{path}: schema {payload.get('schema')!r} is not {SCHEMA!r}")
    if verify:
        recorded = payload.get("sha256")
        if not recorded:
            raise ValueError(f"{path}: artifact carries no sha256")
        actual = payload_sha256(payload)
        if actual != recorded:
            raise ValueError(
                f"{path}: payload hash mismatch (recorded {recorded[:12]}..., "
                f"computed {actual[:12]}...)")
    return SolverRom.from_dict(payload)


def manifest(path: str | Path) -> dict[str, Any]:
    """Identity of an artifact, for a validation record."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "schema": payload.get("schema"),
        "device": payload.get("device"),
        "sha256": payload.get("sha256"),
        "verified": payload.get("sha256") == payload_sha256(payload),
    }
