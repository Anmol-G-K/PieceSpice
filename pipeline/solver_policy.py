"""Stage 7: the solver acceptance policy, read from ``config/acceptance.yaml``.

Adds a ``solver:`` section to the existing policy so the engine's own bar lives
in the same file as the model's, with the same rule: **every number is labelled
``recorded_baseline`` or ``project_decision``**, and a malformed file fails
closed instead of falling back silently.

The bar is declared once and deliberately **not** tuned to what the solver
currently achieves.  A gate that is set just under the measurement is not a
gate; it is a description.
"""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_SCHEMA = "piecespice.acceptance/1.0"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "config" / "acceptance.yaml"

SOURCE_LABELS = frozenset(("recorded_baseline", "project_decision"))

#: Used only when the policy file is absent.  Kept identical to the YAML.
FALLBACK_SOLVER_BARS: dict[str, float] = {
    "equivalence_rel_pct_max": 2.0,
    "min_solver_speedup": 1.05,
}
FALLBACK_SOLVER_SOURCES: dict[str, str] = {
    "solver.equivalence_rel_pct_max": "project_decision",
    "solver.min_solver_speedup": "project_decision",
}


class SolverPolicyError(ValueError):
    """The solver section is missing or malformed."""


def _entry(section: dict, key: str, where: str) -> float:
    entry = section.get(key)
    if not isinstance(entry, dict) or "value" not in entry or "source" not in entry:
        raise SolverPolicyError(
            f"acceptance policy: {where}.{key} must carry 'value' and 'source'")
    if entry["source"] not in SOURCE_LABELS:
        raise SolverPolicyError(
            f"acceptance policy: {where}.{key}.source must be one of "
            f"{sorted(SOURCE_LABELS)}")
    value = entry["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SolverPolicyError(
            f"acceptance policy: {where}.{key}.value must be a number")
    if value <= 0.0:
        raise SolverPolicyError(
            f"acceptance policy: {where}.{key}.value must be positive")
    return float(value)


def load_solver_bars(path: str | Path | None = None) -> tuple[dict[str, float],
                                                              dict[str, str]]:
    """Return ``(bars, sources)`` from the policy's ``solver:`` section.

    Raises when the section is present but malformed.  An absent *file* uses the
    documented fallback; an absent *section* in a present file is an error,
    because a silently defaulted bar is indistinguishable from a declared one.
    """
    target = Path(path) if path is not None else DEFAULT_PATH
    if not target.is_file():
        return dict(FALLBACK_SOLVER_BARS), dict(FALLBACK_SOLVER_SOURCES)
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SolverPolicyError("acceptance policy: root must be a mapping")
    if raw.get("schema") != DEFAULT_SCHEMA:
        raise SolverPolicyError(
            f"acceptance policy: schema must be {DEFAULT_SCHEMA!r}")
    section = raw.get("solver")
    if not isinstance(section, dict):
        raise SolverPolicyError(
            "acceptance policy: missing/invalid section 'solver'")
    bars: dict[str, float] = {}
    sources: dict[str, str] = {}
    for key in ("equivalence_rel_pct_max", "min_solver_speedup"):
        bars[key] = _entry(section, key, "solver")
        sources[f"solver.{key}"] = section[key]["source"]
    return bars, sources
