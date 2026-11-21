"""Versioned, overridable acceptance bars for the unified-ROM gates.

The YAML policy at ``config/acceptance.yaml`` is the source of truth for the
G1/G3/G4/G5/G6 bars and the speed bootstrap settings.  Loading it is
fail-closed: a malformed file raises ``ValueError`` and is never replaced by
defaults.  The built-in values below are used *only* when the file is absent.

``pipeline.unified_rom`` keeps its historical module constants as
backward-compatible aliases for existing callers; they mirror these fallback
values and are checked against the shipped policy by the test suite.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from common.provenance import sha256_bytes

__all__ = [
    "AcceptancePolicy",
    "DEFAULT_PATH",
    "DEFAULT_SCHEMA",
    "FALLBACK_DPT_BARS_PCT",
    "FALLBACK_STATIC_BARS_PCT",
    "load_acceptance",
]

DEFAULT_SCHEMA = "piecespice.acceptance/1.0"
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "acceptance.yaml"
DEFAULT_OVERRIDE_SEMANTICS = (
    "Every value in this policy may be replaced with an application "
    "requirement by editing the YAML file only; no code change is needed. "
    "A malformed file fails closed."
)
SOURCE_LABELS = frozenset(("recorded_baseline", "project_decision"))

# Mirrors pipeline.unified_rom's public constants (kept there for callers).
# Importing that module here would be circular, so the fallback values live
# here and tests assert the two sets stay identical.
FALLBACK_VENDORS = ("irfs7730pbf", "csd16412q5a", "epc2216", "cpm2-1200")
FALLBACK_METHODS = ("trap/default", "gear-both")
FALLBACK_STATIC_BARS_PCT: dict[str, float] = {
    "irfs7730pbf": 7.0683,
    "csd16412q5a": 8.4935,
    "epc2216": 2.0394,
    "cpm2-1200": 3.4287,
}
FALLBACK_DPT_BARS_PCT: dict[str, dict[str, tuple[float, float]]] = {
    "trap/default": {
        "irfs7730pbf": (10.0, 0.60),
        "csd16412q5a": (5.583, 0.048),
        "epc2216": (4.894, 0.049),
        "cpm2-1200": (3.0, 0.261),
    },
    "gear-both": {
        "irfs7730pbf": (12.771, 0.5699),
        "csd16412q5a": (5.467, 0.0442),
        "epc2216": (5.075, 0.0461),
        "cpm2-1200": (4.122, 0.2176),
    },
}
FALLBACK_WORST_ENDPOINT_PCT = 4.13
FALLBACK_G1_BAND_PP = 0.05
FALLBACK_TIMING_BAND_REL = 0.05
FALLBACK_TIMING_BAND_ABS = 1.0e-12
FALLBACK_CHARGE_BUDGET_PCT = 0.5
FALLBACK_SPEED_MIN_SOLVER_SPEEDUP = 1.05
FALLBACK_SPEED_CONFIDENCE = 0.95
FALLBACK_SPEED_N_RESAMPLES = 2000

_FALLBACK_SOURCES: dict[str, str] = {
    **{f"static.mean_peak_rel_rmse_pct_max.{vendor}": "recorded_baseline"
       for vendor in FALLBACK_VENDORS},
    "static.worst_endpoint_pct": "recorded_baseline",
    "static.g1_band_pp": "project_decision",
    **{f"dpt.{method}.{vendor}": (
        "project_decision" if (method, vendor) in (
            ("trap/default", "irfs7730pbf"), ("trap/default", "cpm2-1200"))
        else "recorded_baseline")
       for method in FALLBACK_METHODS for vendor in FALLBACK_VENDORS},
    "timing.band_rel": "project_decision",
    "timing.band_abs": "project_decision",
    "charge.budget_pct": "recorded_baseline",
    "speed.min_solver_speedup": "project_decision",
    "speed.confidence": "project_decision",
    "speed.n_resamples": "project_decision",
}


def _finite_positive(key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key}: expected a finite positive number")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{key}: expected a finite positive number")
    return out


def _in_open_unit_interval(key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key}: expected a number in (0, 1)")
    out = float(value)
    if not math.isfinite(out) or not 0.0 < out < 1.0:
        raise ValueError(f"{key}: expected a number in (0, 1)")
    return out


@dataclass(frozen=True)
class AcceptancePolicy:
    """Validated acceptance bars plus the provenance of the loaded policy."""

    static_bars_pct: Mapping[str, float]
    worst_endpoint_pct: float
    g1_band_pp: float
    dpt_bars_pct: Mapping[str, Mapping[str, tuple[float, float]]]
    timing_band_rel: float
    timing_band_abs: float
    charge_budget_pct: float
    speed_min_solver_speedup: float
    speed_confidence: float
    speed_n_resamples: int
    sources: Mapping[str, str]
    path: Path | None = None
    schema: str = DEFAULT_SCHEMA
    sha256: str | None = None
    override_semantics: str = DEFAULT_OVERRIDE_SEMANTICS

    def __post_init__(self) -> None:
        static = {
            str(vendor): _finite_positive(
                f"static.mean_peak_rel_rmse_pct_max.{vendor}", value)
            for vendor, value in dict(self.static_bars_pct).items()
        }
        dpt: dict[str, dict[str, tuple[float, float]]] = {}
        for method, rows in dict(self.dpt_bars_pct).items():
            dpt[str(method)] = {}
            for vendor, pair in dict(rows).items():
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError(f"dpt.{method}.{vendor}: expected "
                                     "[vds_pct_max, id_pct_max]")
                dpt[str(method)][str(vendor)] = (
                    _finite_positive(f"dpt.{method}.{vendor}[0]", pair[0]),
                    _finite_positive(f"dpt.{method}.{vendor}[1]", pair[1]),
                )
        confidence = _in_open_unit_interval(
            "speed.confidence", self.speed_confidence)
        band_rel = _in_open_unit_interval("timing.band_rel", self.timing_band_rel)
        speed_min = _finite_positive(
            "speed.min_solver_speedup", self.speed_min_solver_speedup)
        if speed_min <= 1.0:
            raise ValueError("speed.min_solver_speedup: must be greater than 1")
        n_resamples = self.speed_n_resamples
        if (isinstance(n_resamples, bool) or not isinstance(n_resamples, int)
                or n_resamples < 1):
            raise ValueError("speed.n_resamples: expected a positive integer")
        sources = dict(self.sources)
        required = (
            {f"static.mean_peak_rel_rmse_pct_max.{vendor}" for vendor in static}
            | {f"dpt.{method}.{vendor}" for method, rows in dpt.items()
               for vendor in rows}
            | {"static.worst_endpoint_pct", "static.g1_band_pp",
               "timing.band_rel", "timing.band_abs", "charge.budget_pct",
               "speed.min_solver_speedup", "speed.confidence",
               "speed.n_resamples"}
        )
        missing = sorted(required - set(sources))
        if missing:
            raise ValueError(f"acceptance policy missing source labels: {missing}")
        bad = {key: value for key, value in sources.items()
               if value not in SOURCE_LABELS}
        if bad:
            raise ValueError(f"acceptance policy has invalid source labels: {bad}")
        object.__setattr__(self, "static_bars_pct", static)
        object.__setattr__(self, "dpt_bars_pct", dpt)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "speed_confidence", confidence)
        object.__setattr__(self, "timing_band_rel", band_rel)
        object.__setattr__(self, "speed_n_resamples", n_resamples)

    def as_record(self) -> dict[str, Any]:
        """JSON-safe provenance/value snapshot for a validation record."""
        return {
            "path": str(self.path) if self.path is not None else None,
            "schema": self.schema,
            "sha256": self.sha256,
            "sources": dict(self.sources),
            "override_semantics": self.override_semantics,
            "values": {
                "static_bars_pct": dict(self.static_bars_pct),
                "worst_endpoint_pct": self.worst_endpoint_pct,
                "g1_band_pp": self.g1_band_pp,
                "dpt_bars_pct": {method: dict(rows)
                                 for method, rows in self.dpt_bars_pct.items()},
                "timing_band_rel": self.timing_band_rel,
                "timing_band_abs": self.timing_band_abs,
                "charge_budget_pct": self.charge_budget_pct,
                "speed_min_solver_speedup": self.speed_min_solver_speedup,
                "speed_confidence": self.speed_confidence,
                "speed_n_resamples": self.speed_n_resamples,
            },
        }


def _fallback_policy() -> AcceptancePolicy:
    """Documented fallback: only used when the policy file is absent."""
    return AcceptancePolicy(
        static_bars_pct=dict(FALLBACK_STATIC_BARS_PCT),
        worst_endpoint_pct=FALLBACK_WORST_ENDPOINT_PCT,
        g1_band_pp=FALLBACK_G1_BAND_PP,
        dpt_bars_pct={method: dict(rows)
                      for method, rows in FALLBACK_DPT_BARS_PCT.items()},
        timing_band_rel=FALLBACK_TIMING_BAND_REL,
        timing_band_abs=FALLBACK_TIMING_BAND_ABS,
        charge_budget_pct=FALLBACK_CHARGE_BUDGET_PCT,
        speed_min_solver_speedup=FALLBACK_SPEED_MIN_SOLVER_SPEEDUP,
        speed_confidence=FALLBACK_SPEED_CONFIDENCE,
        speed_n_resamples=FALLBACK_SPEED_N_RESAMPLES,
        sources=dict(_FALLBACK_SOURCES),
    )


def _require_section(root: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    section = root.get(key)
    if not isinstance(section, Mapping):
        raise ValueError(f"acceptance policy: missing/invalid section {key!r}")
    return section


def _require_entry(section: Mapping[str, Any], key: str,
                   where: str) -> Mapping[str, Any]:
    entry = section.get(key)
    if (not isinstance(entry, Mapping) or "value" not in entry
            or "source" not in entry):
        raise ValueError(f"acceptance policy: {where}.{key} must carry "
                         "'value' and 'source'")
    return entry


def _source_label(where: str, value: Any) -> str:
    if not isinstance(value, str) or value not in SOURCE_LABELS:
        raise ValueError(f"acceptance policy: {where}.source must be one of "
                         f"{sorted(SOURCE_LABELS)}")
    return value


def load_acceptance(path: str | Path | None = None) -> AcceptancePolicy:
    """Load and validate ``config/acceptance.yaml`` (or ``path``).

    A missing file returns the documented fallback policy; a present but
    malformed file raises ``ValueError``.  This distinction is deliberate:
    overrides must never be silently ignored.
    """
    target = Path(path) if path is not None else DEFAULT_PATH
    if not target.is_file():
        return _fallback_policy()
    try:
        raw_bytes = target.read_bytes()
        text = raw_bytes.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"acceptance policy: cannot read {target}: {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"acceptance policy: invalid YAML in {target}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("acceptance policy: root must be a mapping")
    if raw.get("schema") != DEFAULT_SCHEMA:
        raise ValueError("acceptance policy: schema must be "
                         f"{DEFAULT_SCHEMA!r}")
    semantics = raw.get("override_semantics")
    if not isinstance(semantics, str) or not semantics.strip():
        raise ValueError("acceptance policy: override_semantics must be a "
                         "non-empty string")

    sources: dict[str, str] = {}
    static_section = _require_section(raw, "static")
    mean_section = _require_section(static_section, "mean_peak_rel_rmse_pct_max")
    static_bars: dict[str, float] = {}
    for vendor in FALLBACK_VENDORS:
        key = f"static.mean_peak_rel_rmse_pct_max.{vendor}"
        entry = _require_entry(mean_section, vendor, "static.mean_peak_rel_rmse_pct_max")
        static_bars[vendor] = _finite_positive(key, entry["value"])
        sources[key] = _source_label(key, entry["source"])
    for field in ("worst_endpoint_pct", "g1_band_pp"):
        entry = _require_entry(static_section, field, "static")
        key = f"static.{field}"
        sources[key] = _source_label(key, entry["source"])
    worst_endpoint = _finite_positive("static.worst_endpoint_pct",
                                      static_section["worst_endpoint_pct"]["value"])
    g1_band = _finite_positive("static.g1_band_pp",
                               static_section["g1_band_pp"]["value"])

    dpt_section = _require_section(raw, "dpt")
    dpt_bars: dict[str, dict[str, tuple[float, float]]] = {}
    for method in FALLBACK_METHODS:
        method_section = _require_section(dpt_section, method)
        dpt_bars[method] = {}
        for vendor in FALLBACK_VENDORS:
            key = f"dpt.{method}.{vendor}"
            entry = _require_entry(method_section, vendor, f"dpt.{method}")
            pair = entry["value"]
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"acceptance policy: {key} must be a "
                                 "[vds_pct_max, id_pct_max] pair")
            dpt_bars[method][vendor] = (
                _finite_positive(f"{key}[0]", pair[0]),
                _finite_positive(f"{key}[1]", pair[1]),
            )
            sources[key] = _source_label(key, entry["source"])

    timing_section = _require_section(raw, "timing")
    charge_section = _require_section(raw, "charge")
    speed_section = _require_section(raw, "speed")
    scalar_entries = {
        "timing.band_rel": _require_entry(timing_section, "band_rel", "timing"),
        "timing.band_abs": _require_entry(timing_section, "band_abs", "timing"),
        "charge.budget_pct": _require_entry(
            charge_section, "budget_pct", "charge"),
        "speed.min_solver_speedup": _require_entry(
            speed_section, "min_solver_speedup", "speed"),
        "speed.confidence": _require_entry(
            speed_section, "confidence", "speed"),
        "speed.n_resamples": _require_entry(
            speed_section, "n_resamples", "speed"),
    }
    for key, entry in scalar_entries.items():
        sources[key] = _source_label(key, entry["source"])

    return AcceptancePolicy(
        static_bars_pct=static_bars,
        worst_endpoint_pct=worst_endpoint,
        g1_band_pp=g1_band,
        dpt_bars_pct=dpt_bars,
        timing_band_rel=scalar_entries["timing.band_rel"]["value"],
        timing_band_abs=_finite_positive(
            "timing.band_abs", scalar_entries["timing.band_abs"]["value"]),
        charge_budget_pct=_finite_positive(
            "charge.budget_pct", scalar_entries["charge.budget_pct"]["value"]),
        speed_min_solver_speedup=scalar_entries[
            "speed.min_solver_speedup"]["value"],
        speed_confidence=scalar_entries["speed.confidence"]["value"],
        speed_n_resamples=scalar_entries["speed.n_resamples"]["value"],
        sources=sources,
        path=target,
        schema=str(raw["schema"]),
        sha256=sha256_bytes(raw_bytes),
        override_semantics=semantics,
    )
