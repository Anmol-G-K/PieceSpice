"""Fixed-topology scalar calibration (T3, spec 4.3 / topology study section 5).

The only tunable scalars are the four named WOL gate-side knobs, in the fixed
order ``A_GD, O_GD, A_GS, A_RGI``. There is no positional tuple anywhere in
this module and no Qoss/C_DS scalar: that part of the ROM is frozen.

Objective (WOL Eq. 1)::

    J(theta) = sum_c sum_m |emp_c,m - sim_c,m(theta)| / |emp_c,m|

Golden values are supplied once per condition in ``condition["emp"]``; only
candidate ROM points are re-simulated through ``condition["simulate"]``.
Non-finite metrics (missing crossings) and zero references are skipped, never
replaced by zero. Non-completing runs are penalised and flagged, never treated
as fast results. The reported theta is the best **held-out** point visited,
not the best fit-set point.

A condition is a mapping with these fields:

``name`` (str, optional), ``emp`` (mapping metric -> golden value, required),
``simulate`` (callable theta -> :class:`ConditionResult`, required),
``held_out`` (bool, default False), ``weight`` (float, default 1.0),
``completed`` (bool, default True; golden completion guard).

No emitter, netlist, config or gate code lives here. The DPT adapter at the
bottom only composes existing builders, benches and metrics.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import numpy as np
from scipy.optimize import minimize

from common.types import (
    PWLModelSpec,
    UNIFIED_TUNING_IDENTITY,
    UNIFIED_TUNING_KEYS,
    UNIFIED_TUNING_PROVENANCE,
)

_LOGGER = logging.getLogger(__name__)

#: Declared solver-method labels. The trap card is LTspice's default; the
#: Gear card is written into the ROM deck only for a matched-gear-both row.
#: The caller must cache the golden arm at the same label (G0).
_METHOD_OPTIONS = {"trap/default": "", "gear-both": ".options method=gear"}

#: Algorithmic penalty for an incomplete/unusable candidate. It carries no
#: physical meaning and is never reported as a waveform or fidelity number.
_INCOMPLETE_PENALTY = 1.0e6
_BOUND_TOL = 1e-12

__all__ = [
    "ConditionResult",
    "TuningResult",
    "tune_scalars",
    "apply_scalars",
    "derive_o_gd_bound",
    "make_dpt_simulator",
]

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConditionResult:
    """One candidate simulation at one calibration condition."""

    metrics: Dict[str, float]
    completed: bool
    points: int | None = None
    iterations: int | None = None
    t_last: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "completed": bool(self.completed),
            "points": self.points,
            "iterations": self.iterations,
            "t_last": self.t_last,
        }


@dataclass(frozen=True)
class TuningResult:
    """Bounded-search outcome, ready for the unified validation record."""

    vendor: str
    method: str
    theta: Dict[str, float]
    bounds: Dict[str, tuple[float, float]]
    weights: Dict[str, float]
    objective_fit: float
    objective_held_out: float | None
    baseline_fit: float
    baseline_held_out: float | None
    evaluations: int
    optimizer: str
    golden_completion: Dict[str, bool]
    condition_results: Dict[str, Dict[str, Any]]
    held_out_scores: Dict[str, float]
    verdict: str
    provenance: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vendor": self.vendor,
            "method": self.method,
            "theta": dict(self.theta),
            "bounds": {k: tuple(v) for k, v in self.bounds.items()},
            "weights": dict(self.weights),
            "objective_fit": self.objective_fit,
            "objective_held_out": self.objective_held_out,
            "baseline_fit": self.baseline_fit,
            "baseline_held_out": self.baseline_held_out,
            "evaluations": self.evaluations,
            "optimizer": self.optimizer,
            "golden_completion": dict(self.golden_completion),
            "condition_results": {k: dict(v)
                                  for k, v in self.condition_results.items()},
            "held_out_scores": dict(self.held_out_scores),
            "verdict": self.verdict,
            "provenance": dict(self.provenance),
        }


# ---------------------------------------------------------------------------
# Named-parameter validation (no positional tuple is ever accepted)
# ---------------------------------------------------------------------------

def _normalise_theta(theta: Mapping[str, Any]) -> Dict[str, float]:
    if not isinstance(theta, Mapping):
        raise TypeError("theta must be a named mapping, not a positional tuple")
    unknown = [k for k in theta if k not in UNIFIED_TUNING_KEYS]
    if unknown:
        raise ValueError(f"theta has unknown fields {unknown}")
    out: Dict[str, float] = {}
    for key in UNIFIED_TUNING_KEYS:
        if key not in theta:
            raise ValueError(f"theta missing named field {key!r}")
        value = theta[key]
        if isinstance(value, bool):
            raise ValueError(f"theta[{key!r}] must be a real number")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"theta[{key!r}] must be a real number") from exc
        if not math.isfinite(number):
            raise ValueError(f"theta[{key!r}] must be finite")
        out[key] = number
    if out["O_GD"] < 0.0:
        raise ValueError("theta[O_GD] must be >= 0 (monotone Q_RSS offset)")
    for key in ("A_GD", "A_GS", "A_RGI"):
        if out[key] <= 0.0:
            raise ValueError(f"theta[{key!r}] must be > 0")
    return out


def _normalise_bounds(bounds: Mapping[str, Any]) -> Dict[str, tuple[float, float]]:
    if not isinstance(bounds, Mapping):
        raise TypeError("bounds must be a named mapping, not a positional tuple")
    unknown = [k for k in bounds if k not in UNIFIED_TUNING_KEYS]
    if unknown:
        raise ValueError(f"bounds has unknown fields {unknown}")
    out: Dict[str, tuple[float, float]] = {}
    for key in UNIFIED_TUNING_KEYS:
        if key not in bounds:
            raise ValueError(f"bounds missing named field {key!r}")
        pair = bounds[key]
        if (isinstance(pair, (str, bytes)) or not isinstance(pair, (list, tuple))
                or len(pair) != 2):
            raise ValueError(f"bounds[{key!r}] must be a (lo, hi) pair")
        try:
            lo, hi = float(pair[0]), float(pair[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"bounds[{key!r}] must be numeric") from exc
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"bounds[{key!r}] must be finite with lo < hi")
        if key == "O_GD" and lo < 0.0:
            raise ValueError("bounds[O_GD] lower bound must be >= 0")
        out[key] = (lo, hi)
    return out


def _check_bounds(theta: Mapping[str, float],
                  bounds: Mapping[str, tuple[float, float]], label: str) -> None:
    for key in UNIFIED_TUNING_KEYS:
        lo, hi = bounds[key]
        value = float(theta[key])
        if not (lo - _BOUND_TOL <= value <= hi + _BOUND_TOL):
            raise ValueError(
                f"{label}: theta[{key!r}]={value:.6e} outside "
                f"[{lo:.6e}, {hi:.6e}]")


def _normalise_weights(weights: Mapping[str, Any] | None):
    """Return (metric_weights, condition_weights)."""
    if weights is None:
        return {}, {}
    if not isinstance(weights, Mapping):
        raise TypeError("weights must be a mapping")
    block = weights.get("metrics", weights)
    if not isinstance(block, Mapping):
        raise TypeError("weights['metrics'] must be a mapping")
    metric: Dict[str, float] = {}
    for name, value in block.items():
        if name == "conditions":
            continue
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"weights['{name}'] must be finite and >= 0")
        metric[str(name)] = number
    condition: Dict[str, float] = {}
    named = weights.get("conditions", {})
    if not isinstance(named, Mapping):
        raise TypeError("weights['conditions'] must be a mapping")
    for name, value in named.items():
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"weights['conditions']['{name}'] must be >= 0")
        condition[str(name)] = number
    return metric, condition


@dataclass(frozen=True)
class _Condition:
    name: str
    emp: Dict[str, Any]
    simulate: Callable[[Dict[str, float]], Any]
    held_out: bool
    weight: float
    completed: bool


def _normalise_condition(raw: Mapping[str, Any], index: int) -> _Condition:
    if not isinstance(raw, Mapping):
        raise TypeError(f"conditions[{index}] must be a mapping")
    name = str(raw.get("name", f"c{index}"))
    emp = raw.get("emp", raw.get("empirical", raw.get("golden")))
    if not isinstance(emp, Mapping) or not emp:
        raise ValueError(f"condition {name!r} needs a non-empty 'emp' mapping")
    simulate = raw.get("simulate")
    if not callable(simulate):
        raise ValueError(f"condition {name!r} needs simulate(theta) callable")
    weight = float(raw.get("weight", 1.0))
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(f"condition {name!r} weight must be finite and >= 0")
    return _Condition(
        name=name,
        emp=dict(emp),
        simulate=simulate,
        held_out=bool(raw.get("held_out", False)),
        weight=weight,
        completed=bool(raw.get("completed", True)),
    )


def _coerce_result(result: Any) -> ConditionResult:
    if isinstance(result, ConditionResult):
        return result
    if isinstance(result, Mapping):
        metrics = result.get("metrics", result)
        if not isinstance(metrics, Mapping):
            raise TypeError("simulate() metrics must be a mapping")
        return ConditionResult(
            metrics=dict(metrics),
            completed=bool(result.get("completed", True)),
            points=result.get("points"),
            iterations=result.get("iterations"),
            t_last=result.get("t_last"),
        )
    raise TypeError("simulate() must return ConditionResult or a mapping")


def _relative_error(emp: Any, sim: Any) -> float | None:
    """WOL per-metric term; None means the term is not evaluable."""
    if emp is None or sim is None:
        return None
    try:
        reference, value = float(emp), float(sim)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(reference) and math.isfinite(value)):
        return None
    if reference == 0.0:
        return None
    return abs(reference - value) / abs(reference)


@dataclass(frozen=True)
class _Score:
    fit: float
    held_out: float | None
    details: Dict[str, Dict[str, Any]]


def _score_theta(theta: Dict[str, float], conditions: list[_Condition],
                 metric_weights: Mapping[str, float],
                 condition_weights: Mapping[str, float]) -> _Score:
    has_held_out = any(c.held_out for c in conditions)
    fit = 0.0
    held = 0.0
    details: Dict[str, Dict[str, Any]] = {}
    for cond in conditions:
        weight = condition_weights.get(cond.name, cond.weight)
        if not cond.completed:
            details[cond.name] = {
                "golden_completed": False, "completed": False,
                "score": None, "skipped": True,
            }
            continue
        try:
            result = _coerce_result(cond.simulate(dict(theta)))
        except Exception as exc:  # solver abort/timeout is a failure, not a number
            _LOGGER.warning("%s: candidate simulation failed: %s", cond.name, exc)
            details[cond.name] = {
                "golden_completed": True, "completed": False, "score": None,
                "error": str(exc),
            }
            fit += _INCOMPLETE_PENALTY * weight
            if cond.held_out:
                held += _INCOMPLETE_PENALTY * weight
            continue
        if not result.completed:
            details[cond.name] = {
                "golden_completed": True, "completed": False, "score": None,
                "points": result.points, "iterations": result.iterations,
                "t_last": result.t_last,
            }
            fit += _INCOMPLETE_PENALTY * weight
            if cond.held_out:
                held += _INCOMPLETE_PENALTY * weight
            continue
        total = 0.0
        used = 0
        skipped: list[str] = []
        for metric, reference in cond.emp.items():
            w_metric = metric_weights.get(str(metric), 1.0)
            if w_metric == 0.0:
                skipped.append(str(metric))
                continue
            term = _relative_error(reference, result.metrics.get(metric))
            if term is None:
                skipped.append(str(metric))
                continue
            total += w_metric * term
            used += 1
        if used == 0:
            details[cond.name] = {
                "golden_completed": True, "completed": True, "score": None,
                "skipped_metrics": skipped,
            }
            fit += _INCOMPLETE_PENALTY * weight
            if cond.held_out:
                held += _INCOMPLETE_PENALTY * weight
            continue
        details[cond.name] = {
            "golden_completed": True, "completed": True, "score": total,
            "metrics_used": used, "skipped_metrics": skipped,
            "points": result.points, "iterations": result.iterations,
            "t_last": result.t_last,
        }
        if cond.held_out:
            held += total
        else:
            fit += total
    return _Score(fit=fit, held_out=held if has_held_out else None,
                  details=details)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class _BudgetExceeded(Exception):
    """Internal stop signal: max_evals is exhausted; keep the best so far."""


def tune_scalars(
    vendor: str,
    *,
    method: str,
    conditions: list[dict],
    bounds: dict[str, tuple[float, float]],
    weights: dict | None = None,
    max_evals: int,
) -> TuningResult:
    """Minimise WOL Eq. 1 over the four named fixed-topology scalars.

    ``method`` is a declared equal-sandbox label; it is recorded verbatim and
    handed to the condition simulator. Powell (bounded, derivative-free)
    starts at the mathematical identity, never at a WOL Table 3 value. The
    returned theta is the best held-out objective seen during the search.
    """
    if method not in _METHOD_OPTIONS:
        raise ValueError(f"method must be one of {sorted(_METHOD_OPTIONS)}")
    named_bounds = _normalise_bounds(bounds)
    metric_weights, condition_weights = _normalise_weights(weights)
    conds = [_normalise_condition(raw, i) for i, raw in enumerate(conditions)]
    if not conds:
        raise ValueError("at least one calibration condition is required")
    budget = int(max_evals)
    if budget < 1:
        raise ValueError("max_evals must be at least 1")
    held_out_names = {c.name for c in conds if c.held_out}
    evaluable_names = {c.name for c in conds if c.completed}

    state: dict[str, Any] = {
        "count": 0, "cache": {}, "best": None, "best_rank": float("inf"),
    }

    def evaluate(theta: Mapping[str, Any]) -> _Score:
        named = _normalise_theta(theta)
        _check_bounds(named, named_bounds, vendor)
        key = tuple(named[k] for k in UNIFIED_TUNING_KEYS)
        cached = state["cache"].get(key)
        if cached is not None:
            return cached
        if state["count"] >= budget:
            raise _BudgetExceeded
        state["count"] += 1
        score = _score_theta(named, conds, metric_weights, condition_weights)
        state["cache"][key] = score
        rank = score.held_out if score.held_out is not None else score.fit
        if math.isfinite(rank) and rank < state["best_rank"]:
            state["best_rank"] = rank
            state["best"] = (named, score)
        return score

    identity = dict(UNIFIED_TUNING_IDENTITY)
    try:
        baseline = evaluate(identity)
    except _BudgetExceeded as exc:
        raise ValueError("max_evals too small to evaluate the identity baseline") from exc

    # Powell works in unit coordinates: O_GD is in farads (~1e-12) while the
    # A_* multipliers are order 1. Optimising the raw vector makes the farad
    # axis invisible to Powell's relative step/tolerance logic. Bounds stay
    # per named field; the unit map is bijective inside them.
    def to_unit(theta: Mapping[str, float]) -> list[float]:
        return [
            (float(theta[k]) - named_bounds[k][0])
            / (named_bounds[k][1] - named_bounds[k][0])
            for k in UNIFIED_TUNING_KEYS
        ]

    def to_physical(x) -> Dict[str, float]:
        return {
            k: named_bounds[k][0]
            + float(v) * (named_bounds[k][1] - named_bounds[k][0])
            for k, v in zip(UNIFIED_TUNING_KEYS, x)
        }

    initial = [min(1.0, max(0.0, u)) for u in to_unit(identity)]
    limits = [(0.0, 1.0)] * len(UNIFIED_TUNING_KEYS)

    def objective(x) -> float:
        return evaluate(to_physical(x)).fit

    try:
        minimize(objective, initial, method="Powell", bounds=limits,
                 options={"maxfev": budget})
    except _BudgetExceeded:
        _LOGGER.info(
            "%s: tuning budget exhausted; best held-out point kept", vendor)

    best_theta, best_score = state["best"] or (identity, baseline)
    gold_flags = {c.name: c.completed for c in conds}
    held_scores = {
        name: details["score"]
        for name, details in best_score.details.items()
        if name in held_out_names and details.get("score") is not None
    }

    if not evaluable_names or not best_score.details:
        verdict = "infeasible"
    elif best_score.held_out is None or baseline.held_out is None:
        verdict = "infeasible"  # no held-out arm: cannot certify generalisation
    elif not (math.isfinite(best_score.held_out)
              and math.isfinite(baseline.held_out)):
        verdict = "infeasible"
    elif not all(best_score.details[name].get("completed", False)
                 for name in evaluable_names):
        verdict = "reject"  # candidate failed a completion guard
    elif best_score.held_out >= baseline.held_out:
        verdict = "reject"  # no held-out gain over the identity transform
    else:
        verdict = "accept"

    provenance = dict(UNIFIED_TUNING_PROVENANCE)
    provenance["objective"] = (
        "WOL Eq.1: sum_c sum_m |emp-sim|/|emp|; non-finite/zero-reference "
        "terms skipped, never zero-filled")
    provenance["method_label"] = (
        f"{method} (declared; golden cache must use the same method card)")
    provenance["optimizer"] = (
        "scipy.optimize.minimize/Powell, bounded in unit-normalised field "
        "coordinates (avoids the farad-vs-dimensionless scale mismatch), "
        "identity start; WOL Table 3 numbers are scale hints only")
    provenance["incomplete_penalty"] = (
        f"{_INCOMPLETE_PENALTY:g} per unusable condition: algorithmic, not a "
        "physical or measured value")
    provenance["held_out_selection"] = (
        "theta with the lowest held-out objective seen during the search")

    return TuningResult(
        vendor=vendor,
        method=method,
        theta=dict(best_theta),
        bounds=dict(named_bounds),
        weights=dict(metric_weights),
        objective_fit=float(best_score.fit),
        objective_held_out=best_score.held_out,
        baseline_fit=float(baseline.fit),
        baseline_held_out=baseline.held_out,
        evaluations=int(state["count"]),
        optimizer="scipy.optimize.minimize/Powell",
        golden_completion=gold_flags,
        condition_results={k: dict(v) for k, v in best_score.details.items()},
        held_out_scores=held_scores,
        verdict=verdict,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Config/curve-derived O_GD budget (spec 4.3 / UNIFIED_TUNING_PROVENANCE)
# ---------------------------------------------------------------------------

def derive_o_gd_bound(spec: PWLModelSpec) -> float:
    """Upper bound [F] for the additive ``O_GD`` charge offset.

    The physical cap is one full Q_RSS at the top of the curve-4 sweep,
    ``c_rss_lin * Vds_cap_max`` (``UNIFIED_TUNING_PROVENANCE["offset_bound"]``).
    ``Vds_cap_max`` comes from this spec's Q_RSS voltage grid when the charge
    tables are populated; producer specs may instead carry the config-derived
    cap in ``theta_bounds["O_GD"]``. Both are legitimate sources, so the larger
    is used: a drifted literal bracket can never shrink a valid curve budget,
    and a config-derived producer cap is never discarded just because the
    emitted Q_RSS grid is empty.
    """
    c_rss = float(spec.metadata["c_rss_lin"])
    v_cap = max((float(v) for v in spec.q_rss_vs_v), default=0.0)
    curve_cap = max(c_rss * v_cap, c_rss)
    raw_bounds = spec.metadata.get("theta_bounds")
    meta_cap = 0.0
    if isinstance(raw_bounds, Mapping) and "O_GD" in raw_bounds:
        pair = raw_bounds["O_GD"]
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            meta_cap = float(pair[1])
    qgd_ref = max(curve_cap, meta_cap)
    if not math.isfinite(qgd_ref) or qgd_ref <= 0.0:
        raise ValueError("cannot derive a positive O_GD bound from curve 4")
    return qgd_ref


def _validate_charge_table(voltage, charge, label: str) -> None:
    if len(voltage) != len(charge):
        raise ValueError(f"{label}: V and Q tables differ in length")
    if any(float(b) <= float(a) for a, b in zip(voltage, voltage[1:])):
        raise ValueError(f"{label}: voltage grid must be strictly ascending")
    values = [float(q) for q in charge]
    if any(not math.isfinite(q) for q in values):
        raise ValueError(f"{label}: charge must be finite")
    if any(q < 0.0 for q in values):
        raise ValueError(f"{label}: charge must be >= 0")
    if any(b < a for a, b in zip(values, values[1:])):
        raise ValueError(f"{label}: Q(V) must be monotone non-decreasing")


def apply_scalars(spec: PWLModelSpec, theta: Mapping[str, float]) -> PWLModelSpec:
    """Return a copy of *spec* with only the four WOL scalars applied.

    ``Q_RSS`` gets the bounded, non-negative ``O_GD`` offset as ``O_GD*V``
    (zero at V=0, hence monotone and anchor-preserving) plus the ``A_GD``
    scale. The linear fallback mirrors WOL Eq. 2:
    ``c_rss_lin' = A_GD*(c_rss_lin + O_GD)``. ``Q_ISS``/``c_iss_lin`` take
    ``A_GS``; ``r_gate`` takes ``A_RGI``. Qoss/C_DS, channel breakpoints,
    rds_on, v_th, the body diode, element count and pins are untouched.
    """
    named = _normalise_theta(theta)
    raw_bounds = spec.metadata.get("theta_bounds")
    if raw_bounds is not None:
        bounds = _normalise_bounds(raw_bounds)
        # O_GD is capped by the curve-derived Q_RSS budget, never by a
        # literal search bracket that may have drifted into theta_bounds.
        # A_* multiplier brackets remain the per-vendor metadata values.
        if spec.metadata.get("c_rss_lin") is not None:
            bounds["O_GD"] = (0.0, derive_o_gd_bound(spec))
        _check_bounds(named, bounds, spec.device_name)

    meta = spec.metadata
    for key in ("r_gate", "c_iss_lin", "c_rss_lin"):
        if key not in meta:
            raise ValueError(f"{spec.device_name}: metadata missing {key!r}; "
                             "no hidden scalar default in the tuning layer")
    r_gate = float(meta["r_gate"])
    c_iss = float(meta["c_iss_lin"])
    c_rss = float(meta["c_rss_lin"])
    if not math.isfinite(r_gate) or r_gate <= 0.0:
        raise ValueError("r_gate must be > 0 before and after A_RGI")
    if (not math.isfinite(c_iss) or not math.isfinite(c_rss)
            or c_iss < 0.0 or c_rss < 0.0):
        raise ValueError("c_iss_lin/c_rss_lin must be finite and >= 0")

    rss_v = list(spec.q_rss_vs_v)
    rss_q = list(spec.q_rss_charge_c)
    iss_v = list(spec.q_iss_vs_v)
    iss_q = list(spec.q_iss_charge_c)
    _validate_charge_table(rss_v, rss_q, "Q_RSS")
    _validate_charge_table(iss_v, iss_q, "Q_ISS")

    a_gd, o_gd, a_gs, a_rgi = (named[k] for k in UNIFIED_TUNING_KEYS)
    new_rss_q = [a_gd * (float(q) + o_gd * max(float(v), 0.0))
                 for v, q in zip(rss_v, rss_q)]
    new_iss_q = [a_gs * float(q) for q in iss_q]
    _validate_charge_table(rss_v, new_rss_q, "Q_RSS after O_GD")
    _validate_charge_table(iss_v, new_iss_q, "Q_ISS after A_GS")

    new_meta = dict(meta)
    new_meta["r_gate"] = a_rgi * r_gate
    new_meta["c_rss_lin"] = a_gd * (c_rss + o_gd)
    new_meta["c_iss_lin"] = a_gs * c_iss
    new_meta["theta"] = dict(named)
    new_meta["urom_tuning_theta"] = dict(named)
    provenance = dict(meta.get("urom_provenance") or {})
    provenance["applied_theta"] = (
        "WOL Eqs.2-4 on gate-side scalars only; O_GD applied as O_GD*V on "
        "Q_RSS(V) (zero at V=0) and additively to c_rss_lin; C_DS/Qoss frozen")
    new_meta["urom_provenance"] = provenance

    tuned = spec.model_copy(deep=True)
    tuned.q_rss_charge_c = new_rss_q
    tuned.q_iss_charge_c = new_iss_q
    tuned.metadata = new_meta
    return tuned


def _emit_tuned_body(spec: PWLModelSpec) -> str:
    """Emit a tuned Tier-2 body, forwarding the C1 hinge width explicitly.

    ``build_tier2_subcircuit`` takes ``c1_eps``, ``name`` and ``subckt_name``
    as keyword-only parameters. Routing the epsilon through a dynamically
    built ``**kwargs`` makes a type checker treat it as assignable to the
    string parameters, so the two cases are spelled out here: no epsilon uses
    the closed Tier-2 C0 emitter, and a finite value is forwarded only as
    ``c1_eps``. Kept out of the DPT adapter so the branch can be pinned
    offline, without a live simulation.
    """
    from models.netlist_builder import build_tier2_subcircuit

    eps = spec.metadata.get("urom_c1_eps", spec.metadata.get("c1_eps"))
    if eps is None:
        return build_tier2_subcircuit(spec)
    return build_tier2_subcircuit(spec, c1_eps=float(eps))


def make_dpt_simulator(
    vendor: str,
    spec: PWLModelSpec,
    *,
    method: str = "trap/default",
    condition: Mapping[str, Any] | None = None,
    out_dir: str | Path = "reports/models",
    timeout_s: float = 120.0,
) -> Callable[[Mapping[str, float]], ConditionResult]:
    """Build ``theta -> ConditionResult`` from the existing DPT bench.

    The emitted Tier-2 body is rebuilt for each candidate with
    :func:`apply_scalars`; the bench, decks and completion guards are shared
    with the existing double-pulse testbench, so the DUT body is the only
    changed deck region. ``condition`` may override any key of
    ``config/inputs/<vendor>.yaml:dpt`` plus ``temperature``; pin order is
    taken semantically from that YAML and the 3-terminal emitter's Tj port is
    out of scope (the bench temperature card carries temperature instead).

    Completion uses the equal-sandbox guard from spec section 1.3 item 5:
    ``t_last >= 0.999*t_end``, at least half the expected sample points and
    a positive Newton-iteration count from the solver log.
    """
    if method not in _METHOD_OPTIONS:
        raise ValueError(f"method must be one of {sorted(_METHOD_OPTIONS)}")

    from config.yaml_io import load_input
    from simulation.engine import SpiceEngine, TranDirective
    from simulation.netlist_text import NetlistBuilder
    from simulation.testbenches.double_pulse import (
        DoublePulseRun,
        _gate_pwl,
        extract_dpt_metrics,
    )

    cfg = load_input(vendor)
    dpt = dict(cfg["dpt"])
    overrides = dict(condition or {})
    temperature = float(overrides.get(
        "temperature", cfg["conditions"]["temperature"]))
    for key in list(dpt):
        if key in overrides:
            dpt[key] = overrides[key]
    pin_order = [p for p in cfg["pins"]["pin_order"] if p.lower() != "tj"]
    name = f"{spec.device_name}_PIECESPICE_T2"
    method_card = _METHOD_OPTIONS[method]
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def simulate(theta: Mapping[str, float]) -> ConditionResult:
        tuned = apply_scalars(spec, theta)
        digest = hashlib.sha1(
            repr(dict(_normalise_theta(theta))).encode("ascii")
        ).hexdigest()[:12]
        lib = out_path / f"{spec.device_name}_PIECESPICE_T2_tune_{digest}.lib"
        lib.write_text(_emit_tuned_body(tuned), encoding="ascii")

        nb = NetlistBuilder(f"TUNE_DPT_{vendor}_{digest}")
        nb.include(lib)
        nb.v_dc("bus", "vbus", "0", float(dpt["vbus"]))
        nb.l("load", "vbus", "drain", float(dpt["l_load"]))
        nb.raw_line("Dclamp drain vbus DCLAMP")
        nb.raw_line(".model DCLAMP D(IS=1e-12 N=1 RS=1e-3 CJO=0)")
        nb.raw_line("Vgs gate 0 PWL(" + _gate_pwl(
            float(dpt["t1_on"]), float(dpt["t1_width"]),
            float(dpt["t2_on"]), float(dpt["t2_width"]),
            float(dpt["edge_time"]), float(dpt["vgs_on"]),
            float(dpt["t_end"])) + ")")
        nb.x_instance("dut", name, pin_order, temperature=temperature)
        if method_card:
            nb.raw_line(method_card)

        engine = SpiceEngine(temperature=temperature, timeout_s=timeout_s)
        started = time.perf_counter()
        result = engine.run_transient(
            nb.render(),
            TranDirective(step_time=float(dpt["max_step"]),
                          end_time=float(dpt["t_end"]),
                          max_step=float(dpt["max_step"])),
            timeout_s=timeout_s,
        )
        wall = time.perf_counter() - started

        if ("V(drain)" not in result.traces
                or "I(Lload)" not in result.traces
                or "V(gate)" not in result.traces):
            return ConditionResult(metrics={}, completed=False)
        times = np.asarray(result.get("time"), dtype=float)
        match = re.search(r"Total iterations:\s*(\d+)", result.log_text)
        iterations = int(match.group(1)) if match else 0
        expected = int(math.ceil(float(dpt["t_end"])
                                 / float(dpt["max_step"]))) + 1
        t_last = float(times[-1]) if times.size else 0.0
        points = int(times.size)
        completed = bool(
            t_last >= 0.999 * float(dpt["t_end"])
            and points >= 0.5 * expected
            and iterations > 0
        )
        # A trace that stops inside/before the second pulse has no usable
        # dt for extract_dpt_metrics (np.diff would be empty, so np.min of
        # dt raises). Penalise such candidates as incomplete; never let a
        # short/aborted solver result leak an exception to the caller.
        second_points = int(np.count_nonzero(times >= float(dpt["t2_on"])))
        if not completed or second_points < 2:
            return ConditionResult(
                metrics={}, completed=False, points=points,
                iterations=iterations, t_last=t_last,
            )
        run = DoublePulseRun(
            time=times,
            v_ds=np.asarray(result.get("V(drain)"), dtype=float),
            i_d=np.abs(np.asarray(result.get("I(Lload)"), dtype=float)),
            v_gs=np.asarray(result.get("V(gate)"), dtype=float),
            v_bus=float(dpt["vbus"]),
            i_load=float(dpt["i_load_target"]),
            l_load=float(dpt["l_load"]),
            vgs_on=float(dpt["vgs_on"]),
            wall_clock_s=float(wall),
            t2_on=float(dpt["t2_on"]),
        )
        return ConditionResult(
            metrics=extract_dpt_metrics(run),
            completed=completed,
            points=points,
            iterations=iterations,
            t_last=t_last,
        )

    return simulate
