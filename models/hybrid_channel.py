"""Measured ``hinge_c`` hybrid-channel emitter (standalone, Tier-3 phase 2).

This module implements only the data-driven channel form that the controlled
temperature experiment in ``local://hybrid_channel_feasibility.md`` (2026-11-20)
identified as the sole candidate holding both static targets:

    Id(Vgs, Vds) = y0(Vgs)
                   + sum_m a_m(Vgs) * (H(Vds; k_m, w) - H(Vds_low; k_m, w))
    H(V; k, w)   = 0.5 * ((V - k) + sqrt((V - k)^2 + w^2))

i.e. piecewise affine in Vds with smoothed knees, and coefficients interpolated
in Vgs with *delta-slope* chained soft-abs ramps (absolute slopes accumulated
and diverged 21x in the repository audit recorded in AGENTS.md).

Knee count and locations are chosen by a deterministic greedy residual scan
over the supplied Vds grid: at each step the candidate knee minimising the
global relative RMSE is added; the scan stops at ``rel_tolerance`` (0.005 in
the repository). The smoothing width is the median positive grid spacing, so
no vendor-specific number is embedded. Nothing here is fitted to a hardcoded
knee count; 8/5 knees and w = 0.04*Vds_max were properties of the one-off
feasibility experiment, not of this emitter.

Measured limitation (do not hide): on the 2-pulse DPT the experimental
hinge_c channel *regressed* versus the closed Tier-2 channel -- Vds RMSE
TI 11.20 % vs 6.06 %, CPM2 7.10 % vs 2.79 %. The static bar is met; transient
waveform fidelity no worse than Tier-2 is not demonstrated.

Sign convention: ``G_CH di ss VALUE={expr}`` with a positive VALUE conducts
current out of ``di`` into ``ss``; the returned expression reproduces the
supplied positive quasi-static drain-current rows without any sign inversion
(source-to-drain conduction direction of the extracted family). The lower Vds
endpoint row is reproduced exactly by construction (shifted hinges), which
also pins the model at Vds = 0 for the usual Curve-1 grid.

No builder/pipeline wiring lives here. The expression is ASCII and is intended
for ``G_CH di ss VALUE={expr}`` inside the simplis subcircuit.
"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = ["HybridChannelFit", "fit_hybrid_channel", "evaluate_hybrid_channel"]

# Internally generated tokens used by the offline expression evaluator.
_GATE_VAR = "__ps_gate"
_DRAIN_VAR = "__ps_drain"
_NODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_(),.]*$")
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"abs": np.abs, "sqrt": np.sqrt}


@dataclass(frozen=True)
class HybridChannelFit:
    """Fitted hinge_c channel and the SPICE VALUE expression it emits.

    ``vds_knees`` contains one list per Vgs row; the greedy fit shares one
    knee set across the family, so all rows carry the same list.
    ``max_abs_fit_error_a`` is measured by evaluating ``expression`` itself on
    every supplied (Vgs, Vds) sample, not from an internal coefficient path.
    """

    expression: str
    vgs_knots: list[float]
    vds_knees: list[list[float]]
    smoothing_width_v: float
    max_abs_fit_error_a: float
    meta: dict


def _fmt(x: float) -> str:
    """SPICE-safe ASCII float (same precision as the other model emitters)."""
    return f"{float(x):.9e}"


def _node_token(value: Any, label: str) -> str:
    """Restrict caller-supplied node tokens to an ASCII, injection-free form."""
    if (not isinstance(value, str) or not _NODE_RE.fullmatch(value)
            or "__ps_" in value):
        raise ValueError(f"{label} must be a plain ASCII node expression, got {value!r}")
    return value


def _hinge(v: np.ndarray, knee: float, width: float) -> np.ndarray:
    """Smoothed ReLU H(V; k, w) = 0.5*((V-k)+sqrt((V-k)^2+w^2))."""
    x = v - knee
    return 0.5 * (x + np.sqrt(x * x + width * width))


def _delta_slopes(knots: Sequence[float], values: Sequence[float]) -> list[float]:
    """Slope changes for the chained soft-abs ramp; first term is the slope.

    This is the repository invariant: emit ``delta = slope - prev_slope``,
    never the accumulated absolute slope (which diverged 21x in the audit).
    """
    out: list[float] = []
    prev = 0.0
    for i in range(1, len(knots)):
        dv = float(knots[i]) - float(knots[i - 1])
        slope = (float(values[i]) - float(values[i - 1])) / dv if dv else 0.0
        out.append(slope if i == 1 else slope - prev)
        prev = slope
    return out


def _ramp_expr(knots: Sequence[float], values: Sequence[float], term: str) -> str:
    """C0 PWL through (knots, values) as chained soft-abs delta-slope ramps."""
    expr = _fmt(values[0])
    for i, delta in enumerate(_delta_slopes(knots, values), start=1):
        if delta == 0.0:
            continue
        pivot = _fmt(knots[i - 1])
        shifted = f"(({term})-({pivot}))"
        expr += f" + {_fmt(delta)}*({shifted}+abs({shifted}))/2"
    return expr


def _eval_ast(node: ast.AST, names: dict[str, np.ndarray]) -> Any:
    """Evaluate the restricted arithmetic subset emitted by this module."""
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, names)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        if node.id in _FUNCS:
            return _FUNCS[node.id]
        raise ValueError(f"unsupported name in emitted expression: {node.id}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_ast(node.left, names),
                                       _eval_ast(node.right, names))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_ast(node.operand, names))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS and len(node.args) == 1
            and not node.keywords):
        return _FUNCS[node.func.id](_eval_ast(node.args[0], names))
    raise ValueError(f"unsupported syntax in emitted expression: {ast.dump(node)}")


def _eval_expression(expression: str, gate: str, drain: str,
                     vgs: Any, vds: Any) -> np.ndarray:
    """Evaluate an emitted VALUE string with the two node tokens substituted."""
    pattern = re.compile("|".join(re.escape(t) for t in
                                  sorted({gate, drain}, key=len, reverse=True)))
    mapping = {gate: _GATE_VAR, drain: _DRAIN_VAR}
    code = pattern.sub(lambda m: mapping[m.group(0)], expression)
    code = code.replace("^", "**")
    tree = ast.parse(code, mode="eval")
    g, d = np.broadcast_arrays(np.asarray(vgs, dtype=float),
                               np.asarray(vds, dtype=float))
    value = _eval_ast(tree, {_GATE_VAR: g, _DRAIN_VAR: d})
    return np.asarray(value, dtype=float)


def evaluate_hybrid_channel(fit: HybridChannelFit, vgs: Any, vds: Any) -> np.ndarray:
    """Numerically evaluate the *emitted expression* on supplied operating points.

    Used by the offline unit test to prove the ASCII VALUE string, not a
    parallel coefficient path, reproduces the input rows. Scalars or arrays
    are accepted; shapes are broadcast.
    """
    return _eval_expression(fit.expression, fit.meta["gate_node"],
                            fit.meta["drain_node"], vgs, vds)


def fit_hybrid_channel(vgs_values: Sequence[float],
                       vds_grid: Sequence[float],
                       id_rows: Sequence[Sequence[float]],
                       *,
                       gate_node: str = "V(gi,ss)",
                       drain_node: str = "V(di,ss)",
                       rel_tolerance: float = 0.005) -> HybridChannelFit:
    """Greedy-fit the shared-knee smoothed-hinge channel to a Curve-1 family.

    Args:
        vgs_values: gate-voltage knots [V], one per supplied family row.
        vds_grid: strictly increasing drain-voltage samples [V]; nonuniform
            spacing is allowed.
        id_rows: Id[Vgs_i, Vds_j] in amperes, shape (len(vgs), len(vds)).
        gate_node: SPICE expression for Vgs (default ``V(gi,ss)``).
        drain_node: SPICE expression for Vds (default ``V(di,ss)``).
        rel_tolerance: stop criterion, RMSE normalised by the global
            peak-to-peak span of the supplied Id rows (repository metric).

    Raises:
        ValueError: malformed inputs.
        RuntimeError: the greedy scan exhausted the grid without reaching
            ``rel_tolerance`` (no silent approximation).
    """
    gate = _node_token(gate_node, "gate_node")
    drain = _node_token(drain_node, "drain_node")
    if gate == drain:
        raise ValueError("gate_node and drain_node must differ")
    tol = float(rel_tolerance)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("rel_tolerance must be finite and non-negative")

    knots = np.asarray(vgs_values, dtype=float).ravel()
    vd = np.asarray(vds_grid, dtype=float).ravel()
    target = np.asarray(id_rows, dtype=float)
    if vd.size < 2 or knots.size < 1:
        raise ValueError("need at least one Vgs row and two Vds samples")
    if target.shape != (knots.size, vd.size):
        raise ValueError(
            f"id_rows shape {target.shape} != {(knots.size, vd.size)}")
    if not (np.all(np.isfinite(knots)) and np.all(np.isfinite(vd))
            and np.all(np.isfinite(target))):
        raise ValueError("non-finite value in Vgs, Vds or Id rows")
    if not np.all(np.diff(vd) > 0.0):
        raise ValueError("vds_grid must be strictly increasing")
    order = np.argsort(knots, kind="stable")
    knots = knots[order]
    target = target[order]
    if np.any(np.diff(knots) <= 0.0):
        raise ValueError("vgs_values must not contain duplicates")

    spacings = np.diff(vd)
    width = float(np.median(spacings[spacings > 0.0]))
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("cannot derive a positive smoothing width from Vds grid")

    scale = float(np.ptp(target))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Id family has no dynamic range; refusing a degenerate fit")

    # Shifted hinges pin the lower Vds endpoint exactly and keep Id at Vds=0
    # at the supplied (near-zero) value instead of extrapolating a constant.
    x = vd[None, :] - vd[:, None]
    hinge_grid = 0.5 * (x + np.sqrt(x * x + width * width))
    x0 = float(vd[0]) - vd
    h_at_v0 = 0.5 * (x0 + np.sqrt(x0 * x0 + width * width))
    basis = hinge_grid - h_at_v0[:, None]

    residual_target = target.T - target[:, 0]          # (n_vds, n_vgs)
    design = np.empty((vd.size, 0), dtype=float)
    coef = np.empty((0, knots.size), dtype=float)
    selected: list[int] = []
    residual = -residual_target.copy()

    def rmse(r: np.ndarray) -> float:
        return float(np.sqrt(np.mean(r * r)))

    achieved = rmse(residual)
    while achieved > tol * scale:
        if len(selected) >= vd.size:
            break
        best_idx: int | None = None
        best_err = achieved
        for idx in range(vd.size):
            if idx in selected:
                continue
            trial = np.column_stack((design, basis[idx]))
            trial_coef, *_ = np.linalg.lstsq(trial, residual_target, rcond=None)
            err = rmse(trial @ trial_coef - residual_target)
            # Strict improvement, candidates scanned in ascending Vds: ties
            # resolve to the lowest knee voltage, so the rule is deterministic.
            if err < best_err - 1e-14 * scale:
                best_idx, best_err = idx, err
        if best_idx is None:
            break
        selected.append(best_idx)
        design = np.column_stack((design, basis[best_idx]))
        coef, *_ = np.linalg.lstsq(design, residual_target, rcond=None)
        residual = design @ coef - residual_target
        achieved = rmse(residual)

    if achieved > tol * scale:
        raise RuntimeError(
            f"hinge_c greedy fit did not reach rel RMSE {tol:.6g} "
            f"(best {achieved / scale:.6g} after {len(selected)} grid knees); "
            "refusing to emit an unfitted channel")

    y0 = target[:, 0]
    rows = np.column_stack((y0, coef.T))               # (n_vgs, 1 + n_knees)
    terms = [_ramp_expr(knots, y0, gate)]
    for m, idx in enumerate(selected):
        amp = rows[:, m + 1]
        hinge = (
            f"0.5*((({drain})-({_fmt(vd[idx])}))"
            f"+sqrt((({drain})-({_fmt(vd[idx])}))^2+({_fmt(width)})^2))")
        terms.append(
            f"({_ramp_expr(knots, amp, gate)})*(({hinge})-({_fmt(h_at_v0[idx])}))")
    expression = "(" + "+".join(terms) + ")"
    expression.encode("ascii")  # hard gate: emitted VALUE must be ASCII

    prediction = _eval_expression(expression, gate, drain, knots[:, None],
                                  np.broadcast_to(vd, target.shape))
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("emitted expression evaluates to a non-finite current")
    max_err = float(np.max(np.abs(prediction - target)))
    rel_rmse = float(np.sqrt(np.mean((prediction - target) ** 2)) / scale)
    if rel_rmse > tol * (1.0 + 1e-9):
        raise RuntimeError(
            "emitted expression does not reproduce the supplied rows "
            f"(measured rel RMSE {rel_rmse:.6g}, tolerance {tol:.6g})")

    meta = {
        "kind": "hinge_c",
        "gate_node": gate,
        "drain_node": drain,
        "emission": "G_CH di ss VALUE={expression}",
        "stop_criterion": "global_rms_error / peak_to_peak(Id rows)",
        "rel_tolerance": tol,
        "id_peak_to_peak_a": scale,
        "knee_selection": ("deterministic greedy: scan supplied Vds grid "
                           "in ascending order, add knee minimising global "
                           "RMSE, tie -> lowest Vds"),
        "knee_scope": "shared across the Vgs family (one per-row list in vds_knees)",
        "n_knees": len(selected),
        "smoothing_width_derivation": {
            "method": "median_positive_vds_spacing",
            "n_positive_spacings": int(np.count_nonzero(spacings > 0.0)),
            "vds_span_v": float(vd[-1] - vd[0]),
            "width_to_span_ratio": float(width / (vd[-1] - vd[0])),
        },
        "vgs_interpolation": (
            "chained soft-abs ramps with delta slopes; absolute slopes "
            "accumulate and diverged 21x in the repository audit"),
        "lower_endpoint_policy": "exact first Vds sample via shifted hinges",
        "vgs_coefficients": [[float(v) for v in row] for row in rows],
        "current_sign": ("positive VALUE = drain-to-source conduction of "
                         "G_CH di ss; supplied positive quasi-static "
                         "drain-current rows emitted without sign inversion"),
        "achieved_rel_rmse": rel_rmse,
        "per_row_max_abs_error_a": [
            float(v) for v in np.max(np.abs(prediction - target), axis=1)],
        "known_limitation": (
            "measured 2-pulse DPT Vds RMSE regressed vs closed Tier-2: "
            "TI 11.20% vs 6.06%, CPM2 7.10% vs 2.79% "
            "(local://hybrid_channel_feasibility.md, 2026-11-20). "
            "Static fidelity only; transient superiority not demonstrated."),
    }
    return HybridChannelFit(
        expression=expression,
        vgs_knots=[float(v) for v in knots],
        vds_knees=[[float(vd[i]) for i in selected] for _ in range(knots.size)],
        smoothing_width_v=width,
        max_abs_fit_error_a=max_err,
        meta=meta,
    )
