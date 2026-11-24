"""C1-continuous knee smoothing kernels (Plan §2.4).

Piecewise models have second-derivative discontinuities at segment
boundaries; Newton-Raphson can oscillate on the kinks. These kernels replace
hard ``max(0, x)`` / hard piecewise transitions with smooth, finite-Jacobian
equivalents whose smoothing width epsilon is a model parameter:

* :func:`softplus`     - smooth max(0, x);  derivative = sigmoid(x/eps)
* :func:`smooth_relu_product` - chained softplus ramp (used by charge tables)
* :func:`tanh_saturation` - tanh knee for channel saturation (plan eq. §2.4)
* :func:`pwl_smooth`   - evaluate a PWL breakpoint table with C1 blending
* :func:`hyp_expr`     - ASCII SPICE smoothed hinge 0.5*(x+sqrt(x*x+eps*eps))
* :func:`hyp_value`    - numpy mirror of :func:`hyp_expr` (one formula, no copy)
* :func:`delta_ramp_chain` - anchored (Q(0)=0) ASCII delta-slope ramp emitter
* :func:`eps_from_grid` - data-derived width (median spacing, half-min cap)
* :func:`emit_qoss_expr` / :func:`emit_transfer_ramp` - Tier-2 emitter wrappers
  over :func:`delta_ramp_chain` (semantic SPICE terminals, no new nodes)

All kernels satisfy: as eps -> 0 they converge to their hard counterparts,
and d/dx is bounded and continuous everywhere. Device-agnostic.
Emitter helpers return ASCII only. Their epsilon is always supplied by the
caller (finite, >= 0, in the grid's units): there is no magic default width.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

__all__ = [
    "fit_tanh_knee_ls",
    "fit_tanh_plateau",
    "smooth_relu_product",
    "softplus",
    "sigmoid",
    "hyp_expr",
    "hyp_value",
    "delta_ramp_chain",
    "eps_from_grid",
    "emit_qoss_expr",
    "emit_transfer_ramp",
    "tanh_saturation",
    "pwl_smooth",
    "fit_tanh_knee",
    "TanhKneeFit",
]


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically safe logistic sigmoid."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    if out.ndim == 0:
        return float(out)
    return out


def softplus(x: np.ndarray | float, eps: float = 0.05) -> np.ndarray | float:
    """Smooth max(0, x): eps*ln(1+exp(x/eps)).

    For x << -eps this underflows to exactly 0 (no overflow); for x >> eps it
    converges to x. eps sets the transition width in x-units.
    """
    x = np.asarray(x, dtype=float)
    z = x / max(float(eps), 1e-15)
    # log-sum-exp trick: eps*log1p(exp(z)) == max(z,0)*eps + eps*log1p(exp(-|z|))
    out = np.maximum(z, 0.0) * eps + eps * np.log1p(np.exp(-np.abs(z)))
    if out.ndim == 0:
        return float(out)
    return out


def _validate_eps(eps: float) -> float:
    """Return a finite, non-negative smoothing width in grid units [V]."""
    e = float(eps)
    if not math.isfinite(e) or e < 0.0:
        raise ValueError("smoothing eps must be finite and >= 0")
    return 0.0 if e == 0.0 else e   # normalise -0.0 to +0.0


def _fmt_spice(value: float) -> str:
    """Emit one scalar at repository precision (%.6e).

    Mirrors ``models.netlist_builder._fmt``. Coefficients are never rounded
    before this print-time format (pwl-approximation-pitfalls.md, section 5).
    """
    return f"{float(value):.6e}"


def eps_from_grid(xs: Sequence[float]) -> float:
    """Data-derived hinge width: median positive spacing, capped at half-min.

    Implements the P7 rule (unified C1 study section 3): no vendor-specific
    number is embedded. ``xs`` is an ascending grid in volts; only strictly
    positive spacings are considered. Returns 0.0 (legacy C0) for fewer than
    two points or no positive spacing. Non-finite inputs raise ValueError;
    there is no magic fallback epsilon.
    """
    b = np.asarray(xs, dtype=float)
    if b.ndim != 1:
        raise ValueError("eps_from_grid expects a 1-D grid")
    if not np.all(np.isfinite(b)):
        raise ValueError("eps_from_grid: grid contains NaN/Inf")
    d = np.diff(b)
    d = d[np.isfinite(d) & (d > 0.0)]
    if d.size == 0:
        return 0.0
    return float(min(float(np.median(d)), 0.5 * float(np.min(d))))


def hyp_expr(x: str, eps: float = 0.0) -> str:
    """ASCII SPICE expression for the smoothed hinge H(x, eps).

    ``H(x, eps) = 0.5*(x + sqrt(x*x + eps*eps))`` - the hyperbola-smoothed
    (Smedes) ReLU, the C1 counterpart of ``0.5*(x + abs(x))``.

    Args:
        x: An already-formed SPICE expression string, e.g.
           ``"V(di,ss)-1.000000e+00"``. It is parenthesised before
           substitution so any operator inside it binds safely.
        eps: Hinge width in the same units as ``x`` (volts for charge and
             transfer ramps), formatted at %.6e. Must be finite and >= 0.

    Units/edge cases: eps = 0 gives the exact C0 hinge with no offset; the
    sqrt argument is a sum of squares, so no NaN is produced for finite x;
    dH/dx lies in [0, 1], so a delta-slope chain stays monotone when its data
    are monotone. Returns ASCII only.
    """
    e = _validate_eps(eps)
    v = f"({x})"
    return f"0.5*({v}+sqrt({v}*{v}+{_fmt_spice(e * e)}))"


def hyp_value(x: np.ndarray | float, eps: float = 0.0
              ) -> np.ndarray | float:
    """Numeric mirror of :func:`hyp_expr`: 0.5*(x + sqrt(x*x + eps*eps)).

    One hinge formula for both worlds: emitters use :func:`hyp_expr` for
    SPICE text, offline mirrors use this function for numpy arrays. eps = 0
    is the hard ReLU and matches the C0 emitter; eps > 0 is finite-checked
    and non-negative exactly as in :func:`hyp_expr`.
    """
    e = _validate_eps(eps)
    xv = np.asarray(x, dtype=float)
    out = (0.5 * (xv + np.abs(xv)) if e == 0.0 else
           0.5 * (xv + np.sqrt(xv * xv + e * e)))
    return float(out) if np.ndim(x) == 0 else out


def _chain_coefficients(xs: Sequence[float], ys: Sequence[float]):
    """Return (grid, q0, delta slopes) for a delta-slope ramp chain.

    Segment slope s_i = (ys[i]-ys[i-1])/(xs[i]-xs[i-1]); s_0 = 0. Each ramp
    at xs[i-1] carries delta_i = s_i - s_{i-1} (DELTA, never the absolute
    slope: absolute slopes diverge ~21x - AGENTS.md). Non-finite data raise
    ValueError; duplicate x values get a zero slope rather than a NaN.
    """
    b = np.asarray(xs, dtype=float)
    q = np.asarray(ys, dtype=float)
    if b.ndim != 1 or q.ndim != 1 or b.size != q.size:
        raise ValueError("xs and ys must be 1-D sequences of equal length")
    if b.size == 0:
        raise ValueError("need at least one (x, y) point")
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(q)):
        raise ValueError("ramp data contain NaN/Inf")
    seg = np.zeros_like(q)
    if b.size > 1:
        dx = np.diff(b)
        dy = np.diff(q)
        seg[1:] = np.divide(dy, dx, out=np.zeros_like(dy), where=dx != 0.0)
    prev = np.concatenate(([0.0], seg[:-1]))
    return b, float(q[0]), seg - prev


def smooth_relu_product(
    x: np.ndarray | float,
    xs: Sequence[float],
    ys: Sequence[float],
    eps: float = 0.0,
) -> np.ndarray | float:
    """Chained softplus/hyp ramp evaluation (numpy mirror of the emitter).

    ``Q(x) = ys[0] + sum_i delta_i * H(x - xs[i-1], eps)`` where
    ``H(d, eps) = 0.5*(d + sqrt(d*d + eps*eps))`` is the smooth-relu ramp used
    by charge tables. For eps > 0 only the injected hinge offset at x = 0 is
    subtracted, preserving the table baseline ``ys[0]`` (the tanh knee width
    for a saturation table); a plain hyp swap injects H(-v, eps) > 0 at zero.

    Args:
        x: Evaluation point(s), in the grid's units (volts).
        xs: Ascending breakpoint grid [V].
        ys: Monotone non-decreasing table values [C] for charge or [A] for a
            transfer drive.
        eps: Hinge width [V]; finite and >= 0. eps = 0 reproduces the legacy
             C0 soft-abs chain exactly; no magic default width is assumed.

    Units/edge cases: monotone ``ys`` plus delta slopes keeps dQ/dx >= 0
    because every H' lies in [0, 1]; all inputs are finite-checked and the
    sqrt argument is a sum of squares, so no NaN is emitted. Q(0) = ys[0]
    for both eps = 0 and eps -> 0+, so the A/B switch is C0-continuous for a
    non-zero baseline; Q(0) = 0 when the supplied charge table is anchored.
    """
    e = _validate_eps(eps)
    b, q0, delta = _chain_coefficients(xs, ys)
    d = delta[1:]
    breaks = b[:-1]
    xv = np.asarray(x, dtype=float)
    shift = xv[..., np.newaxis] - breaks
    ramp = 0.5 * (shift + np.abs(shift)) if e == 0.0 else \
        0.5 * (shift + np.sqrt(shift * shift + e * e))
    value = q0 + ramp @ d
    if e > 0.0:
        z = -breaks
        zramp = 0.5 * (z + np.sqrt(z * z + e * e))
        value = value - zramp @ d
    if np.ndim(x) == 0:
        return float(value)
    return value


def delta_ramp_chain(
    term: str,
    xs: Sequence[float],
    ys: Sequence[float],
    eps: float = 0.0,
) -> str:
    """Anchored ASCII delta-slope ramp chain for a SPICE ``Q={...}`` element.

    ``Q(t) = q0 + sum_i delta_i * H(t - xs[i-1], eps)``, with per-segment
    DELTA slopes (s_i - s_{i-1}) and ``H`` from :func:`hyp_expr`. For eps > 0
    only the injected hinge offset at t = 0 is subtracted from the anchor, so
    Q(0) = q0 exactly at emitted precision - the shifted-hinge rule from C1
    study P1. This keeps eps -> 0+ continuous with the eps = 0 C0 chain even
    when q0 != 0 (the channel saturation table's tanh knee width).

    Args:
        term: SPICE voltage/current expression string, e.g. ``"V(di,ss)"``;
              ASCII only. Used verbatim inside each ramp.
        xs: Ascending breakpoint grid [V].
        ys: Monotone non-decreasing values [C] (charge) or [A] (drive).
        eps: Hinge width [V]; finite and >= 0, normally from
             :func:`eps_from_grid`. eps = 0 emits the legacy C0 soft-abs
             template used by today's emitters, bit-for-bit at %.6e.

    Units/edge cases: eps >= 0 (negative/NaN raise ValueError); monotone ys
    with delta slopes gives monotone Q; Q(0) = q0 for eps > 0 and eps = 0, so
    a non-zero baseline (channel knee width) survives the A/B switch and an
    anchored charge table (q0 = 0) still has Q(0) = 0; all data are
    finite-checked and the sqrt argument is a sum of squares, so no NaN is
    emitted. Every number is formatted at %.6e, never pre-rounded.
    """
    e = _validate_eps(eps)
    b, q0, delta = _chain_coefficients(xs, ys)
    if not isinstance(term, str) or not term:
        raise ValueError("term must be a non-empty SPICE expression string")
    term.encode("ascii")   # no non-ASCII tokens in emitted decks

    if e == 0.0:
        # Legacy C0 chain (models.netlist_builder._ramp_chain template):
        # identical coefficients, breakpoints and parenthesisation at %.6e.
        expr = [_fmt_spice(q0)]
        for i in range(1, len(b)):
            if abs(delta[i]) > 0.0 or i == 1:
                bp = _fmt_spice(b[i - 1])
                expr.append(
                    f" + {_fmt_spice(delta[i])}*(({term}-{bp})"
                    f"+abs(({term})-{bp}))/2")
        return "".join(expr)

    z = -b[:-1]
    h0 = 0.5 * (z + np.sqrt(z * z + e * e))
    anchor = q0 - float(np.dot(delta[1:], h0))   # preserve baseline q0
    expr = [_fmt_spice(anchor)]
    for i in range(1, len(b)):
        if abs(delta[i]) > 0.0 or i == 1:
            hinge = hyp_expr(f"({term}-{_fmt_spice(b[i - 1])})", e)
            expr.append(f" + {_fmt_spice(delta[i])}*{hinge}")
    return "".join(expr)


def emit_qoss_expr(v: Sequence[float], q: Sequence[float], eps: float = 0.0,
                   *, term: str = "V(di,ss)") -> str:
    """Tier-2 C_QOSS charge expression: anchored delta-ramp Qoss(Vds).

    Thin wrapper over :func:`delta_ramp_chain` with the semantic drain-source
    terminal. eps comes from the caller (normally :func:`eps_from_grid`);
    eps = 0 reproduces the closed Tier-2 C0 chain. Q(0) = 0 is guaranteed by
    the shared chain's shifted hinge. ASCII only; no new nodes.
    """
    return delta_ramp_chain(term, v, q, eps)


def emit_transfer_ramp(xs: Sequence[float], ys: Sequence[float],
                       eps: float = 0.0, *, term: str = "V(gi,ss)") -> str:
    """Tier-2 channel drive ramp: anchored delta-ramp PWL(Vgs).

    Thin wrapper over :func:`delta_ramp_chain` with the semantic gate-source
    terminal. The closed Tier-2 tanh knee, plateau blend and Vth anchor stay
    in the caller; this emits only the transfer/saturation ramps. eps = 0
    reproduces today's C0 text bit-for-bit at %.6e. ASCII only; no nodes.
    """
    return delta_ramp_chain(term, xs, ys, eps)


def tanh_saturation(
    v_ds: np.ndarray | float,
    v_ds_sat: np.ndarray | float,
    gm: np.ndarray | float,
) -> np.ndarray | float:
    """Channel-current saturation knee: Id = gm * Vds_sat * tanh(Vds/Vds_sat).

    Plan §2.4's hyperbolic-tangent form - smooth, monotonic, saturating at
    gm*Vds_sat with C-infinite derivative continuity at Vds=0 (odd function).
    """
    ratio = np.asarray(v_ds, dtype=float) / np.maximum(
        np.asarray(v_ds_sat, dtype=float), 1e-15)
    return np.asarray(gm, dtype=float) * np.asarray(v_ds_sat, dtype=float) \
        * np.tanh(ratio)


def pwl_smooth(
    x: np.ndarray | float,
    breaks: np.ndarray,
    values: np.ndarray,
    eps: float = 0.02,
) -> np.ndarray | float:
    """Evaluate a breakpoint table with C1-smooth transitions.

    Between breakpoints the result is linear; within +/-eps of a breakpoint
    the corner is rounded by a smoothstep blend, so d/dx is continuous.

    Args:
        x: Evaluation point(s).
        breaks: Breakpoint x-coordinates [K+1], ascending.
        values: y at each breakpoint [K+1].
        eps: Half-width of the rounding region around each break (in x units).
    """
    import logging
    _logger = logging.getLogger(__name__)

    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    b = np.asarray(breaks, dtype=float)
    vals = np.asarray(values, dtype=float)

    min_seg = float(np.min(np.diff(b))) if len(b) > 1 else 0.0
    if eps > min_seg / 2:
        _logger.warning(
            "pwl_smooth: eps=%g exceeds half the smallest segment width "
            "(%g); rounding regions overlap and values may overshoot the "
            "table range.", eps, min_seg / 2)

    seg = np.clip(np.searchsorted(b[1:-1], x_arr, side="right"), 0, len(vals) - 2)
    x0, x1 = b[seg], b[seg + 1]
    y0, y1 = vals[seg], vals[seg + 1]
    t = np.where(x1 > x0, (x_arr - x0) / np.where(x1 > x0, x1 - x0, 1.0), 0.0)

    # smoothstep blending near each interior break
    result = y0 + t * (y1 - y0)
    for i in range(1, len(b) - 1):
        bi = b[i]
        lo, hi = bi - eps, bi + eps
        if hi <= lo:
            continue
        w = np.clip((x_arr - lo) / (hi - lo), 0.0, 1.0)
        s = w * w * (3.0 - 2.0 * w)          # smoothstep
        # value from left segment extended, and right segment extended:
        left_val = vals[i - 1] + (x_arr - b[i - 1]) * (
            (vals[i] - vals[i - 1]) / max(bi - b[i - 1], 1e-300))
        right_val = vals[i] + (x_arr - bi) * (
            (vals[i + 1] - vals[i]) / max(b[i + 1] - bi, 1e-300))
        blend = (1 - s) * left_val + s * right_val
        inside = (x_arr > lo) & (x_arr < hi)
        result = np.where(inside, blend, result)

    if np.ndim(x) == 0:
        return float(result[0])
    return result


# ---------------------------------------------------------------------------
# Library-level tanh knee fitting (notebooks call this; no inline math there)
# ---------------------------------------------------------------------------

@dataclass
class TanhKneeFit:
    """Result of fitting a tanh saturation knee to a device curve."""

    id_sat: float          # plateau current [A]
    v_ds_sat: float        # voltage where Id reaches tanh(1) ~ 76% [V]
    gm: float              # transconductance scale = id_sat / v_ds_sat [A/V]
    max_abs_deviation_a: float   # worst |tanh - source curve| [A]


def fit_tanh_knee_ls(v, i, i_sat, *, s_min=1e-3, s_max=200.0) -> float:
    """Least-squares knee width for ``i ≈ i_sat·tanh(v/s)`` with fixed plateau.

    ``fit_tanh_knee`` locates the knee from the tanh(1) crossing; that point
    estimator is noisy on measured curves. Here the full sampled curve is
    used, which is the parameter actually fitted in the Tier-2 emitter.
    """
    v = np.asarray(v, dtype=float)
    i = np.asarray(i, dtype=float)
    plateau = max(float(i_sat), 1e-12)
    def loss(s: float) -> float:
        return float(np.sum((plateau * np.tanh(v / max(s, 1e-12)) - i) ** 2))
    return float(minimize_scalar(
        loss, bounds=(max(s_min, 1e-12), s_max), method="bounded").x)


def fit_tanh_plateau(v, i, i_sat, *, s_max=200.0) -> Tuple[float, float]:
    """Fit both plateau and knee of ``i ≈ I·tanh(v/s)`` to a sampled curve.

    The last extracted sample is not always the true tanh plateau (the TI
    BSIM and IR MODPEX curves are still rising at the sweep ceiling). This
    fit estimates the plateau as a parameter; callers decide whether to use
    it via the ratio ``I_fit / I_end``.
    """
    from scipy.optimize import least_squares

    v = np.asarray(v, dtype=float)
    i = np.asarray(i, dtype=float)
    end = max(float(i_sat), 1e-12)
    lo = [1e-6, 0.01]
    hi = [max(end * 10.0, 1.0), s_max]
    seed_s = fit_tanh_knee_ls(v, i, end, s_max=s_max)
    x0 = [float(np.clip(end, lo[0], hi[0])),
          float(np.clip(seed_s, lo[1], hi[1]))]
    res = least_squares(
        lambda p: p[0] * np.tanh(v / max(p[1], 1e-12)) - i,
        x0=x0, bounds=(lo, hi))
    return float(res.x[0]), float(res.x[1])

def fit_tanh_knee(v_pwl: np.ndarray, i_pwl: np.ndarray) -> TanhKneeFit:
    """Fit a tanh saturation knee to a PWL/compressed output curve.

    Derivation (library-owned so notebooks don't reimplement it):
      * Id_sat  = final plateau current of the PWL
      * Vds_sat = Vds where the PWL reaches tanh(1)=0.7616 of Id_sat,
                  i.e. the measured knee location
      * gm      = Id_sat / Vds_sat

    The fitted model coincides with the source curve along both tails;
    deviation concentrates at the knee.
    """
    v = np.asarray(v_pwl, dtype=float).ravel()
    i = np.asarray(i_pwl, dtype=float).ravel()
    if v.size != i.size or v.size < 2:
        raise ValueError("need matching v/i arrays with >= 2 samples")

    order = np.argsort(v)
    v, i = v[order], i[order]
    i = np.abs(i)

    id_sat = float(i[-1])
    if id_sat <= 0:
        raise ValueError("curve must reach positive current to fit a knee")

    knee_frac = float(np.tanh(1.0))
    v_ds_sat = float(np.interp(knee_frac * id_sat, i, v))
    v_ds_sat = max(v_ds_sat, 1e-3)
    gm = id_sat / v_ds_sat

    dense_v = np.linspace(v[0], v[-1], 400)
    dense_fit = np.asarray(tanh_saturation(dense_v, v_ds_sat=v_ds_sat, gm=gm))
    dev = float(np.max(np.abs(
        dense_fit - np.interp(dense_v, v, i))))

    return TanhKneeFit(id_sat=id_sat, v_ds_sat=v_ds_sat, gm=gm,
                       max_abs_deviation_a=dev)
