"""C1-continuous knee smoothing kernels (Plan §2.4).

Piecewise models have second-derivative discontinuities at segment
boundaries; Newton-Raphson can oscillate on the kinks. These kernels replace
hard ``max(0, x)`` / hard piecewise transitions with smooth, finite-Jacobian
equivalents whose smoothing width epsilon is a model parameter:

* :func:`softplus`     - smooth max(0, x);  derivative = sigmoid(x/eps)
* :func:`smooth_relu_product` - chained softplus ramp (used by charge tables)
* :func:`tanh_saturation` - tanh knee for channel saturation (plan eq. §2.4)
* :func:`pwl_smooth`   - evaluate a PWL breakpoint table with C1 blending

All kernels satisfy: as eps -> 0 they converge to their hard counterparts,
and d/dx is bounded and continuous everywhere. Device-agnostic.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "softplus",
    "sigmoid",
    "tanh_saturation",
    "pwl_smooth",
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
        left_line = vals[i - 1] + (bi - b[i - 1]) / (bi - b[i - 1]) * 0  # unused
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
