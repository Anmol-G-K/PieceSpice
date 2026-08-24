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

from dataclasses import dataclass

import numpy as np

__all__ = [
    "softplus",
    "sigmoid",
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
