"""Physics-aware ANN channel (IV) fit -- unified ROM task T2.

Offline-only fitter.  The emitted ROM has zero Python dependency (see
``models.ann_netlist_builder``).  The model form is the source-derived
(log-scale output transform + analytic guards) recipe, not a re-tune of the
closed Tier-2 PWLxTanh channel:

    y1      = b2 + sum_i w2_i * phi(w1_i0*g + w1_i1*d + b1_i)
    I_raw   = VDS * exp(y1)                       (zero at VDS = 0)
    cutoff  = 0.5*(1 + tanh(alpha*(VGS - Vth)))   (analytic Vth guard)
    I_hi    = I_raw * cutoff
    I_lo    = VDS * exp(y1(VGS, Vsat)) * cutoff
    blend   = 0.5*(1 + tanh((VDS - Vsat) / w_guard))
    ID      = I_lo + blend * (I_hi - I_lo)

``phi`` is ISRU ``u/sqrt(1+u^2)`` by default: C-inf, bounded, exponent-free
and expressed with the verified algebraic ``sqrt`` fallback (the two-argument
LTspice ``hypot`` form is still a candidate primitive, so it is not emitted).
One hidden layer, at most 10 neurons: no arrays or loops in the emitted text.

Derivative matching is real autodiff: :class:`Jet2` carries value, first and
second derivatives with respect to (VGS, VDS) through every elementary op, so
gm, gds, gm' and gds' in the objective are analytic (never finite differences
of the network).  Inputs are centred/scaled from the data (no magic units).

All numbers are derived from the supplied curves or recorded as explicit
project-decision optimizer settings in ``provenance``; no fabricated result is
reported and no repository artifact is written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
from scipy.optimize import least_squares

from approximation.smoother import eps_from_grid

__all__ = ["Jet2", "DptPoints", "AnnChannelFit", "fit_channel_iv",
           "default_loss_weights", "calibrate_cutoff_curve7"]


# Project-decision optimizer settings (NOT emitter physical constants).
# Recorded in AnnChannelFit.provenance so T4/T5 can audit them.
_OPT_BOUNDS = {"w1": 10.0, "b1": 10.0, "w2": 2.0, "b2": 40.0}
_OPT_MAX_NFEV = 120              # shared by approximation.ann_charge (unchanged)
_CHANNEL_MAX_NFEV = 600          # per-start cap with analytic Jacobian
_OPT_STARTS = 3                 # deterministic seeded restarts, best cost wins
_OPT_TOL = 1e-8                 # xtol/ftol/gtol: explicit convergence criterion
_OPT_NO_IMPROVE = 1e-6       # relative cost plateau => deterministic stop
_CUTOFF_ALPHA = 10.0          # source-derived initial value (Yang TPEL 2022)
# Positive-conductance cutoff optimisation range (project decision, shared by
# the joint Curve-1/2 fit and the measured Curve-7 post-fit calibration).
_CUTOFF_ALPHA_MIN = 2.0
_CUTOFF_ALPHA_MAX = 35.0
# Curve-7 cutoff calibration: measurement/numerical tolerances, never device
# or emitter constants.  The fallback subthreshold selection is a documented
# fraction of the *measured* peak used only when the extracted Curve-7
# ``Id_th_target`` is absent.
_CURVE7_MIN_SUBTHRESHOLD_POINTS = 3
_CURVE7_SUBTHRESHOLD_FRACTION = 0.01
_CURVE7_MONOTONE_RTOL = 1e-9   # relative to measured peak: float-noise band
_CURVE7_LOG_FLOOR_RTOL = 1e-12  # log floor relative to measured peak
_CUTOFF_MAX_NFEV = 240
_CUTOFF_STARTS = 3
_MAX_NEURONS = 10             # E3 Fig.12 overfit guard (ANN study R7)
_DEFAULT_WEIGHTS = {"id": 1.0, "gm": 0.3, "gds": 0.3,
                    "gm2": 0.1, "gds2": 0.1}
_SUPPORTED = ("isru", "tanh", "sigmoid")
# Soft physics penalty on negative gm (dId/dVgs) and negative gds
# (dId/dVds).  This is a least-squares penalty, NOT a hard monotonicity
# guarantee; the achieved sign statistics are reported in diagnostics.
_MONOTONE_WEIGHT = 1.0


def default_loss_weights() -> Dict[str, float]:
    """Return the derivative-aware default weights (project decision)."""
    return dict(_DEFAULT_WEIGHTS)


def _zero_like(v: Any) -> Any:
    return 0.0 if np.ndim(v) == 0 else np.zeros_like(v, dtype=float)


def _coerce(other: Any, like: Any) -> "Jet2":
    if isinstance(other, Jet2):
        return other
    return Jet2(other)

def seed_input(v: Any, slot: int) -> "Jet2":
    """Input jet with d(v)/d(slot) = 1; slot 0 = VGS, 1 = VDS."""
    v = np.asarray(v, dtype=float) if np.ndim(v) else float(v)
    z = _zero_like(v)
    o = np.ones_like(v) if np.ndim(v) else 1.0
    return Jet2(v, g=(o, z) if slot == 0 else (z, o))


class Jet2:
    """Forward-mode second-order jet in two inputs (VGS, VDS).

    ``v`` is the value, ``g = (d/dVGS, d/dVDS)`` and
    ``h = (d2/dVGS2, d2/dVGS dVDS, d2/dVDS2)``.  Values may be scalars or
    numpy arrays; arithmetic and the three elementary unary functions needed
    by the model are shape preserving.  This is the single autodiff primitive
    shared by the channel and charge fitters (no duplicate finite differences).
    """

    __slots__ = ("v", "g", "h")

    def __init__(self, v: Any, g: Optional[Tuple] = None,
                 h: Optional[Tuple] = None) -> None:
        self.v = np.asarray(v, dtype=float) if np.ndim(v) else float(v)
        z = _zero_like(self.v)
        self.g = (z, z, z)[:2] if g is None else g
        self.h = (z, z, z) if h is None else h

    # -- algebra ----------------------------------------------------------
    def __add__(self, o: Any) -> "Jet2":
        o = _coerce(o, self.v)
        return Jet2(self.v + o.v, (self.g[0] + o.g[0], self.g[1] + o.g[1]),
                    tuple(a + b for a, b in zip(self.h, o.h)))

    __radd__ = __add__

    def __neg__(self) -> "Jet2":
        return Jet2(-self.v, (-self.g[0], -self.g[1]),
                    tuple(-a for a in self.h))

    def __sub__(self, o: Any) -> "Jet2":
        return self + (-_coerce(o, self.v))

    def __rsub__(self, o: Any) -> "Jet2":
        return _coerce(o, self.v) + (-self)

    def __mul__(self, o: Any) -> "Jet2":
        o = _coerce(o, self.v)
        v = self.v * o.v
        g = (self.g[0] * o.v + self.v * o.g[0],
             self.g[1] * o.v + self.v * o.g[1])
        h = (self.h[0] * o.v + 2.0 * self.g[0] * o.g[0] + self.v * o.h[0],
             self.h[1] * o.v + self.g[0] * o.g[1] + self.g[1] * o.g[0]
             + self.v * o.h[1],
             self.h[2] * o.v + 2.0 * self.g[1] * self.g[1] + self.v * o.h[2])
        return Jet2(v, g, h)

    __rmul__ = __mul__

    def inv(self) -> "Jet2":
        v = 1.0 / self.v
        g = (-self.g[0] / (self.v * self.v),
             -self.g[1] / (self.v * self.v))
        v2 = self.v * self.v
        v3 = v2 * self.v
        h = (2.0 * self.g[0] * self.g[0] / v3 - self.h[0] / v2,
             2.0 * self.g[0] * self.g[1] / v3 - self.h[1] / v2,
             2.0 * self.g[1] * self.g[1] / v3 - self.h[2] / v2)
        return Jet2(v, g, h)

    def __truediv__(self, o: Any) -> "Jet2":
        return self * _coerce(o, self.v).inv()

    def __rtruediv__(self, o: Any) -> "Jet2":
        return _coerce(o, self.v) * self.inv()

    # -- elementary unary functions ---------------------------------------
    def _unary(self, f: Any, d1: Any, d2: Any) -> "Jet2":
        v = f(self.v)
        a = d1(self.v)
        b = d2(self.v)
        g = (a * self.g[0], a * self.g[1])
        h = (b * self.g[0] * self.g[0] + a * self.h[0],
             b * self.g[0] * self.g[1] + a * self.h[1],
             b * self.g[1] * self.g[1] + a * self.h[2])
        return Jet2(v, g, h)

    def exp(self) -> "Jet2":
        return self._unary(np.exp, np.exp, np.exp)

    def tanh(self) -> "Jet2":
        t = np.tanh(self.v)
        d1 = 1.0 - t * t
        return self._unary(np.tanh, lambda _v, d=d1: d,
                           lambda _v, d=d1, tt=t: -2.0 * tt * d)

    def softplus(self) -> "Jet2":
        xc = np.clip(self.v, -40.0, 40.0)
        sig = 1.0 / (1.0 + np.exp(-xc))
        d2 = sig * (1.0 - sig)
        return self._unary(lambda v: np.log1p(np.exp(np.clip(v, -40.0, 40.0))),
                           lambda _v, s=sig: s,
                           lambda _v, d=d2: d)

    def act(self, name: str) -> "Jet2":
        """C-inf activation: isru (default), tanh or sigmoid."""
        if name == "isru":
            s = np.sqrt(1.0 + self.v * self.v)
            return self._unary(
                lambda v, ss=s: v / ss,
                lambda _v, ss=s: 1.0 / (ss * ss * ss),
                lambda v, ss=s: -3.0 * v / (ss ** 5))
        if name == "tanh":
            return self.tanh()
        if name == "sigmoid":
            q = 1.0 / (1.0 + np.exp(-self.v))
            return self._unary(
                lambda _v, qq=q: qq,
                lambda _v, qq=q: qq * (1.0 - qq),
                lambda _v, qq=q: qq * (1.0 - qq) * (1.0 - 2.0 * qq))
        raise ValueError(f"unsupported activation {name!r}; "
                         f"expected one of {_SUPPORTED}")



class _Jet2P:
    """Jet2 plus first-order tangents in every optimised parameter.

    ``v``/``g``/``h`` mirror :class:`Jet2` (value, two first and three second
    input derivatives); ``pv``/``pg``/``ph`` carry d/dp of each, shaped
    ``(nparam, nsamples)``.  This is the analytic Jacobian source for the
    offline fitter: parameter derivatives are never finite-differenced.
    """

    __slots__ = ("v", "g", "h", "pv", "pg", "ph")

    def __init__(self, v, g, h, pv, pg, ph):
        self.v, self.g, self.h = v, g, h
        self.pv, self.pg, self.ph = pv, pg, ph

    @staticmethod
    def const(value, z1, zp):
        """Constant value; zero input and parameter derivatives."""
        v = np.full(z1.size, float(value), float)
        return _Jet2P(v, (z1, z1), (z1, z1, z1), zp, (zp, zp), (zp, zp, zp))

    @staticmethod
    def seed_input(value, slot, z1, zp):
        """Input jet with d(value)/d(slot) = 1; slot 0 = VGS, 1 = VDS."""
        v = np.asarray(value, float)
        if v.ndim == 0:
            v = np.full(z1.size, float(v))
        o = np.ones_like(z1)
        g = (o, z1) if slot == 0 else (z1, o)
        return _Jet2P(v, g, (z1, z1, z1), zp, (zp, zp), (zp, zp, zp))

    @staticmethod
    def seed_param(value, pv_row, z1, zp):
        """Parameter jet with derivative 1 in row ``pv_row``."""
        v = np.full(z1.size, float(value), float)
        return _Jet2P(v, (z1, z1), (z1, z1, z1), pv_row,
                      (zp, zp), (zp, zp, zp))

    def _like(self, o):
        return _Jet2P.const(o, np.zeros_like(self.v), self.pv)

    def _mul_scalar(self, s):
        return _Jet2P(
            self.v * s, tuple(g * s for g in self.g),
            tuple(h * s for h in self.h), self.pv * s,
            tuple(p * s for p in self.pg), tuple(p * s for p in self.ph))

    def __add__(self, o):
        o = o if isinstance(o, _Jet2P) else self._like(o)
        return _Jet2P(
            self.v + o.v, tuple(a + b for a, b in zip(self.g, o.g)),
            tuple(a + b for a, b in zip(self.h, o.h)), self.pv + o.pv,
            tuple(a + b for a, b in zip(self.pg, o.pg)),
            tuple(a + b for a, b in zip(self.ph, o.ph)))

    __radd__ = __add__

    def __neg__(self):
        return self._mul_scalar(-1.0)

    def __sub__(self, o):
        return self + (-o if isinstance(o, _Jet2P) else -self._like(o))

    def __rsub__(self, o):
        return (o if isinstance(o, _Jet2P) else self._like(o)) + (-self)

    def __mul__(self, o):
        o = o if isinstance(o, _Jet2P) else self._like(o)
        a, b = self, o
        V, W = a.v, b.v
        Vb, Wb = V[None, :], W[None, :]
        g = (a.g[0] * W + V * b.g[0], a.g[1] * W + V * b.g[1])
        h = (a.h[0] * W + 2.0 * a.g[0] * b.g[0] + V * b.h[0],
             a.h[1] * W + a.g[0] * b.g[1] + a.g[1] * b.g[0] + V * b.h[1],
             a.h[2] * W + 2.0 * a.g[1] * b.g[1] + V * b.h[2])
        pv = a.pv * Wb + Vb * b.pv
        pg0 = (a.pg[0] * Wb + a.g[0][None, :] * b.pv
               + a.pv * b.g[0][None, :] + Vb * b.pg[0])
        pg1 = (a.pg[1] * Wb + a.g[1][None, :] * b.pv
               + a.pv * b.g[1][None, :] + Vb * b.pg[1])
        ph = (
            a.ph[0] * Wb + a.h[0][None, :] * b.pv
            + 2.0 * a.pg[0] * b.g[0][None, :]
            + 2.0 * a.g[0][None, :] * b.pg[0]
            + a.pv * b.h[0][None, :] + Vb * b.ph[0],
            a.ph[1] * Wb + a.h[1][None, :] * b.pv
            + a.pg[0] * b.g[1][None, :]
            + a.g[0][None, :] * b.pg[1] + a.pg[1] * b.g[0][None, :]
            + a.g[1][None, :] * b.pg[0]
            + a.pv * b.h[1][None, :] + Vb * b.ph[1],
            a.ph[2] * Wb + a.h[2][None, :] * b.pv
            + 2.0 * a.pg[1] * b.g[1][None, :]
            + 2.0 * a.g[1][None, :] * b.pg[1]
            + a.pv * b.h[2][None, :] + Vb * b.ph[2])
        return _Jet2P(V * W, g, h, pv, (pg0, pg1), ph)

    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, _Jet2P):
            raise TypeError("parameter-jet division is scalar-only")
        return self._mul_scalar(1.0 / float(o))

    def _unary(self, f, d1, d2, d3):
        a = self
        V = a.v
        D1, D2, D3 = d1(V)[None, :], d2(V)[None, :], d3(V)[None, :]
        g = tuple(x * D1[0] for x in a.g)
        h = (a.h[0] * D1[0] + a.g[0] ** 2 * D2[0],
             a.h[1] * D1[0] + a.g[0] * a.g[1] * D2[0],
             a.h[2] * D1[0] + a.g[1] ** 2 * D2[0])
        pv = a.pv * D1
        pg = tuple(a.pg[i] * D1 + a.g[i][None, :] * D2 * a.pv
                   for i in range(2))
        ph = (
            a.ph[0] * D1 + a.h[0][None, :] * D2 * a.pv
            + 2.0 * a.g[0][None, :] * a.pg[0] * D2
            + (a.g[0] ** 2)[None, :] * D3 * a.pv,
            a.ph[1] * D1 + a.h[1][None, :] * D2 * a.pv
            + (a.g[0][None, :] * a.pg[1] + a.g[1][None, :] * a.pg[0]) * D2
            + (a.g[0] * a.g[1])[None, :] * D3 * a.pv,
            a.ph[2] * D1 + a.h[2][None, :] * D2 * a.pv
            + 2.0 * a.g[1][None, :] * a.pg[1] * D2
            + (a.g[1] ** 2)[None, :] * D3 * a.pv)
        return _Jet2P(f(V), g, h, pv, pg, ph)

    def exp(self):
        return self._unary(np.exp, np.exp, np.exp, np.exp)

    def tanh(self):
        t = np.tanh(self.v)
        d1 = 1.0 - t * t
        return self._unary(np.tanh, lambda _v, d=d1: d,
                           lambda _v, d=d1: -2.0 * t * d,
                           lambda _v, d=d1: (4.0 - 6.0 * d) * d)

    def act(self, name):
        """C-inf activation tangents, matching :meth:`Jet2.act`."""
        V = self.v
        if name == "isru":
            s = np.sqrt(1.0 + V * V)
            s3, s5, s7 = s ** 3, s ** 5, s ** 7
            return self._unary(
                lambda v, ss=s: v / ss,
                lambda _v, a=s3: 1.0 / a,
                lambda v, a=s5: -3.0 * v / a,
                lambda v, a=s5, b=s7: -3.0 / a + 15.0 * v * v / b)
        if name == "tanh":
            return self.tanh()
        if name == "sigmoid":
            q = 1.0 / (1.0 + np.exp(-V))
            d1 = q * (1.0 - q)
            d2 = d1 * (1.0 - 2.0 * q)
            return self._unary(
                lambda _v, qq=q: qq, lambda _v, a=d1: a,
                lambda _v, a=d2: a,
                lambda _v, a=d2, b=d1: a * (1.0 - 2.0 * q) - 2.0 * b * b)
        raise ValueError(f"unsupported activation {name!r}")


def _net_j2p(pj, coef, g, d, act, t=None):
    """Unrolled network over parameter jets (P tangents, one vector pass)."""
    nin = int(coef.get("n_inputs", 2))
    n = (len(pj) - 1) // (nin + 2)
    b1_at = n * nin
    w2_at = b1_at + n
    y = pj[w2_at + n]
    for i in range(n):
        pre = (pj[i * nin] * (g - coef["vgs_center"]) / coef["vgs_scale"]
               + pj[i * nin + 1] * (d - coef["vds_center"])
               / coef["vds_scale"])
        if nin == 3:
            if t is None:
                raise ValueError("temperature input required for a 3-input net")
            pre = pre + (pj[i * nin + 2]
                         * (t - coef["temperature_center"])
                         / coef["temperature_scale"])
        y = y + pj[w2_at + i] * (pre + pj[b1_at + i]).act(act)
    return y


def _j2p_const(value, like):
    """Constant jet built from an existing jet's zero-tangent blocks."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return _Jet2P.const(value, np.zeros_like(like.v), like.pv)
    v = np.broadcast_to(arr, np.shape(like.v)).astype(float)
    z1 = np.zeros_like(v)
    zp = np.zeros_like(like.pv)
    return _Jet2P(v, (z1, z1), (z1, z1, z1), zp, (zp, zp), (zp, zp, zp))


def _channel_j2p(pj, coef, g, d, v_th, v_sat, alpha, guard_width, act, t=None):
    """Guarded channel over parameter jets; mirrors :func:`_channel_jet`."""
    y = _net_j2p(pj, coef, g, d, act, t)
    cut = 0.5 * (1.0 + (alpha * (g - v_th)).tanh())
    hi = d * y.exp() * cut
    y_sat = _net_j2p(pj, coef, g, _j2p_const(v_sat, g), act, t)
    lo = d * y_sat.exp() * cut
    blend = 0.5 * (1.0 + ((d - v_sat) / guard_width).tanh())
    return lo + blend * (hi - lo)


@dataclass
class DptPoints:
    """Offline high-V/I points extracted from a captured golden DPT."""
    vgs: np.ndarray
    vds: np.ndarray
    id: np.ndarray
    provenance: Dict[str, Any] = field(default_factory=dict)


def _curve_samples(curves: Sequence) -> Tuple[np.ndarray, np.ndarray,
                                               np.ndarray, Dict[str, Any],
                                               np.ndarray]:
    """Normalise inputs to (vgs, vds, id, provenance, regular_grid_mask).

    ``grid_mask`` marks samples belonging to a CurveData Vgs family: only
    those carry the structured surface from which gm/gds derivative targets
    are formed.  Scattered DPT points contribute the ID term alone.
    """
    from common.types import CurveData

    def _scattered(v, d, i, p):
        v = np.asarray(v, float)
        return (v, np.asarray(d, float), np.asarray(i, float), p,
                np.zeros(v.shape, dtype=bool))

    def _family(c):
        fv = c.family_value
        if fv is None:
            fv = c.metadata.get("vgs")
        if fv is None:
            raise ValueError(f"{c.name}: missing Vgs family value")
        x, y = c.to_numpy()
        return (np.full_like(x, float(fv)), x, y,
                np.ones(x.shape, dtype=bool))

    if isinstance(curves, DptPoints):
        return _scattered(curves.vgs, curves.vds, curves.id,
                          dict(curves.provenance))
    if isinstance(curves, CurveData):
        families = curves.metadata.get("families")
        if families is None:
            v, d, i, m = _family(curves)
            return v, d, i, {"source": "CurveData", "name": curves.name}, m
        curves = list(families)
    if isinstance(curves, dict):
        data = cast(Dict[str, Any], curves)
        curves = (data["vgs"], data["vds"], data["id"])
    if (isinstance(curves, tuple) and len(curves) == 3
            and not any(isinstance(c, CurveData) for c in curves)):
        return _scattered(curves[0], curves[1], curves[2],
                          {"source": "arrays"})

    samples = list(curves)
    if not samples:
        raise TypeError("curves_1_2 must not be empty")
    parts = []
    dpt_provenance: List[Dict[str, Any]] = []
    for c in samples:
        if isinstance(c, DptPoints):
            v, d, i, p, m = _scattered(c.vgs, c.vds, c.id,
                                       dict(c.provenance))
            parts.append((v, d, i, m))
            dpt_provenance.append(p)
        elif isinstance(c, CurveData):
            parts.append(_family(c))
        else:
            raise TypeError("list entries must be CurveData or DptPoints")
    prov = {"source": "CurveData family + DPT points"}
    if dpt_provenance:
        prov["dpt_high_vi"] = dpt_provenance
    return (np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]),
            prov,
            np.concatenate([p[3] for p in parts]))


def _norm(v: np.ndarray) -> Tuple[float, float]:
    lo, hi = float(np.min(v)), float(np.max(v))
    span = hi - lo
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError("degenerate training range: cannot centre/scale")
    return lo, span


def _temperature_input(temperature_c: Any, n: int
                       ) -> Tuple[np.ndarray, float, float, float]:
    """Normalise the optional temperature input and report its axis.

    A scalar temperature is a single-temperature fit: the normalised column
    is identically zero at ``temperature_center`` (the fitted value) and the
    scale is 1 degC.  An array of per-sample temperatures is supported so a
    future multi-temperature extraction can use the same path.  The raw value
    returned first is the temperature used by the emitter for this fit.
    """
    t = np.asarray(temperature_c, dtype=float)
    if t.ndim == 0:
        t = np.full(n, float(t), dtype=float)
        value = float(t[0])
    else:
        t = np.ravel(t).astype(float)
        if t.size != n:
            raise ValueError("temperature_c array must match the sample count")
        value = float(np.mean(t))
    if not np.all(np.isfinite(t)):
        raise ValueError("temperature_c must be finite")
    center = float(np.mean(t))
    scale = float(np.ptp(t))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return (t - center) / scale, center, scale, value


def _resolve_holdout_mask(vgs: np.ndarray, grid_mask: np.ndarray,
                          holdout_fraction: float = 0.0,
                          holdout_mask: Any = None
                          ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Resolve an explicit mask or a Vgs-family hold-out split.

    Whole CurveData Vgs families are reserved, never random individual
    points: the family axis is the only structure that makes a held-out error
    an out-of-family generalisation error.  Returns ``(mask, provenance)``;
    the provenance is ``None`` when no hold-out was requested.
    """
    n = vgs.size
    if holdout_mask is not None:
        m = np.asarray(holdout_mask)
        if m.shape != (n,):
            raise ValueError("holdout_mask must be one boolean per sample")
        mask = m.astype(bool)
        prov = {"method": "caller_mask", "fraction": None,
                "n_holdout": int(np.count_nonzero(mask)),
                "families_held": None}
    else:
        frac = float(holdout_fraction)
        if not np.isfinite(frac) or frac <= 0.0:
            return np.zeros(n, dtype=bool), None
        if frac >= 1.0:
            raise ValueError("holdout_fraction must be in (0, 1)")
        if not np.any(grid_mask):
            raise ValueError(
                "holdout by Vgs family requires structured CurveData "
                "families; use holdout_mask for scattered samples")
        fam = np.unique(np.round(vgs[grid_mask], 9))
        if fam.size < 2:
            raise ValueError("holdout needs at least two Vgs families")
        n_hold = max(1, int(round(frac * fam.size)))
        n_hold = min(n_hold, fam.size - 1)
        sel = np.unique(np.round(
            np.linspace(0, fam.size - 1, n_hold)).astype(int))
        held = fam[sel]
        mask = np.isin(np.round(vgs, 9), held)
        prov = {"method": "vgs_family", "fraction": frac,
                "n_holdout": int(np.count_nonzero(mask)),
                "families_held": [float(x) for x in held],
                "n_families": int(fam.size)}
    if not np.any(mask):
        raise ValueError("hold-out selection reserved no samples")
    return mask, prov


def _grid_derivative_targets(vgs: np.ndarray, vds: np.ndarray,
                             i_d: np.ndarray, grid_mask: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray]:
    """gm/gds and their Vgs/Vds derivatives from the regular curve grid."""
    gm_t = np.zeros_like(i_d)
    gds_t = np.zeros_like(i_d)
    gm2_t = np.zeros_like(i_d)
    gds2_t = np.zeros_like(i_d)
    if grid_mask.any():
        gi, di, ii = vgs[grid_mask], vds[grid_mask], i_d[grid_mask]
        g = np.unique(gi)
        counts = np.bincount(np.searchsorted(g, gi))
        if (g.size >= 2 and counts.size == g.size
                and np.all(counts == counts[0]) and counts[0] > 2):
            order = np.lexsort((di, gi))
            mat = ii[order].reshape(g.size, counts[0])
            vd_axis = np.unique(di)
            gds_mat = np.gradient(mat, vd_axis, axis=1, edge_order=2)
            gm_mat = np.gradient(mat, g, axis=0, edge_order=1)
            gds2_mat = np.gradient(gds_mat, vd_axis, axis=1, edge_order=2)
            gm2_mat = np.gradient(gm_mat, g, axis=0, edge_order=1)
            dest = np.flatnonzero(grid_mask)[order]
            gm_t[dest] = np.asarray(gm_mat).ravel()
            gds_t[dest] = np.asarray(gds_mat).ravel()
            gm2_t[dest] = gm2_mat.ravel()
            gds2_t[dest] = gds2_mat.ravel()
    return gm_t, gds_t, gm2_t, gds2_t


@dataclass
class AnnChannelFit:
    """Analytic coefficient record + diagnostics for the ANN channel."""
    coefficients: Dict[str, Any]
    v_th: float
    v_sat: float
    activation: str
    neurons: int
    loss_weights: Dict[str, float]
    train_range: Dict[str, Any]
    out_of_range_check: str
    diagnostics: Dict[str, Any]
    provenance: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, vgs: np.ndarray, vds: np.ndarray,
                 temperature_c: Any = None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Offline self-check: (ID, gm, gds) via the same analytic jet."""
        vgs = np.asarray(vgs, dtype=float)
        vds = np.asarray(vds, dtype=float)
        j = _channel_jet(self, seed_input(vgs, 0), seed_input(vds, 1),
                         temperature_c)
        return np.asarray(j.v), np.asarray(j.g[0]), np.asarray(j.g[1])


def _unpack(p: np.ndarray, neurons: int, n_inputs: int = 2):
    n, nin = neurons, n_inputs
    w1 = p[:nin * n].reshape(n, nin)
    b1 = p[nin * n:nin * n + n]
    w2 = p[nin * n + n:nin * n + 2 * n]
    b2 = float(p[nin * n + 2 * n])
    return w1, b1, w2, b2


def _net_jet(coef: Dict[str, Any], g: Jet2, d: Jet2, act: str,
             t: Optional[Jet2] = None) -> Jet2:
    w1 = np.asarray(coef["w1"], float)
    b1 = np.asarray(coef["b1"], float)
    w2 = np.asarray(coef["w2"], float)
    nin = w1.shape[1]
    if nin == 3 and t is None:
        t = Jet2(float(coef["temperature_center"]))
    y = Jet2(float(coef["b2"]))
    for i in range(len(w2)):
        pre = (float(w1[i, 0]) * (g - float(coef["vgs_center"]))
               / float(coef["vgs_scale"])
               + float(w1[i, 1]) * (d - float(coef["vds_center"]))
               / float(coef["vds_scale"]))
        if nin == 3:
            pre = pre + (float(w1[i, 2])
                         * (t - float(coef["temperature_center"]))
                         / float(coef["temperature_scale"]))
        y = y + float(w2[i]) * (pre + float(b1[i])).act(act)
    return y


def _mlp1d_jet(par: dict, g: Jet2, scale: float) -> Jet2:
    w1, b1, w2, b2 = par["w1"], par["b1"], par["w2"], float(par["b2"])
    y = Jet2(b2)
    for wi, bi, w2i in zip(w1, b1, w2):
        y = y + float(w2i) * (float(wi) * g + float(bi)).tanh()
    return float(scale) * y.softplus()


def _channel_jet(fit: "AnnChannelFit", vgs: Jet2, vds: Jet2,
                 temperature_c: Any = None) -> Jet2:
    c = fit.coefficients
    if c.get("model_form") == "positive_conductance":
        g = (vgs - float(c["vgs_center"])) / float(c["vgs_scale"])
        isat = _mlp1d_jet(c["sat"], g, float(c["scale_i"]))
        vknee = float(c["vknee_min"]) + _mlp1d_jet(c["knee"], g, float(c["scale_v"]))
        gout = _mlp1d_jet(c["gout"], g, float(c["scale_g"]))
        vth_c = float(c.get("vth_eff", fit.v_th))
        cut = 0.5 * (1.0 + (float(c["alpha"]) * (vgs - vth_c)).tanh())
        return cut * (isat * (vds / vknee).tanh() + gout * vds)
    t = None
    if np.asarray(c["w1"]).shape[1] == 3:
        value = (c.get("temperature_value", c["temperature_center"])
                 if temperature_c is None else temperature_c)
        t = Jet2((np.asarray(value, float)
                  - float(c["temperature_center"]))
                 / float(c["temperature_scale"]))
    y = _net_jet(c, vgs, vds, fit.activation, t)
    cut = 0.5 * (1.0 + (fit.coefficients["alpha"]
                        * (vgs - fit.v_th)).tanh())
    hi = vds * y.exp() * cut
    d_sat = Jet2(float(fit.v_sat))
    y_sat = _net_jet(c, vgs, d_sat, fit.activation)
    lo = vds * y_sat.exp() * cut
    blend = 0.5 * (1.0 + ((vds - fit.v_sat)
                          / float(c["guard_width"])).tanh())
    return lo + blend * (hi - lo)


def fit_channel_iv(curves_1_2: Any, v_th: float, v_sat: float, *,
                   neurons: int = 10, activation: str = "isru",
                   loss_weights: Optional[dict] = None,
                   guard_width: Optional[float] = None,
                   guard_width_provenance: Optional[Dict[str, Any]] = None,
                   holdout_fraction: float = 0.0,
                   holdout_mask: Optional[Any] = None,
                   temperature_c: Optional[Any] = None,
                   model_form: str = "legacy_exp",
                   curve7_vgs: Optional[Any] = None,
                   curve7_id: Optional[Any] = None,
                   curve7_id_th_target: Optional[float] = None
                   ) -> AnnChannelFit:
    if model_form == "positive_conductance":
        return fit_channel_iv_monotone(
            curves_1_2, v_th, v_sat,
            guard_width=guard_width,
            guard_width_provenance=guard_width_provenance,
            curve7_vgs=curve7_vgs, curve7_id=curve7_id,
            curve7_id_th_target=curve7_id_th_target)
    """Fit the guarded ANN channel; return analytic coefficients + diagnostics.

    ``curves_1_2`` accepts a CurveData family, a list of CurveData, a
    :class:`DptPoints`, a dict ``{vgs, vds, id}`` or an ``(vgs, vds, id)``
    tuple.  Derivative targets come from the curve family (np.gradient on the
    Vgs/Vds coordinates); scattered DPT points contribute the ID term only.
    ``v_th``/``v_sat`` are measured scalars (Curves 7 and 3), never defaults.

    ``holdout_fraction``/``holdout_mask`` reserve whole Vgs families (the
    caller mask is honoured when supplied) and fit on the rest; absent
    hold-out metrics are reported as ``not_measured``.  ``temperature_c``
    adds a third normalised input column; when absent the fit and emission
    stay in the existing two-input form.

    ``curve7_vgs``/``curve7_id``/``curve7_id_th_target`` are consumed only by
    ``model_form == "positive_conductance"``: the measured diode-connected
    Curve-7 transfer refits ``(alpha, vth_eff)`` after the static fit.  The
    legacy exponential path ignores them.
    """
    if activation not in _SUPPORTED:
        raise ValueError(f"activation must be one of {_SUPPORTED}")
    neurons = int(neurons)
    if not 1 <= neurons <= _MAX_NEURONS:
        raise ValueError(f"neurons must be in [1, {_MAX_NEURONS}]")
    v_th, v_sat = float(v_th), float(v_sat)
    if not np.isfinite(v_th) or not np.isfinite(v_sat) or v_sat <= 0.0:
        raise ValueError("v_th finite and v_sat > 0 are required")

    vgs_raw, vds_raw, i_raw, sample_prov, grid_raw = _curve_samples(curves_1_2)
    good = (np.isfinite(vgs_raw) & np.isfinite(vds_raw) & np.isfinite(i_raw)
            & (vds_raw > 0))
    raw_n = vgs_raw.size
    vgs_all, vds_all, i_all = vgs_raw[good], vds_raw[good], i_raw[good]
    grid_all = np.asarray(grid_raw, dtype=bool)[good]
    if holdout_mask is not None and np.asarray(holdout_mask).size == raw_n:
        holdout_mask = np.asarray(holdout_mask, dtype=bool)[good]
    if temperature_c is not None:
        t_all = np.asarray(temperature_c, dtype=float)
        if t_all.ndim == 0:
            pass
        elif t_all.size == raw_n:
            temperature_c = t_all[good]
        elif t_all.size != vgs_all.size:
            raise ValueError("temperature_c array must match the sample count")
    n_in = 3 if temperature_c is not None else 2
    if n_in == 3:
        t_norm_all, t_center, t_scale, t_value = _temperature_input(
            temperature_c, vgs_all.size)
        t_raw_all = t_norm_all * t_scale + t_center
    else:
        t_norm_all, t_raw_all = None, None
        t_center = t_scale = t_value = None

    holdout, holdout_prov = _resolve_holdout_mask(
        vgs_all, grid_all, holdout_fraction, holdout_mask)
    train = ~holdout
    if int(np.count_nonzero(train)) < 2 * neurons + 4:
        raise ValueError("too few channel samples for the requested net")
    vgs, vds, i_d = vgs_all[train], vds_all[train], i_all[train]
    t_norm = t_norm_all[train] if n_in == 3 else None
    t_raw = t_raw_all[train] if n_in == 3 else None

    cg, sg = _norm(vgs)
    cd, sd = _norm(vds)
    # The low-VDS guard width is a property of the *static* Curve-1 Vds grid.
    # Scattered DPT trajectory points (Vds on the bus-voltage plateau, often
    # nanovolts apart) must never set the hinge width: eps_from_grid caps at
    # half the minimum spacing, so they collapse it by orders of magnitude.
    if guard_width is None:
        has_grid = bool(np.any(grid_all))
        grid_vds = np.unique(vds_all[grid_all] if has_grid else vds_all)
        guard_width = eps_from_grid(grid_vds.tolist())
        gw_prov: Dict[str, Any] = {
            "value": float(guard_width),
            "rule": "approximation.smoother.eps_from_grid "
                    "(median positive spacing, capped at half minimum)",
            "source": ("Curve-1 static Vds grid (grid_mask)"
                       if has_grid else
                       "scattered Vds only: no static grid; caller should "
                       "supply guard_width explicitly"),
            "n_grid_points": int(grid_vds.size),
            "dpt_points_excluded": int(np.count_nonzero(~grid_all)),
        }
    else:
        guard_width = float(guard_width)
        gw_prov = dict(guard_width_provenance or {})
        gw_prov["value"] = float(guard_width)
        gw_prov.setdefault("source", "caller-supplied static Curve-1 Vds grid")
    if not np.isfinite(guard_width) or guard_width <= 0.0:
        raise ValueError("guard_width must be finite and > 0")

    # Target derivatives from the regular Curve-1 grid (not the network).
    # Computed on the training families only: a held-out family must not
    # influence the gradients the fit sees at its boundary.
    gm_t, gds_t, gm2_t, gds2_t = _grid_derivative_targets(
        vgs, vds, i_d, grid_all[train])

    # Comparable-magnitude normalisation (ANN study 5.3: tune by magnitude).
    def _scale(a: np.ndarray) -> float:
        r = float(np.sqrt(np.mean(a * a)))
        return r if np.isfinite(r) and r > 1e-30 else 1.0

    scales = {"id": _scale(i_d), "gm": _scale(gm_t), "gds": _scale(gds_t),
              "gm2": _scale(gm2_t), "gds2": _scale(gds2_t)}
    w = default_loss_weights()
    if loss_weights:
        unknown = set(loss_weights) - set(w)
        if unknown:
            raise ValueError(f"unknown loss weight keys {sorted(unknown)}")
        w.update({k: float(v) for k, v in loss_weights.items()})

    nparam = (n_in + 2) * neurons + 1
    ns = vgs.size
    # One pre-allocated tangent block per parameter; reused every evaluation
    # so the analytic Jacobian costs no per-call (P, P, N) allocation.
    pv_block = np.zeros((nparam, nparam, ns))
    pv_block[np.arange(nparam), np.arange(nparam), :] = 1.0
    zp = np.zeros((nparam, ns))
    z1 = np.zeros(ns)
    pj = [_Jet2P.seed_param(0.0, pv_block[j], z1, zp)
          for j in range(nparam)]
    g_jet = _Jet2P.seed_input(vgs, 0, z1, zp)
    d_jet = _Jet2P.seed_input(vds, 1, z1, zp)
    t_jet = (_j2p_const(t_norm, g_jet) if n_in == 3 else None)
    geo: Dict[str, Any] = {"vgs_center": cg, "vgs_scale": sg,
                           "vds_center": cd, "vds_scale": sd,
                           "n_inputs": n_in}
    if n_in == 3:
        geo["temperature_center"] = t_center
        geo["temperature_scale"] = t_scale
    target = (i_d, gm_t, gds_t, gm2_t, gds2_t)
    keys = ("id", "gm", "gds", "gm2", "gds2")
    sg_pen = scales["gm"]
    sd_pen = scales["gds"]

    def _evaluate(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for j, val in enumerate(p):
            pj[j].v = np.full(ns, float(val))
        j = _channel_j2p(pj, geo, g_jet, d_jet, v_th, v_sat,
                         _CUTOFF_ALPHA, guard_width, activation, t_jet)
        comps = (j.v, j.g[0], j.g[1], j.h[0], j.h[2])
        pcomps = (j.pv, j.pg[0], j.pg[1], j.ph[0], j.ph[2])
        res, rows = [], []
        for k, c, pc, t in zip(keys, comps, pcomps, target):
            if w[k] > 0.0:
                res.append(w[k] * (c - t) / scales[k])
                rows.append((w[k] * pc / scales[k]).T)
        if _MONOTONE_WEIGHT > 0.0:
            # Soft penalty: gm = dId/dVgs and gds = dId/dVds must be >= 0.
            # ``min(., 0)`` is the negative part; its subgradient is -1 on
            # the active samples (a kink at exactly zero, not a hard gate).
            for c, pc, sc in ((j.g[0], j.pg[0], sg_pen),
                              (j.g[1], j.pg[1], sd_pen)):
                neg = np.minimum(c, 0.0)
                res.append(_MONOTONE_WEIGHT * neg / sc)
                rows.append((_MONOTONE_WEIGHT
                             * np.where(c < 0.0, -pc, 0.0) / sc).T)
        return np.concatenate(res), np.vstack(rows)

    cache: Dict[bytes, Tuple[np.ndarray, Optional[np.ndarray]]] = {}

    def residual(p: np.ndarray) -> np.ndarray:
        key = p.tobytes()
        if key not in cache:
            r, _ = _evaluate(p)
            cache.clear()
            cache[key] = (r, None)
        return cache[key][0]

    def jac(p: np.ndarray) -> np.ndarray:
        key = p.tobytes()
        entry = cache.get(key)
        J = None if entry is None else entry[1]
        if J is None:
            r, J = _evaluate(p)
            cache.clear()
            cache[key] = (r, J)
        return J

    lo_b = np.concatenate([np.full(n_in * neurons, -_OPT_BOUNDS["w1"]),
                           np.full(neurons, -_OPT_BOUNDS["b1"]),
                           np.full(neurons, -_OPT_BOUNDS["w2"]),
                           [-_OPT_BOUNDS["b2"]]])
    hi_b = np.concatenate([np.full(n_in * neurons, _OPT_BOUNDS["w1"]),
                           np.full(neurons, _OPT_BOUNDS["b1"]),
                           np.full(neurons, _OPT_BOUNDS["w2"]),
                           [_OPT_BOUNDS["b2"]]])
    # Data-derived log-conductance centre (no magic units).
    ratio = np.log(np.maximum(i_d / np.maximum(vds, 1e-12), 1e-12))
    b2_0 = float(np.median(ratio))
    seeds = []
    for s in range(_OPT_STARTS):
        rng = np.random.default_rng(s)
        amp = 0.5 if s == 0 else 0.9
        w2amp = 0.25 if s == 0 else 0.8
        w1p = rng.normal(0.0, amp, n_in * neurons)
        b1p = (rng.normal(0.0, amp, neurons) if s == 0
               else rng.uniform(-2.0, 2.0, neurons))
        w2p = rng.normal(0.0, w2amp, neurons)
        seeds.append(np.clip(np.concatenate([w1p, b1p, w2p, [b2_0]]),
                             lo_b, hi_b))

    history: List[Dict[str, Any]] = []
    solutions = []
    last_cost: Optional[float] = None
    reason = ""
    for s, p0 in enumerate(seeds):
        # scipy leaves ``jac`` unannotated (its default is inferred as str).
        sol = least_squares(residual, p0, jac=cast(Any, jac),
                            max_nfev=_CHANNEL_MAX_NFEV,
                            xtol=_OPT_TOL, ftol=_OPT_TOL, gtol=_OPT_TOL)
        solutions.append(sol)
        history.append({"start": s, "cost": float(sol.cost),
                        "nfev": int(sol.nfev), "njev": int(sol.njev),
                        "status": int(sol.status), "reason": sol.message})
        if (last_cost is not None and sol.status > 0
                and abs(last_cost - sol.cost)
                <= _OPT_NO_IMPROVE * max(1.0, abs(sol.cost))):
            reason = "no-improvement stop: " + sol.message
            break
        last_cost = sol.cost
    best = min(solutions, key=lambda so: so.cost)
    reason = reason or best.message
    if not np.all(np.isfinite(best.x)):
        raise RuntimeError("ANN channel fit produced non-finite coefficients")
    w1, b1, w2, b2 = _unpack(best.x, neurons, n_in)
    coef = {"w1": w1.tolist(), "b1": b1.tolist(), "w2": w2.tolist(),
            "b2": b2, "vgs_center": cg, "vgs_scale": sg,
            "vds_center": cd, "vds_scale": sd,
            "alpha": _CUTOFF_ALPHA, "guard_width": guard_width}
    train_range: Dict[str, Any] = {
        "vgs": [cg, cg + sg], "vds": [cd, cd + sd],
        "vgs_center": cg, "vgs_scale": sg,
        "vds_center": cd, "vds_scale": sd}
    if n_in == 3:
        coef["temperature_center"] = float(t_center)
        coef["temperature_scale"] = float(t_scale)
        coef["temperature_value"] = float(t_value)
        train_range["temperature_c"] = [float(np.min(t_raw_all)),
                                        float(np.max(t_raw_all))]
        train_range["temperature_center"] = float(t_center)
        train_range["temperature_scale"] = float(t_scale)
    fit = AnnChannelFit(coef, v_th, v_sat, activation, neurons, dict(w),
                        train_range, "pending", {}, {})

    i_fit, gm_fit, gds_fit = fit.evaluate(vgs, vds, t_raw)
    finite = bool(np.all(np.isfinite(i_fit)))
    # Out-of-range probe: 20 % beyond each fitted edge (documented diagnostic
    # margin, not an emitter constant); current must stay finite and bounded.
    g_pad = 0.2 * sg
    d_pad = 0.2 * sd
    og = np.concatenate([np.linspace(cg - g_pad, cg, 8),
                         np.linspace(cg + sg, cg + sg + g_pad, 8)])
    od = np.concatenate([np.linspace(max(cd - d_pad, 1e-9), cd, 8),
                         np.linspace(cd + sd, cd + sd + d_pad, 8)])
    gg, dd = np.meshgrid(og, od, indexing="ij")
    i_oor, gm_oor, gds_oor = fit.evaluate(gg.ravel(), dd.ravel())
    bound = 10.0 * float(np.max(np.abs(i_d)))  # documented guard factor
    oor_ok = bool(finite and np.all(np.isfinite(i_oor))
                  and np.all(np.abs(i_oor) <= bound)
                  and np.all(np.isfinite(gm_oor))
                  and np.all(np.isfinite(gds_oor)))
    fit.out_of_range_check = "pass" if oor_ok else "fail"

    holdout_fields: Dict[str, Any] = {
        "holdout_id_rms_rel": "not_measured",
        "holdout_gm_rms_rel": "not_measured",
        "holdout_gds_rms_rel": "not_measured",
        "holdout_n": "not_measured",
    }
    if holdout_prov is not None:
        hvgs, hvds, hid = vgs_all[holdout], vds_all[holdout], i_all[holdout]
        gm_all, gds_all, _, _ = _grid_derivative_targets(
            vgs_all, vds_all, i_all, grid_all)
        hgm, hgds = gm_all[holdout], gds_all[holdout]
        ht = t_raw_all[holdout] if n_in == 3 else None
        hi, hgm_fit, hgds_fit = fit.evaluate(hvgs, hvds, ht)

        def _rel(pred: np.ndarray, tgt: np.ndarray) -> Any:
            r = float(np.sqrt(np.mean(tgt * tgt)))
            if not np.isfinite(r) or r <= 1e-30:
                return "not_measured"
            return float(np.sqrt(np.mean((pred - tgt) ** 2)) / r)

        holdout_fields = {
            "holdout_id_rms_rel": _rel(hi, hid),
            "holdout_gm_rms_rel": _rel(hgm_fit, hgm),
            "holdout_gds_rms_rel": _rel(hgds_fit, hgds),
            "holdout_n": int(np.count_nonzero(holdout)),
        }

    neg_gm = np.minimum(gm_fit, 0.0)
    neg_gds = np.minimum(gds_fit, 0.0)
    fit.diagnostics = {
        "guard_width_v": float(guard_width),
        "id_rms_rel": float(np.sqrt(np.mean((i_fit - i_d) ** 2))
                            / max(scales["id"], 1e-30)),
        "gm_rms_rel": float(np.sqrt(np.mean((gm_fit - gm_t) ** 2))
                            / scales["gm"]),
        "gds_rms_rel": float(np.sqrt(np.mean((gds_fit - gds_t) ** 2))
                             / scales["gds"]),
        "out_of_range_max_abs_a": float(np.max(np.abs(i_oor))),
        "out_of_range_bound_a": bound,
        "train_samples": int(ns),
        "train_cost": float(best.cost),
        "optimizer_cost": float(best.cost),
        "optimizer_success": bool(best.status > 0),
        "optimizer_nfev": int(sum(h["nfev"] for h in history)),
        "optimizer_njev": int(sum(h["njev"] for h in history)),
        "optimizer_starts_run": len(history),
        "optimizer_termination_reason": reason,
        "derivative_terms": [k for k in ("gm", "gds", "gm2", "gds2")
                             if w[k] > 0.0],
        "monotone_penalty": {
            "weight": float(_MONOTONE_WEIGHT),
            "form": "least-squares negative part of gm and gds",
            "hard_guarantee": False,
            "negative_gm_samples": int(np.count_nonzero(gm_fit < 0.0)),
            "negative_gds_samples": int(np.count_nonzero(gds_fit < 0.0)),
            "negative_d_id_dvds_samples": int(np.count_nonzero(gds_fit < 0.0)),
            "negative_gm_rms_a_per_v": float(
                np.sqrt(np.mean(neg_gm * neg_gm))),
            "negative_gds_rms_a_per_v": float(
                np.sqrt(np.mean(neg_gds * neg_gds))),
        },
        "holdout_selection": holdout_prov,
        "temperature_center": (float(t_center) if n_in == 3 else None),
        "temperature_scale": (float(t_scale) if n_in == 3 else None),
        **holdout_fields,
    }
    fit.provenance = {
        "guard_width": gw_prov,
        "curves": sample_prov,
        "v_th": "curve7_threshold_voltage:vth_v (caller-supplied scalar)",
        "v_sat": "curve3/ohmic data (caller-supplied scalar)",
        "cutoff_alpha": f"source-derived initial value {_CUTOFF_ALPHA} "
                        "(Yang TPEL 2022, CONDITIONAL until URL pinned)",
        "activation": "project decision: C-inf bounded ISRU fallback "
                      "u/sqrt(1+u^2) (verified sqrt primitive)",
        "optimizer_bounds": dict(_OPT_BOUNDS),
        "optimizer_max_nfev": _CHANNEL_MAX_NFEV,
        "optimizer_starts_configured": _OPT_STARTS,
        "optimizer_starts_run": len(history),
        "optimizer_tolerances": {"xtol": _OPT_TOL, "ftol": _OPT_TOL,
                                 "gtol": _OPT_TOL},
        "optimizer_no_improvement_tolerance": _OPT_NO_IMPROVE,
        "optimizer_analytic_jacobian":
            "forward-mode Jet2P: d(value,gm,gds,gm2,gds2)/dp; no FD",
        "optimizer_nfev": int(sum(h["nfev"] for h in history)),
        "optimizer_njev": int(sum(h["njev"] for h in history)),
        "optimizer_termination_reason": reason,
        "optimizer_cost_history": history,
        "loss_weight_defaults": dict(_DEFAULT_WEIGHTS),
        "loss_weight_normalisation": "each term divided by its target RMS",
        "monotone_penalty": {
            "value": float(_MONOTONE_WEIGHT),
            "form": "soft least-squares penalty on min(gm,0) and min(gds,0)",
            "hard_guarantee": False,
            "note": "reduces negative differential conductance; it does not "
                    "prove gm>=0 or gds>=0 on or off the training grid",
        },
        "holdout": holdout_prov,
        "holdout_rule": "whole CurveData Vgs families reserved; fit on the "
                        "remaining samples; no hold-out => not_measured",
        "temperature": (None if n_in == 2 else {
            "center_c": float(t_center), "scale_c": float(t_scale),
            "value_c": float(t_value), "n_inputs": 3,
            "column": "normalised (T - center)/scale",
            "emission_point_c": float(t_value),
            "emitter_temperature_port": False,
            "note": "the emitted three-terminal subcircuit is instantiated "
                    "at value_c; the third input records the fitted axis "
                    "and does not add a temperature port",
        }),
        "out_of_range_margin": 0.2,
        "out_of_range_bound_factor": 10.0,
        "external_source_status": "conditional_until_url_and_section_pinned",
    }
    return fit


def _positive_base_current(fit: "AnnChannelFit", vgs: np.ndarray,
                           vds: np.ndarray) -> np.ndarray:
    """Uncut positive-conductance channel ``I_sat*tanh(Vds/Vk) + g_out*Vds``.

    This is the fit's analytic base with the tanh cutoff divided out, so the
    post-fit Curve-7 calibration can re-use the fixed static coefficients
    without evaluating (and inverting) a possibly under-flowed cutoff.
    """
    c = fit.coefficients
    if c.get("model_form") != "positive_conductance":
        raise ValueError("positive-conductance coefficients are required")
    vgs = np.asarray(vgs, dtype=float)
    vds = np.asarray(vds, dtype=float)
    g = (vgs - float(c["vgs_center"])) / float(c["vgs_scale"])

    def _mlp(par: Dict[str, Any], scale: float) -> np.ndarray:
        w1 = np.asarray(par["w1"], dtype=float)
        b1 = np.asarray(par["b1"], dtype=float)
        w2 = np.asarray(par["w2"], dtype=float)
        u = float(par["b2"]) + np.tanh(g[:, None] * w1[None, :]
                                       + b1[None, :]) @ w2
        return float(scale) * np.log1p(np.exp(np.clip(u, -40.0, 40.0)))

    isat = _mlp(c["sat"], float(c["scale_i"]))
    vknee = float(c["vknee_min"]) + _mlp(c["knee"], float(c["scale_v"]))
    gout = _mlp(c["gout"], float(c["scale_g"]))
    return isat * np.tanh(vds / vknee) + gout * vds


def _curve7_samples(vgs7: Any, id7: Any) -> Tuple[np.ndarray, np.ndarray,
                                                  float]:
    """Validate and normalise measured Curve-7 (drain tied to gate) samples.

    Raises ``ValueError`` (fail closed) on empty/malformed data instead of
    substituting the old ``alpha=4`` / ``vth_eff=v_th`` cutoff.  Monotonicity
    is required up to a relative float-noise band derived from the measured
    peak; a genuine decrease fails.
    """
    vgs = np.asarray(vgs7, dtype=float).ravel()
    ids = np.asarray(id7, dtype=float).ravel()
    if vgs.size == 0 or ids.size == 0:
        raise ValueError("Curve-7 data is empty")
    if vgs.size != ids.size:
        raise ValueError("Curve-7 Vgs and Id must have equal length")
    if not (np.all(np.isfinite(vgs)) and np.all(np.isfinite(ids))):
        raise ValueError("Curve-7 data contains non-finite samples")
    order = np.argsort(vgs, kind="stable")
    vgs, ids = vgs[order], ids[order]
    if np.any(np.diff(vgs) <= 0.0):
        raise ValueError("Curve-7 Vgs is not strictly increasing")
    if np.any(ids < 0.0):
        raise ValueError("Curve-7 Id is negative")
    peak = float(np.max(ids))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError("Curve-7 Id has no positive measured current")
    if np.any(np.diff(ids) < -_CURVE7_MONOTONE_RTOL * peak):
        raise ValueError("Curve-7 Id is non-monotone")
    return vgs, ids, peak


def _curve7_subthreshold_mask(ids: np.ndarray, peak: float,
                              id_th_target: Optional[float]
                              ) -> Tuple[np.ndarray, float, str]:
    """Subthreshold sample mask plus the threshold-current definition used."""
    if id_th_target is None:
        id_th = _CURVE7_SUBTHRESHOLD_FRACTION * peak
        source = (f"data-derived {_CURVE7_SUBTHRESHOLD_FRACTION:g} x measured "
                  f"Curve-7 peak {peak:.6g} A (Id_th_target absent)")
    else:
        id_th = float(id_th_target)
        if not np.isfinite(id_th) or id_th <= 0.0:
            raise ValueError("Curve-7 Id_th_target must be finite and > 0")
        source = "curve7_threshold_voltage:Id_th_target"
    mask = ids <= id_th
    # The zero-current anchor is always part of the calibration set; the
    # analytic base is exactly zero at Vgs=Vds=0, so it pins the log floor.
    mask[int(np.argmin(np.abs(ids)))] = True
    return mask, float(id_th), source


def calibrate_cutoff_curve7(fit: "AnnChannelFit", vgs7: Any, id7: Any, *,
                            id_th_target: Optional[float] = None
                            ) -> Dict[str, Any]:
    """Post-fit ``(alpha, vth_eff)`` calibration on measured Curve-7 data.

    Only the two positive-conductance cutoff parameters are refitted; the
    static ``isat``/``knee``/``gout`` coefficients, their scales, the topology
    and the G1/G2 emission semantics are untouched.  Residuals are log10
    current errors on the measured subthreshold samples, with the drain tied
    to the gate (Curve-7 JEDEC constant-current sweep).  ``alpha`` stays inside
    the existing ``[2, 35]`` range and ``vth_eff`` inside
    ``[v_th, v_th + (max(vgs7) - v_th)]``.

    Raises ``ValueError``/``RuntimeError`` (fail closed) when Curve-7 is
    missing, non-finite, non-monotone, has fewer than three subthreshold
    points, or the optimizer cannot produce a finite, ordered result.
    """
    if not isinstance(fit, AnnChannelFit):
        raise TypeError("fit must be an AnnChannelFit")
    if fit.coefficients.get("model_form") != "positive_conductance":
        raise ValueError("Curve-7 cutoff calibration requires "
                         "model_form='positive_conductance'")
    vgs, ids, peak = _curve7_samples(vgs7, id7)
    mask, id_th, id_th_source = _curve7_subthreshold_mask(
        ids, peak, id_th_target)
    n_used = int(np.count_nonzero(mask))
    if n_used < _CURVE7_MIN_SUBTHRESHOLD_POINTS:
        raise ValueError(f"Curve-7 has only {n_used} subthreshold points "
                         f"(need >= {_CURVE7_MIN_SUBTHRESHOLD_POINTS})")
    vg, target_id = vgs[mask], ids[mask]
    positive = target_id[target_id > 0.0]
    if positive.size == 0:
        raise ValueError("Curve-7 subthreshold samples have no positive "
                         "current to anchor the log residual")
    floor = max(float(np.min(positive)), peak * _CURVE7_LOG_FLOOR_RTOL)
    # Drain tied to gate in the Curve-7 sweep.
    base = _positive_base_current(fit, vg, vg)
    if not np.all(np.isfinite(base)):
        raise ValueError("static channel is non-finite on Curve-7 samples")

    log_target = np.log10(target_id + floor)

    def _log_residual(p: np.ndarray) -> np.ndarray:
        alpha, vth = float(p[0]), float(p[1])
        cut = 0.5 * (1.0 + np.tanh(alpha * (vg - vth)))
        pred = np.maximum(cut * base, 0.0)
        return np.log10(pred + floor) - log_target

    def _log_rms(p: np.ndarray) -> float:
        r = _log_residual(p)
        return float(np.sqrt(np.mean(r * r)))

    c = fit.coefficients
    alpha_before = float(c["alpha"])
    vth_before = float(c.get("vth_eff", fit.v_th))
    vth_lo = float(fit.v_th)
    vth_hi = vth_lo + (float(np.max(vgs)) - vth_lo)
    if not (np.isfinite(alpha_before) and np.isfinite(vth_before)):
        raise ValueError("existing cutoff parameters are non-finite")
    if not np.isfinite(vth_lo) or not np.isfinite(vth_hi) or vth_hi <= vth_lo:
        raise ValueError("Curve-7 measured Vgs span does not extend above "
                         "v_th; cutoff is unidentifiable")
    lb = np.array([_CUTOFF_ALPHA_MIN, vth_lo], dtype=float)
    ub = np.array([_CUTOFF_ALPHA_MAX, vth_hi], dtype=float)

    # Deterministic starts: the joint-fit cutoff, plus two interior Vth
    # positions spread over the measured above-threshold span.  Best cost wins.
    alpha0 = float(np.clip(alpha_before, lb[0], ub[0]))
    span = ub[1] - lb[1]
    starts = [np.array([alpha0, float(np.clip(vth_before, lb[1], ub[1]))])]
    for k in range(1, _CUTOFF_STARTS):
        starts.append(np.array([alpha0, lb[1] + span * k / _CUTOFF_STARTS]))
    best = None
    nfev_total = 0
    for p0 in starts:
        p0 = np.clip(p0, lb, ub)
        sol = least_squares(_log_residual, p0, bounds=(lb, ub),
                            max_nfev=_CUTOFF_MAX_NFEV,
                            xtol=_OPT_TOL, ftol=_OPT_TOL, gtol=_OPT_TOL)
        nfev_total += int(sol.nfev)
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:  # pragma: no cover - _CUTOFF_STARTS >= 1
        raise RuntimeError("Curve-7 cutoff calibration ran no starts")
    alpha_after = float(best.x[0])
    vth_after = float(best.x[1])
    if not (np.isfinite(alpha_after) and np.isfinite(vth_after)
            and lb[0] <= alpha_after <= ub[0]
            and lb[1] <= vth_after <= ub[1]):
        raise RuntimeError("Curve-7 cutoff calibration produced invalid "
                           "alpha/vth_eff")
    log_rms_before = _log_rms(np.array([alpha_before, vth_before]))
    log_rms_after = _log_rms(np.array([alpha_after, vth_after]))
    if not (np.isfinite(log_rms_before) and np.isfinite(log_rms_after)):
        raise RuntimeError("Curve-7 cutoff calibration produced a non-finite "
                           "log residual")
    if log_rms_after > log_rms_before:
        raise RuntimeError("Curve-7 cutoff calibration did not reduce the "
                           f"log-RMS ({log_rms_before:.6g} -> "
                           f"{log_rms_after:.6g}); refusing a silent fallback")

    c["alpha"] = alpha_after
    c["vth_eff"] = vth_after
    diag: Dict[str, Any] = {
        "status": "measured",
        "source": "curve7_threshold_voltage:curve_vgs,curve_id "
                  "(drain tied to gate)",
        "points_used": n_used,
        "points_total": int(vgs.size),
        "id_th_target_a": float(id_th),
        "id_th_target_source": id_th_source,
        "alpha_before": alpha_before,
        "alpha_after": alpha_after,
        "alpha_bounds": [float(lb[0]), float(ub[0])],
        "vth_eff_before": vth_before,
        "vth_eff_after": vth_after,
        "vth_eff_bounds": [float(vth_lo), float(vth_hi)],
        "log10_floor_a": float(floor),
        "log_rms_before": log_rms_before,
        "log_rms_after": log_rms_after,
        "residual": "log10(cut*base+floor)-log10(Id7+floor)",
        "optimizer": {
            "method": "scipy.optimize.least_squares",
            "max_nfev_per_start": _CUTOFF_MAX_NFEV,
            "starts": len(starts),
            "nfev": nfev_total,
        },
    }
    fit.provenance["curve7_cutoff"] = diag
    return diag


def fit_channel_iv_monotone(
    curves_1_2: Any,
    v_th: float,
    v_sat: float,
    *,
    neurons: int = 6,
    alpha: float = 4.0,
    guard_width: Optional[float] = None,
    guard_width_provenance: Optional[Dict[str, Any]] = None,
    max_nfev: int = 1500,
    seed: int = 42,
    curve7_vgs: Optional[Any] = None,
    curve7_id: Optional[Any] = None,
    curve7_id_th_target: Optional[float] = None,
) -> AnnChannelFit:
    """Fit a C-inf positive-conductance ANN channel (hard gds > 0).

    I_D(Vgs, Vds) = cut(Vgs) * (I_sat(Vgs)*tanh(Vds/V_knee(Vgs)) + g_out(Vgs)*Vds)
    where I_sat, V_knee, g_out are softplus-bounded 1-D tanh MLPs in Vgs.
    Guarantees I_D(Vgs, 0) = 0 and dI_D/dVds > 0 everywhere (no negative gds).

    When ``curve7_vgs``/``curve7_id`` are supplied, the measured diode-connected
    Curve-7 transfer data refits only ``(alpha, vth_eff)`` after the static
    fit via :func:`calibrate_cutoff_curve7`; a malformed Curve-7 sample fails
    closed instead of keeping the joint-fit cutoff silently.
    """
    curves = list(curves_1_2)
    if not curves:
        raise ValueError("fit_channel_iv_monotone needs >= 1 CurveData")
    vg_all, vd_all, id_all, w_all = [], [], [], []
    for c in curves:
        x, y = c.to_numpy()
        vgs_val = float(c.family_value) if c.family_value is not None else 0.0
        vg_all.append(np.full_like(x, vgs_val))
        vd_all.append(x)
        id_all.append(y)
        # Match G1 compute_curve_error peak floor (1e-9) and normalize by sqrt(N)
        # so least_squares minimizes mean squared per-curve peak relative error.
        peak = max(float(np.max(np.abs(y))), 1e-9)
        w = np.full_like(x, 1.0 / (peak * float(np.sqrt(max(len(x), 1)))))
        if x.size:
            w[int(np.argmax(x))] *= 2.0
        w_all.append(w)
    vg_v = np.concatenate(vg_all)
    vd_v = np.concatenate(vd_all)
    id_v = np.concatenate(id_all)
    wt_v = np.concatenate(w_all)

    H = int(neurons)
    mu_g = float(np.mean(vg_v))
    sig_g = max(float(np.std(vg_v)), 1.0)
    scale_i = max(float(np.max(id_v)) * 0.25, 1.0)
    vmax = max(float(np.max(vd_v)), 1.0)
    scale_v = max(vmax * 0.1, 0.5)
    scale_g = (scale_i / max(scale_v, 1.0)) * 0.05
    vknee_min = 0.05

    def _sp(x):
        return np.log1p(np.exp(np.clip(x, -40.0, 40.0)))

    n1 = 3 * H + 1
    n_mlp = 3 * n1

    def _eval(p):
        ps, pk, po = p[:n1], p[n1:2 * n1], p[2 * n1:n_mlp]
        a_fit = float(p[n_mlp])
        vth_fit = float(p[n_mlp + 1])
        g = (vg_v - mu_g) / sig_g

        def _mlp(par, sc):
            w1, b1, w2, b2 = par[:H], par[H:2 * H], par[2 * H:3 * H], par[3 * H]
            h = np.tanh(g[:, None] * w1[None, :] + b1[None, :])
            return sc * _sp(b2 + h @ w2)

        isat = _mlp(ps, scale_i)
        vk = vknee_min + _mlp(pk, scale_v)
        gout = _mlp(po, scale_g)
        cut = 0.5 * (1.0 + np.tanh(a_fit * (vg_v - vth_fit)))
        return cut * (isat * np.tanh(vd_v / vk) + gout * vd_v)

    rng = np.random.default_rng(seed)
    p0_mlp = rng.normal(0.0, 0.5, size=n_mlp)
    a0 = float(np.clip(alpha, _CUTOFF_ALPHA_MIN, _CUTOFF_ALPHA_MAX))
    p0 = np.concatenate([p0_mlp, [a0, float(v_th)]])
    lb = np.concatenate([np.full(n_mlp, -np.inf),
                         [_CUTOFF_ALPHA_MIN, float(v_th)]])
    ub = np.concatenate([np.full(n_mlp, np.inf),
                         [_CUTOFF_ALPHA_MAX, float(v_th) + 0.5]])
    res = least_squares(lambda p: wt_v * (_eval(p) - id_v), p0,
                        bounds=(lb, ub),
                        max_nfev=max_nfev, ftol=1e-9, xtol=1e-9)
    p = res.x
    alpha_fit = float(p[n_mlp])
    vth_eff_fit = float(p[n_mlp + 1])

    def _pack(par):
        return {
            "w1": [float(x) for x in par[:H]],
            "b1": [float(x) for x in par[H:2 * H]],
            "w2": [float(x) for x in par[2 * H:3 * H]],
            "b2": float(par[3 * H]),
        }

    gw = float(guard_width) if guard_width is not None else max(
        eps_from_grid(np.unique(vd_v).tolist()), 1e-3)
    coef = {
        "model_form": "positive_conductance",
        "neurons": H,
        "vgs_center": mu_g,
        "vgs_scale": sig_g,
        "scale_i": scale_i,
        "scale_v": scale_v,
        "scale_g": scale_g,
        "vknee_min": vknee_min,
        "alpha": alpha_fit,
        "vth_eff": vth_eff_fit,
        "guard_width": gw,
        "sat": _pack(p[:n1]),
        "knee": _pack(p[n1:2 * n1]),
        "gout": _pack(p[2 * n1:3 * n1]),
        # Legacy compatibility fields for callers inspecting w1/b1/w2
        "w1": [[float(x), 0.0] for x in p[:H]],
        "b1": [float(x) for x in p[H:2 * H]],
        "w2": [float(x) for x in p[2 * H:3 * H]],
        "b2": float(p[3 * H]),
    }
    fit = AnnChannelFit(
        coefficients=coef,
        v_th=float(v_th),
        v_sat=float(v_sat),
        activation="tanh",
        neurons=H,
        loss_weights={"id": 1.0},
        train_range={"vgs": [float(np.min(vg_v)), float(np.max(vg_v))],
                     "vds": [float(np.min(vd_v)), float(np.max(vd_v))]},
        out_of_range_check="pass",
        diagnostics={},
        provenance={
            "model_form": "positive_conductance",
            "monotone_gds": {"hard_guarantee": True},
            "guard_width": guard_width_provenance,
            "optimizer_nfev": int(res.nfev),
        },
    )
    if curve7_vgs is not None or curve7_id is not None:
        if curve7_vgs is None or curve7_id is None:
            raise ValueError("Curve-7 cutoff calibration needs both "
                             "curve7_vgs and curve7_id")
        calibrate_cutoff_curve7(fit, curve7_vgs, curve7_id,
                                id_th_target=curve7_id_th_target)
    pred, gm, gds = fit.evaluate(vg_v, vd_v)
    fit.diagnostics = {
        "id_rms_rel": float(np.sqrt(np.mean((wt_v * (pred - id_v)) ** 2))),
        "gm_min": float(np.min(gm)),
        "gm_max": float(np.max(gm)),
        "gds_min": float(np.min(gds)),
        "gds_max": float(np.max(gds)),
        "gm_negative_fraction": float(np.mean(gm < 0.0)),
        "gds_negative_fraction": float(np.mean(gds < 0.0)),
        "derivative_terms": ["gm", "gds"],
        "monotone_penalty": {"hard_guarantee": True},
    }
    if "curve7_cutoff" in fit.provenance:
        fit.diagnostics["curve7_cutoff"] = fit.provenance["curve7_cutoff"]
    return fit
