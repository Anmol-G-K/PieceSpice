"""Physics-aware ANN charge (QV) fit -- unified ROM task T2.

Offline-only fitter.  Charge is trained in log space with an explicit Q0
offset (the charge-shift fix), then sampled back to monotone Q(V) tables in
the exact ``q_oss_*`` / ``q_iss_*`` / ``q_rss_*`` shape consumed by the closed
Tier-2 emitter.  Derivative terms dQ/dV and d2Q/dV2 are analytic via the
shared second-order jet in :mod:`approximation.ann_channel` (no finite
differences of the network).  Tables are emitted by the shared
``delta_ramp_chain`` in :mod:`approximation.smoother` with the caller's
C1 epsilon, so Q(0) = 0 and the delta-slope invariant are preserved.

The extracted Curve-4 YAML stores only cap equivalents at one reference bias;
there is no full C(V) sweep in the repository.  Where a sweep is supplied the
fitter integrates it; otherwise it anchors a monotone charge curve at the
measured equivalent Q(Cref) = Cref*Vref.  This limitation is recorded in
``provenance`` and reported by the orchestrator -- no charge surface is
fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import least_squares

from approximation.ann_channel import (Jet2, _OPT_BOUNDS, _OPT_MAX_NFEV,
                                       seed_input)
from approximation.smoother import eps_from_grid, smooth_relu_product

__all__ = ["AnnChargeFit", "fit_charge_qv", "default_charge_loss_weights"]

_TABLES = ("oss", "iss", "rss")
_SPEC_FIELDS = {
    "oss": ("q_oss_vs_v", "q_oss_charge_c"),
    "iss": ("q_iss_vs_v", "q_iss_charge_c"),
    "rss": ("q_rss_vs_v", "q_rss_charge_c"),
}
_DEFAULT_CHARGE_W = {"q": 1.0, "dq": 0.2, "d2q": 0.05}
_ACTIVATIONS = ("isru", "tanh", "sigmoid")


def default_charge_loss_weights() -> Dict[str, float]:
    return dict(_DEFAULT_CHARGE_W)


@dataclass
class AnnChargeFit:
    """Analytic log-net records, sampled monotone tables and diagnostics."""
    tables: Dict[str, Dict[str, list]]
    q0: Optional[float]
    activation: str
    neurons: int
    loss_weights: Dict[str, float]
    coefficients: Dict[str, Any]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, table: str, v: Any, eps: float = 0.0) -> Any:
        """Numpy mirror of the emitted Q= chain (finite, monotone)."""
        if table not in self.tables:
            raise KeyError(f"unknown charge table {table!r}")
        t = self.tables[table]
        return smooth_relu_product(v, t["v"], t["q"], eps)

    def as_spec_tables(self) -> Dict[str, list]:
        """Return the six canonical PWLModelSpec charge fields."""
        out: Dict[str, list] = {}
        for name, (vk, qk) in _SPEC_FIELDS.items():
            out[vk] = list(self.tables[name]["v"])
            out[qk] = list(self.tables[name]["q"])
        return out

    def monotone(self) -> bool:
        return all(
            np.all(np.diff(np.asarray(self.tables[n]["q"], float)) >= 0.0)
            and float(self.tables[n]["q"][0]) == 0.0 for n in _TABLES)


def _grid_meta(meta: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for key in ("ciss_f", "coss_f", "crss_f", "cap_ref_vds"):
        val = meta.get(key)
        out[key] = float(val) if val is not None else None
    return out


def _dedupe(x: np.ndarray, y: np.ndarray) -> tuple:
    """Collapse duplicate coordinates (curve5 repeats Vgs at the plateau)."""
    ux, inv = np.unique(x, return_inverse=True)
    if ux.size == x.size:
        return x, y
    sums = np.bincount(inv, weights=y)
    cnt = np.bincount(inv)
    return ux, sums / np.maximum(cnt, 1)

def _integrate(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Trapezoid C(V) -> anchored monotone Q(V)."""
    q = np.concatenate(([0.0], np.cumsum(0.5 * (y[1:] + y[:-1])
                                         * np.diff(x))))
    return np.maximum.accumulate(q)


def _anchor(vref: Optional[float], cref: Optional[float]
            ) -> Optional[tuple]:
    """Source-derived linear charge anchor Q(Cref) = Cref*Vref."""
    if vref is None or cref is None or vref <= 0 or cref <= 0:
        return None
    return np.array([0.0, float(vref)]), np.array([0.0, float(vref * cref)])


def _prepare_data(curve4, plateau):
    """Return {table: (x, Q)} plus the per-table source description."""
    from common.types import CurveData

    if not isinstance(curve4, CurveData) or not isinstance(plateau, CurveData):
        raise TypeError("fit_charge_qv expects two CurveData containers")
    x4, y4 = curve4.to_numpy()
    x4 = np.asarray(x4, float)
    y4 = np.asarray(y4, float)
    meta = _grid_meta(curve4.metadata)
    kind = str(curve4.metadata.get("y_kind", "capacitance")).lower()
    good = np.isfinite(x4) & np.isfinite(y4) & (y4 >= 0.0)
    x4, y4 = x4[good], y4[good]
    if x4.size < 2:
        raise ValueError("curve4 needs at least two finite samples")
    order = np.argsort(x4)
    x4, y4 = _dedupe(x4[order], y4[order])

    if kind == "charge":
        qoss = np.maximum.accumulate(y4 - y4[0])
    else:
        qoss = _integrate(x4, y4)

    xg, yg = plateau.to_numpy()
    xg = np.asarray(xg, float)
    yg = np.asarray(yg, float)
    gg = np.isfinite(xg) & np.isfinite(yg)
    xg, yg = xg[gg], yg[gg]
    order = np.argsort(xg)
    if xg.size >= 2:
        xg, yg = _dedupe(xg[order], yg[order])
        qiss = np.maximum.accumulate(yg - yg[0])
        qiss = np.maximum(qiss, 0.0)
    else:
        qiss = None

    data: Dict[str, tuple] = {"oss": (x4, qoss)}
    src: Dict[str, str] = {
        "oss": f"curve4 {curve4.name}: "
               + ("integrated C(V)" if kind != "charge" else "Q(V) table")}
    iss = (xg, qiss) if qiss is not None else _anchor(
        meta["cap_ref_vds"], meta["ciss_f"])
    if iss is not None:
        data["iss"] = iss
        src["iss"] = ("curve5 gate-charge plateau" if qiss is not None
                      else "ciss_f * cap_ref_vds anchor")
    rss = _anchor(meta["cap_ref_vds"], meta["crss_f"])
    if rss is not None:
        data["rss"] = rss
        src["rss"] = "crss_f * cap_ref_vds anchor"
    return data, src, meta


def _fit_one(x: np.ndarray, q: np.ndarray, q0: float, neurons: int,
             activation: str, weights: Dict[str, float], cx: float, sx: float
             ) -> Dict[str, Any]:
    """Fit log(Q + q0) with analytic dQ/dV and d2Q/dV2 residuals."""
    tgt = np.log(np.maximum(q + q0, 1e-300))
    dq_t = np.gradient(q, x, edge_order=1) if x.size > 2 else \
        np.full_like(q, (q[-1] - q[0]) / max(x[-1] - x[0], 1e-30))
    d2q_t = np.gradient(dq_t, x, edge_order=1) if x.size > 2 else \
        np.zeros_like(q)
    dq_t = np.nan_to_num(dq_t, nan=0.0, posinf=0.0, neginf=0.0)
    d2q_t = np.nan_to_num(d2q_t, nan=0.0, posinf=0.0, neginf=0.0)

    def _scale(a: np.ndarray) -> float:
        r = float(np.sqrt(np.mean(a * a)))
        return r if np.isfinite(r) and r > 1e-30 else 1.0

    sc = {"q": _scale(tgt), "dq": _scale(dq_t), "d2q": _scale(d2q_t)}
    p0 = np.concatenate([
        np.random.default_rng(0).normal(0.0, 0.5, 2 * neurons),
        np.random.default_rng(1).normal(0.0, 0.5, neurons),
        np.random.default_rng(2).normal(0.0, 0.25, neurons),
        [float(np.median(tgt))],
    ])
    nb = 2 * neurons
    lo = np.concatenate([np.full(nb, -_OPT_BOUNDS["w1"]),
                         np.full(neurons, -_OPT_BOUNDS["b1"]),
                         np.full(neurons, -_OPT_BOUNDS["w2"]),
                         [-_OPT_BOUNDS["b2"]]])
    hi = np.concatenate([np.full(nb, _OPT_BOUNDS["w1"]),
                         np.full(neurons, _OPT_BOUNDS["b1"]),
                         np.full(neurons, _OPT_BOUNDS["w2"]),
                         [_OPT_BOUNDS["b2"]]])

    def residual(p: np.ndarray) -> np.ndarray:
        w1 = p[:nb].reshape(neurons, 2)
        b1 = p[nb:nb + neurons]
        w2 = p[nb + neurons:nb + 2 * neurons]
        y = Jet2(float(p[-1]))
        xn = (seed_input(x, 0) - cx) / sx
        for i in range(neurons):
            pre = float(w1[i, 0]) * xn + float(b1[i])
            y = y + float(w2[i]) * pre.act(activation)
        qjet = y.exp() - q0
        parts = [weights["q"] * (np.asarray(y.v) - tgt) / sc["q"]]
        if weights["dq"]:
            parts.append(weights["dq"] * (np.asarray(qjet.g[0]) - dq_t)
                         / sc["dq"])
        if weights["d2q"]:
            parts.append(weights["d2q"] * (np.asarray(qjet.h[0]) - d2q_t)
                         / sc["d2q"])
        return np.concatenate(parts)

    sol = least_squares(residual, p0, bounds=(lo, hi),
                        max_nfev=_OPT_MAX_NFEV, xtol=1e-10, ftol=1e-10,
                        gtol=1e-10)
    if not np.all(np.isfinite(sol.x)):
        raise RuntimeError("ANN charge fit produced non-finite coefficients")
    return {
        "w1": sol.x[:nb].reshape(neurons, 2).tolist(),
        "b1": sol.x[nb:nb + neurons].tolist(),
        "w2": sol.x[nb + neurons:nb + 2 * neurons].tolist(),
        "b2": float(sol.x[-1]),
        "x_center": cx, "x_scale": sx,
        "cost": float(sol.cost), "success": bool(sol.success),
    }


def _sample_table(x: np.ndarray, q: np.ndarray) -> Dict[str, list]:
    """Sample a monotone, zero-anchored Q(V) table for the emitter chain."""
    xmax = float(np.max(x))
    if xmax <= 0.0:
        raise ValueError("charge voltage grid must extend above zero")
    n = int(min(32, max(8, x.size)))
    xs = np.linspace(0.0, xmax, n)
    qs = np.interp(xs, np.concatenate(([0.0], x)),
                   np.concatenate(([0.0], q)))
    qs = np.maximum.accumulate(np.maximum(qs - qs[0], 0.0))
    qs[0] = 0.0
    return {"v": [float(v) for v in xs], "q": [float(v) for v in qs]}


def fit_charge_qv(curve4, plateau, *, q0: Optional[float] = None,
                  loss_weights: Optional[dict] = None) -> AnnChargeFit:
    """Fit log-scale, Q0-offset charge nets and return monotone Q(V) tables.

    ``curve4`` carries capacitance samples (y_kind='capacitance', default) or
    an already-integrated Q(V) curve (y_kind='charge'); cap equivalents and
    the reference voltage travel in ``curve4.metadata``.  ``plateau`` is the
    Curve-5 gate-charge curve (Vgs, Qg in coulombs).  Returns an
    :class:`AnnChargeFit` whose :meth:`as_spec_tables` maps directly onto the
    declared-but-previously-empty ``q_iss_*``/``q_rss_*``/``q_oss_*`` fields.
    """
    data, sources, meta = _prepare_data(curve4, plateau)
    w = default_charge_loss_weights()
    if loss_weights:
        unknown = set(loss_weights) - set(w)
        if unknown:
            raise ValueError(f"unknown charge loss weights {sorted(unknown)}")
        w.update({k: float(v) for k, v in loss_weights.items()})
    if any(v < 0 for v in w.values()):
        raise ValueError("charge loss weights must be non-negative")

    activation = str(curve4.metadata.get("activation", "isru"))
    if activation not in _ACTIVATIONS:
        raise ValueError(f"activation must be one of {_ACTIVATIONS}")
    neurons = int(curve4.metadata.get("neurons", 6))
    if not 1 <= neurons <= 10:
        raise ValueError("charge neurons must be in [1, 10]")

    tables: Dict[str, Dict[str, list]] = {}
    coeffs: Dict[str, Any] = {}
    eps_by: Dict[str, float] = {}
    q0_used: Dict[str, float] = {}
    diag: Dict[str, Any] = {}
    for name in _TABLES:
        if name not in data:
            continue
        x, q = data[name]
        q = np.maximum.accumulate(np.maximum(np.asarray(q, float), 0.0))
        pos = np.diff(q)
        pos = pos[pos > 0.0]
        if q0 is None:
            q0_i = float(np.median(pos)) if pos.size else float(np.max(q))
        else:
            q0_i = float(q0)
        if not np.isfinite(q0_i) or q0_i <= 0.0:
            raise ValueError(f"{name}: charge offset q0 must be finite > 0")
        cx, sx = (float(np.min(x)), float(np.max(x) - np.min(x)))
        if sx <= 0.0:
            raise ValueError(f"{name}: degenerate charge voltage grid")
        final = _fit_one(x, q, q0_i, neurons, activation, w, cx, sx)
        coeffs[name] = final
        # Re-evaluate the analytic net on the data grid (for diagnostics),
        # then sample a monotone table for emission.
        tables[name] = _sample_table(x, q)
        eps_by[name] = eps_from_grid(tables[name]["v"])
        q0_used[name] = q0_i
        diag[name] = {"source": sources[name], "q_max_c": float(np.max(q)),
                      "q0_c": q0_i, "eps_v": eps_by[name],
                      "optimizer_cost": final["cost"],
                      "optimizer_success": final["success"]}

    fit = AnnChargeFit(tables, (float(q0) if q0 is not None else None),
                       activation, neurons, dict(w), coeffs, diag, {})
    fit.provenance = {
        "curve4": curve4.name,
        "plateau": plateau.name,
        "cap_equivalents": meta,
        "table_sources": sources,
        "q0_rule": "caller-supplied, else median positive charge increment",
        "q0_used": q0_used,
        "eps_rule": "median_positive_grid_spacing_cap_half_min_segment "
                    "(approximation.smoother.eps_from_grid)",
        "log_transform": "log(Q + q0); Q = exp(y) - q0",
        "derivative_terms": [k for k in ("dq", "d2q") if w[k] > 0.0],
        "q0_offset": "Tung SISPAD 2024 Eq.5 charge-shift fix "
                     "(CONDITIONAL until URL pinned)",
        "external_source_status": "conditional_until_url_and_section_pinned",
        "limitation": "Curve-4 YAML has cap equivalents at one Vds only; "
                      "where no C(V) sweep is supplied the Q table is the "
                      "source-derived linear Q(Cref)=Cref*Vref anchor, not "
                      "a learned 2-D Q(Vgs,Vds) surface.",
    }
    return fit
