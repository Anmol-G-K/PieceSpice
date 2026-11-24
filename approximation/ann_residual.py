"""Signed C0+ANN residual channel fit (Phase VI hybrid basis).

The closed Tier-2 channel is never replaced.  This module learns only the
offline residual ``I_res = I_golden - I_C0`` as

    I_res(Vgs,Vds) = Vds * (b2 + sum_i w2_i*phi(w1_i0*Vgs
                                                + w1_i1*Vds + b1_i)) * cutoff

with ``cutoff = 0.5*(1 + tanh(alpha*(Vgs - Vth)))``.  The explicit ``Vds``
factor pins I_res(Vgs,0) = 0; the tanh cutoff suppresses the correction below
threshold; no exponential or positive-only transform is used, so the residual
can be signed.  First and second input derivatives are analytic (used for the
solver-facing diagnostics); the bounded optimiser uses the analytic parameter
Jacobian.  Nothing here runs a simulator or writes a repository artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
from scipy.optimize import least_squares

from approximation.ann_channel import (_curve_samples, _grid_derivative_targets,
                                       _resolve_holdout_mask,
                                       _temperature_input)

__all__ = ["AnnResidualFit", "fit_channel_residual", "residual_with_derivatives"]

# Project-decision optimiser settings (NOT emitter physical constants); all
# recorded in AnnResidualFit.provenance so the fit can be audited offline.
_OPT_BOUNDS = {"w1": 10.0, "b1": 10.0, "w2": 2.0, "b2": 40.0}
_OPT_MAX_NFEV = 400
_OPT_STARTS = 2
_OPT_TOL = 1e-10
_CUTOFF_ALPHA = 10.0          # source-derived initial value (Yang TPEL 2022)
_MAX_NEURONS = 10             # E3 Fig. 12 overfit guard (ANN study R7)
_ACTIVATIONS = ("isru", "tanh", "sigmoid")


def _activation(u: np.ndarray, name: str, order: int = 0) -> np.ndarray:
    """phi and its first/second derivative for the supported activations."""
    if name == "isru":
        q = np.sqrt(1.0 + u * u)
        if order == 0:
            return u / q
        if order == 1:
            return q ** -3
        return -3.0 * u * q ** -5
    if name == "tanh":
        t = np.tanh(u)
        if order == 0:
            return t
        d = 1.0 - t * t
        return d if order == 1 else -2.0 * t * d
    if name == "sigmoid":
        # Stable both tails: exp(+u) would overflow for large positive u.
        s = np.where(u >= 0.0, 1.0 / (1.0 + np.exp(-np.abs(u))),
                     np.exp(-np.abs(u)) / (1.0 + np.exp(-np.abs(u))))
        d = s * (1.0 - s)
        if order == 0:
            return s
        return d if order == 1 else d * (1.0 - 2.0 * s)
    raise ValueError(f"unsupported activation {name!r}; expected {_ACTIVATIONS}")


def _cutoff(vgs: np.ndarray, v_th: float, alpha: float, order: int = 0
            ) -> np.ndarray:
    """Tanh Vth gate and its first/second Vgs derivative."""
    t = np.tanh(alpha * (vgs - v_th))
    if order == 0:
        return 0.5 * (1.0 + t)
    c1 = 0.5 * alpha * (1.0 - t * t)
    return c1 if order == 1 else -2.0 * alpha * t * c1


@dataclass
class AnnResidualFit:
    """Analytic signed-residual coefficients plus audit diagnostics."""

    coefficients: Dict[str, Any]
    v_th: float
    activation: str
    neurons: int
    alpha: float = _CUTOFF_ALPHA
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, vgs: Any, vds: Any,
                 temperature_c: Any = None) -> np.ndarray:
        """Return I_res on broadcast (vgs, vds) grids."""
        return residual_with_derivatives(self, vgs, vds, temperature_c)[0]

    def evaluate_with_derivatives(self, vgs: Any, vds: Any,
                                  temperature_c: Any = None
                                  ) -> Tuple[np.ndarray, ...]:
        """Return (I, dI/dVgs, dI/dVds, d2I/dVgs2, d2I/dVds2, d2I/dVgsdVds)."""
        return residual_with_derivatives(self, vgs, vds, temperature_c)


def residual_with_derivatives(fit: AnnResidualFit, vgs: Any, vds: Any,
                              temperature_c: Any = None
                              ) -> Tuple[np.ndarray, ...]:
    """Analytic value and first/second input derivatives of the residual."""
    c = fit.coefficients
    w1 = np.asarray(c["w1"], float)
    b1 = np.asarray(c["b1"], float)
    w2 = np.asarray(c["w2"], float)
    g = np.asarray(vgs, float)
    d = np.asarray(vds, float)
    g, d = np.broadcast_arrays(g, d)
    nin = w1.shape[1]
    if w1.shape != (fit.neurons, nin) or nin not in (2, 3) \
            or b1.shape != (fit.neurons,) or w2.shape != (fit.neurons,):
        raise ValueError("residual coefficients do not match neurons")
    act = fit.activation
    if act not in _ACTIVATIONS:
        raise ValueError(f"unsupported activation {act!r}")

    u = w1[:, 0:1] * g + w1[:, 1:2] * d + b1[:, None]
    if nin == 3:
        tval = (c.get("temperature_value", c["temperature_center"])
                if temperature_c is None else temperature_c)
        tn = ((np.asarray(tval, float)
               - float(c["temperature_center"]))
              / float(c["temperature_scale"]))
        u = u + w1[:, 2:3] * tn
    phi = _activation(u, act, 0)
    phi1 = _activation(u, act, 1)
    phi2 = _activation(u, act, 2)
    y = float(c["b2"]) + np.einsum("i,i...->...", w2, phi)
    y_g = np.einsum("i,i,i...->...", w2, w1[:, 0], phi1)
    y_d = np.einsum("i,i,i...->...", w2, w1[:, 1], phi1)
    y_gg = np.einsum("i,i,i...->...", w2, w1[:, 0] ** 2, phi2)
    y_dd = np.einsum("i,i,i...->...", w2, w1[:, 1] ** 2, phi2)
    y_gd = np.einsum("i,i,i,i...->...", w2, w1[:, 0], w1[:, 1], phi2)

    cut = _cutoff(g, float(fit.v_th), float(fit.alpha), 0)
    cut1 = _cutoff(g, float(fit.v_th), float(fit.alpha), 1)
    cut2 = _cutoff(g, float(fit.v_th), float(fit.alpha), 2)

    value = d * y * cut
    dg = d * (y_g * cut + y * cut1)
    dd = y * cut + d * y_d * cut
    dgg = d * (y_gg * cut + 2.0 * y_g * cut1 + y * cut2)
    ddd = 2.0 * y_d * cut + d * y_dd * cut
    dgd = y_g * cut + y * cut1 + d * (y_gd * cut + y_d * cut1)
    return (value, dg, dd, dgg, ddd, dgd)


def _normalised_model(p: np.ndarray, n: int, gn: np.ndarray, dn: np.ndarray,
                      vds: np.ndarray, cut: np.ndarray, activation: str,
                      *, with_jac: bool = False,
                      tn: np.ndarray | None = None):
    """Residual value (and analytic d/dp) on the centred/scaled grid."""
    nin = 2 if tn is None else 3
    w1 = p[:nin * n].reshape(n, nin)
    b1 = p[nin * n:nin * n + n]
    w2 = p[nin * n + n:nin * n + 2 * n]
    b2 = float(p[nin * n + 2 * n])
    u = w1[:, 0:1] * gn + w1[:, 1:2] * dn + b1[:, None]
    if nin == 3:
        u = u + w1[:, 2:3] * tn
    phi = _activation(u, activation, 0)
    phi1 = _activation(u, activation, 1)
    drive = vds * cut
    value = drive * (b2 + np.einsum("i,i...->...", w2, phi))
    if not with_jac:
        return value, None
    cols = []
    for i in range(n):
        base = drive * w2[i] * phi1[i]
        cols += [np.broadcast_to(base * gn, value.shape),
                 np.broadcast_to(base * dn, value.shape)]
        if nin == 3:
            cols.append(np.broadcast_to(base * tn, value.shape))
    for i in range(n):
        cols.append(np.broadcast_to(drive * w2[i] * phi1[i], value.shape))
    for i in range(n):
        cols.append(np.broadcast_to(drive * phi[i], value.shape))
    cols.append(np.broadcast_to(drive, value.shape))
    jac = np.stack([np.asarray(col).ravel() for col in cols], axis=1)
    return value, jac


def _fd_derivative_check(fit: AnnResidualFit, g: np.ndarray, d: np.ndarray
                         ) -> Dict[str, Any]:
    """Compare analytic input derivatives to central differences (offline)."""
    idx = np.linspace(0, g.size - 1, min(8, g.size)).astype(int)
    gs, ds = g[idx], d[idx]
    h_g = 1e-4 * max(float(np.ptp(g)), 1.0)
    h_d = 1e-4 * max(float(np.ptp(d)), 1.0)
    analytic = residual_with_derivatives(fit, gs, ds)
    gp = residual_with_derivatives(fit, gs + h_g, ds)
    gm = residual_with_derivatives(fit, gs - h_g, ds)
    dp = residual_with_derivatives(fit, gs, ds + h_d)
    dm = residual_with_derivatives(fit, gs, ds - h_d)
    fd_g = (gp[0] - gm[0]) / (2.0 * h_g)
    fd_d = (dp[0] - dm[0]) / (2.0 * h_d)
    fd_gg = (gp[0] - 2.0 * analytic[0] + gm[0]) / (h_g * h_g)
    fd_dd = (dp[0] - 2.0 * analytic[0] + dm[0]) / (h_d * h_d)
    fd_gd = (residual_with_derivatives(fit, gs + h_g, ds + h_d)[0]
             - residual_with_derivatives(fit, gs + h_g, ds - h_d)[0]
             - residual_with_derivatives(fit, gs - h_g, ds + h_d)[0]
             + residual_with_derivatives(fit, gs - h_g, ds - h_d)[0]) \
        / (4.0 * h_g * h_d)
    errs = [float(np.max(np.abs(a - b)))
            for a, b in zip(analytic[1:], (fd_g, fd_d, fd_gg, fd_dd, fd_gd))]
    return {
        "probe_points": int(gs.size), "step_vgs_v": h_g, "step_vds_v": h_d,
        "max_abs_diff": dict(zip(("dg", "dd", "dgg", "ddd", "dgd"), errs)),
    }


def fit_channel_residual(curves: Any, spec: Any, *, v_th: float,
                         activation: str = "isru", neurons: int = 6,
                         guard_width: Optional[float] = None,
                         guard_width_provenance: Optional[Dict[str, Any]] = None,
                         c0_current: Any = None,
                         c0_eps: float = 0.0,
                         spec_source: Optional[str] = None,
                         holdout_fraction: float = 0.0,
                         holdout_mask: Optional[Any] = None,
                         temperature_c: Optional[Any] = None) -> AnnResidualFit:
    """Fit ``I_golden - I_C0`` on Curve-1 families plus optional DPT points.

    ``curves`` accepts the containers understood by
    :func:`approximation.ann_channel._curve_samples` (CurveData family list
    and/or :class:`DptPoints`).  ``spec`` is the PWLModelSpec consumed by the
    closed Tier-2 evaluator; a ``c0_current(vgs, vds)`` callable may be passed
    instead for synthetic tests.  ``guard_width`` must be the existing static
    Curve-1 derivation (never one inferred from dense DPT samples).  No file
    is written and no simulator is started.

    ``holdout_fraction``/``holdout_mask`` reserve whole Vgs families (mask
    wins when supplied) and fit on the rest; absent hold-out metrics are
    ``not_measured``.  ``temperature_c`` adds a third normalised input column
    and its centre/scale to the coefficients; absent, the fit stays in the
    existing two-input form.
    """
    if activation not in _ACTIVATIONS:
        raise ValueError(f"activation must be one of {_ACTIVATIONS}")
    neurons = int(neurons)
    if not 1 <= neurons <= _MAX_NEURONS:
        raise ValueError(f"neurons must be in [1, {_MAX_NEURONS}]")
    v_th = float(v_th)
    if not np.isfinite(v_th):
        raise ValueError("v_th must be finite")
    if guard_width is not None:
        guard_width = float(guard_width)
        if not np.isfinite(guard_width) or guard_width <= 0.0:
            raise ValueError("guard_width must be finite and > 0")

    vgs_raw, vds_raw, i_raw, sample_prov, grid_raw = _curve_samples(curves)
    good = (np.isfinite(vgs_raw) & np.isfinite(vds_raw) & np.isfinite(i_raw)
            & (vds_raw > 0.0))
    raw_n = vgs_raw.size
    vgs_all, vds_all, golden = vgs_raw[good], vds_raw[good], i_raw[good]
    grid_all = np.asarray(grid_raw, dtype=bool)[good]
    if holdout_mask is not None and np.asarray(holdout_mask).size == raw_n:
        holdout_mask = np.asarray(holdout_mask, dtype=bool)[good]
    if temperature_c is not None:
        t_all_in = np.asarray(temperature_c, dtype=float)
        if t_all_in.ndim == 0:
            pass
        elif t_all_in.size == raw_n:
            temperature_c = t_all_in[good]
        elif t_all_in.size != vgs_all.size:
            raise ValueError("temperature_c array must match the sample count")
    n_in = 3 if temperature_c is not None else 2

    if c0_current is None:
        if spec is None:
            raise ValueError("spec is required when c0_current is not supplied")
        from models.netlist_builder import evaluate_tier2_channel
        c0 = lambda g, d: evaluate_tier2_channel(spec, g, d, eps=c0_eps)
    else:
        c0 = c0_current
    c0_i = np.asarray(c0(vgs_all, vds_all), dtype=float)
    if c0_i.shape != golden.shape or not np.all(np.isfinite(c0_i)):
        raise ValueError("C0 evaluator must return finite currents per sample")
    target_all = golden - c0_i

    holdout, holdout_prov = _resolve_holdout_mask(
        vgs_all, grid_all, holdout_fraction, holdout_mask)
    train = ~holdout
    if int(np.count_nonzero(train)) < 2 * neurons + 4:
        raise ValueError("too few residual samples for the requested net")
    vgs, vds = vgs_all[train], vds_all[train]
    target = target_all[train]
    grid_mask = grid_all[train]
    if n_in == 3:
        t_norm_all, t_center, t_scale, t_value = _temperature_input(
            temperature_c, vgs_all.size)
        t_raw_all = t_norm_all * t_scale + t_center
        t_norm = t_norm_all[train]
        t_raw = t_raw_all[train]
    else:
        t_norm_all = t_raw_all = None
        t_norm = t_raw = None
        t_center = t_scale = t_value = None

    target_rms = float(np.sqrt(np.mean(target * target)))
    if not np.isfinite(target_rms) or target_rms <= 0.0:
        raise ValueError("residual target is zero or non-finite; nothing to fit")

    cg, sg = float(np.min(vgs)), float(np.ptp(vgs))
    cd, sd = float(np.min(vds)), float(np.ptp(vds))
    if sg <= 0.0 or sd <= 0.0:
        raise ValueError("degenerate residual training range")
    gn = (vgs - cg) / sg
    dn = (vds - cd) / sd
    cut = _cutoff(vgs, v_th, _CUTOFF_ALPHA, 0)
    y_target = target / target_rms
    lo = np.concatenate([np.full(n_in * neurons, -_OPT_BOUNDS["w1"]),
                         np.full(neurons, -_OPT_BOUNDS["b1"]),
                         np.full(neurons, -_OPT_BOUNDS["w2"]),
                         [-_OPT_BOUNDS["b2"]]])
    hi = -lo
    b2_0 = float(np.clip(np.median(target / np.maximum(vds, 1e-12)),
                         -_OPT_BOUNDS["b2"], _OPT_BOUNDS["b2"]))
    seeds = []
    for s in range(_OPT_STARTS):
        rng = np.random.default_rng(s)
        w1 = rng.normal(0.0, 0.5 if s == 0 else 1.0, n_in * neurons)
        b1 = rng.normal(0.0, 0.5, neurons)
        w2 = rng.normal(0.0, 0.05 if s == 0 else 0.15, neurons)
        seeds.append(np.clip(np.concatenate([w1, b1, w2, [b2_0]]), lo, hi))

    def residual_vec(p: np.ndarray) -> np.ndarray:
        val, _ = _normalised_model(p, neurons, gn, dn, vds, cut,
                                   activation, tn=t_norm)
        return (val / target_rms - y_target).ravel()

    def jacobian(p: np.ndarray) -> np.ndarray:
        _, jac = _normalised_model(p, neurons, gn, dn, vds, cut,
                                   activation, with_jac=True, tn=t_norm)
        if jac is None:
            raise RuntimeError("analytic residual Jacobian was not produced")
        return jac / target_rms

    history = []
    solutions = []
    for s, p0 in enumerate(seeds):
        sol = least_squares(residual_vec, p0, jac=cast(Any, jacobian),
                            bounds=(lo, hi), max_nfev=_OPT_MAX_NFEV,
                            xtol=_OPT_TOL, ftol=_OPT_TOL, gtol=_OPT_TOL)
        solutions.append(sol)
        history.append({"start": s, "cost": float(sol.cost),
                        "nfev": int(sol.nfev), "njev": int(sol.njev),
                        "status": int(sol.status), "reason": sol.message})
    best = min(solutions, key=lambda so: so.cost)
    if not np.all(np.isfinite(best.x)):
        raise RuntimeError("ANN residual fit produced non-finite coefficients")

    w1 = best.x[:n_in * neurons].reshape(neurons, n_in)
    b1 = best.x[n_in * neurons:n_in * neurons + neurons]
    w2 = best.x[n_in * neurons + neurons:n_in * neurons + 2 * neurons]
    # Absorb the centring/scaling into raw (Vgs, Vds, T) coefficients so the
    # emitted expression is literally phi(w1_i0*VGS + w1_i1*VDS + b1_i).
    raw_w1 = np.column_stack((w1[:, 0] / sg, w1[:, 1] / sd))
    raw_b1 = b1 - w1[:, 0] * (cg / sg) - w1[:, 1] * (cd / sd)
    if n_in == 3:
        raw_w1 = np.column_stack((raw_w1, w1[:, 2] / t_scale))
        raw_b1 = raw_b1 - w1[:, 2] * (t_center / t_scale)
    coef = {
        "w1": raw_w1.tolist(), "b1": raw_b1.tolist(),
        "w2": w2.tolist(), "b2": float(best.x[n_in * neurons + 2 * neurons]),
        "vgs_center": cg, "vgs_scale": sg,
        "vds_center": cd, "vds_scale": sd,
        "alpha": _CUTOFF_ALPHA,
    }
    if n_in == 3:
        coef["temperature_center"] = float(t_center)
        coef["temperature_scale"] = float(t_scale)
        coef["temperature_value"] = float(t_value)
    if guard_width is not None:
        coef["guard_width"] = guard_width
    fit = AnnResidualFit(coef, v_th, activation, neurons, _CUTOFF_ALPHA)

    model = np.asarray(fit.evaluate(vgs, vds, t_raw), dtype=float)
    if not np.all(np.isfinite(model)):
        raise RuntimeError("ANN residual evaluation produced non-finite values")
    deriv = _fd_derivative_check(fit, vgs, vds)
    zero_vds_max = float(np.max(np.abs(fit.evaluate(np.linspace(cg, cg + sg, 11),
                                                     np.zeros(11)))))
    below = float(np.max(np.abs(fit.evaluate(
        np.full(9, v_th - 0.5), np.linspace(0.1, cd + sd, 9)))))
    above = float(np.max(np.abs(fit.evaluate(
        np.full(9, v_th + 0.5), np.linspace(0.1, cd + sd, 9)))))

    holdout_fields: Dict[str, Any] = {
        "holdout_id_rms_rel": "not_measured",
        "holdout_gm_rms_rel": "not_measured",
        "holdout_gds_rms_rel": "not_measured",
        "holdout_n": "not_measured",
    }
    if holdout_prov is not None:
        hgm_t, hgds_t, _, _ = _grid_derivative_targets(
            vgs_all, vds_all, target_all, grid_all)
        hvgs, hvds = vgs_all[holdout], vds_all[holdout]
        ht = t_raw_all[holdout] if n_in == 3 else None
        hm = np.asarray(fit.evaluate(hvgs, hvds, ht), dtype=float)
        hder = fit.evaluate_with_derivatives(hvgs, hvds, ht)
        hgm_fit, hgds_fit = hder[1], hder[2]

        def _rel(pred: np.ndarray, tgt: np.ndarray) -> Any:
            r = float(np.sqrt(np.mean(tgt * tgt)))
            if not np.isfinite(r) or r <= 1e-30:
                return "not_measured"
            return float(np.sqrt(np.mean((pred - tgt) ** 2)) / r)

        holdout_fields = {
            "holdout_id_rms_rel": _rel(hm, target_all[holdout]),
            "holdout_gm_rms_rel": _rel(hgm_fit, hgm_t[holdout]),
            "holdout_gds_rms_rel": _rel(hgds_fit, hgds_t[holdout]),
            "holdout_n": int(np.count_nonzero(holdout)),
        }

    fit.diagnostics = {
        "train_samples": int(vgs.size),
        "curve1_samples": int(np.count_nonzero(grid_mask)),
        "dpt_samples": int(np.count_nonzero(~grid_mask)),
        "target_rms_a": target_rms,
        "residual_rms_a": float(np.sqrt(np.mean((model - target) ** 2))),
        "residual_rel_rmse": float(np.sqrt(np.mean((model - target) ** 2))
                                   / target_rms),
        "negative_target_samples": int(np.count_nonzero(target < 0.0)),
        "positive_target_samples": int(np.count_nonzero(target > 0.0)),
        "zero_vds_max_abs_a": zero_vds_max,
        "below_vth_max_abs_a": below,
        "above_vth_max_abs_a": above,
        "cutoff_attenuation": (below / above if above > 0.0 else None),
        "analytic_vs_finite_difference": deriv,
        "optimizer_cost": float(best.cost),
        "optimizer_starts_run": len(history),
        "optimizer_nfev": int(sum(h["nfev"] for h in history)),
        "optimizer_njev": int(sum(h["njev"] for h in history)),
        "optimizer_success": bool(best.status > 0),
        "optimizer_termination_reason": best.message,
        "derivative_terms": ["dI_res/dVgs", "dI_res/dVds",
                             "d2I_res/dVgs2", "d2I_res/dVds2",
                             "d2I_res/dVgsdVds"],
        "holdout_selection": holdout_prov,
        "temperature_center": (float(t_center) if n_in == 3 else None),
        "temperature_scale": (float(t_scale) if n_in == 3 else None),
        **holdout_fields,
    }
    fit.provenance = {
        "target": "I_golden - I_C0; C0 = closed Tier-2 G_CH evaluated by "
                  "models.netlist_builder.evaluate_tier2_channel "
                  f"(eps={c0_eps:g}); no simulated currents are fabricated",
        "c0_spec": spec_source or (getattr(spec, "device_name", None)
                                   if spec is not None else "callable"),
        "data_sources": sample_prov,
        "n_samples": int(vgs.size),
        "n_curve1_samples": int(np.count_nonzero(grid_mask)),
        "n_dpt_samples": int(np.count_nonzero(~grid_mask)),
        "activation": activation,
        "neurons": neurons,
        "v_th": "curve7_threshold_voltage:vth_v (caller-supplied scalar)",
        "cutoff_alpha": f"{_CUTOFF_ALPHA:g} source-derived initial value "
                        "(Yang TPEL 2022, CONDITIONAL until URL pinned)",
        "guard_width": dict(guard_width_provenance or {
            "value": guard_width, "source": "caller-supplied static grid",
            "dpt_points_excluded": bool(np.any(~grid_mask))}),
        "normalisation": "inputs centred/scaled for the fit, then absorbed "
                         "into raw Vgs/Vds coefficients for emission",
        "optimiser": "scipy.optimize.least_squares, bounded, deterministic "
                     "seeded starts, analytic parameter Jacobian",
        "optimizer_bounds": dict(_OPT_BOUNDS),
        "optimizer_max_nfev": _OPT_MAX_NFEV,
        "optimizer_starts_configured": _OPT_STARTS,
        "optimizer_tolerances": {"xtol": _OPT_TOL, "ftol": _OPT_TOL,
                                 "gtol": _OPT_TOL},
        "optimizer_history": history,
        "holdout": holdout_prov,
        "holdout_rule": "whole CurveData Vgs families reserved; fit on the "
                        "remaining samples; no hold-out => not_measured",
        "temperature": (None if n_in == 2 else {
            "center_c": float(t_center), "scale_c": float(t_scale),
            "value_c": float(t_value), "n_inputs": 3,
            "column": "normalised (T - center)/scale",
            "emission_point_c": float(t_value),
            "emitter_temperature_port": False,
        }),
        "analytic_derivatives": "closed forms of phi', phi'' and tanh-cutoff "
                                "derivatives; central-difference audit in "
                                "diagnostics.analytic_vs_finite_difference",
        "limitation": "offline static fit only; no fidelity/speed gate runs "
                      "in this module",
        "external_source_status": "conditional_until_url_and_section_pinned",
    }
    return fit
