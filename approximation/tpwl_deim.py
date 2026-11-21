"""Tier-3 TPWL terminal model with POD-DEIM anchor selection (v2).

Formulation (validated on all four configured vendors):

* The device is described at its terminals as
      i_d = i_ch(vgs, vds) + d/dt Qd(vgs, vds)
      i_g =          d/dt Qg(vgs, vds)
  with i_ch, Qd, Qg piecewise-affine over s linearization cells in the
  scaled operating plane, blended with the Rewienski-White weights.
* Cell anchors are seeded by DEIM (Chaturantabut & Sorensen, Algorithm 1)
  applied to the POD basis of the operating-trajectory snapshots, then
  completed by farthest-point covering.
* The paper's weight rule is used verbatim: with d_i the distance to
  anchor i and m = min d_i, w_i = (exp(d_i)/m)**(-25), normalised —
  i.e. softmax(-25 d), evaluated with a min-shift for numerical safety.

Why the earlier internal-node ODE fit was abandoned (measured evidence,
2026-11-19): trajectory-local least-squares Jacobians are rank-deficient
(the switching trajectory is a 1-D curve; design condition numbers
1e9-1e20), fitted eigenvalues 1e8-1e12 are sampling artifacts, and any
stabilising clip destroys the fit (one-step error 1e4-1e5). The terminal
charge formulation carries the dynamics through dQ/dt integrated by the
SPICE solver itself, which needs no fitted ODE at all.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter

from approximation.charge_surface import ChargeSurface
from simulation.testbenches.dpt_trajectory import Trajectory

# Paper-printed constant (Rewienski & White 2003, section 4.1, step 3)
WEIGHT_EXPONENT = 25.0


# ---------------------------------------------------------------------------
# DEIM — Algorithm 1 verbatim
# ---------------------------------------------------------------------------

def deim_indices(basis: np.ndarray) -> np.ndarray:
    """DEIM point selection (Sorensen & Embree, arXiv:1407.5516, Alg. 1)."""
    m, k = basis.shape
    if k < 1 or k > m:
        raise ValueError(f"DEIM needs 1 <= k <= m; got {k}, {m}")
    p = [int(np.argmax(np.abs(basis[:, 0])))]
    for j in range(1, k):
        v = basis[:, j]
        c = np.linalg.solve(basis[p, :j], v[p])
        r = v - basis[:, :j] @ c
        p.append(int(np.argmax(np.abs(r))))
    if len(set(p)) != len(p):
        raise RuntimeError("DEIM produced duplicate indices")
    return np.asarray(p, dtype=int)


# ---------------------------------------------------------------------------
# Rewienski-White weights
# ---------------------------------------------------------------------------

def paper_weights(anchors: np.ndarray, ws: np.ndarray) -> np.ndarray:
    """Normalised weights (exp(d_i)/m)**(-25) with min-shift stability.

    anchors: (s, d); ws: (d, K).  Returns (s, K).
    """
    d = np.linalg.norm(anchors.T[:, :, None] - ws[:, None, :], axis=0)
    logits = -WEIGHT_EXPONENT * d
    logits -= logits.max(axis=0, keepdims=True)
    w = np.exp(logits)
    return w / w.sum(axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# Model container
# ---------------------------------------------------------------------------

@dataclass
class TPWLTerminalModel:
    """Piecewise-affine terminal model of one vendor device."""
    vendor: str
    subcircuit: str
    pin_roles: list[str]
    anchors: np.ndarray        # (s, 2) scaled operating points
    v_ref: np.ndarray          # (2,) raw operating-point reference
    v_span: np.ndarray         # (2,) raw operating-point spans
    channel: np.ndarray        # (s, 3) [a, g, h]      A, A per scaled V
    charge_d: np.ndarray       # (s, 2) [c, d]        C per scaled V
    charge_g: np.ndarray       # (s, 3) [q0, q1, q2]  C, C per scaled V
    edge_window: int           # samples; two configured gate edges
    qoss_v: np.ndarray | None = None   # AC-derived Qoss table (V)
    qoss_q: np.ndarray | None = None   # AC-derived Qoss table (C)
    coss_v: np.ndarray | None = None   # raw AC Coss(Vds) curve (V)
    coss_c: np.ndarray | None = None   # raw AC Coss(Vds) curve (F)
    v_block: float | None = None       # extracted Vth (Curve 7)
    v_width: float | None = None       # transfer-sweep voltage resolution
    meta: dict = field(default_factory=dict)
    qg_v: np.ndarray | None = None     # DPT gate-charge table (V)
    qg_q: np.ndarray | None = None     # DPT gate-charge table (C)
    qoss_surface: ChargeSurface | None = None   # measured 2-D Qoss(Vds, Vgs)
    temperature_c: float | None = None          # surface slice temperature

    def scaled(self, u: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(u) - self.v_ref[:, None]) / \
            self.v_span[:, None]

    def predict(self, t: np.ndarray, u: np.ndarray,
                ) -> tuple[np.ndarray, np.ndarray]:
        """Terminal currents along a given (t, vgs, vds) trajectory."""
        ws = self.scaled(u)
        w = paper_weights(self.anchors, ws)
        dw = ws[:, None, :] - self.anchors.T[:, :, None]   # (2, s, K)
        id_ch = np.einsum("sk,sk->k", w, self.channel[:, 0][:, None]
                          + self.channel[:, 1][:, None] * dw[0]
                          + self.channel[:, 2][:, None] * dw[1])
        if self.v_block is not None and self.v_width is not None:
            id_ch = id_ch * np.clip(
                (u[0] - self.v_block) / self.v_width, 0.0, 1.0)
        qd = np.einsum("sk,sk->k", w, self.charge_d[:, 0][:, None] * dw[0]
                       + self.charge_d[:, 1][:, None] * dw[1])
        qg = np.einsum("sk,sk->k", w, self.charge_g[:, 0][:, None]
                       + self.charge_g[:, 1][:, None] * dw[0]
                       + self.charge_g[:, 2][:, None] * dw[1])
        id_cap = np.gradient(qd, t)
        if self.qoss_v is not None and self.qoss_q is not None:
            qoss_t = np.interp(u[1], self.qoss_v, self.qoss_q)
            id_cap = id_cap + np.gradient(qoss_t, t)
        return id_ch + id_cap, qg


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def _split_index(vgs: np.ndarray) -> int:
    high = float(np.max(vgs))
    lvl = 0.5 * high
    rising = np.flatnonzero((vgs[:-1] < lvl) & (vgs[1:] >= lvl)) + 1
    if len(rising) < 2:
        raise RuntimeError("fewer than two Vgs rising edges in trajectory")
    return int(rising[1])


def _operating_scaler(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference and peak-to-peak span of each operating coordinate."""
    ref = u.mean(axis=1)
    span = np.ptp(u, axis=1)
    span[span == 0] = 1.0
    return ref, span

def _farthest_completion(ws: np.ndarray, seeds: list[int],
                         n: int) -> list[int]:
    """Farthest-point covering seeded by the DEIM indices."""
    pts = list(seeds)
    dmin = np.full(ws.shape[1], np.inf)
    for i in pts:
        dmin = np.minimum(dmin, np.linalg.norm(ws - ws[:, i:i + 1], axis=0))
    while len(pts) < n:
        i = int(np.argmax(dmin))
        pts.append(i)
        dmin = np.minimum(dmin,
                          np.linalg.norm(ws - ws[:, i:i + 1], axis=0))
    return pts


def _anchors_for(ws: np.ndarray, s: int) -> list[int]:
    """DEIM-seeded anchor indices (POD basis over trajectory samples)."""
    vt = np.linalg.svd(ws, full_matrices=False)[2].T      # (K, d)
    k = min(s, vt.shape[1])
    seeds = [int(i) for i in deim_indices(vt[:, :k])]
    return _farthest_completion(ws, seeds, s)


def _fit_cells(ws, dws, id_, qg, anchors_idx, is_static, is_dpt):
    """Per-cell fit from two data sources.

    * channel affine: extracted Curve-1 static family + DPT samples;
    * drain-charge slopes and gate charge: DPT samples only, where the
      sweep rates needed to separate displacement current are observable.
    """
    P = ws[:, anchors_idx]                               # (2, s)
    cell = np.argmin(
        np.linalg.norm(ws[:, None, :] - P[:, :, None], axis=0), axis=0)
    n_cells = len(anchors_idx)
    channel = np.zeros((n_cells, 3))
    charge_d = np.zeros((n_cells, 2))
    charge_g = np.zeros((n_cells, 3))
    for p in range(n_cells):
        rows = np.flatnonzero(cell == p)
        static = rows[is_static[rows]]
        dpt = rows[is_dpt[rows]]
        if len(static) + len(dpt) < 5:
            dd = np.linalg.norm(ws - ws[:, anchors_idx[p]:anchors_idx[p] + 1],
                                axis=0)
            rows = np.argsort(dd)[:5]
            static = rows[is_static[rows]]
            dpt = rows[is_dpt[rows]]
        anchor = ws[:, anchors_idx[p]:anchors_idx[p] + 1]
        # Channel uses the extracted static family only: DPT samples at
        # Vgs ~ 0 carry the VM-02 mirror-network transient (100 A while
        # Vds sweeps) and corrupt the quasi-static channel (IRFS fit_id
        # exploded to 31.5 when they were included). The DPT residual is
        # handled entirely by the charge slopes and the Qoss base.
        ch_rows = static if len(static) >= 3 else rows
        dv = (ws[:, ch_rows] - anchor).T
        Fc = np.column_stack([np.ones(len(ch_rows)), dv[:, 0], dv[:, 1]])
        solc, *_ = np.linalg.lstsq(Fc, id_[ch_rows], rcond=None)
        channel[p] = solc
        if len(dpt) >= 3:
            dvd = (ws[:, dpt] - anchor).T
            resid = id_[dpt] - (solc[0] + solc[1] * dvd[:, 0]
                                + solc[2] * dvd[:, 1])
            Fs = np.column_stack([dws[0, dpt], dws[1, dpt]])
            sols, *_ = np.linalg.lstsq(Fs, resid, rcond=None)
            charge_d[p] = sols
            Fq = np.column_stack([np.ones(len(dpt)), dvd[:, 0], dvd[:, 1]])
            solq, *_ = np.linalg.lstsq(Fq, qg[dpt], rcond=None)
            charge_g[p] = solq
    return channel, charge_d, charge_g


def _rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    span = float(np.ptp(ref)) or 1.0
    return float(np.sqrt(np.mean((pred - ref) ** 2)) / span)


def _trajectory_series(traj: Trajectory, *, v_ref: np.ndarray,
                       v_span: np.ndarray,
                       qoss_v: np.ndarray | None,
                       qoss_q: np.ndarray | None) -> dict:
    """DPT samples of one trajectory in the shared operating coordinates.

    Extra training conditions reuse the primary trajectory's reference/span
    so every cell anchor lives on one operating plane.
    """
    split = _split_index(traj.input[0])
    t, u, y = traj.time, traj.input, traj.output
    ws = (u - v_ref[:, None]) / v_span[:, None]
    wtr = ws[:, :split]

    dt = float(np.median(np.diff(t)))
    window = max(3, int(2.0 * traj.edge_time / dt) + 1)
    window += (window + 1) % 2
    window = min(window, split - 1 + (split - 1) % 2)
    if window < 3 or window > split:
        raise RuntimeError(
            f"{traj.vendor}: trajectory has {split} pre-split samples; "
            f"cannot smooth with window {window}")
    wsm = savgol_filter(wtr, window, 2, axis=1)
    dws = np.gradient(wsm, t[:split], axis=1)
    qg = cumulative_trapezoid(y[1, :split], t[:split], initial=0.0)

    id_fit = y[0, :split]
    if qoss_v is not None and qoss_q is not None:
        qoss_t = np.interp(u[1, :split], qoss_v, qoss_q)
        id_fit = y[0, :split] - np.gradient(qoss_t, t[:split])
    return {"split": split, "t": t, "u": u, "y": y, "wtr": wtr,
            "dws": dws, "id_fit": id_fit, "qg": qg, "window": window}


def _gate_charge_table(t: np.ndarray, vgs: np.ndarray, ig: np.ndarray,
                       split: int) -> tuple[np.ndarray, np.ndarray]:
    """Monotone Qg(Vgs) from the DPT gate-current integral, anchored at 0.

    Only the first turn-on segment is used: it is the gate-charge sweep.  At
    a Miller plateau the gate voltage is flat while charge accumulates, so the
    maximum accumulated charge at a repeated Vgs is kept (the physical
    vertical charge step); the result is then projected non-decreasing, which
    is the conservation property the emitted ``C_QG`` requires.
    """
    q = cumulative_trapezoid(np.asarray(ig, dtype=float)[:split],
                             np.asarray(t, dtype=float)[:split], initial=0.0)
    v = np.asarray(vgs, dtype=float)[:split]
    hi = int(np.argmax(v))
    if hi < 2:
        raise RuntimeError("gate-charge table needs a turn-on segment")
    v = v[:hi + 1]
    q = q[:hi + 1]
    uniq, inv = np.unique(v, return_inverse=True)
    qv = np.full(uniq.size, -np.inf)
    np.maximum.at(qv, inv, q)
    if not np.all(np.isfinite(qv)):
        raise RuntimeError("gate-charge table has empty voltage bins")
    # Anchor Q(0 V) = 0 without inventing a sample: insert the zero bias on
    # the measured integral when the sweep starts above or below zero.
    if not np.any(uniq == 0.0):
        zero_q = float(np.interp(0.0, uniq, qv))
        pos = int(np.searchsorted(uniq, 0.0))
        uniq = np.insert(uniq, pos, 0.0)
        qv = np.insert(qv, pos, zero_q)
    zero_pos = int(np.flatnonzero(uniq == 0.0)[0])
    qv = qv - float(qv[zero_pos])
    qv = np.maximum.accumulate(qv)
    if uniq.size < 2:
        raise RuntimeError("gate-charge table needs >=2 voltage levels")
    return uniq, qv


def build_model(traj: Trajectory, coss_curve=None,
                v_block: float | None = None,
                v_width: float | None = None,
                s_grid: tuple[int, ...] = (8, 16, 24, 32),
                include_static: bool = True,
                selection_error_fn: Callable[[TPWLTerminalModel], float]
                | None = None,
                extra_trajectories: Sequence[Trajectory] = (),
                holdout_trajectory: bool = False) -> TPWLTerminalModel:
    """Fit channel on Curve-1 static family + DPT; select s by static error.

    DPT alone drives only two Vgs levels and gave 65-1e9 % error on the
    Curve-1 family; the static family is the project's headline metric.

    ``selection_error_fn`` replaces the function-fit static RMSE as the cell
    selection key with a caller-measured error (Tier-3 passes the *emitted*
    channel's static RMSE).  The record says which source was used.

    ``extra_trajectories`` are real additional DPT captures (e.g. other
    training conditions); their samples are concatenated with the primary
    trajectory while the static family stays a single shared data set.
    ``holdout_trajectory`` reserves the last extra trajectory for a blocked
    cross-validation diagnostic instead of fitting it.
    """
    primary_split = _split_index(traj.input[0])
    v_ref, v_span = _operating_scaler(traj.input[:, :primary_split])

    qoss_v = qoss_q = None
    if coss_curve is not None:
        from approximation.charge_model import integrate_charge
        cv, cc = np.asarray(coss_curve[0]), np.asarray(coss_curve[1])
        cm = integrate_charge(cv, cc)
        qoss_v, qoss_q = cm.v, cm.q
    coss_v = None if coss_curve is None else np.asarray(coss_curve[0])
    coss_c = None if coss_curve is None else np.asarray(coss_curve[1])

    extras = list(extra_trajectories)
    series = [_trajectory_series(traj, v_ref=v_ref, v_span=v_span,
                                 qoss_v=qoss_v, qoss_q=qoss_q)]
    for extra in extras:
        series.append(_trajectory_series(
            extra, v_ref=v_ref, v_span=v_span,
            qoss_v=qoss_v, qoss_q=qoss_q))

    holdout = None
    fit_series = series
    if holdout_trajectory and extras:
        holdout = series[-1]
        fit_series = series[:-1]

    sources = [f"{traj.vendor}:primary"]
    for i, extra in enumerate(extras):
        role = ("holdout" if holdout is not None and i == len(extras) - 1
                else "training")
        sources.append(f"{extra.vendor}:{role}:{i + 1}")

    wtr = np.hstack([s["wtr"] for s in fit_series])
    dws = np.hstack([s["dws"] for s in fit_series])
    id_fit = np.hstack([s["id_fit"] for s in fit_series])
    qg = np.hstack([s["qg"] for s in fit_series])

    # combine sources: static family once (channel coverage) + every DPT
    ws_all, dws_all, id_all, qg_all = wtr, dws, id_fit, qg
    is_static = np.zeros(wtr.shape[1], dtype=bool)
    is_dpt = np.ones(wtr.shape[1], dtype=bool)
    static_u = static_id = None
    if include_static:
        from approximation.static_curves import static_samples
        static_u, static_id, _ = static_samples(traj.vendor)
        ws_static = (static_u - v_ref[:, None]) / v_span[:, None]
        ws_all = np.hstack([wtr, ws_static])
        dws_all = np.hstack([dws, np.zeros_like(ws_static)])
        id_all = np.hstack([id_fit, static_id])
        qg_all = np.hstack([qg, np.zeros(ws_static.shape[1])])
        is_static = np.hstack([
            np.zeros(wtr.shape[1], dtype=bool),
            np.ones(ws_static.shape[1], dtype=bool)])
        is_dpt = ~is_static

    try:
        qg_v, qg_q = _gate_charge_table(
            series[0]["t"], series[0]["u"][0], series[0]["y"][1],
            series[0]["split"])
    except RuntimeError:
        # Optional for the closed 1-D emitter; a V2 gate-charge emission
        # fails closed in the emitter when the table is missing.
        qg_v = qg_q = None

    sv = np.linalg.svd(ws_all, compute_uv=False)
    energy = (sv ** 2 / np.sum(sv ** 2)).tolist()

    best = None
    best_key = float("inf")
    for s in s_grid:
        idx = _anchors_for(ws_all, s)
        channel, charge_d, charge_g = _fit_cells(
            ws_all, dws_all, id_all, qg_all, idx, is_static, is_dpt)
        cand = TPWLTerminalModel(
            vendor=traj.vendor, subcircuit=traj.subcircuit,
            pin_roles=traj.pin_roles, anchors=ws_all[:, idx].T,
            v_ref=v_ref, v_span=v_span, channel=channel,
            charge_d=charge_d, charge_g=charge_g,
            edge_window=series[0]["window"],
            qoss_v=qoss_v, qoss_q=qoss_q, coss_v=coss_v, coss_c=coss_c,
            v_block=v_block, v_width=v_width, qg_v=qg_v, qg_q=qg_q,
            meta={"pod_energy": energy, "cells": s,
                  "deim_seed_count": min(s, len(energy)),
                  "qoss_base": coss_curve is not None,
                  "static_samples": int(is_static.sum()),
                  "dpt_samples": int(is_dpt.sum()),
                  "n_trajectories": len(series),
                  "trajectory_sources": list(sources),
                  "qg_table": qg_v is not None})
        if static_u is not None:
            sp = cand.predict(np.linspace(0, 1, static_u.shape[1]),
                              static_u)[0]
            st_err = _rmse(sp, static_id)
            cand.meta["static_rmse_rel"] = st_err
        else:
            st_err = float("inf")
        if selection_error_fn is not None:
            sel_err = float(selection_error_fn(cand))
            cand.meta["emitted_static_rmse_rel"] = sel_err
            cand.meta["selection_error_source"] = "emitted_static_rmse_rel"
        else:
            sel_err = st_err
            cand.meta["selection_error_source"] = "static_rmse_rel"
        cand.meta["selection_error_rel"] = sel_err
        id_pred, qg_pred = cand.predict(traj.time[primary_split:],
                                        traj.input[:, primary_split:])
        cand.meta["heldout_id_rmse"] = _rmse(
            id_pred, traj.output[0, primary_split:])
        qg_ref = cumulative_trapezoid(
            traj.output[1, primary_split:], traj.time[primary_split:],
            initial=0.0)
        cand.meta["heldout_qg_rmse"] = _rmse(qg_pred, qg_ref)
        key = sel_err if np.isfinite(sel_err) else float("inf")
        if best is None or key < best_key:
            best = cand
            best_key = key
    if best is None:
        raise RuntimeError(f"{traj.vendor}: no valid TPWL candidate")

    if holdout is not None:
        hs = holdout["split"]
        id_pred, qg_pred = best.predict(holdout["t"][hs:],
                                        holdout["u"][:, hs:])
        best.meta["holdout_id_rmse_rel"] = _rmse(
            id_pred, holdout["y"][0, hs:])
        qg_ref = cumulative_trapezoid(
            holdout["y"][1, hs:], holdout["t"][hs:], initial=0.0)
        best.meta["holdout_qg_rmse_rel"] = _rmse(qg_pred, qg_ref)
        best.meta["cv_status"] = "holdout"
    else:
        best.meta["holdout_id_rmse_rel"] = "not_measured"
        best.meta["holdout_qg_rmse_rel"] = "not_measured"
        best.meta["cv_status"] = "not_measured"
        if holdout_trajectory:
            best.meta["cv_reason"] = (
                "holdout_trajectory requested but no extra_trajectories "
                "supplied; no cross-validation split is possible")
        else:
            best.meta["cv_reason"] = "holdout_trajectory not requested"
    return best
