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

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter

from tier3.trajectory import Trajectory

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


def build_model(traj: Trajectory, coss_curve=None,
                v_block: float | None = None,
                v_width: float | None = None,
                s_grid: tuple[int, ...] = (8, 16, 24, 32),
                include_static: bool = True) -> TPWLTerminalModel:
    """Fit channel on Curve-1 static family + DPT; select s by static error.

    DPT alone drives only two Vgs levels and gave 65-1e9 % error on the
    Curve-1 family; the static family is the project's headline metric.
    """
    split = _split_index(traj.input[0])
    t, u, y = traj.time, traj.input, traj.output
    v_ref, v_span = _operating_scaler(u[:, :split])
    ws = (u - v_ref[:, None]) / v_span[:, None]
    wtr = ws[:, :split]

    dt = float(np.median(np.diff(t)))
    window = max(3, int(2.0 * traj.edge_time / dt) + 1)
    window += (window + 1) % 2
    window = min(window, split - 1 + (split - 1) % 2)
    wsm = savgol_filter(wtr, window, 2, axis=1)
    dws = np.gradient(wsm, t[:split], axis=1)
    qg = cumulative_trapezoid(y[1, :split], t[:split], initial=0.0)

    qoss_v = qoss_q = None
    id_fit = y[0, :split]
    if coss_curve is not None:
        from approximation.charge_model import integrate_charge
        cv, cc = np.asarray(coss_curve[0]), np.asarray(coss_curve[1])
        cm = integrate_charge(cv, cc)
        qoss_v, qoss_q = cm.v, cm.q
        qoss_t = np.interp(u[1, :split], qoss_v, qoss_q)
        id_fit = y[0, :split] - np.gradient(qoss_t, t[:split])
    coss_v = None if coss_curve is None else np.asarray(coss_curve[0])
    coss_c = None if coss_curve is None else np.asarray(coss_curve[1])

    # combine sources: static family for channel coverage, DPT for charges
    ws_all, dws_all, id_all, qg_all = wtr, dws, id_fit, qg
    is_static = np.zeros(wtr.shape[1], dtype=bool)
    is_dpt = np.ones(wtr.shape[1], dtype=bool)
    static_u = static_id = None
    if include_static:
        from tier3.static_curves import static_samples
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

    sv = np.linalg.svd(ws_all, compute_uv=False)
    energy = (sv ** 2 / np.sum(sv ** 2)).tolist()

    best = None
    for s in s_grid:
        idx = _anchors_for(ws_all, s)
        channel, charge_d, charge_g = _fit_cells(
            ws_all, dws_all, id_all, qg_all, idx, is_static, is_dpt)
        cand = TPWLTerminalModel(
            vendor=traj.vendor, subcircuit=traj.subcircuit,
            pin_roles=traj.pin_roles, anchors=ws_all[:, idx].T,
            v_ref=v_ref, v_span=v_span, channel=channel,
            charge_d=charge_d, charge_g=charge_g, edge_window=window,
            qoss_v=qoss_v, qoss_q=qoss_q, coss_v=coss_v, coss_c=coss_c,
            v_block=v_block, v_width=v_width,
            meta={"pod_energy": energy, "cells": s,
                  "deim_seed_count": min(s, len(energy)),
                  "qoss_base": coss_curve is not None,
                  "static_samples": int(is_static.sum())})
        if static_u is not None:
            sp = cand.predict(np.linspace(0, 1, static_u.shape[1]),
                              static_u)[0]
            st_err = _rmse(sp, static_id)
            cand.meta["static_rmse_rel"] = st_err
        else:
            st_err = float("inf")
        id_pred, qg_pred = cand.predict(t[split:], u[:, split:])
        cand.meta["heldout_id_rmse"] = _rmse(id_pred, y[0, split:])
        qg_ref = cumulative_trapezoid(y[1, split:], t[split:], initial=0.0)
        cand.meta["heldout_qg_rmse"] = _rmse(qg_pred, qg_ref)
        if best is None or st_err < best.meta.get(
                "static_rmse_rel", float("inf")):
            best = cand
    if best is None:
        raise RuntimeError(f"{traj.vendor}: no valid TPWL candidate")
    return best
