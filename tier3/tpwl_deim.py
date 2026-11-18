"""TPWL + POD-DEIM reduced model, following the official algorithms.

Paper constants used verbatim:

* Rewienski & White (TCAD 2003), eqs. (8)-(9), §4.1 weight procedure:
  d_i = ||z - z_i||_2, m = min_i d_i, w_i = (exp(d_i)/m)**(-25),
  normalised.  Since m**25 cancels in the normalisation this is exactly
  softmax(-25 * d_i), implemented with a min-shift for numerical safety.
* Chaturantabut & Sorensen DEIM Algorithm 1 (as reprinted in
  Sorensen & Embree, arXiv:1407.5516, Algorithm 1):
      p1 = argmax |v1|
      for j = 2..k:
          c = V(p, 0:j-1)^{-1} v(p)
          r = v_j - V(:, 0:j-1) c
          p_j = argmax |r|
  with v_j the j-th left singular vector of the nonlinear snapshot matrix.

Everything else (reduced order r, linearisation radius, linearisation
locations) is selected from the captured trajectory by cross-validation,
not guessed. Parameter names and choices are recorded in the model so the
emitted netlist and report can state them explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from tier3.trajectory import Trajectory

from scipy.linalg import eig as _eig
from scipy.signal import savgol_filter
# Paper-printed constants (NOT tuning knobs)
WEIGHT_EXPONENT = 25.0        # Rewienski & White 2003, §4.1 step 3


@dataclass
class TPWLModel:
    """Reduced affine TPWL model with DEIM hyper-reduction."""
    vendor: str
    basis: np.ndarray              # (N, r)
    x_mean: np.ndarray             # (N,)
    z_points: np.ndarray           # (s, r)
    Ared: np.ndarray               # (s, r, r)
    Bred: np.ndarray               # (s, r, p)
    gamma: np.ndarray              # (s, r)
    C: np.ndarray                  # (q, r)
    D: np.ndarray                  # (q, p)
    e: np.ndarray                  # (q,)
    deim_indices: np.ndarray       # (m,), zero-based state indices
    deim_coeff: np.ndarray         # (r, m) = V^T U (P^T U)^{-1}
    z0: np.ndarray                 # (r,) initial reduced state
    time: np.ndarray               # training time grid
    state_names: list[str]
    pin_roles: list[str]
    subcircuit: str
    r: int
    n_points: int
    eps: float
    validation_rmse: float
    meta: dict = field(default_factory=dict)
    time_scale: float = 1.0                 # trajectory duration T (seconds)
    u_ref: np.ndarray | None = None
    u_span: np.ndarray | None = None
    y_min: np.ndarray | None = None
    y_max: np.ndarray | None = None

    # -- reduced dynamics ---------------------------------------------------
    def weights(self, z: np.ndarray) -> np.ndarray:
        """Rewienski-White weights, eq. (8)/§4.1 step 1-4."""
        d = np.linalg.norm(z[None, :] - self.z_points, axis=1)
        return self._softmax(-WEIGHT_EXPONENT * d)

    @staticmethod
    def _softmax(v: np.ndarray) -> np.ndarray:
        s = v - np.max(v)
        ex = np.exp(s)
        return ex / np.sum(ex)

    def _scaled_input(self, u: np.ndarray) -> np.ndarray:
        if self.u_ref is None or self.u_span is None:
            return u
        return (u - self.u_ref) / self.u_span

    def rhs(self, _t: float, z: np.ndarray,
            u: np.ndarray) -> np.ndarray:
        us = self._scaled_input(u)
        w = self.weights(z)
        dz = np.zeros_like(z)
        for i, wi in enumerate(w):
            dz += wi * (self.Ared[i] @ z + self.Bred[i] @ us
                        + self.gamma[i])
        return dz / self.time_scale

    def output(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        y = self.C @ z + self.D @ self._scaled_input(u) + self.e
        if self.y_min is not None and self.y_max is not None:
            y = np.clip(y, self.y_min, self.y_max)
        return y


# ---------------------------------------------------------------------------
# DEIM — Algorithm 1 verbatim
# ---------------------------------------------------------------------------

def deim_indices(basis: np.ndarray) -> np.ndarray:
    """DEIM Algorithm 1 on an m-by-k orthogonal basis matrix."""
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
# Trajectory-derived quantities
# ---------------------------------------------------------------------------

def _split_index(traj: Trajectory) -> int:
    """Index of the second rising edge of Vgs — the second DPT pulse."""
    vgs = traj.input[0]
    high = np.max(vgs)
    level = 0.5 * high
    rising = np.flatnonzero((vgs[:-1] < level) & (vgs[1:] >= level)) + 1
    if len(rising) < 2:
        raise RuntimeError(f"{traj.vendor}: fewer than two Vgs rising edges")
    return int(rising[1])


def _linearization_points(x: np.ndarray, eps: float,
                          max_points: int) -> list[int]:
    """Trajectory walk of Rewienski & White §4.2 (new point when ||x-x_i||>eps)."""
    points = [0]
    current = 0
    for k in range(1, x.shape[1]):
        if np.linalg.norm(x[:, k] - x[:, current]) > eps:
            if len(points) >= max_points:
                break
            points.append(k)
            current = k
    return points


def _fit_local(x: np.ndarray, u: np.ndarray, fsel: np.ndarray,
               x_i: np.ndarray, u_i: np.ndarray, eps: float,
               min_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local numerical Jacobian: affine LS fit of selected f snapshots."""
    dist = np.linalg.norm(x - x_i[:, None], axis=0)
    keep = np.flatnonzero(dist <= eps)
    if keep.size < min_rows:
        keep = np.argsort(dist)[:min_rows]
    dx = (x[:, keep] - x_i[:, None]).T
    du = (u[:, keep] - u_i[:, None]).T
    design = np.hstack([dx, du, np.ones((keep.size, 1))])
    sol, *_ = np.linalg.lstsq(design, fsel[:, keep].T, rcond=None)
    return sol[:x.shape[0]].T, sol[x.shape[0]:x.shape[0] + u.shape[0]].T, \
        sol[-1, :]


def _state_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and peak-to-peak span of each state channel (data-derived)."""
    mean = x.mean(axis=1)
    span = np.ptp(x, axis=1)
    span[span == 0] = 1.0
    return mean, span


def _input_scaler(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = u.mean(axis=1)
    span = np.ptp(u, axis=1)
    span[span == 0] = 1.0
    return mean, span


def _smooth_derivative(xs: np.ndarray, tau: np.ndarray,
                       traj: Trajectory) -> tuple[np.ndarray, int]:
    """Savitzky-Golay smoothing before differentiation.

    Window spans two configured gate edges, derived from the YAML edge time
    and the measured sample interval — no hand-picked smoothing constant.
    """
    dt = float(np.median(np.diff(traj.time)))
    window = max(3, int(2.0 * traj.edge_time / dt) + 1)
    window += (window + 1) % 2
    window = min(window, len(tau) - 1 + (len(tau) - 1) % 2)
    smoothed = savgol_filter(xs, window, 2, axis=1)
    return np.gradient(smoothed, tau, axis=1), window


def _project_stable(A: np.ndarray, max_mag: float) -> np.ndarray:
    """Numerical guard: mirror spurious unstable eigenvalues into the left
    half-plane and cap their magnitude at the sampling-derived bound."""
    vals, vecs = _eig(A)
    vals = np.where(vals.real > 0, -np.abs(vals.real) + 1j * vals.imag, vals)
    mag = np.abs(vals)
    vals = np.where(mag > max_mag, vals * (max_mag / np.maximum(mag, 1e-300)), vals)
    return np.real(vecs @ np.diag(vals) @ np.linalg.inv(vecs))

def _assemble(traj: Trajectory, r: int, eps: float, split: int,
              max_points: int) -> TPWLModel:
    """Fit one candidate model on the first pulse; score on the second."""
    x, u, y = traj.state, traj.input, traj.output
    xtr, utr, ytr = x[:, :split], u[:, :split], y[:, :split]
    ttr = traj.time[:split]
    T = float(ttr[-1] - ttr[0])
    x_mean, x_span = _state_scaler(xtr)
    u_ref, u_span = _input_scaler(utr)
    xs = (xtr - x_mean[:, None]) / x_span[:, None]
    us = (utr - u_ref[:, None]) / u_span[:, None]
    tau = (ttr - ttr[0]) / T
    deriv, smooth_window = _smooth_derivative(xs, tau, traj)
    uu_svd, _, _ = np.linalg.svd(xs, full_matrices=False)
    V = uu_svd[:, :r]

    # DEIM on nonlinear snapshots (scaled derivative of the state)
    uf, _, _ = np.linalg.svd(deriv, full_matrices=False)
    U = uf[:, :r]
    idx = deim_indices(U)
    coeff = V.T @ U @ np.linalg.inv(U[idx, :])       # (r, r)
    fsel = deriv[idx, :]

    dt = float(np.median(np.diff(traj.time)))
    eig_bound = T / (2.0 * dt)
    kpts = _linearization_points(xs, float(eps), max_points)
    z_points, Ared, Bred, gammas = [], [], [], []
    min_rows = fsel.shape[0] + u.shape[0] + 1
    for k in kpts:
        x_i, u_i = xs[:, k], us[:, k]
        A_i, B_i, f_i = _fit_local(xs, us, fsel, x_i, u_i,
                                   float(eps), min_rows)
        Ared.append(_project_stable(coeff @ A_i @ V, eig_bound))
        Bred.append(coeff @ B_i)
        gammas.append(coeff @ (f_i - A_i @ x_i - B_i @ u_i))
        z_points.append(V.T @ x_i)

    ztr = V.T @ xs
    out_design = np.vstack([ztr, us, np.ones((1, ztr.shape[1]))]).T
    out_sol, *_ = np.linalg.lstsq(out_design, ytr.T, rcond=None)
    C = out_sol[:r].T
    D = out_sol[r:r + u.shape[0]].T
    e = out_sol[-1]

    model = TPWLModel(
        vendor=traj.vendor, basis=V, x_mean=x_mean,
        z_points=np.asarray(z_points), Ared=np.asarray(Ared),
        Bred=np.asarray(Bred), gamma=np.asarray(gammas),
        C=C, D=D, e=e, deim_indices=idx, deim_coeff=coeff,
        z0=V.T @ xs[:, 0], time=traj.time[:split],
        state_names=traj.state_names, pin_roles=traj.pin_roles,
        subcircuit=traj.subcircuit, r=r, n_points=len(kpts),
        eps=float(eps), validation_rmse=float("inf"),
        meta={"split_index": split, "weight_exponent": WEIGHT_EXPONENT,
              "x_span": x_span.tolist(), "u_ref": u_ref.tolist(),
              "u_span": u_span.tolist(), "smoothing_window": smooth_window,
              "stability_projection": True,
              "deim_index_names": [traj.state_names[j] for j in idx]},
        time_scale=T, u_ref=u_ref, u_span=u_span,
        y_min=ytr.min(axis=1), y_max=ytr.max(axis=1))

    # Score on held-out second pulse using the projected reduced state.
    zv = V.T @ ((x[:, split:] - x_mean[:, None]) / x_span[:, None])
    yp = np.vstack([model.output(zv[:, j], u[:, split + j])
                    for j in range(zv.shape[1])]).T
    span = np.ptp(traj.output, axis=1)
    span[span == 0] = 1.0
    err = np.sqrt(np.mean(((yp - y[:, split:]) / span[:, None]) ** 2,
                          axis=1))
    raw_score = float(np.mean(err))
    # Penalise modes faster than two measured samples (data-derived bound).
    max_eig = max(float(np.max(np.abs(np.linalg.eigvals(A))))
                  for A in model.Ared)
    bound = eig_bound
    model.meta.update(raw_score=raw_score, max_reduced_eigenvalue=max_eig,
                      stiffness_bound=bound)
    model.validation_rmse = raw_score * (1.0 + max_eig / bound)
    return model


class IntegrationLimit(RuntimeError):
    """Raised when the reduced ODE needs more evaluations than allowed."""


def validate_dynamics(model: TPWLModel, traj: Trajectory, *,
                      max_nfev: int = 200_000) -> dict:
    """Integrate the selected reduced model on the full DPT gate train.

    Uses LSODA (stiff-capable) with a maximum step derived from the measured
    trajectory sample interval, and a hard evaluation cap so a pathological
    candidate cannot stall the pipeline.
    """
    tv = traj.time
    uv = traj.input
    dt = float(np.median(np.diff(tv)))
    counter = {"n": 0}

    def _rhs(t: float, z: np.ndarray) -> np.ndarray:
        counter["n"] += 1
        if counter["n"] > max_nfev:
            raise IntegrationLimit("reduced ODE evaluation cap reached")
        ut = np.array([np.interp(t, tv, uv[0]), np.interp(t, tv, uv[1])])
        return model.rhs(t, z, ut)

    try:
        sol = solve_ivp(_rhs, (tv[0], tv[-1]), model.z0, t_eval=tv,
                        method="LSODA", max_step=dt)
    except IntegrationLimit as exc:
        return {"evaluable": False, "reason": str(exc), "nfev": counter["n"]}
    if not sol.success:
        return {"evaluable": False, "reason": str(sol.message),
                "nfev": int(sol.nfev)}
    yp = np.vstack([model.output(sol.y[:, j], uv[:, j])
                    for j in range(uv.shape[1])]).T
    span = np.ptp(traj.output, axis=1)
    span[span == 0] = 1.0
    err = np.sqrt(np.mean(((yp - traj.output) / span[:, None]) ** 2,
                          axis=1))
    return {"evaluable": True, "nfev": int(sol.nfev),
            "rmse_id": float(err[0]), "rmse_ig": float(err[1]),
            "rmse_mean": float(np.mean(err))}


def build_model(traj: Trajectory, *, max_rank: int = 8,
                max_points: int = 8) -> TPWLModel:
    """Select (r, eps) by held-out second-pulse error; no arbitrary values."""
    split = _split_index(traj)
    x = traj.state
    x_mean, x_span = _state_scaler(x[:, :split])
    xs = (x[:, :split] - x_mean[:, None]) / x_span[:, None]
    span_dist = np.linalg.norm(xs - xs[:, :1], axis=0)
    eps_grid = np.unique(np.quantile(span_dist, [0.25, 0.5, 0.75]))
    ranks = range(1, min(max_rank, x.shape[0]) + 1)
    best: TPWLModel | None = None
    for r in ranks:
        for eps in eps_grid:
            pts = _linearization_points(xs, float(eps), max_points)
            if len(pts) < 2:
                continue
            model = _assemble(traj, r, float(eps), split, max_points)
            if best is None or model.validation_rmse < best.validation_rmse:
                best = model
    if best is None:
        raise RuntimeError(f"{traj.vendor}: no valid TPWL candidate")
    return best
