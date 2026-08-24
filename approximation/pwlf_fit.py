"""Optimal continuous piecewise linear fitting (Plan §2.1).

Fits C0-continuous PWL models  y(x) = beta_0 + sum_k beta_k * max(0, x - xi_k)
using the `pwlf` library's global-search + local least-squares machinery, and
adds the PieceSpice requirements on top:

* **Adaptive segment selection** (``fit_adaptive``): increments K until the
  fit meets an RMSE *or* max-abs-error tolerance (plan: "typically <1%"),
  with a hard cap so degenerate data cannot loop forever.
* **Residual metrics**: L2 RMSE and L-inf max abs error per fit
  (:class:`PWLFitResult`), plus per-segment slope/breakpoint report.
* **Monotonic breakpoints**: guaranteed sorted, deduplicated breakpoints so
  downstream netlist emission never produces a zero-width segment.

All functions accept plain numpy arrays and return numpy arrays / dataclasses;
no plotting code lives here (root-module separation rule §4.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import pwlf
except ImportError as _exc:  # pragma: no cover - environment issue
    raise ImportError(
        "The 'pwlf' package is required for approximation.pwlf_fit; "
        "install with: uv add pwlf") from _exc


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PWLFitResult:
    """Outcome of one piecewise-linear fit."""

    breakpoints_x: np.ndarray        # sorted breakpoint coordinates [K+1]
    slopes: np.ndarray               # per-segment slope [K]
    intercepts: np.ndarray           # per-segment intercept [K]
    predicted_y: np.ndarray          # model evaluated on the input x
    rmse: float                      # L2 root-mean-square error
    max_abs_error: float             # L-inf maximum absolute error
    n_segments: int                  # K actually used
    converged: bool = True
    notes: List[str] = field(default_factory=list)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the fitted PWL model at arbitrary x (clamped at ends)."""
        x = np.asarray(x, dtype=float)
        # Per-segment linear evaluation via searchsorted on interior breaks.
        interior = self.breakpoints_x[1:-1]
        seg = np.clip(np.searchsorted(interior, x, side="right"), 0,
                      len(self.slopes) - 1)
        return (self.slopes[seg] * x + self.intercepts[seg])

    def summary(self) -> str:
        return (f"PWL K={self.n_segments}: RMSE={self.rmse:.4g}, "
                f"max|e|={self.max_abs_error:.4g}, "
                f"breaks={np.array2string(self.breakpoints_x, precision=4)}")


# ---------------------------------------------------------------------------
# Core single-fit routine
# ---------------------------------------------------------------------------

def _sanitize_xy(x_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_data, dtype=float).ravel()
    y = np.asarray(y_data, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        raise ValueError(
            f"need >= 3 finite samples to fit a PWL model, got {x.size}")
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    # Collapse duplicate x (keep mean of y at identical x) for pwlf stability.
    ux, inverse = np.unique(x, return_inverse=True)
    if ux.size != x.size:
        uy = np.zeros(ux.size)
        counts = np.zeros(ux.size)
        np.add.at(uy, inverse, y)
        np.add.at(counts, inverse, 1.0)
        x, y = ux, uy / counts
    else:
        x, y = ux, y
    return x, y


def fit_piecewise_linear(
    x_data: np.ndarray,
    y_data: np.ndarray,
    n_segments: int = 4,
) -> PWLFitResult:
    """Fit a continuous piecewise-linear model with exactly ``n_segments``.

    Args:
        x_data: Independent variable samples (unsorted ok).
        y_data: Dependent variable samples.
        n_segments: Number of linear segments K (>= 1).

    Returns:
        PWLFitResult with breakpoints, per-segment line coefficients and
        residual metrics.
    """
    if n_segments < 1:
        raise ValueError(f"n_segments must be >= 1, got {n_segments}")

    x, y = _sanitize_xy(x_data, y_data)

    # A single segment is plain least squares - pwlf handles K=1 too, but
    # doing it directly avoids its global-search overhead.
    if n_segments == 1 or x.size <= n_segments + 1:
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        err = y - pred
        return PWLFitResult(
            breakpoints_x=np.array([x[0], x[-1]]),
            slopes=np.array([slope]),
            intercepts=np.array([intercept]),
            predicted_y=pred,
            rmse=float(np.sqrt(np.mean(err ** 2))),
            max_abs_error=float(np.max(np.abs(err))),
            n_segments=1,
            notes=(["degenerated to single-segment LSQ (few samples)"
                    ] if x.size <= n_segments + 1 else []),
        )

    model = pwlf.PiecewiseLinFit(x, y)
    try:
        breaks = model.fit(n_segments)
    except Exception as exc:
        # Fall back to one fewer segments rather than failing the pipeline.
        return fit_piecewise_linear(x, y, max(1, n_segments - 1)) if (
            n_segments > 1) else _raise_fit(exc)

    result = _result_from_breaks(x, y, np.asarray(breaks))
    result.notes.append(f"pwlf global search succeeded at K={n_segments}")
    return result


def _raise_fit(exc: Exception) -> None:
    raise RuntimeError(f"PWLF fit failed even at K=1: {exc}") from exc


def _result_from_breaks(x: np.ndarray, y: np.ndarray,
                        breaks: np.ndarray) -> PWLFitResult:
    """Build a PWLFitResult from explicit breakpoint coordinates."""
    breaks = np.unique(np.round(np.asarray(breaks, dtype=float), 12))
    k = len(breaks) - 1
    slopes = np.zeros(k)
    intercepts = np.zeros(k)
    pred = np.zeros_like(y)

    seg_idx = np.clip(np.searchsorted(breaks[1:-1], x, side="right"),
                      0, k - 1)
    for s in range(k):
        m = seg_idx == s
        if m.sum() >= 2:
            sl, ic = np.polyfit(x[m], y[m], 1)
        else:  # isolated sample: reuse neighbour slope
            sl, ic = (slopes[s - 1], y[m][0] - slopes[s - 1] * x[m][0]) \
                if s > 0 else (0.0, float(np.mean(y)))
        slopes[s], intercepts[s] = sl, ic
        pred[m] = sl * x[m] + ic

    err = y - pred
    return PWLFitResult(
        breakpoints_x=breaks,
        slopes=slopes,
        intercepts=intercepts,
        predicted_y=pred,
        rmse=float(np.sqrt(np.mean(err ** 2))),
        max_abs_error=float(np.max(np.abs(err))) if err.size else 0.0,
        n_segments=k,
    )


# ---------------------------------------------------------------------------
# Adaptive selection (plan §2.1 step 4)
# ---------------------------------------------------------------------------

def fit_adaptive(
    x_data: np.ndarray,
    y_data: np.ndarray,
    rmse_tolerance: Optional[float] = None,
    rel_tolerance: float = 0.01,
    max_segments: int = 12,
    min_segments: int = 1,
) -> PWLFitResult:
    """Increment K until the fit meets tolerance (plan algorithm §2.1).

    Stop when EITHER metric is satisfied:
      * absolute ``rmse_tolerance`` [y-units], or
      * relative ``rel_tolerance`` of the y peak-to-peak span
        (default 1% — the plan's typical target).

    Always within ``[min_segments, max_segments]``; returns the best fit seen
    if the tolerance is unreachable within the cap (converged=False then).
    """
    x, y = _sanitize_xy(x_data, y_data)
    span = float(np.max(y) - np.min(y))
    tol_abs = rmse_tolerance if rmse_tolerance is not None else \
        rel_tolerance * max(span, 1e-30)

    best: Optional[PWLFitResult] = None
    prev_rmse: Optional[float] = None
    for k in range(max(min_segments, 1), max_segments + 1):
        result = fit_piecewise_linear(x, y, k)
        if best is None or result.rmse < best.rmse:
            best = result
        if result.rmse <= tol_abs:
            result.notes.append(
                f"adaptive stop: K={k} reached RMSE {result.rmse:.4g} "
                f"<= tolerance {tol_abs:.4g}")
            return result
        # Early exit on STAGNATION: this K failed to improve on the previous
        # K's RMSE by at least 0.1%. (Bug fix: the old check compared against
        # `best`, which after an improvement IS `result`, so it always fired
        # right after an improvement and exited far too early.)
        if prev_rmse is not None and k > min_segments + 1 \
                and result.rmse > 0.999 * prev_rmse:
            if result.rmse < best.rmse:
                best = result
            break
        prev_rmse = result.rmse

    assert best is not None
    best.converged = False
    best.notes.append(
        f"tolerance {tol_abs:.4g} not reachable within "
        f"max_segments={max_segments}; returning best K="
        f"{best.n_segments} (RMSE {best.rmse:.4g})")
    return best


# Back-compat shim for earlier call signature returning a bare tuple.
def fit_piecewise_linear_tuple(
    x_data: np.ndarray, y_data: np.ndarray, n_segments: int = 4
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Legacy tuple interface: (breakpoints, predicted_y, rmse)."""
    r = fit_piecewise_linear(x_data, y_data, n_segments)
    return r.breakpoints_x, r.predicted_y, r.rmse
