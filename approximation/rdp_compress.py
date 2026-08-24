"""Ramer-Douglas-Peucker (RDP) polyline simplification (Numpy 2.0+ compatible).

Prunes high-density simulation data while preserving critical non-linear
inflection points (threshold knee, saturation transition — plan §2.2).

Enhancements over a bare RDP:

* **Iterative core** (explicit stack) — no RecursionError on 10^5-sample
  sweeps.
* **Scale-normalized epsilon** option: ``relative=True`` interprets epsilon
  as a fraction of the curve's bounding-box diagonal, so one tolerance works
  across curves with wildly different units (V/A/pF).
* **Guarantees**: first/last points always retained; output x strictly
  increasing; NaN/inf samples dropped before simplification.

Pure numerical module - no plotting code (§4.2).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["compress_rdp", "rdp_epsilon_for_max_error"]


def _pldist_2d(points: np.ndarray, start: np.ndarray,
               end: np.ndarray) -> np.ndarray:
    """Perpendicular distances from 2D points to the line start -> end."""
    if np.allclose(start, end):
        return np.linalg.norm(points - start, axis=1)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    numerator = np.abs(
        dy * points[:, 0] - dx * points[:, 1]
        + end[0] * start[1] - end[1] * start[0])
    return numerator / max(float(np.hypot(dx, dy)), 1e-300)


def _rdp_iterative(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Explicit-stack RDP (no recursion-depth limits)."""
    n = len(points)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True

    stack: list[tuple[int, int]] = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        seg = points[lo:hi + 1]
        dists = _pldist_2d(seg[1:-1], seg[0], seg[-1])
        local_max = int(np.argmax(dists))
        max_dist = dists[local_max]
        if max_dist > epsilon:
            split = lo + 1 + local_max
            keep[split] = True
            stack.append((lo, split))
            stack.append((split, hi))
    return points[keep]


def compress_rdp(
    x_data: np.ndarray,
    y_data: np.ndarray,
    epsilon: float = 0.01,
    relative: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simplify an (x, y) characteristic curve with Ramer-Douglas-Peucker.

    Args:
        x_data: 1D x coordinates (any order; sorted internally, output is
            ascending).
        y_data: 1D y coordinates.
        epsilon: Pruning distance tolerance.
        relative: When True, ``epsilon`` is interpreted as a fraction of the
            curve's bounding-box diagonal (e.g. 0.01 = 1% of diagonal), which
            makes the tolerance transferable between curves of different
            physical scales.

    Returns:
        (x_compressed, y_compressed) with the first and last points always
        retained and x strictly ascending.
    """
    x = np.asarray(x_data, dtype=float).ravel()
    y = np.asarray(y_data, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(
            f"x/y length mismatch: {x.size} vs {y.size}")
    if x.size < 3:
        return x.copy(), y.copy()

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]

    eps = float(epsilon)
    if relative:
        dx_span = x.max() - x.min()
        dy_span = y.max() - y.min()
        diag = float(np.hypot(dx_span, dy_span))
        eps = epsilon * max(diag, 1e-300)

    points = np.column_stack([x, y])
    simplified = _rdp_iterative(points, eps)

    # Guarantee strict monotonic x (RDP on sorted input already satisfies
    # this, but dedupe defensively after numeric edge cases).
    xs, ys = simplified[:, 0], simplified[:, 1]
    keep_x = np.concatenate([[True], np.diff(xs) > 0])
    xs, ys = xs[keep_x], ys[keep_x]

    # Always retain the true final sample even if it duplicates in x.
    if xs[-1] != x[-1]:
        xs = np.append(xs, x[-1])
        ys = np.append(ys, y[-1])

    return xs, ys


def rdp_epsilon_for_max_error(
    x_data: np.ndarray,
    y_data: np.ndarray,
    max_error: float,
    low: float = 1e-9,
    high: float = 1e3,
    iterations: int = 40,
) -> float:
    """Binary-search the largest epsilon whose RDP result stays within
    ``max_error`` absolute deviation from the original curve.

    Useful to hit a datasheet-fidelity budget exactly instead of guessing a
    tolerance. Returns the epsilon value (not the compressed curve).
    """
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)

    def peak_deviation(eps: float) -> float:
        xc, yc = compress_rdp(x, y, epsilon=eps)
        interp = np.interp(x, xc, yc)
        return float(np.max(np.abs(interp - y)))

    lo, hi = low, high
    best = low
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        if peak_deviation(mid) <= max_error:
            best = mid
            lo = mid
        else:
            hi = mid
    return best
