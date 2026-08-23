"""Ramer-Douglas-Peucker (RDP) polyline simplification (Numpy 2.0+ compatible)."""

from typing import Tuple
import numpy as np


def _pldist_2d(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Compute perpendicular distances from 2D points to line segment (start -> end)."""
    if np.allclose(start, end):
        return np.linalg.norm(points - start, axis=1)

    # Line equation: (y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1 = 0
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    numerator = np.abs(dy * points[:, 0] - dx * points[:, 1] + end[0] * start[1] - end[1] * start[0])
    denominator = np.hypot(dx, dy)
    return numerator / max(1e-15, denominator)


def _rdp_recursive(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Recursive core for 2D RDP point reduction."""
    if len(points) <= 2:
        return points

    dists = _pldist_2d(points[1:-1], points[0], points[-1])
    if len(dists) == 0:
        return points

    max_idx = int(np.argmax(dists)) + 1
    max_dist = dists[max_idx - 1]

    if max_dist > epsilon:
        left = _rdp_recursive(points[:max_idx + 1], epsilon)
        right = _rdp_recursive(points[max_idx:], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return np.vstack([points[0], points[-1]])


def compress_rdp(
    x_data: np.ndarray,
    y_data: np.ndarray,
    epsilon: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """Simplify (x, y) characteristic curves using native 2D Ramer-Douglas-Peucker algorithm.

    Args:
        x_data: 1D array of x coordinates
        y_data: 1D array of y coordinates
        epsilon: Distance tolerance for point pruning

    Returns:
        x_compressed, y_compressed
    """
    points = np.column_stack([x_data, y_data])
    simplified = _rdp_recursive(points, epsilon=epsilon)
    return simplified[:, 0], simplified[:, 1]
