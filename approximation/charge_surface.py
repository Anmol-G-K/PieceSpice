"""Two-dimensional terminal charge surface for the Tier-3 emitter.

The closed 1-D ``Qoss(Vds)`` table cannot describe charge that depends on the
gate bias as well.  This module stores a measured charge surface on the
(Vgs, Vds) plane (optionally one slice per temperature) and exposes the two
operations the emitter and the validation path need:

* ``charge(vds, vgs)`` -- bilinear interpolation of Q [C];
* ``capacitance(vds, vgs)`` -- dQ/dVds at fixed Vgs [F].

Only measured/synthetic tabulated data are accepted.  Construction is
fail-closed: axes must be strictly increasing, shapes must match, and the
charge must be non-decreasing along Vds (a negative dQ/dVds would emit a
negative capacitance).  Queries outside the tabulated box raise instead of
silently extrapolating.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _increasing_axis(values: np.ndarray, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError(f"ChargeSurface {name} axis needs >=2 points")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"ChargeSurface {name} axis has non-finite values")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError(
            f"ChargeSurface {name} axis must be strictly increasing")
    return axis


def _bracket(axis: np.ndarray, values: np.ndarray, name: str
             ) -> tuple[np.ndarray, np.ndarray]:
    """Cell index and fraction for *values*, failing closed outside *axis*."""
    if np.any(values < axis[0]) or np.any(values > axis[-1]):
        lo, hi = float(np.min(values)), float(np.max(values))
        raise ValueError(
            f"ChargeSurface query outside the measured {name} range "
            f"[{axis[0]:.6g}, {axis[-1]:.6g}] V (got {lo:.6g}..{hi:.6g} V)")
    idx = np.searchsorted(axis, values, side="right") - 1
    idx = np.clip(idx, 0, axis.size - 2)
    frac = (values - axis[idx]) / (axis[idx + 1] - axis[idx])
    return idx, frac


@dataclass
class ChargeSurface:
    """Tabulated Q(vds, vgs[, temperature]) with bilinear evaluation.

    ``q`` is shaped ``(n_vgs, n_vds)``; with a temperature axis it is
    ``(n_temperature, n_vgs, n_vds)``.  ``temperature_c`` is optional and, when
    present, must have one entry per leading ``q`` slice.
    """

    vds: np.ndarray
    vgs: np.ndarray
    q: np.ndarray
    temperature_c: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.vds = _increasing_axis(self.vds, "vds")
        self.vgs = _increasing_axis(self.vgs, "vgs")
        self.q = np.asarray(self.q, dtype=float)
        if self.temperature_c is not None:
            self.temperature_c = _increasing_axis(
                self.temperature_c, "temperature_c")
            if self.q.ndim != 3:
                raise ValueError(
                    "ChargeSurface with temperature_c needs q shaped "
                    "(n_temperature, n_vgs, n_vds)")
            if self.q.shape[0] != self.temperature_c.size:
                raise ValueError(
                    "ChargeSurface temperature axis and q slices disagree: "
                    f"{self.temperature_c.size} vs {self.q.shape[0]}")
            expected = (self.vgs.size, self.vds.size)
        else:
            if self.q.ndim != 2:
                raise ValueError(
                    "ChargeSurface without temperature_c needs q shaped "
                    "(n_vgs, n_vds)")
            expected = (self.vgs.size, self.vds.size)
        if self.q.shape[-2:] != expected:
            raise ValueError(
                f"ChargeSurface q shape {self.q.shape} does not match "
                f"(vgs, vds) = {expected}")
        if not np.all(np.isfinite(self.q)):
            raise ValueError("ChargeSurface q has non-finite values")
        if np.any(np.diff(self.q, axis=-1) < 0.0):
            raise ValueError(
                "ChargeSurface charge is non-monotone in Vds; refusing a "
                "negative capacitance")
        if not (np.all(np.isfinite(self.temperature_c))
                if self.temperature_c is not None else True):
            raise ValueError("ChargeSurface temperature axis is non-finite")

    # -- temperature handling -------------------------------------------------
    def _slice(self, temperature: float | None) -> np.ndarray:
        if self.temperature_c is None:
            if temperature is not None:
                raise ValueError(
                    "ChargeSurface has no temperature axis; a temperature "
                    "query would be an unmeasured extrapolation")
            return self.q
        if temperature is None:
            raise ValueError(
                "ChargeSurface has a temperature axis; a temperature is "
                "required for every query")
        t = float(temperature)
        if not np.isfinite(t):
            raise ValueError("ChargeSurface temperature query is non-finite")
        axis = self.temperature_c
        if t < axis[0] or t > axis[-1]:
            raise ValueError(
                f"ChargeSurface query outside the measured temperature range "
                f"[{axis[0]:.6g}, {axis[-1]:.6g}] degC (got {t:.6g})")
        k = int(np.searchsorted(axis, t, side="right") - 1)
        k = min(max(k, 0), axis.size - 2)
        w = (t - axis[k]) / (axis[k + 1] - axis[k])
        return (1.0 - w) * self.q[k] + w * self.q[k + 1]

    # -- public evaluation ----------------------------------------------------
    def charge(self, vds, vgs, temperature: float | None = None):
        """Interpolated terminal charge [C] at (vds, vgs)."""
        vds_a = np.asarray(vds, dtype=float)
        vgs_a = np.asarray(vgs, dtype=float)
        vds_b, vgs_b = np.broadcast_arrays(vds_a, vgs_a)
        q = self._slice(temperature)
        i, tx = _bracket(self.vgs, vgs_b, "vgs")
        j, ty = _bracket(self.vds, vds_b, "vds")
        q00 = q[i, j]
        q10 = q[i + 1, j]
        q01 = q[i, j + 1]
        q11 = q[i + 1, j + 1]
        value = ((1.0 - tx) * (1.0 - ty) * q00
                 + tx * (1.0 - ty) * q10
                 + (1.0 - tx) * ty * q01
                 + tx * ty * q11)
        return float(value) if np.ndim(value) == 0 else value

    def capacitance(self, vds, vgs, temperature: float | None = None):
        """dQ/dVds at fixed Vgs [F] (bilinear cell slope)."""
        vds_a = np.asarray(vds, dtype=float)
        vgs_a = np.asarray(vgs, dtype=float)
        vds_b, vgs_b = np.broadcast_arrays(vds_a, vgs_a)
        q = self._slice(temperature)
        i, tx = _bracket(self.vgs, vgs_b, "vgs")
        j, _ = _bracket(self.vds, vds_b, "vds")
        slope = ((1.0 - tx) * (q[i, j + 1] - q[i, j])
                 + tx * (q[i + 1, j + 1] - q[i + 1, j])) \
            / (self.vds[j + 1] - self.vds[j])
        return float(slope) if np.ndim(slope) == 0 else slope

    def qoss_table(self, vgs: float = 0.0, temperature: float | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
        """Qoss(vds) slice at fixed Vgs for the 1-D SPICE charge emission."""
        q = self.charge(self.vds, float(vgs), temperature=temperature)
        return self.vds.copy(), np.asarray(q, dtype=float)
