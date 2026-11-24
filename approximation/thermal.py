"""Foster/Cauer thermal networks for Tier-4 offline loss injection.

The emitted Tier-4 switch has no electrical loss element, so conduction and
switching losses are computed offline by :mod:`approximation.loss_lut` and
injected into a thermal network here.  This module owns *only* the network
arithmetic: every resistance, time constant and ambient temperature comes from
an explicit caller/config mapping.  A missing or malformed network raises
:class:`ValueError`; no default ambient temperature and no fitted thermal
resistance are invented.

Topologies
----------
``foster``
    A parallel bank of ``R_i`` (K/W) in series with ``tau_i = R_i * C_i``
    (s).  The step response is the standard analytic sum
    ``Tj(t) = Tamb + p * sum_i R_i * (1 - exp(-t / tau_i))``.

``cauer``
    A series R ladder from ambient to the junction with shunt C at every node
    (``C_i = tau_i / R_i``).  The step response has no simple analytic sum, so
    it is integrated with the exact matrix exponential of the small linear
    state-space model.  ``steady_state_tj`` remains ``Tamb + p * sum_i R_i``
    for both topologies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

NAN = float("nan")
TOPOLOGIES = ("foster", "cauer")


def _values(value: Any, name: str) -> tuple[float, ...]:
    """Normalize a scalar or sequence of finite floats; empty is an error."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        if isinstance(value, np.ndarray):
            raw = list(value.reshape(-1))
        elif isinstance(value, (int, float, np.integer, np.floating)):
            raw = [value]
        else:
            try:
                raw = list(value)
            except TypeError as exc:
                raise ValueError(f"thermal {name} must be a number or sequence") from exc
    else:
        raw = list(value)
    out: list[float] = []
    for item in raw:
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"thermal {name} contains a non-numeric value: {item!r}") from exc
        if not math.isfinite(number):
            raise ValueError(f"thermal {name} must be finite; got {item!r}")
        out.append(number)
    if not out:
        raise ValueError(f"thermal {name} must be non-empty")
    return tuple(out)


@dataclass(frozen=True)
class ThermalNetwork:
    """Explicit Foster or Cauer thermal network (no fitted defaults).

    ``r_th_k_per_w`` and ``tau_s`` are per-stage values; ``tau_s`` is
    ``R_i * C_i`` for either topology.  ``ambient_c`` is required because a
    fabricated ambient would silently change every junction-temperature
    result.
    """

    r_th_k_per_w: tuple[float, ...] | float
    tau_s: tuple[float, ...] | float
    ambient_c: float
    topology: str = "foster"
    source: dict[str, Any] = field(default_factory=dict)
    name: str = "thermal_network"

    def __post_init__(self) -> None:
        r_th = _values(self.r_th_k_per_w, "r_th_k_per_w")
        tau = _values(self.tau_s, "tau_s")
        if len(r_th) != len(tau):
            raise ValueError(
                "thermal r_th_k_per_w and tau_s must have equal lengths "
                f"(got {len(r_th)} and {len(tau)})")
        if any(value <= 0.0 for value in r_th):
            raise ValueError("thermal r_th_k_per_w values must be positive")
        if any(value <= 0.0 for value in tau):
            raise ValueError("thermal tau_s values must be positive")
        try:
            ambient = float(self.ambient_c)
        except (TypeError, ValueError) as exc:
            raise ValueError("thermal ambient_c must be numeric") from exc
        if not math.isfinite(ambient):
            raise ValueError("thermal ambient_c must be finite")
        topology = str(self.topology).strip().lower()
        if topology not in TOPOLOGIES:
            raise ValueError(
                f"thermal topology must be one of {TOPOLOGIES}; got {self.topology!r}")
        object.__setattr__(self, "r_th_k_per_w", r_th)
        object.__setattr__(self, "tau_s", tau)
        object.__setattr__(self, "ambient_c", ambient)
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "source", dict(self.source or {}))

    # -- derived quantities --------------------------------------------------
    @property
    def r_th_total_k_per_w(self) -> float:
        """Junction-to-ambient steady-state resistance (K/W)."""
        return float(sum(self.r_th_k_per_w))

    @property
    def c_th_f_per_k(self) -> tuple[float, ...]:
        """Per-stage thermal capacitance implied by ``tau_i = R_i * C_i``."""
        return tuple(float(tau) / float(r)
                     for r, tau in zip(self.r_th_k_per_w, self.tau_s))

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe network description (values only, no computed Tj)."""
        return {
            "name": self.name,
            "topology": self.topology,
            "r_th_k_per_w": list(self.r_th_k_per_w),
            "tau_s": list(self.tau_s),
            "c_th_f_per_k": list(self.c_th_f_per_k),
            "r_th_total_k_per_w": self.r_th_total_k_per_w,
            "ambient_c": self.ambient_c,
            "source": dict(self.source),
            "rule": ("Foster: analytic parallel-RC sum. Cauer: series-R ladder "
                     "with shunt C solved by matrix exponential. Every step "
                     "response starts at ambient; no t < 0 extrapolation."),
        }

    # -- thermal arithmetic --------------------------------------------------
    def steady_state_tj(self, p_loss_w: float) -> float:
        """``Tamb + p_loss * sum(R_th)``; NaN for a non-finite power."""
        try:
            power = float(p_loss_w)
        except (TypeError, ValueError):
            return NAN
        if not math.isfinite(power):
            return NAN
        return float(self.ambient_c) + power * self.r_th_total_k_per_w

    def step_response(self, p_loss_w: float, t: float | Sequence[float]):
        """Junction temperature after a power step applied at ``t = 0``.

        A scalar time returns ``float``; a sequence returns ``numpy.ndarray``.
        Negative/non-finite times and non-finite powers return NaN rather than
        an extrapolated temperature.
        """
        try:
            power = float(p_loss_w)
        except (TypeError, ValueError):
            power = NAN
        times, scalar = _time_values(t)
        if not math.isfinite(power):
            values = np.full(times.shape, NAN, dtype=float)
        else:
            values = self._response(power, times)
        return float(values[0]) if scalar else values

    def _response(self, power: float, times: np.ndarray) -> np.ndarray:
        out = np.full(times.shape, NAN, dtype=float)
        valid = np.isfinite(times) & (times >= 0.0)
        if not np.any(valid):
            return out
        if self.topology == "foster":
            rise = np.zeros(int(valid.sum()), dtype=float)
            for r_th, tau in zip(self.r_th_k_per_w, self.tau_s):
                rise += float(r_th) * (
                    1.0 - np.exp(-times[valid] / float(tau)))
            out[valid] = self.ambient_c + power * rise
            return out
        out[valid] = self.ambient_c + power * self._cauer_rise(times[valid])
        return out

    def _cauer_rise(self, times: np.ndarray) -> np.ndarray:
        """Exact Cauer ladder step response via the state-transition matrix.

        State ``x_i`` is the temperature rise at node ``i`` (node 0 is
        ambient).  ``C_i dx_i/dt = (x_{i-1} - x_i)/R_i
        - (x_i - x_{i+1})/R_{i+1}`` with ``x_0 = x_{n+1} = 0`` and the power
        injected at node ``n``.  ``x(t) = A^-1 (expm(A t) - I) b``.
        """
        from scipy.linalg import expm

        n = len(self.r_th_k_per_w)
        r = [float(v) for v in self.r_th_k_per_w]
        c = [float(v) for v in self.c_th_f_per_k]
        a = np.zeros((n, n), dtype=float)
        for i in range(n):
            a[i, i] = -1.0 / (r[i] * c[i])
            if i + 1 < n:
                a[i, i] -= 1.0 / (r[i + 1] * c[i])
                a[i, i + 1] = 1.0 / (r[i + 1] * c[i])
            if i > 0:
                a[i, i - 1] = 1.0 / (r[i] * c[i])
        b = np.zeros(n, dtype=float)
        b[-1] = 1.0 / c[-1]
        eye = np.eye(n)
        out = np.empty(times.size, dtype=float)
        for k, time in enumerate(times):
            transient = expm(a * float(time)) - eye
            out[k] = float(np.linalg.solve(a, transient @ b)[-1])
        return out

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], *,
                     name: str = "thermal_network",
                     source: Mapping[str, Any] | None = None) -> "ThermalNetwork":
        """Build from ``config.thermal``; missing ambient/R/tau is an error."""
        if not isinstance(mapping, Mapping):
            raise ValueError("thermal config must be a mapping")
        missing = [key for key in ("ambient_c",)
                   if key not in mapping]
        if "r_th_k_per_w" not in mapping and "r_th" not in mapping:
            missing.append("r_th_k_per_w")
        if "tau_s" not in mapping and "tau" not in mapping:
            missing.append("tau_s")
        if missing:
            raise ValueError(
                "thermal config is missing required key(s): "
                + ", ".join(missing))
        return cls(
            r_th_k_per_w=mapping.get("r_th_k_per_w", mapping.get("r_th")),
            tau_s=mapping.get("tau_s", mapping.get("tau")),
            ambient_c=mapping["ambient_c"],
            topology=str(mapping.get("topology", "foster")),
            source=dict(source or {}),
            name=name,
        )


def _time_values(t: float | Sequence[float]) -> tuple[np.ndarray, bool]:
    if isinstance(t, (int, float, np.integer, np.floating)):
        return np.asarray([float(t)], dtype=float), True
    try:
        values = np.asarray(t, dtype=float)
    except (TypeError, ValueError):
        return np.asarray([NAN], dtype=float), True
    if values.ndim == 0:
        return values.reshape(1), True
    return values.reshape(-1), False
