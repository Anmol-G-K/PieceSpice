"""Switching energy loss calculation Eon, Eoff, Eoss."""

import numpy as np
from scipy.integrate import trapezoid
from pydantic import BaseModel


class SwitchingLossMetrics(BaseModel):
    e_on_uj: float
    e_off_uj: float
    total_switching_energy_uj: float


def compute_switching_energy(
    time: np.ndarray,
    v_ds: np.ndarray,
    i_d: np.ndarray,
    t_start: float,
    t_end: float
) -> float:
    """Integrate instantaneous power P(t) = v_ds(t) * i_d(t) over interval [t_start, t_end]."""
    mask = (time >= t_start) & (time <= t_end)
    if not np.any(mask):
        return 0.0

    t_sub = time[mask]
    p_inst = v_ds[mask] * i_d[mask]
    energy_joules = float(trapezoid(p_inst, t_sub))
    return energy_joules * 1e6  # Return in microjoules (uJ)
