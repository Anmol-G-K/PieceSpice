"""Charge-conserving capacitance modeling Q(V) = ∫ C(v) dv."""

from typing import Tuple
import numpy as np
from scipy.integrate import cumulative_trapezoid


def compute_charge_qoss(
    v_ds: np.ndarray,
    c_oss: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Integrate non-linear Coss capacitance over voltage to obtain charge Qoss(V).

    Args:
        v_ds: 1D array of drain-source voltage sweep [V]
        c_oss: 1D array of output capacitance [F]

    Returns:
        v_ds: Voltage points
        q_oss: Integrated charge array [C]
        e_oss: Total stored capacitive energy at max voltage [J]
    """
    # Integrate Q(V) = ∫ C(v) dv
    q_oss = cumulative_trapezoid(c_oss, v_ds, initial=0.0)

    # Integrate Eoss(V) = ∫ v * C(v) dv
    e_oss_curve = cumulative_trapezoid(v_ds * c_oss, v_ds, initial=0.0)
    total_e_oss = float(e_oss_curve[-1])

    return v_ds, q_oss, total_e_oss
