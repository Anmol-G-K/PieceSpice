"""Switching timing dynamics extraction (tr, tf, td_on, td_off)."""

from typing import Tuple
import numpy as np
from pydantic import BaseModel


class SwitchingTimeMetrics(BaseModel):
    rise_time_ns: float
    fall_time_ns: float
    delay_on_ns: float
    delay_off_ns: float
    dv_dt_max_v_per_ns: float
    di_dt_max_a_per_ns: float


def _crossing_time(t: np.ndarray, y: np.ndarray, level: float,
                   rising: bool = True):
    """First time *y* crosses *level* in the given direction, interpolated.

    Returns None when no crossing exists.
    """
    diff = (y - level)
    if rising:
        sign_change = np.where((diff[:-1] < 0) & (diff[1:] >= 0))[0]
    else:
        sign_change = np.where((diff[:-1] > 0) & (diff[1:] <= 0))[0]
    if len(sign_change) == 0:
        return None
    i = sign_change[0]
    y0, y1 = y[i], y[i + 1]
    frac = 0.0 if y1 == y0 else abs(level - y0) / abs(y1 - y0)
    return float(t[i] + frac * (t[i + 1] - t[i]))


def _ns(seconds: float) -> float:
    """Seconds -> ns; NaN survives (means 'crossing not present')."""
    if not np.isfinite(seconds):
        return float("nan")
    return max(seconds, 0.0) * 1e9


def extract_switching_times(
    time: np.ndarray,
    v_ds: np.ndarray,
    i_d: np.ndarray,
    v_gs: np.ndarray,
    i_load: float | None = None,
    v_bus: float | None = None,
) -> SwitchingTimeMetrics:
    """Extract standard dynamic switching timing figures.

    Definitions (per IEC 60747-8 conventions):
      td(on):  v_gs crosses 10% of its final plateau until v_ds falls to
               90% of its off-state value.
      tr:      v_ds from 90% down to 10% of the off-state value.
      td(off): v_gs drops to 90% of plateau until v_ds rises to 10% of bus.
      tf:      v_ds from 10% up to 90% of bus.

    A timing figure whose level crossing is absent (e.g. a golden capture
    truncated before turn-off) is returned as NaN, never 0.0: zero is a
    measured value and made the relative-error gates report absurd numbers.

    Args:
        i_load: Expected on-state load current [A]. If None, inferred as the
            median of the last quarter of i_d.
        v_bus: Off-state drain voltage [V]. If None, inferred as the max v_ds.
    """
    t = np.asarray(time, dtype=float)

    if v_bus is None:
        v_bus = float(np.max(v_ds))
    if i_load is None:
        tail = i_d[len(i_d)//4:]
        i_load = float(np.median(tail))

    # 50% duty => half the samples sit at 0 V; a high percentile gives
    # the true on-state gate level for any duty cycle.
    v_gs_plateau = float(np.percentile(v_gs, 95))
    gs_lo = 0.1 * v_gs_plateau
    gs_hi = 0.9 * v_gs_plateau

    ds_on_level = 0.1 * v_bus     # drain collapsed (device on)
    ds_off_level = 0.9 * v_bus    # drain high (device off)

    # --- turn-on event: gate rising edge ---
    t_gs_10_on = _crossing_time(t, v_gs, gs_lo, rising=True)
    t_gs_90_on = _crossing_time(t, v_gs, gs_hi, rising=True)
    t_ds_90_on = _crossing_time(t, v_ds, ds_off_level, rising=False)
    t_ds_10_on = _crossing_time(t, v_ds, ds_on_level, rising=False)

    td_on = (
        (t_ds_90_on - t_gs_10_on)
        if (t_gs_10_on is not None and t_ds_90_on is not None)
        else float("nan"))
    tr = (
        (t_ds_10_on - t_ds_90_on)
        if (t_ds_10_on is not None and t_ds_90_on is not None)
        else float("nan"))

    # --- turn-off event: use the LAST crossings (falling gate edge) ---
    def last_crossing(level, falling=True, sig=None):
        sig = v_gs if sig is None else sig
        d = sig - level
        direction = ((d[:-1] > 0) & (d[1:] <= 0)) if falling else                     ((d[:-1] < 0) & (d[1:] >= 0))
        idxs = np.where(direction)[0]
        if len(idxs) == 0:
            return None
        i = idxs[-1]
        y0, y1 = sig[i], sig[i + 1]
        frac = 0.0 if y1 == y0 else abs(level - y0) / abs(y1 - y0)
        return float(t[i] + frac * (t[i + 1] - t[i]))

    t_gs_90_off = last_crossing(gs_hi, falling=True)
    t_gs_10_off = last_crossing(gs_lo, falling=True)
    t_ds_10_off = last_crossing(ds_on_level, falling=False, sig=v_ds)
    t_ds_90_off = last_crossing(ds_off_level, falling=False, sig=v_ds)

    td_off = (
        (t_ds_10_off - t_gs_90_off)
        if (t_gs_90_off is not None and t_ds_10_off is not None)
        else float("nan"))
    tf = (
        (t_ds_90_off - t_ds_10_off)
        if (t_ds_10_off is not None and t_ds_90_off is not None)
        else float("nan"))

    ns = 1e9
    dt = np.gradient(t)
    dv_dt = np.abs(np.gradient(v_ds)) / np.maximum(dt, 1e-15)
    di_dt = np.abs(np.gradient(i_d)) / np.maximum(dt, 1e-15)

    return SwitchingTimeMetrics(
        rise_time_ns=_ns(tr),
        fall_time_ns=_ns(tf),
        delay_on_ns=_ns(td_on),
        delay_off_ns=_ns(td_off),
        dv_dt_max_v_per_ns=float(np.max(dv_dt) * ns),
        di_dt_max_a_per_ns=float(np.max(di_dt) * ns),
    )
