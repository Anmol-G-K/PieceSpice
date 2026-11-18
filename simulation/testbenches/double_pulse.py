"""Clamped-inductive double-pulse testbench (Milestone 4).

Circuit::

    Vbus -- Lload -- drain(DUT) -- source -- 0
    Dclamp from drain (anode) to vbus (cathode): clamps the drain to the bus
    while the inductor freewheels.  The gate is driven by two pulses: the
    first charges the inductor, the second is the device under test.

The engine only sees netlist text; the bench owns the topology and the
waveform extraction.  ``i_d`` is the inductor current (equal to the device
current while the channel conducts).

No conditions are hardcoded here: every voltage, current, inductance and
pulse time comes from ``config/inputs/<vendor>.yaml`` -> ``dpt``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from simulation.engine import SpiceEngine, TranDirective
from simulation.netlist_text import NetlistBuilder


@dataclass
class DoublePulseRun:
    """Raw waveforms and measured simulator wall time of one DPT run."""

    time: np.ndarray
    v_ds: np.ndarray
    i_d: np.ndarray
    v_gs: np.ndarray
    v_bus: float
    i_load: float
    l_load: float
    vgs_on: float
    wall_clock_s: float
    t2_on: float = 0.0             # second pulse starts; used for DPT metrics


def _gate_pwl(t1_on: float, t1_width: float, t2_on: float,
              t2_width: float, edge: float, vgs_on: float,
              t_end: float) -> str:
    """Two-pulse gate waveform as monotone (time, voltage) pairs."""
    pts = [
        (0.0, 0.0),
        (t1_on - edge, 0.0),
        (t1_on, vgs_on),
        (t1_on + t1_width - edge, vgs_on),
        (t1_on + t1_width, 0.0),
        (t2_on - edge, 0.0),
        (t2_on, vgs_on),
        (t2_on + t2_width - edge, vgs_on),
        (t2_on + t2_width, 0.0),
        (t_end, 0.0),
    ]
    times = [p[0] for p in pts]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("DPT pulse timings must be strictly increasing")
    return " ".join(f"{t:.9g},{v:.9g}" for t, v in pts)


def run_double_pulse(
    library_path,
    subcircuit_name: str,
    pin_order: list[str],
    *,
    vbus: float,
    vgs_on: float,
    i_load_target: float,
    l_load: float,
    t1_on: float,
    t1_width: float,
    t2_on: float,
    t2_width: float,
    edge_time: float,
    t_end: float,
    max_step: float,
    temperature: float = 25.0,
) -> DoublePulseRun:
    """Simulate the clamped-inductive DPT and return its waveforms."""
    nb = NetlistBuilder(f"DPT_{subcircuit_name}")
    nb.include(library_path)
    nb.v_dc("bus", "vbus", "0", vbus)
    nb.l("load", "vbus", "drain", l_load)
    nb.raw_line("Dclamp drain vbus DCLAMP")
    nb.raw_line(".model DCLAMP D(IS=1e-12 N=1 RS=1e-3 CJO=0)")
    nb.raw_line("Vgs gate 0 PWL(" + _gate_pwl(
        t1_on, t1_width, t2_on, t2_width, edge_time, vgs_on, t_end) + ")")
    nb.x_instance("dut", subcircuit_name, pin_order, temperature=temperature)

    directive = TranDirective(step_time=max_step, end_time=t_end,
                              max_step=max_step)
    t0 = time.perf_counter()
    result = SpiceEngine(temperature=temperature).run_transient(
        nb.render(), directive)
    elapsed = time.perf_counter() - t0

    if "V(drain)" not in result.traces or "I(Lload)" not in result.traces:
        raise KeyError(
            "DPT requires V(drain) and I(Lload); got "
            f"{sorted(result.traces)}")

    return DoublePulseRun(
        time=np.asarray(result.get("time"), dtype=float),
        v_ds=np.asarray(result.get("V(drain)"), dtype=float),
        i_d=np.abs(np.asarray(result.get("I(Lload)"), dtype=float)),
        v_gs=np.asarray(result.get("V(gate)"), dtype=float),
        v_bus=float(vbus), i_load=float(i_load_target),
        l_load=float(l_load), vgs_on=float(vgs_on),
        t2_on=float(t2_on),
        wall_clock_s=float(elapsed))


def _cross(t: np.ndarray, y: np.ndarray, level: float, *,
           rising: bool, last: bool = False) -> float | None:
    """Interpolated crossing time; None when the level is not crossed."""
    d = y - level
    idx = np.where(((d[:-1] < 0) & (d[1:] >= 0)) if rising
                   else ((d[:-1] > 0) & (d[1:] <= 0)))[0]
    if idx.size == 0:
        return None
    i = int(idx[-1] if last else idx[0])
    y0, y1 = y[i], y[i + 1]
    f = 0.0 if y1 == y0 else abs(level - y0) / abs(y1 - y0)
    return float(t[i] + f * (t[i + 1] - t[i]))


def extract_dpt_metrics(run: DoublePulseRun,
                        e_oss_uj: float | None = None) -> dict:
    """Timing, energy and slew-rate figures from a DPT waveform.

    Energy windows start at the gate edge and end at the corresponding
    10 %/90 % drain crossing plus a 100 ns tail; they are edge-delimited,
    not a fixed sample count.  Missing crossings yield NaN.
    """
    from metrics.loss import compute_switching_energy

    plateau = float(np.percentile(run.v_gs, 95))
    d_hi, d_lo = 0.9 * run.v_bus, 0.1 * run.v_bus
    tail = 100e-9

    def ns_or_nan(x):
        return float("nan") if x is None or x < 0 else float(x) * 1e9

    # The first pulse turns the device on at zero inductor current; DPT
    # switching loss uses the SECOND pulse, at full load current.
    g_on = _cross(run.time, run.v_gs, 0.1 * plateau, rising=True, last=True)
    ds90_on = _cross(run.time, run.v_ds, d_hi, rising=False, last=True)
    v_on = _cross(run.time, run.v_ds, d_lo, rising=False, last=True)
    g_off = _cross(run.time, run.v_gs, 0.9 * plateau, rising=False, last=True)
    ds10_off = _cross(run.time, run.v_ds, d_lo, rising=True, last=True)
    v_off = _cross(run.time, run.v_ds, d_hi, rising=True, last=True)
    td_on = None if g_on is None or ds90_on is None else ds90_on - g_on
    tr = None if ds90_on is None or v_on is None else v_on - ds90_on
    td_off = None if g_off is None or ds10_off is None else ds10_off - g_off
    tf = None if ds10_off is None or v_off is None else v_off - ds10_off
    second = run.time >= run.t2_on
    dt = np.diff(run.time[second])
    dvdt = float(np.max(np.abs(np.diff(run.v_ds[second]))) / max(float(np.min(dt)), 1e-15))
    didt = float(np.max(np.abs(np.diff(run.i_d[second]))) / max(float(np.min(dt)), 1e-15))

    def energy(a, b):
        if a is None or b is None or b <= a:
            return float("nan")
        return compute_switching_energy(run.time, run.v_ds, run.i_d,
                                        a, b + tail)

    return {
        "td_on_ns": ns_or_nan(td_on),
        "tr_ns": ns_or_nan(tr),
        "td_off_ns": ns_or_nan(td_off),
        "tf_ns": ns_or_nan(tf),
        "dv_dt_max_v_per_ns": float(dvdt * 1e-9),
        "di_dt_max_a_per_ns": float(didt * 1e-9),
        "e_on_uj": energy(g_on, v_on),
        "e_off_uj": energy(g_off, v_off),
        "e_oss_uj": float("nan") if e_oss_uj is None else float(e_oss_uj),
        "i_peak_a": float(np.max(run.i_d)),
        "wall_clock_s": run.wall_clock_s,
    }
