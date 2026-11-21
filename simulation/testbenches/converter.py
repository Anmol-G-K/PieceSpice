"""Synchronous-buck converter bench for the emitted Tier-4 ideal switch.

The bench runs a real two-switch synchronous buck around the emitted
``.subckt`` (high-side drain=Vin/source=Sw, low-side drain=Sw/source=0) and
extracts an efficiency only from a *complete* run:

* ``Pout = Vout * Iout`` where ``Vout`` is the measured output-node mean over
  one full steady-state switching period and ``Iout`` is the measured load
  branch mean (falling back to ``Vout / Rload`` only when that trace is
  absent);
* ``Pin = Vbus * Iin`` where ``Iin`` is the bus-source branch mean over the
  same window (LTspice source-current sign convention is recorded, not
  hidden);
* ``efficiency = Pout / Pin``.

An aborted, truncated, trace-less or non-converged run returns explicit N/A
(``None`` measurement fields) with a guard record; no efficiency is inferred
from the schedule, from Vbus*Iload, or from a partial waveform.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from common.provenance import sha256_file, sha256_text
from simulation.engine import SpiceEngine, TranDirective
from simulation.netlist_text import NetlistBuilder

SCHEMA = "piecespice.converter/1.0"

# Reused completion-guard fraction from the Tier-3/DPT benches.  Count
# similarity is not a converter gate; these only reject aborted waveforms.
T_LAST_FRACTION = 0.999
MIN_TRACE_POINTS = 3
MIN_WINDOW_POINTS = 3

# Bench schedule choices (project decisions, labelled in provenance).
DEAD_TIME_FRACTION_OF_PERIOD = 0.01
MIN_DEAD_TIME_S = 20.0e-9
DEFAULT_EDGE_TIME_S = 20.0e-9

_ITERATIONS_RE = re.compile(r"Total iterations:\s*(\d+)")


def _num(value: Any) -> float | None:
    """Finite float or None; NaN/Inf are not measurements."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def extract_efficiency(vout_v: float, iout_a: float, iin_a: float,
                       vbus_v: float) -> dict[str, Any]:
    """``Pout = Vout*Iout``, ``Pin = Vbus*Iin``, efficiency from measured means.

    This helper performs no simulation and invents no fallback: any
    non-finite/non-positive input yields ``None`` measurement fields plus an
    explicit status and reason.
    """
    out: dict[str, Any] = {
        "vout_v": _num(vout_v),
        "iout_a": _num(iout_a),
        "iin_a": _num(iin_a),
        "vbus_v": _num(vbus_v),
        "pout_w": None, "pin_w": None, "efficiency": None,
        "status": "invalid_input", "reason": None,
        "rule": "Pout = Vout*Iout; Pin = Vbus*Iin; efficiency = Pout/Pin",
    }
    values = (out["vout_v"], out["iout_a"], out["iin_a"], out["vbus_v"])
    if any(value is None for value in values):
        out["reason"] = "one or more measured means are non-finite"
        return out
    vout, iout, iin, vbus = values  # type: ignore[misc]
    out["pout_w"] = vout * iout
    out["pin_w"] = vbus * iin
    if vout < 0.0 or iout < 0.0:
        out["reason"] = "output voltage/current means must be non-negative"
        return out
    if vbus <= 0.0 or iin < 0.0:
        out["reason"] = "bus voltage/current means are not physical"
        return out
    if iin == 0.0 or out["pin_w"] <= 0.0:
        out.update(status="non_positive_input_power",
                   reason="measured Pin is not positive; efficiency is N/A")
        return out
    out["efficiency"] = out["pout_w"] / out["pin_w"]
    out["status"] = "computed"
    return out


def _iterations(log: str | None) -> int | None:
    match = _ITERATIONS_RE.search(log or "")
    return int(match.group(1)) if match else None


def _pulse_checks(period: float, duty: float, dead_time: float,
                  edge_time: float) -> tuple[float, float]:
    """High/low-side PULSE widths with non-overlap dead time."""
    high_width = duty * period - 2.0 * edge_time
    low_width = (1.0 - duty) * period - 2.0 * dead_time - 2.0 * edge_time
    if high_width <= 0.0 or low_width <= 0.0:
        raise ValueError(
            "duty/edge/dead-time leave no positive pulse width "
            f"(high={high_width:g}, low={low_width:g})")
    return high_width, low_width


def _switch_line(name: str, subcircuit: str, pin_roles: Sequence[str],
                 nodes: Mapping[str, str]) -> str:
    conn: list[str] = []
    for role in pin_roles:
        key = str(role).strip().lower()
        if key not in nodes:
            raise ValueError(
                f"converter bench supports drain/gate/source switch pins; "
                f"got unknown role {role!r}")
        conn.append(nodes[key])
    return f"X{name} {' '.join(conn)} {subcircuit}"


def _trace(result: Any, name: str) -> np.ndarray | None:
    try:
        values = np.asarray(result.get(name), dtype=float)
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    return values if values.ndim == 1 else None


def _measure_window(time_trace: np.ndarray, start_s: float
                    ) -> tuple[np.ndarray | None, dict[str, Any]]:
    finite = np.isfinite(time_trace)
    mask = finite & (time_trace >= start_s)
    points = int(mask.sum())
    meta = {
        "window_s": [float(start_s), float(time_trace[finite][-1])
                     if np.any(finite) else None],
        "points": points,
        "method": ("last full switching period before the measured endpoint; "
                   "startup cycles are excluded"),
        "minimum_points": MIN_WINDOW_POINTS,
    }
    if points < MIN_WINDOW_POINTS:
        return None, {**meta, "status": "insufficient_window_points"}
    return mask, {**meta, "status": "ok"}


def run_buck_converter(
    library_path: str | Path,
    subcircuit_name: str,
    pin_roles: Sequence[str],
    *,
    vbus_v: float,
    duty: float,
    f_sw_hz: float,
    l_h: float,
    c_out_f: float,
    r_load_ohm: float,
    vgs_on_v: float,
    cycles: int = 20,
    temperature: float = 25.0,
    max_step: float | None = None,
    edge_time: float = DEFAULT_EDGE_TIME_S,
    timeout_s: float = 90.0,
    runner: Callable[[str, TranDirective], Any] | None = None,
) -> dict[str, Any]:
    """Run a synchronous buck around one emitted switch and extract efficiency.

    ``runner`` may be supplied for a synthetic/offline execution path (it must
    accept ``(netlist_text, TranDirective)`` and return an LTspice-like result
    with ``get(name)``/``traces``/``log_text``); the default constructs the
    real ``SpiceEngine``.  An incomplete run returns N/A measurement fields.
    """
    vbus = _num(vbus_v)
    rate = _num(f_sw_hz)
    duty_ratio = _num(duty)
    inductance = _num(l_h)
    capacitance = _num(c_out_f)
    load = _num(r_load_ohm)
    gate = _num(vgs_on_v)
    edge = _num(edge_time)
    conditions = {
        "vbus_v": vbus, "duty": duty_ratio, "f_sw_hz": rate,
        "l_h": inductance, "c_out_f": capacitance, "r_load_ohm": load,
        "vgs_on_v": gate, "cycles": int(cycles), "temperature_c": _num(temperature),
        "max_step_s": _num(max_step), "edge_time_s": edge,
        "subcircuit": str(subcircuit_name), "pin_roles": list(pin_roles),
    }
    base: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "invalid_input",
        "completed": False,
        "conditions": conditions,
        "f_sw_hz": rate,
        "duty": duty_ratio,
        "library": {"path": str(library_path), "available": False,
                    "sha256": None},
        "netlist_sha256": None,
        "guard": {"completed": False},
        "extraction": None,
        "vout_v": None, "iout_a": None, "iin_a": None,
        "vbus_v": vbus, "pout_w": None, "pin_w": None, "efficiency": None,
        "reason": None,
    }
    if (vbus is None or rate is None or duty_ratio is None
            or inductance is None or capacitance is None or load is None
            or gate is None or edge is None):
        base["reason"] = "all converter conditions must be finite"
        return base
    if vbus <= 0.0 or rate <= 0.0 or inductance <= 0.0 or capacitance <= 0.0 \
            or load <= 0.0 or gate <= 0.0 or edge <= 0.0:
        base["reason"] = "converter conditions must be positive"
        return base
    if not 0.0 < duty_ratio < 1.0:
        base["reason"] = "duty must be strictly between 0 and 1"
        return base

    period = 1.0 / rate
    end = float(cycles) * period
    dead_time = max(MIN_DEAD_TIME_S, DEAD_TIME_FRACTION_OF_PERIOD * period)
    try:
        high_width, low_width = _pulse_checks(
            period, duty_ratio, dead_time, edge)
        library = Path(library_path)
        base["library"]["path"] = str(library)
        base["library"]["available"] = library.is_file()
        if not library.is_file():
            base["status"] = "invalid_input"
            base["reason"] = f"library not found: {library}"
            return base
        base["library"]["sha256"] = sha256_file(library)
        nb = NetlistBuilder(f"BUCK_{subcircuit_name}")
        nb.include(library)
        nb.v_dc("bus", "vin", "0", vbus)
        nb.raw_line(_switch_line("hs", str(subcircuit_name), pin_roles, {
            "drain": "vin", "gate": "ghs", "source": "sw"}))
        nb.raw_line(_switch_line("ls", str(subcircuit_name), pin_roles, {
            "drain": "sw", "gate": "gls", "source": "0"}))
        nb.v_pulse("ghs", "ghs", "0", v1=0.0, v2=gate, td=0.0,
                   tr=edge, tf=edge, pw=high_width, period=period)
        nb.v_pulse("gls", "gls", "0", v1=0.0, v2=gate,
                   td=duty_ratio * period + dead_time,
                   tr=edge, tf=edge, pw=low_width, period=period)
        nb.l("out", "sw", "vout", inductance)
        nb.c("out", "vout", "0", capacitance)
        nb.r("load", "vout", "0", load)
        netlist = nb.render()
        base["netlist_sha256"] = sha256_text(netlist)
        base["schedule"] = {
            "period_s": period, "t_end_s": end,
            "dead_time_s": dead_time, "edge_time_s": edge,
            "high_pulse_width_s": high_width,
            "low_pulse_width_s": low_width,
            "rule": ("high/low-side PULSE widths from duty, cycles, edge "
                     "time and a project-decision non-overlap dead time"),
        }
        step_time = float(max_step) if max_step is not None else period / 100.0
        directive = TranDirective(step_time=step_time, end_time=end,
                                  max_step=max_step)
    except (ValueError, OSError) as exc:
        base["status"] = "invalid_input"
        base["reason"] = f"{type(exc).__name__}: {exc}"
        return base

    started = time.perf_counter()
    try:
        if runner is None:
            result = SpiceEngine(temperature=temperature,
                                 timeout_s=timeout_s).run_transient(
                                     netlist, directive)
        else:
            result = runner(netlist, directive)
    except Exception as exc:  # solver/backend failures are N/A, never faked
        base.update(status="error",
                    reason=f"{type(exc).__name__}: {exc}"[:300])
        return base
    wall = time.perf_counter() - started

    time_trace = _trace(result, "time")
    vout_trace = _trace(result, "V(vout)")
    iin_trace = _trace(result, "I(Vbus)")
    missing = [name for name, trace in (
        ("time", time_trace), ("V(vout)", vout_trace),
        ("I(Vbus)", iin_trace)) if trace is None]
    log = getattr(result, "log_text", "") or ""
    guard: dict[str, Any] = {
        "executed": True,
        "completed": False,
        "trace_points": 0 if time_trace is None else int(time_trace.size),
        "points": 0 if time_trace is None else int(time_trace.size),
        "t_last_s": (None if time_trace is None or not time_trace.size
                     else float(time_trace[-1])),
        "t_last": (None if time_trace is None or not time_trace.size
                   else float(time_trace[-1])),
        "t_end_s": end,
        "iterations": _iterations(log),
        "wall_clock_s": wall,
        "rule": (f"t_last >= {T_LAST_FRACTION:g}*t_end, iterations > 0, "
                 f"trace_points >= {MIN_TRACE_POINTS}, and every required "
                 "trace present"),
        "reason": None,
    }
    if missing:
        guard["reason"] = "missing required trace(s): " + ", ".join(missing)
        base.update(status="missing_trace", guard=guard,
                    reason=guard["reason"])
        return base
    if time_trace is None or vout_trace is None or iin_trace is None:
        guard["reason"] = "missing required trace(s): " + ", ".join(missing)
        base.update(status="missing_trace", guard=guard,
                    reason=guard["reason"])
        return base
    iterations = guard["iterations"]
    t_last = guard["t_last_s"]
    complete = bool(
        t_last is not None
        and t_last >= T_LAST_FRACTION * end
        and iterations is not None and iterations > 0
        and guard["trace_points"] >= MIN_TRACE_POINTS
        and vout_trace.size == time_trace.size
        and iin_trace.size == time_trace.size
    )
    guard["completed"] = complete
    if not complete:
        guard["reason"] = (
            "incomplete converter run: "
            f"t_last={t_last}, iterations={iterations}, "
            f"points={guard['trace_points']}, required t_end={end}")
        base.update(status="incomplete", guard=guard,
                    reason=guard["reason"])
        return base

    start = end - period
    mask, extraction = _measure_window(time_trace, start)
    base["extraction"] = extraction
    if mask is None:
        guard["completed"] = False
        guard["reason"] = "no complete steady-state window inside the trace"
        base.update(status="incomplete", guard=guard,
                    reason=guard["reason"])
        return base

    vout_mean = float(np.mean(vout_trace[mask]))
    iin_raw_mean = float(np.mean(iin_trace[mask]))
    # LTspice reports a delivering V-source current as negative.
    iin_mean = -iin_raw_mean if iin_raw_mean < 0.0 else iin_raw_mean
    load_trace = _trace(result, "I(Rload)")
    if load_trace is not None and load_trace.size == time_trace.size:
        iout_mean = abs(float(np.mean(load_trace[mask])))
        iout_source = "I(Rload) mean over the extraction window"
    else:
        iout_mean = vout_mean / load
        iout_source = "Vout/Rload (I(Rload) trace absent)"
    measured = extract_efficiency(vout_mean, iout_mean, iin_mean, vbus)
    base.update({
        "status": ("measured" if measured["status"] == "computed"
                   else "invalid_measurement"),
        "completed": True,
        "guard": guard,
        "extraction": {
            **extraction,
            "iout_source": iout_source,
            "iin_raw_mean_a": iin_raw_mean,
            "iin_sign_rule": ("LTspice V-source current is negative when "
                              "delivering bus power; iin = -mean when the "
                              "raw mean is negative"),
            "vout_source": "V(vout) mean over the extraction window",
            "iin_source": "I(Vbus) mean over the extraction window",
        },
        "vout_v": measured["vout_v"],
        "iout_a": measured["iout_a"],
        "iin_a": measured["iin_a"],
        "pout_w": measured["pout_w"],
        "pin_w": measured["pin_w"],
        "efficiency": measured["efficiency"],
        "reason": measured["reason"],
        "measurement_status": measured["status"],
        "rule": measured["rule"],
    })
    return base
