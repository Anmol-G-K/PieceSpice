"""Measured runtime comparison: vendor subcircuit vs Tier-2 ROM.

Two timings are recorded per run:

* ``wall_clock_s`` — full ``SpiceEngine.run_transient`` call, including
  process launch, ``.include`` inlining, netlist write and ``.raw`` parsing.
* ``solver_elapsed_s`` — LTspice's own "Total elapsed time" from the `.log`,
  i.e. the solver portion only.

The switching circuit is identical for both models. With ``max_step=None``
LTspice chooses its own timestep; that is the condition under which a smooth
reduced model can actually be faster.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np

from simulation.engine import SpiceEngine, TranDirective
from simulation.netlist_text import NetlistBuilder


@dataclass(frozen=True)
class SpeedupRun:
    """One measured simulator run."""

    wall_clock_s: float
    solver_elapsed_s: float
    iterations: int
    trace_points: int
    trace_names: int
    log_text: str = ""


def _elapsed(log: str) -> float:
    match = re.search(r"Total elapsed time:\s*([\d.]+)", log)
    return float("nan") if match is None else float(match.group(1))


def _iterations(log: str) -> int:
    match = re.search(r"Total iterations:\s*(\d+)", log)
    return -1 if match is None else int(match.group(1))




def _pulse_train(t0: float, first_width: float, width: float, period: float,
                 n: int, edge: float, vgs_on: float) -> str:
    pts = [(0.0, 0.0)]
    for k in range(n):
        w = first_width if k == 0 else width
        a = t0 + k * period
        pts += [(a - edge, 0.0), (a, vgs_on),
                (a + w - edge, vgs_on), (a + w, 0.0)]
    times = [p[0] for p in pts]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("pulse timings must be increasing")
    pts.append((times[-1] + 2e-6, 0.0))
    return " ".join(f"{t:.9g},{v:.9g}" for t, v in pts)


def run_inductive_benchmark(
    library_path,
    subcircuit_name: str,
    pin_order: list[str],
    *,
    vbus: float,
    vgs_on: float,
    i_load: float,
    l_load: float,
    pulse_width: float,
    first_pulse_width: float,
    pulse_period: float,
    n_pulses: int,
    edge_time: float,
    max_step: float | None = None,
    temperature: float = 25.0,
    repeats: int = 3,
    timeout_s: float = 90.0,
) -> dict:
    """DPT-style clamped-inductive load with an N-pulse gate train."""
    last_width = first_pulse_width if n_pulses == 1 else pulse_width
    end = 1.0e-6 + (n_pulses - 1) * pulse_period + last_width + 2e-6
    nb = NetlistBuilder(f"SPEEDUP_DPT_{subcircuit_name}")
    nb.include(library_path)
    nb.v_dc("bus", "vbus", "0", vbus)
    nb.l("load", "vbus", "drain", l_load)
    nb.raw_line("Dclamp drain vbus DCLAMP")
    nb.raw_line(".model DCLAMP D(IS=1e-12 N=1 RS=1e-3 CJO=0)")
    nb.raw_line("Vgs gate 0 PWL(" + _pulse_train(
        1e-6, first_pulse_width, pulse_width, pulse_period,
        n_pulses, edge_time, vgs_on) + ")")
    nb.x_instance("dut", subcircuit_name, pin_order, temperature=temperature)

    directive = TranDirective(step_time=max_step or edge_time, end_time=end,
                              max_step=max_step)
    engine = SpiceEngine(temperature=temperature, timeout_s=timeout_s)
    netlist = nb.render()
    runs: list[SpeedupRun] = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        result = engine.run_transient(netlist, directive)
        wall = time.perf_counter() - t0
        log = result.log_text
        runs.append(SpeedupRun(
            wall_clock_s=float(wall), solver_elapsed_s=_elapsed(log),
            iterations=_iterations(log),
            trace_points=len(result.get("time")),
            trace_names=len(result.traces), log_text=log))

    median = lambda name: float(np.median([getattr(r, name) for r in runs]))
    return {"library": str(library_path), "subcircuit": subcircuit_name,
            "n_pulses": int(n_pulses), "max_step": max_step,
            "end_time_s": end, "repeats": int(repeats),
            "wall_clock_s": median("wall_clock_s"),
            "solver_elapsed_s": median("solver_elapsed_s"),
            "iterations": median("iterations"),
            "trace_points": median("trace_points"),
            "trace_names": median("trace_names")}
def run_switching_benchmark(
    library_path,
    subcircuit_name: str,
    pin_order: list[str],
    *,
    vbus: float,
    vgs_on: float,
    i_load: float,
    cycles: int = 20,
    f_sw: float = 50_000.0,
    duty: float = 0.5,
    max_step: float | None = None,
    temperature: float = 25.0,
    repeats: int = 3,
    timeout_s: float = 90.0,
) -> dict:
    """Run golden/ROM switching circuit and return measured runtime summary.

    ``max_step=None`` emits ``.tran 0 <end> 0`` so LTspice selects the step;
    a finite value pins the step for the forced-grid control experiment.
    """
    period = 1.0 / float(f_sw)
    end = float(cycles) * period
    r_load = float(vbus) / float(i_load)
    edge = 20e-9

    nb = NetlistBuilder(f"SPEEDUP_{subcircuit_name}")
    nb.include(library_path)
    nb.v_dc("bus", "vbus", "0", vbus)
    nb.r("load", "vbus", "drain", r_load)
    nb.v_pulse("gs", "gate", "0", v1=0.0, v2=vgs_on,
               td=period * 0.25, tr=edge, tf=edge,
               pw=duty * period - 2 * edge, period=period)
    nb.x_instance("dut", subcircuit_name, pin_order, temperature=temperature)

    directive = TranDirective(step_time=max_step or edge, end_time=end,
                              max_step=max_step)
    engine = SpiceEngine(temperature=temperature, timeout_s=timeout_s)
    netlist = nb.render()
    runs: list[SpeedupRun] = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        result = engine.run_transient(netlist, directive)
        wall = time.perf_counter() - t0
        log = result.log_text
        runs.append(SpeedupRun(
            wall_clock_s=float(wall),
            solver_elapsed_s=_elapsed(log),
            iterations=_iterations(log),
            trace_points=len(result.get("time")),
            trace_names=len(result.traces),
            log_text=log))

    def median(name):
        return float(np.median([getattr(r, name) for r in runs]))

    return {
        "library": str(library_path),
        "subcircuit": subcircuit_name,
        "cycles": int(cycles),
        "max_step": max_step,
        "end_time_s": end,
        "repeats": int(repeats),
        "wall_clock_s": median("wall_clock_s"),
        "solver_elapsed_s": median("solver_elapsed_s"),
        "iterations": median("iterations"),
        "trace_points": median("trace_points"),
        "trace_names": median("trace_names"),
    }
