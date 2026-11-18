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
    engine = SpiceEngine(temperature=temperature)
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
