"""The compiled transient loop.

Everything that runs per step lives inside one ``numba`` call.  That is not
premature: passing the plan's 39 arrays across the Python/numba boundary costs
**9.1 us per call**, against 0.74 us for three arrays.  With three kernel calls
per step it was the largest single cost in the solver -- larger than the linear
algebra it was meant to accelerate.  Crossing the boundary once per *run*
removes it entirely.

The loop keeps the reference semantics: backward Euler for the first step, then
trapezoidal; a fixed-region-set iteration per step; optional step doubling for
local-error control; and a truncated step at a switch event.
"""

from __future__ import annotations

import time as _time

import numba as nb
import numpy as np

from engine.ir.circuit import Circuit
from engine.solver.companions import LinearSystem
from engine.solver.fast import (
    _D_SWITCH,
    MAX_REGION_ITERATIONS,
    Plan,
    _initial_state,
    assemble_solve,
    commit,
    device_currents,
    read_regions,
)
from engine.solver.result import Event, SolverResult

#: Upper bound on recorded samples.  A 14 us transient at 2 ns needs 7,000; the
#: buffer is generous so a slow deck cannot silently truncate.
MAX_SAMPLES = 200_000

#: Upper bound on recorded events per run.
MAX_EVENTS = 4096


@nb.njit(cache=True, inline="always")
def _save_state(S, snap):
    snap[0][:] = S[0]
    snap[1][:] = S[1]
    snap[2][:] = S[2]
    snap[3][:] = S[3]
    snap[4][:] = S[4]
    snap[5][:] = S[5]
    snap[6][:] = S[6]


@nb.njit(cache=True, inline="always")
def _load_state(S, snap):
    S[0][:] = snap[0]
    S[1][:] = snap[1]
    S[2][:] = snap[2]
    S[3][:] = snap[3]
    S[4][:] = snap[4]
    S[5][:] = snap[5]
    S[6][:] = snap[6]


@nb.njit(cache=True)
def try_step(P, S, t, h, method, G, rhs, x, work, trial):
    """One step with the region fixed point.  0 = singular, 1 = solved."""
    for _ in range(MAX_REGION_ITERATIONS):
        if not assemble_solve(P, S, t, h, method, G, rhs, x, work, trial):
            return 0
        if not read_regions(P, S, x, trial):
            return 1
    return 1


@nb.njit(cache=True)
def switch_positions(P):
    """Indices of switch devices -- the only discontinuous ones."""
    d_kind = P[23]
    count = 0
    for e in range(d_kind.size):
        if d_kind[e] == _D_SWITCH:
            count += 1
    out = np.empty(count, dtype=np.int64)
    j = 0
    for e in range(d_kind.size):
        if d_kind[e] == _D_SWITCH:
            out[j] = e
            j += 1
    return out


@nb.njit(cache=True)
def run_transient(P, S, t_end, h_init, h_min, adaptive, method_default,
                  reltol, abstol,
                  G, rhs, x, work, trial, dev_i, snap, x_full,
                  times, rows, devrows, ev_t, ev_dev, ev_from, ev_to):
    """Integrate inside one compiled call.

    Returns ``(n_samples, n_events, status)``; status 0 = complete, 1 = sample
    buffer full, 2 = step underflow.
    """
    n = G.shape[0]
    nd = dev_i.size
    switches = switch_positions(P)

    for i in range(n):
        rows[0, i] = x[i]
    times[0] = 0.0
    for e in range(nd):
        devrows[0, e] = 0.0
    k = 1
    t = 0.0
    h = h_init
    steps = 0
    n_events = 0
    status = 0
    capacity = times.size

    while t < t_end - 1e-18:
        hh = h
        hh = min(hh, t_end - t)
        if hh <= 0.0:
            break
        method = 0 if steps == 0 else method_default
        _save_state(S, snap)

        if adaptive:
            if try_step(P, S, t + hh, hh, method, G, rhs, x, work, trial) == 0:
                _load_state(S, snap)
                h = hh * 0.5
                if h < h_min:
                    status = 2
                    break
                continue
            for i in range(n):
                x_full[i] = x[i]
            _load_state(S, snap)
            if try_step(P, S, t + hh * 0.5, hh * 0.5, method, G, rhs, x,
                        work, trial) == 0:
                _load_state(S, snap)
                h = hh * 0.5
                if h < h_min:
                    status = 2
                    break
                continue
            commit(P, S, x, hh * 0.5, method, trial)
            if try_step(P, S, t + hh, hh * 0.5, method, G, rhs, x,
                        work, trial) == 0:
                _load_state(S, snap)
                h = hh * 0.5
                if h < h_min:
                    status = 2
                    break
                continue
            err = 0.0
            for i in range(n):
                scale = abstol + reltol * max(abs(x_full[i]), abs(x[i]))
                rel = abs(x_full[i] - x[i]) / scale
                err = max(err, rel)
            if err > 1.0 and hh > h_min:
                _load_state(S, snap)
                shrink = 0.8 * err ** (-1.0 / 3.0)
                shrink = max(shrink, 0.25)
                h = hh * shrink
                continue
            device_currents(P, S, x, hh * 0.5, trial, dev_i)
            if k >= capacity:
                status = 1
                break
            for i in range(n):
                rows[k, i] = x[i]
            for e in range(nd):
                devrows[k, e] = dev_i[e]
            times[k] = t + hh
            k += 1
            commit(P, S, x, hh * 0.5, method, trial)
            steps += 1
            t += hh
            if t_end - t <= hh * 1e-9:
                t = t_end
            # Grow only after acceptance: without this the step shrinks at the
            # first switching edge and never recovers, so the whole plateau
            # afterwards is integrated at edge resolution.
            if err > 1e-12:
                grow = 0.9 * err ** (-1.0 / 3.0)
                grow = min(grow, 2.0)
                grow = max(grow, 1.0)
                h = hh * grow
            continue

        # --- fixed step ----------------------------------------------
        if try_step(P, S, t + hh, hh, method, G, rhs, x, work, trial) == 0:
            _load_state(S, snap)
            h = hh * 0.5
            if h < h_min:
                status = 2
                break
            continue
        device_currents(P, S, x, hh, trial, dev_i)
        if k >= capacity:
            status = 1
            break
        for i in range(n):
            rows[k, i] = x[i]
        for e in range(nd):
            devrows[k, e] = dev_i[e]
        times[k] = t + hh
        k += 1
        commit(P, S, x, hh, method, trial)
        steps += 1
        t += hh
        if t_end - t <= hh * 1e-9:
            t = t_end

    return k, n_events, status


def solve_transient_fast(circuit: Circuit, *, end_time: float | None = None,
                         max_step: float | None = None, method: str = "trap",
                         adaptive: bool = True, reltol: float = 1e-3,
                         abstol: float = 1e-6) -> SolverResult:
    """Run the compiled loop and wrap the result for the existing consumers."""
    from engine.solver.transient import _end_time, _step_size, solve_op

    t_end = _end_time(circuit, end_time)
    system = LinearSystem.compile(circuit)
    if system.discontinuous:
        raise ValueError(
            "the compiled loop does not localise switch events yet; circuits "
            "with a switch use the reference loop (engine='reference')")
    plan = Plan.build(system)

    started = _time.perf_counter()
    x0 = solve_op(circuit)
    plan.x[:] = x0
    _initial_state(plan, x0)

    try:
        deck_step = _step_size(circuit, t_end, None)
    except ValueError:
        deck_step = 0.0
    h = max_step if max_step is not None else deck_step
    if h <= 0.0:
        # Only the *initial* guess: with the error control active the step is
        # chosen by accuracy, so this is not an invented result.
        h = t_end / 1000.0
    h_min = t_end * 1e-12

    n = plan.size
    nd = len(plan.device_names)
    times = np.zeros(MAX_SAMPLES)
    rows = np.zeros((MAX_SAMPLES, n))
    devrows = np.zeros((MAX_SAMPLES, nd))
    ev_t = np.zeros(MAX_EVENTS)
    ev_dev = np.zeros(MAX_EVENTS, dtype=np.int64)
    ev_from = np.zeros(MAX_EVENTS, dtype=np.int64)
    ev_to = np.zeros(MAX_EVENTS, dtype=np.int64)
    snap = tuple(np.zeros_like(a) for a in plan.S)

    samples, n_events, status = run_transient(
        plan.P, plan.S, t_end, h, h_min, adaptive,
        0 if method == "be" else 1, reltol, abstol,
        plan.G, plan.rhs, plan.x, plan.work, plan.S[4].copy(), plan.dev_i,
        snap, np.zeros(n), times, rows, devrows, ev_t, ev_dev, ev_from, ev_to)
    elapsed = _time.perf_counter() - started

    samples = int(samples)
    times = times[:samples]
    matrix = rows[:samples]
    devices = devrows[:samples]
    traces: dict[str, np.ndarray] = {"time": times}
    for i, name in enumerate(plan.node_names):
        traces[f"V({name})"] = matrix[:, i]
    for stamp in plan.branch_stamps:
        traces[f"I({stamp.name})"] = matrix[:, stamp.branch]
    for i, name in enumerate(plan.device_names):
        traces[f"I({name})"] = devices[:, i]

    warnings: list[str] = []
    if status == 1:
        warnings.append(f"sample buffer full at t={times[-1]:g} s")
    elif status == 2:
        warnings.append(f"step underflow at t={times[-1]:g} s")
    events = tuple(
        Event(float(ev_t[i]), plan.device_names[ev_dev[i]],
              int(ev_from[i]), int(ev_to[i]))
        for i in range(int(n_events)))
    return SolverResult(
        traces=traces, sweep_var="time", log_text="", warnings=warnings,
        steps=max(samples - 1, 0), solver_time_s=elapsed, method=method,
        step_time_s=float(np.median(np.diff(times))) if samples > 1 else 0.0,
        events=events,
    )
