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
    assemble,
    assemble_static,
    commit,
    device_currents,
    read_regions,
    solve_cached,
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
def try_step(P, S, t, h, method, G, G_static, rhs, x, lu, piv, G_prev,
             cached, tmp, trial):
    """One step with the region fixed point.  0 = singular, 1 = solved."""
    for _ in range(MAX_REGION_ITERATIONS):
        assemble(P, S, t, h, method, G, G_static, rhs, trial)
        if not solve_cached(G, rhs, x, lu, piv, G_prev, cached, tmp):
            return 0
        if not read_regions(P, S, x, trial):
            return 1
    return 1


#: TR local truncation error is (h^3/12) times the third derivative; the third
#: divided difference over three intervals approximates it over 6, so the
#: estimate is LTE ~ LFACTOR * h^3 * DD3.
LTE_FACTOR = 0.5


@nb.njit(cache=True)
def dd3_error(x, hx, ht, tn, h, n, abstol, reltol):
    """Largest normalised local truncation error from the last four solutions.

    The SPICE2 control: a divided-difference estimate from history instead of a
    second (or third) solve.  Step doubling cost two extra assembles and two
    extra solves per accepted step; this costs a handful of subtractions.
    """
    worst = 0.0
    for i in range(n):
        d1a = (x[i] - hx[0, i]) / (tn - ht[0])
        d1b = (hx[0, i] - hx[1, i]) / (ht[0] - ht[1])
        d1c = (hx[1, i] - hx[2, i]) / (ht[1] - ht[2])
        d2a = (d1a - d1b) / (tn - ht[1])
        d2b = (d1b - d1c) / (ht[0] - ht[2])
        d3 = (d2a - d2b) / (tn - ht[2])
        scale = abstol + reltol * max(abs(x[i]), abs(hx[0, i]))
        err = abs(LTE_FACTOR * h * h * h * d3) / scale
        if err > worst:
            worst = err
    return worst


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
def run_transient(P, S, t_end, h_init, h_max, h_min, adaptive, method_default,
                  reltol, abstol,
                  G, G_static, rhs, x, lu, piv, G_prev, cached, tmp,
                  trial, dev_i, snap,
                  times, rows, devrows, ev_t, ev_dev, ev_from, ev_to):
    """Integrate inside one compiled call.

    Returns ``(n_samples, n_events, status)``; status 0 = complete, 1 = sample
    buffer full, 2 = step underflow.
    """
    n = G.shape[0]
    nd = dev_i.size
    switches = switch_positions(P)
    old_on = np.zeros(switches.size, dtype=np.int64)

    for i in range(n):
        rows[0, i] = x[i]
    times[0] = 0.0
    for e in range(nd):
        devrows[0, e] = 0.0
    k = 1
    t = 0.0
    h = h_init
    hx = np.zeros((4, n))
    ht = np.zeros(4)
    n_hist = 0
    steps = 0
    n_events = 0
    status = 0
    capacity = times.size

    while t < t_end - 1e-18:
        hh = h
        if h_max > 0.0 and hh > h_max:
            hh = h_max
        hh = min(hh, t_end - t)
        if hh <= 0.0:
            break
        method = 0 if steps == 0 else method_default
        _save_state(S, snap)

        if adaptive:
            if try_step(P, S, t + hh, hh, method, G, G_static, rhs, x, lu,
                        piv, G_prev, cached, tmp, trial) == 0:
                _load_state(S, snap)
                h = hh * 0.5
                if h < h_min:
                    status = 2
                    break
                continue
            err = 0.0
            if n_hist >= 3:
                err = dd3_error(x, hx, ht, t + hh, hh, n, abstol, reltol)
            if err > 1.0 and hh > h_min:
                _load_state(S, snap)
                shrink = 0.8 * err ** (-1.0 / 3.0)
                if shrink < 0.25:
                    shrink = 0.25
                h = hh * shrink
                continue
            # A switch flipping inside the step is a real discontinuity, and
            # the instant has to be located or the whole transition is placed
            # by the step grid.  The fixed-step branch has always bisected for
            # this; the adaptive branch did not, which is why the arm G7
            # measures never localised an event and why taking four times as
            # many steps left its error unchanged.
            _load_state(S, snap)
            for j in range(switches.size):
                old_on[j] = S[4][switches[j]]
            earliest = hh
            crossed = -1
            for j in range(switches.size):
                e = switches[j]
                if trial[e] == old_on[j]:
                    continue
                lo = 0.0
                hi = hh
                for _ in range(24):
                    mid = 0.5 * (lo + hi)
                    _load_state(S, snap)
                    if try_step(P, S, t + mid, mid, method, G, G_static, rhs,
                                x, lu, piv, G_prev, cached, tmp, trial) == 0:
                        break
                    if trial[e] != old_on[j]:
                        hi = mid
                    else:
                        lo = mid
                if hi < earliest:
                    earliest = hi
                    crossed = e
            if crossed >= 0 and earliest > hh * 1e-9:
                _load_state(S, snap)
                if try_step(P, S, t + earliest, earliest, method, G, G_static,
                            rhs, x, lu, piv, G_prev, cached, tmp, trial) != 0:
                    device_currents(P, S, x, earliest, trial, dev_i)
                    if k >= capacity:
                        status = 1
                        break
                    for i in range(n):
                        rows[k, i] = x[i]
                    for e in range(nd):
                        devrows[k, e] = dev_i[e]
                    times[k] = t + earliest
                    k += 1
                    if n_events < ev_t.size:
                        ev_t[n_events] = t + earliest
                        ev_dev[n_events] = crossed
                        ev_from[n_events] = S[4][crossed]
                        ev_to[n_events] = trial[crossed]
                        n_events += 1
                    commit(P, S, x, earliest, method, trial)
                    steps += 1
                    t += earliest
                    for slot in range(3, 0, -1):
                        ht[slot] = ht[slot - 1]
                        for i in range(n):
                            hx[slot, i] = hx[slot - 1, i]
                    ht[0] = t
                    for i in range(n):
                        hx[0, i] = x[i]
                    n_hist += 1
                    # Do not jump back to the ceiling across a transition:
                    # ramp away from the event the way a step controller
                    # should, or the next step straddles the same edge again.
                    h = earliest
                    continue
                _load_state(S, snap)
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
            for slot in range(3, 0, -1):
                ht[slot] = ht[slot - 1]
                for i in range(n):
                    hx[slot, i] = hx[slot - 1, i]
            ht[0] = t
            for i in range(n):
                hx[0, i] = x[i]
            n_hist += 1
            if t_end - t <= hh * 1e-9:
                t = t_end
            if err > 1e-12:
                grow = 0.9 * err ** (-1.0 / 3.0)
                if grow > 2.0:
                    grow = 2.0
                if grow < 1.0:
                    grow = 1.0
                h = hh * grow
            else:
                h = hh * 2.0
            continue

        # --- fixed step, truncated at the first switch event -----------
        for j in range(switches.size):
            old_on[j] = S[4][switches[j]]
        if try_step(P, S, t + hh, hh, method, G, G_static, rhs, x, lu, piv,
                    G_prev, cached, tmp, trial) == 0:
            _load_state(S, snap)
            h = hh * 0.5
            if h < h_min:
                status = 2
                break
            continue
        # A switch flipping inside the step is a real discontinuity, so the
        # instant is found by bisecting the step fraction rather than snapped to
        # the grid.  Table breakpoints are not treated this way: the current is
        # continuous across them and the region iteration absorbs them.
        earliest = hh
        crossed = -1
        for j in range(switches.size):
            e = switches[j]
            if trial[e] == old_on[j]:
                continue
            lo = 0.0
            hi = hh
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                _load_state(S, snap)
                if try_step(P, S, t + mid, mid, method, G, G_static, rhs, x,
                            lu, piv, G_prev, cached, tmp, trial) == 0:
                    break
                if trial[e] != old_on[j]:
                    hi = mid
                else:
                    lo = mid
            if hi < earliest:
                earliest = hi
                crossed = e
        if crossed >= 0 and earliest > hh * 1e-9:
            _load_state(S, snap)
            if try_step(P, S, t + earliest, earliest, method, G, G_static,
                        rhs, x, lu, piv, G_prev, cached, tmp, trial) == 0:
                _load_state(S, snap)
                h = hh * 0.5
                continue
            device_currents(P, S, x, earliest, trial, dev_i)
            if k >= capacity:
                status = 1
                break
            for i in range(n):
                rows[k, i] = x[i]
            for e in range(nd):
                devrows[k, e] = dev_i[e]
            times[k] = t + earliest
            k += 1
            if n_events < ev_t.size:
                ev_t[n_events] = t + earliest
                ev_dev[n_events] = crossed
                ev_from[n_events] = S[4][crossed]
                ev_to[n_events] = trial[crossed]
                n_events += 1
            commit(P, S, x, earliest, method, trial)
            steps += 1
            t += earliest
            continue
        _load_state(S, snap)
        if try_step(P, S, t + hh, hh, method, G, G_static, rhs, x, lu, piv,
                    G_prev, cached, tmp, trial) == 0:
            _load_state(S, snap)
            h = hh * 0.5
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
    """Compiled integration.

    ``max_step`` is a **ceiling** when ``adaptive`` is on (SPICE's ``.tran``
    meaning) and the exact step when it is off.
    """
    """Run the compiled loop and wrap the result for the existing consumers."""
    from engine.solver.transient import _end_time, _step_size, solve_op

    t_end = _end_time(circuit, end_time)
    system = LinearSystem.compile(circuit)
    if system.discontinuous and adaptive:
        # Event truncation lives in the fixed-step branch; a switch circuit is
        # therefore integrated at the deck's step rather than by error control.
        adaptive = False
    plan = Plan.build(system)

    started = _time.perf_counter()
    x0 = solve_op(circuit)
    plan.x[:] = x0
    _initial_state(plan, x0)

    try:
        deck_step = _step_size(circuit, t_end, None)
    except ValueError:
        deck_step = 0.0
    # ``max_step`` is a ceiling (SPICE's ``.tran`` semantics), not a fixed step:
    # the error control may take smaller steps and never larger ones.
    ceiling = max_step if max_step is not None else 0.0
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
        plan.P, plan.S, t_end, h, ceiling, h_min, adaptive,
        0 if method == "be" else 1, reltol, abstol,
        plan.G, plan.G_static, plan.rhs, plan.x, plan.lu, plan.piv,
        plan.G_prev, plan.cached, plan.tmp,
        plan.S[4].copy(), plan.dev_i, snap,
        times, rows, devrows, ev_t, ev_dev, ev_from, ev_to)
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
