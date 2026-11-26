"""The transient stepping loop.

What the loop does, in order, and why:

1. **Settle the region set** (:func:`_settle`).  Assemble with the current
   piecewise-linear regions, solve, re-read the regions, repeat.  This is a
   fixed point over a *finite discrete* set, so it either settles in a few
   rounds or is chattering -- it is not Newton and it cannot diverge.  A set
   that will not settle halves the step and retries; a set that still will not
   settle raises, naming the devices.
2. **Locate discontinuities** (stage 4).  A switch changing region is a step
   change in conductance, so the instant matters.  The loop bisects on the step
   fraction until the control variable reaches the threshold, truncates the
   step exactly there, records the event, flips the region, and resumes.  Table
   breakpoints are *not* localised: the current is continuous across them, so
   the region fixed-point absorbs them and bisecting would multiply solves for
   no accuracy gain.
3. **Record, advance, continue**, on a variable time grid that includes the
   event instants.

Integration is trapezoidal after one backward-Euler start step, which is the
standard pairing: trapezoidal needs a history term that does not exist at
``t=0``, and starting with BE damps the initial numerical transient.  ``be`` is
exposed so convergence order can be measured instead of asserted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from engine.ir.circuit import Circuit, ElementKind
from engine.solver import companions
from engine.solver.companions import (
    MAX_REGION_ITERATIONS,
    DynamicState,
    LinearSystem,
)
from engine.solver.device import ChargeCapDevice, PwlDevice
from engine.solver.result import Event, SolverResult

#: Analyses the loop can run.
SUPPORTED_METHODS = companions.METHODS

#: Bisection rounds for an event instant.  24 halvings put a 2 ns step
#: resolution below 1e-16 s, far under the tolerance that matters for edge
#: timing; the loop stops early once the bracket is that tight.
EVENT_BISECTIONS = 24

#: Step-halving retries when the solve fails outright.
MAX_STEP_RETRIES = 12

#: Step-halving retries when the region set chatters (a step spanning more than
#: one table cell).  Bounded so a device parked on a breakpoint cannot stall the
#: run; past the budget the entry region is kept and reported.
MAX_REGION_RETRIES = 8


@dataclass(slots=True)
class _Trace:
    """Accumulates a variable-length trace (events make the grid non-uniform)."""

    times: list[float] = field(default_factory=list)
    rows: list[np.ndarray] = field(default_factory=list)

    def append(self, t: float, x: np.ndarray) -> None:
        self.times.append(float(t))
        self.rows.append(np.array(x, dtype=float, copy=True))

    def matrix(self) -> np.ndarray:
        return np.array(self.rows) if self.rows else np.zeros((0, 0))


def _end_time(circuit: Circuit, override: float | None) -> float:
    if override is not None:
        return float(override)
    directive = circuit.analysis("tran")
    if directive is None or directive.end_time is None:
        raise ValueError(
            "no .tran end time: pass end_time= or put a .tran card in the deck")
    return float(directive.end_time)


def _step_size(circuit: Circuit, end_time: float,
               override: float | None) -> float:
    if override is not None:
        step = float(override)
    else:
        directive = circuit.analysis("tran")
        step = 0.0
        if directive is not None:
            if directive.max_step:
                step = float(directive.max_step)
            elif directive.step_time:
                step = float(directive.step_time)
    if step <= 0.0:
        raise ValueError(
            "no step size: pass max_step= or give .tran a max step; "
            "an invented step size is an invented result")
    if step > end_time:
        raise ValueError(
            f"max step {step:g} s is longer than the run {end_time:g} s")
    return step


def _force_devices(system: LinearSystem, regions: dict[str, int]) -> None:
    """Push the trial regions into devices that carry hysteresis state."""
    for device in system.devices:
        force = getattr(device, "force_region", None)
        if force is not None and device.name in regions:
            force(regions[device.name])


def _solve_once(system: LinearSystem, state: DynamicState, *,
                t: float, h: float, method: str,
                G: np.ndarray, rhs: np.ndarray,
                regions: dict[str, int]) -> np.ndarray:
    companions.assemble(system, state, t=t, h=h, method=method,
                        G=G, rhs=rhs, regions=regions)
    return np.linalg.solve(G, rhs)


def _settle(system: LinearSystem, state: DynamicState, *,
            t: float, h: float, method: str,
            G: np.ndarray, rhs: np.ndarray,
            regions: dict[str, int]) -> tuple[np.ndarray, dict[str, int], str]:
    """Solve until the piecewise-linear region set stops changing.

    Returns ``(x, regions, status)`` where status is one of:

    ``"ok"``
        the set is a fixed point of the region map.
    ``"cycle"``
        the set alternates with the previous one.  This is the classic PWL
        breakpoint chatter: the device sits exactly on a cell boundary, so
        assembling with either neighbour lands in the other.  The regions are
        **frozen at the set that produced this solution** and reported, because
        a continuous PWL function is nearly identical on both sides of its own
        breakpoint -- the current is continuous, only the slope jumps -- so
        freezing costs a slope error, not a current error.  Raising instead
        would make a perfectly good run fail on a boundary.
    ``"fail"``
        the solve was singular or non-finite; the caller shortens the step.
    """
    entry = dict(regions)
    trial = dict(regions)
    history: list[dict[str, int]] = []
    x = np.zeros(system.size)
    for _ in range(MAX_REGION_ITERATIONS):
        _force_devices(system, trial)
        try:
            x = _solve_once(system, state, t=t, h=h, method=method,
                            G=G, rhs=rhs, regions=trial)
        except np.linalg.LinAlgError:
            return _split_failure(system, state, t=t, h=h, method=method,
                                  G=G, rhs=rhs, entry=entry, x=x)
        if not np.all(np.isfinite(x)):
            return _split_failure(system, state, t=t, h=h, method=method,
                                  G=G, rhs=rhs, entry=entry, x=x)
        updated = {d.name: d.region(x) for d in system.devices}
        if updated == trial:
            return x, trial, "ok"
        if updated in history:
            return x, trial, "cycle"
        history.append(trial)
        trial = updated
    return x, trial, "frozen"


def _split_failure(system: LinearSystem, state: DynamicState, *, t: float,
                   h: float, method: str, G: np.ndarray, rhs: np.ndarray,
                   entry: dict[str, int], x: np.ndarray
                   ) -> tuple[np.ndarray, dict[str, int], str]:
    """Last resort: re-solve with the region set the step started from.

    That set is always a valid linearisation of the device -- it is the one the
    previous step converged to -- so the step is "explicit region" rather than
    wrong.  The next step re-reads the regions, and if the state moved too far
    the local error control (or a shortened step) catches it.
    """
    _force_devices(system, entry)
    try:
        x_entry = _solve_once(system, state, t=t, h=h, method=method,
                              G=G, rhs=rhs, regions=entry)
    except np.linalg.LinAlgError:
        return x, entry, "fail"
    if not np.all(np.isfinite(x_entry)):
        return x, entry, "fail"
    return x_entry, entry, "frozen"


def solve_op(circuit: Circuit) -> np.ndarray:
    """DC operating point at ``t=0`` (capacitors open, inductors shorted).

    Returns the full unknown vector (node voltages, then branch currents).
    This is the same initial condition LTspice uses with ``uic`` off.
    """
    system = LinearSystem.compile(circuit)
    state = DynamicState.zeros(system)
    G = np.zeros((system.size, system.size))
    rhs = np.zeros(system.size)
    regions: dict[str, int] = {}
    previous: dict[str, int] = {}
    for _ in range(MAX_REGION_ITERATIONS):
        _force_devices(system, regions)
        companions.assemble_op(system, state, t=0.0, G=G, rhs=rhs,
                               regions=regions)
        x = np.linalg.solve(G, rhs)
        if not np.all(np.isfinite(x)):
            raise FloatingPointError(
                "operating point is not finite: the circuit has no DC solution "
                "(check for a floating node or a shorted source)")
        updated = {d.name: d.region(x) for d in system.devices}
        if updated == regions:
            return x
        if updated == previous:
            return x                       # breakpoint chatter: freeze
        previous = regions
        regions = updated
    raise FloatingPointError(
        "operating point region set does not settle: "
        f"{sorted(regions)}")


def _initial_state(system: LinearSystem, x0: np.ndarray,
                   regions: dict[str, int] | None = None) -> DynamicState:
    state = DynamicState.zeros(system)
    for index, stamp in enumerate(system.capacitors):
        state.vc[index] = companions.branch_voltage(stamp, x0)
        state.ic[index] = 0.0                   # open at DC
    for index, stamp in enumerate(system.inductors):
        assert stamp.branch is not None
        state.il[index] = float(x0[stamp.branch])
        state.vl[index] = 0.0                   # short at DC
    state.regions = dict(regions or {})
    for device in system.devices:
        device.reset()
    for device in system.devices:
        force = getattr(device, "force_region", None)
        if force is not None and device.name in state.regions:
            force(state.regions[device.name])
        # A charge capacitor starts at the charge implied by the operating
        # point, not at zero: zero would integrate a spurious dQ/h current on
        # the first step and bias every later sample.
        device.update(x0)
    return state


def _advance(system: LinearSystem, state: DynamicState, x: np.ndarray, *,
             h: float, method: str) -> None:
    """Roll capacitor/inductor/charge history forward to the solved step."""
    ic = companions.capacitor_current(system, state, h=h, method=method, x=x)
    for index, stamp in enumerate(system.capacitors):
        state.vc[index] = companions.branch_voltage(stamp, x)
        state.ic[index] = ic[index]
    for index, stamp in enumerate(system.inductors):
        assert stamp.branch is not None
        state.il[index] = float(x[stamp.branch])
        state.vl[index] = companions.branch_voltage(stamp, x)
    for device in system.devices:
        device.update(x)


def _locate_event(system: LinearSystem, device: PwlDevice, *,
                  state: DynamicState, t0: float, h: float,
                  to_region: int, G: np.ndarray,
                  rhs: np.ndarray) -> float | None:
    """Bisect the step fraction at which *device* crosses its threshold.

    The bracket invariant is always "``lo`` is still in the old region, ``hi``
    is already in the new one", so the direction of the threshold crossing does
    not change the algorithm -- only which side of the comparison means "new".
    Turning on is an upward crossing, turning off a downward one.

    Returns the offset in ``[0, h]`` of the first instant in the new region, or
    ``None`` when no partial solve succeeded.
    """
    threshold = getattr(device, "event_threshold", lambda: 0.0)()
    rising = to_region > 0
    lo, hi = 0.0, h
    tol = max(h * 1e-9, 1e-18)
    for _ in range(EVENT_BISECTIONS):
        mid = 0.5 * (lo + hi)
        x_mid, _, settle_status = _settle(system, state, t=t0 + mid, h=mid,
                                          method="be", G=G, rhs=rhs,
                                          regions=dict(state.regions))
        if settle_status == "fail":
            return None
        reached = device.control(x_mid) >= threshold if rising             else device.control(x_mid) <= threshold
        if reached:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol:
            break
    return hi


def solve_transient(
    circuit: Circuit,
    *,
    end_time: float | None = None,
    max_step: float | None = None,
    method: str = "trap",
    events: bool = True,
    engine: str = "reference",
    adaptive: bool | None = None,
) -> SolverResult:
    """Integrate *circuit* and return traces in the SPICE result shape.

    ``engine`` selects the executor, not the model.  ``"reference"`` (default)
    is the readable Python loop; ``"compiled"`` runs the identical companion
    models through the numba kernel.  The reference stays the default until the
    compiled path has passed the same equivalence tests everywhere -- a fast
    path that disagrees with the definition is not a fast path.

    ``max_step`` pins a fixed step (the SPICE meaning, and what every recorded
    comparison uses).  Without it the step is chosen by local-error control;
    the deck's ``.tran`` max step then only seeds the first step.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unknown integration method {method!r}")
    use_adaptive = (max_step is None) if adaptive is None else adaptive
    if engine == "compiled":
        from engine.solver.fastloop import solve_transient_fast

        return solve_transient_fast(circuit, end_time=end_time,
                                    max_step=max_step, method=method,
                                    adaptive=use_adaptive)
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unknown integration method {method!r}")
    t_end = _end_time(circuit, end_time)
    h_nominal = _step_size(circuit, t_end, max_step)

    system = LinearSystem.compile(circuit)
    size = system.size
    if size == 0:
        raise ValueError("circuit has no unknowns")

    started = time.perf_counter()
    x = solve_op(circuit)
    regions = {d.name: d.region(x) for d in system.devices}
    state = _initial_state(system, x, regions)

    node_names = system.structure.node_index.names
    branch_stamps = [s for s in system.structure.stamps if s.branch is not None]
    trace = _Trace()
    trace.append(0.0, x)

    G = np.zeros((size, size))
    rhs = np.zeros(size)
    warnings: list[str] = []
    events_out: list[Event] = []
    retries = 0
    chatter_cycles = 0
    chatter_frozen = 0
    region_retries = 0
    step_scale = 1.0
    t = 0.0
    step_index = 0
    max_steps = int(50 * t_end / h_nominal) + 100

    while t < t_end and step_index < max_steps:
        step_index += 1
        h = min(h_nominal * step_scale, t_end - t)
        if h <= 0.0:
            break
        step_method = "be" if step_index == 1 else method

        x_new, new_regions, status = _settle(
            system, state, t=t + h, h=h, method=step_method,
            G=G, rhs=rhs, regions=regions)
        ok = status != "fail"
        # A chattering region set is usually a step that spans more than one
        # table cell -- the fix is a shorter step, not a different answer.  Only
        # when halving stops helping is the entry region accepted (and
        # reported), which keeps a boundary event from failing a good run.
        if status in ("cycle", "frozen"):
            if step_scale > 1e-6 and region_retries < MAX_REGION_RETRIES:
                step_scale *= 0.5
                region_retries += 1
                chatter_cycles += 1
                continue
            chatter_frozen += 1
        if not ok:
            if step_scale > 1e-9 and retries < MAX_STEP_RETRIES:
                step_scale *= 0.5
                retries += 1
                continue
            raise FloatingPointError(
                "region set will not settle at t="
                f"{t + h:g} s; chattering devices: "
                f"{sorted(regions)}")

        # Locate discontinuities inside this step and truncate to the first one.
        boundary_events: list[tuple[float, PwlDevice, int, int]] = []
        if events:
            first: tuple[float, PwlDevice, int, int] | None = None
            for device in system.discontinuous:
                old = regions.get(device.name, 0)
                new = new_regions.get(device.name, old)
                if old == new:
                    continue
                offset = _locate_event(
                    system, device, state=state, t0=t, h=h,
                    to_region=new, G=G, rhs=rhs)
                if offset is None:
                    continue
                if offset >= h - h * 1e-6:
                    # The crossing sits on the step boundary: nothing to
                    # truncate, but it is still an event on the timeline.
                    boundary_events.append((t + h, device, old, new))
                    continue
                if first is None or offset < first[0]:
                    first = (offset, device, old, new)
            if first is not None:
                offset, device, old, new = first
                if offset > h * 1e-9:
                    x_cross, cross_regions, cross_status = _settle(
                        system, state, t=t + offset, h=offset,
                        method=step_method, G=G, rhs=rhs, regions=regions)
                    if cross_status in ("cycle", "frozen"):
                        chatter_cycles += 1
                    if cross_status != "fail":
                        _advance(system, state, x_cross, h=offset,
                                 method=step_method)
                        trace.append(t + offset, x_cross)
                        events_out.append(Event(t + offset, device.name,
                                                old, new))
                        t += offset
                        regions = dict(cross_regions)
                        force = getattr(device, "force_region", None)
                        if force is not None:
                            force(new)
                        regions[device.name] = new
                        state.regions = dict(regions)
                        continue

        step_scale = 1.0                       # a settled step restores the grid
        region_retries = 0
        _advance(system, state, x_new, h=h, method=step_method)
        t_next = t + h
        if t_end - t_next <= h * 1e-9:
            t_next = t_end                     # never add a float-drift step
        trace.append(t_next, x_new)
        for when, device, old, new in boundary_events:
            events_out.append(Event(min(when, t_next), device.name, old, new))
        t = t_next
        regions = dict(new_regions)
        state.regions = dict(regions)

    elapsed = time.perf_counter() - started
    if retries:
        warnings.append(
            f"{retries} step(s) were shortened to settle the region set")
    if chatter_cycles:
        warnings.append(
            f"{chatter_cycles} step(s) were shortened to settle a chattering "
            "region set")
    if chatter_frozen:
        warnings.append(
            f"{chatter_frozen} step(s) kept the step's entry region after "
            "the retry budget")
    if step_index >= max_steps:
        warnings.append(
            f"step budget exhausted at t={t:g} s of {t_end:g} s")

    matrix = trace.matrix()
    times = np.asarray(trace.times)
    traces: dict[str, np.ndarray] = {"time": times}
    for i, name in enumerate(node_names):
        traces[f"V({name})"] = matrix[:, i]
    for stamp in branch_stamps:
        assert stamp.branch is not None
        traces[f"I({stamp.name})"] = matrix[:, stamp.branch]
    for name, values in _device_current_traces(system, matrix, times).items():
        traces[f"I({name})"] = values

    return SolverResult(
        traces=traces,
        sweep_var="time",
        log_text="",
        warnings=warnings,
        steps=step_index,
        solver_time_s=elapsed,
        method=method,
        step_time_s=h_nominal,
        events=tuple(events_out),
    )


def _device_current_traces(system: LinearSystem, matrix: np.ndarray,
                           times: np.ndarray) -> dict[str, np.ndarray]:
    """Terminal current of every piecewise-linear device, sample by sample.

    Needed for the bench identity ``I(Lload) - I(Dclamp)``: without the clamp
    diode's current the device current of a solver arm is not the same quantity
    the golden arm reports.  A charge capacitor is integrated from its own
    charge history rather than read pointwise, because that is the only way its
    current is defined.
    """
    out: dict[str, np.ndarray] = {}
    for device in system.devices:
        if isinstance(device, ChargeCapDevice):
            table = device.charge.table
            voltages = np.array([device.voltage(row) for row in matrix])
            charges = np.array([table.at(table.region(v), v) for v in voltages])
            current = np.zeros(len(times))
            if len(times) > 1:
                current[1:] = np.diff(charges) / np.diff(times)
            out[device.name] = current
        else:
            out[device.name] = np.array([device.current(row) for row in matrix])
    return out


def branch_current_traces(circuit: Circuit) -> tuple[str, ...]:
    """Names of elements that produce an ``I(...)`` trace."""
    return tuple(e.name for e in circuit.elements
                 if e.kind in (ElementKind.VSOURCE, ElementKind.INDUCTOR))


__all__ = [
    "SUPPORTED_METHODS",
    "Event",
    "branch_current_traces",
    "solve_op",
    "solve_transient",
]
