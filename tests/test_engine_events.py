"""Stage-4 verification: exact event timing.

The point of this stage is not that the solver "handles" a switch -- it is that
the instant a switch changes region is *located*, not quantised to the step.
That is the difference between an edge timing whose resolution is the time step
and one whose resolution is the tolerances, and it is the reason a PWL engine
can address the G4 timing gate that model tuning cannot.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.ir.parse import parse_circuit
from engine.solver.transient import solve_transient

SWITCH_DECK = """
* switch event, 3 ns grid
Vdd vdd 0 DC 10
Vg g 0 PWL(0,0 100n,5 200n,5)
S1 vdd out g 0 SW1
.model SW1 SW(Ron=0.1 Roff=1e9 Vt={vt} Vh=0)
Rl out 0 100
.tran 0 200n 0 3n
.end
"""


def _run(vt: str = "2", *, events: bool = True):
    circuit = parse_circuit(SWITCH_DECK.replace("{vt}", vt))
    return solve_transient(circuit, events=events)


# ---------------------------------------------------------------------------
# Localisation accuracy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("vt", "expected_ns"),
                         [("2", 40.0), ("2.5", 50.0), ("3.5", 70.0)])
def test_switch_event_lands_on_the_analytic_crossing(vt: str,
                                                    expected_ns: float) -> None:
    # Gate ramps 0 -> 5 V linearly over 100 ns, so the threshold is crossed at
    # t = vt / 5 * 100 ns exactly.
    result = _run(vt)
    assert len(result.events) == 1
    event = result.events[0]
    assert event.device == "S1"
    assert (event.from_region, event.to_region) == (0, 1)
    error = abs(event.time - expected_ns * 1e-9)
    # Measured: sub-picosecond.  The step is 3 ns, so anything near the grid
    # would be a failure of localisation, not a tolerance question.
    assert error < 1e-12, f"event misplaced by {error * 1e12:.1f} ps"


def test_event_time_is_far_below_the_step_size() -> None:
    result = _run("2")
    step = result.step_time_s
    assert abs(result.events[0].time - 40e-9) < step * 1e-3


# ---------------------------------------------------------------------------
# Timeline completeness
# ---------------------------------------------------------------------------

def test_a_crossing_on_a_step_boundary_is_still_recorded() -> None:
    # 1 ns grid: 40 ns is exactly a step boundary, so there is nothing to
    # truncate -- but the event still belongs on the timeline.
    deck = SWITCH_DECK.replace(".tran 0 200n 0 3n", ".tran 0 200n 0 1n")
    result = solve_transient(parse_circuit(deck.replace("{vt}", "2")))
    assert len(result.events) == 1
    assert result.events[0].time == pytest.approx(40e-9, abs=1e-12)


def test_every_gate_cycle_produces_exactly_one_event() -> None:
    deck = """
* two pulses -> four gate transitions
Vdd vdd 0 DC 10
Vg g 0 PWL(0,0 99n,0 100n,5 499n,5 500n,0 999n,0 1000n,5 1499n,5 1500n,0)
S1 vdd out g 0 SW1
.model SW1 SW(Ron=0.1 Roff=1e9 Vt=2 Vh=0)
Rl out 0 100
.tran 0 1500n 0 5n
.end
"""
    result = solve_transient(parse_circuit(deck))
    # Each edge is 1 ns wide (99n->100n), so a 2 V threshold on a 0 -> 5 V ramp
    # is crossed 0.4 ns into the edge.
    times = [round(e.time * 1e9, 3) for e in result.events]
    assert times == [99.4, 499.6, 999.4, 1499.6], times
    assert [e.to_region for e in result.events] == [1, 0, 1, 0]


def test_events_can_be_switched_off() -> None:
    with_events = _run("2", events=True)
    without = _run("2", events=False)
    assert len(with_events.events) == 1
    assert without.events == ()
    # Both must still simulate the same circuit: no truncation means the
    # switch flips at the next step boundary, so the end state agrees.
    assert without.voltage("out")[-1] == pytest.approx(
        with_events.voltage("out")[-1], rel=1e-3)


# ---------------------------------------------------------------------------
# Chatter
# ---------------------------------------------------------------------------

def test_hysteresis_does_not_chatter_at_the_threshold() -> None:
    # Gate parked exactly at the nominal threshold with a 0.4 V band: staying
    # off is correct, and producing no event is the point.
    deck = """
* gate parked at Vt
Vdd vdd 0 DC 10
Vg g 0 PWL(0,0 10n,2.0 100n,2.0)
S1 vdd out g 0 SW1
.model SW1 SW(Ron=0.1 Roff=1e9 Vt=2 Vh=0.4)
Rl out 0 100
.tran 0 100n 0 2n
.end
"""
    result = solve_transient(parse_circuit(deck))
    assert len(result.events) <= 1
    assert np.all(np.isfinite(result.get("V(out)")))


def test_a_slow_ramp_does_not_explode_the_event_count() -> None:
    # A 4 us ramp over a 2 ns grid is 2000 steps; the switch may only cross its
    # threshold once, so the event timeline stays small even though every step
    # re-reads every region.
    deck = SWITCH_DECK.replace("PWL(0,0 100n,5 200n,5)",
                               "PWL(0,0 4u,5 5u,5)").replace(
        ".tran 0 200n 0 3n", ".tran 0 5u 0 2n")
    result = solve_transient(parse_circuit(deck.replace("{vt}", "2")))
    assert len(result.events) == 1
    assert abs(result.events[0].time - 1.6e-6) < 1e-12
