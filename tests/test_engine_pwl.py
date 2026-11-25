"""Stage-3 verification: piecewise-linear devices.

Two things are being protected here.

**The approximation is bounded, not hidden.** A PWL diode is not exponential, so
the test measures how far its table sits from the Shockley curve it replaced and
asserts the bound.  A solver that quietly differs from the physics by 50 % would
still pass a "does it run" test.

**The region fixed point terminates.** A region set that oscillates is the
classic PWL failure mode, so the tests cover a circuit where two devices could
fight over the same node.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.ir.circuit import Circuit, Element, ElementKind, NodeIndex, Waveform
from engine.ir.parse import parse_circuit
from engine.solver.device import (
    ChargeTable,
    DiodeModel,
    PwlTable,
    SwitchDevice,
)
from engine.solver.transient import solve_op, solve_transient

_KB_OVER_Q = 8.617333262e-5
_VT = _KB_OVER_Q * (27.0 + 273.15)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def test_pwl_table_selects_segments_and_clamps_at_the_ends() -> None:
    table = PwlTable((0.0, 1.0, 2.0), (0.0, 10.0, 10.0))
    assert len(table) == 2
    assert table.region(-5.0) == 0          # clamped low
    assert table.region(0.5) == 0
    assert table.region(1.5) == 1
    assert table.region(99.0) == 1          # clamped high
    assert table.slope(0) == pytest.approx(10.0)
    assert table.slope(1) == pytest.approx(0.0)   # physical end segment
    assert table.at(0, 0.25) == pytest.approx(2.5)
    assert table.at(1, 50.0) == pytest.approx(10.0)


def test_pwl_table_rejects_unsorted_points() -> None:
    with pytest.raises(ValueError):
        PwlTable.from_pairs([(0.0, 0.0), (1.0, 1.0), (1.0, 2.0)])


def test_charge_table_must_be_monotone_and_zero_anchored() -> None:
    ChargeTable(PwlTable((0.0, 1.0), (0.0, 1e-9)))            # fine
    with pytest.raises(ValueError, match="non-decreasing"):
        ChargeTable(PwlTable((0.0, 1.0), (1e-9, 0.0)))
    with pytest.raises(ValueError, match="zero-anchored"):
        ChargeTable(PwlTable((0.0, 1.0), (1e-12, 1e-9)))


# ---------------------------------------------------------------------------
# Diode
# ---------------------------------------------------------------------------

def test_diode_table_tracks_shockley_within_a_few_millivolts() -> None:
    isat, n, rs = 1e-12, 1.0, 1e-3
    table = DiodeModel.build(isat=isat, n=n, rs=rs).table
    worst = 0.0
    for current in np.geomspace(1e-9, 10.0, 40):
        v_exact = n * _VT * math.log1p(current / isat) + current * rs
        region = table.region(v_exact)
        v_pwl = (current - table.intercept(region)) / table.slope(region)
        worst = max(worst, abs(v_pwl - v_exact))
    # Measured 2.55 mV worst over 10 decades.  A voltage error of a few mV is
    # the price of linearity; 10 mV is the guard that keeps it honest.
    assert worst < 10e-3, f"PWL diode voltage error {worst * 1e3:.2f} mV"


def test_diode_forward_drop_matches_shockley_in_a_circuit() -> None:
    deck = """
* diode clamp
V1 in 0 DC 5
R1 in out 1k
D1 out 0 DM
.model DM D(IS=1e-12 N=1 RS=1e-3)
.end
"""
    circuit = parse_circuit(deck)
    result = solve_transient(circuit, end_time=50e-6, max_step=1e-7)
    v_out = float(result.voltage("out")[-1])
    current = (5.0 - v_out) / 1e3
    exact = _VT * math.log1p(current / 1e-12) + current * 1e-3
    assert abs(v_out - exact) < 10e-3, (v_out, exact)


def test_diode_blocks_in_reverse() -> None:
    deck = """
* reverse biased diode
V1 in 0 DC -10
R1 in out 1meg
D1 out 0 DM
.model DM D(IS=1e-12 N=1 RS=1e-3)
.end
"""
    result = solve_transient(parse_circuit(deck), end_time=10e-6,
                             max_step=1e-7)
    # A reverse diode must pass essentially nothing: -10 V across 1 MOhm.
    assert result.voltage("out")[-1] == pytest.approx(-10.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Switch
# ---------------------------------------------------------------------------

def _switch(threshold: float = 2.0, hysteresis: float = 0.0) -> SwitchDevice:
    return SwitchDevice("S1", (0, 1, 2, None), ron=0.1, roff=1e9,
                        vt=threshold, vh=hysteresis)


def test_switch_region_follows_the_gate() -> None:
    switch = _switch()
    x = np.zeros(3)
    x[2] = 0.0
    assert switch.region(x) == 0
    x[2] = 5.0
    assert switch.region(x) == 1
    switch.force_region(0)
    x[2] = 1.9
    assert switch.region(x) == 0


def test_hysteresis_holds_the_state_inside_the_band() -> None:
    switch = _switch(threshold=2.0, hysteresis=0.5)
    x = np.zeros(3)
    x[2] = 2.2                       # inside the band, below switch-on
    assert switch.region(x) == 0
    x[2] = 2.4                       # past vt + vh/2
    assert switch.region(x) == 1
    x[2] = 2.0                       # back inside the band, stays on
    assert switch.region(x) == 1
    x[2] = 1.6                       # below vt - vh/2
    assert switch.region(x) == 0


def test_switch_conductances_are_the_model_values() -> None:
    # A supply behind 100 ohm, switched to ground: the node voltage is the
    # divider of Ron (or Roff) against that 100 ohm.
    deck = """
* switch conduction
Vdd dd 0 DC 1
R1 dd d 100
Vg g 0 DC {gate}
S1 d 0 g 0 SW1
.model SW1 SW(Ron=0.5 Roff=1e6 Vt=2 Vh=0)
.end
"""
    on = solve_op(parse_circuit(deck.replace("{gate}", "5")))
    node = parse_circuit(deck.replace("{gate}", "5")).node_index.index("d")
    assert on[node] == pytest.approx(1.0 * 0.5 / 100.5, rel=1e-6)
    off = solve_op(parse_circuit(deck.replace("{gate}", "0")))
    assert off[node] == pytest.approx(1.0 * 1e6 / (100 + 1e6), rel=1e-6)


# ---------------------------------------------------------------------------
# Charge capacitor
# ---------------------------------------------------------------------------

def test_charge_capacitor_passes_exactly_the_charge_in_its_table() -> None:
    """Charge delivered by the source equals the charge the table stores.

    This is the invariant that matters for switching loss: a ROM's Qoss table is
    the whole model, so the solver must integrate exactly that charge and not a
    capacitance approximation of it.
    """
    table = ChargeTable(PwlTable((0.0, 2.0, 4.0), (0.0, 2e-9, 6e-9)))
    circuit = Circuit(
        title="charge cap",
        elements=(
            Element(name="V1", kind=ElementKind.VSOURCE, nodes=("in", "0"),
                    waveform=Waveform.constant(1.0)),
            Element(name="R1", kind=ElementKind.RESISTOR, nodes=("in", "out"),
                    value=1e3),
            Element(name="C1", kind=ElementKind.PWL_CAP, nodes=("out", "0"),
                    data=table),
        ),
        models={},
        node_index=NodeIndex.build(("in", "out")),
    )
    result = solve_transient(circuit, end_time=50e-6, max_step=1e-7)
    v_final = float(result.voltage("out")[-1])
    assert v_final == pytest.approx(1.0, abs=1e-3)      # settles at the source
    delivered = float(np.trapezoid(-result.current("V1"), result.get("time")))
    assert delivered == pytest.approx(table.table.at(0, v_final), rel=1e-3)


def test_charge_capacitor_expression_from_a_deck_is_refused() -> None:
    # A deck-level Q = {..} is an expression; only the ROM's table is stepped.
    from engine.envelope import NotYetImplemented

    deck = """
* charge expression
V1 in 0 DC 1
R1 in out 1k
C1 out 0 Q = {1e-9*(V(out))}
.end
"""
    with pytest.raises(NotYetImplemented):
        solve_transient(parse_circuit(deck), end_time=1e-6, max_step=1e-8)


def test_region_set_settles_for_two_devices_on_one_node() -> None:
    # A diode and a switch share a node; both regions must settle together.
    deck = """
* two pwl devices on one node
V1 in 0 PULSE(0 5 0 10n 10n 2u 4u)
R1 in out 100
D1 out 0 DM
Vg g 0 DC 5
S1 out 0 g 0 SW1
.model DM D(IS=1e-12 N=1 RS=1e-3)
.model SW1 SW(Ron=0.2 Roff=1e7 Vt=2 Vh=0.2)
.tran 0 4u 0 20n
.end
"""
    result = solve_transient(parse_circuit(deck))
    assert not result.warnings
    assert np.all(np.isfinite(result.get("V(out)")))
