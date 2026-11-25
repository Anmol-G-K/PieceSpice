"""The compiled kernel must agree with the reference loop.

``engine="reference"`` is the definition: readable Python, verified against
closed form and LTspice.  ``engine="compiled"`` is the same model executed by a
numba kernel.  A fast path that disagrees with the definition is not a fast
path, so every circuit family is compared directly rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.ir.circuit import Circuit, Element, ElementKind, NodeIndex, Waveform
from engine.ir.parse import parse_circuit
from engine.solver.device import ChargeTable, PwlTable
from engine.solver.transient import solve_transient

DECKS = {
    "rc": ("* rc\nV1 in 0 PWL(0,0 1e-12,1)\nR1 in out 1k\nC1 out 0 1u\n.end",
           5e-3, 1e-5),
    "rl": ("* rl\nV1 in 0 PWL(0,0 1e-12,1)\nR1 in mid 10\nL1 mid 0 1m\n.end",
           5e-4, 1e-6),
    "rlc": ("* rlc\nV1 in 0 PWL(0,0 1e-12,1)\nL1 in a 1m\nR1 a b 10\n"
            "C1 b 0 1u\n.end", 2e-3, 1e-6),
    "diode": ("* diode\nV1 in 0 PWL(0,0 1e-12,5)\nR1 in out 1k\nD1 out 0 DM\n"
              "C1 out 0 10n\n.model DM D(IS=1e-12 N=1 RS=1e-3)\n.end",
              20e-6, 50e-9),
}


@pytest.mark.parametrize("name", sorted(DECKS))
def test_compiled_matches_reference_on_fixed_steps(name: str) -> None:
    deck, end, step = DECKS[name]
    circuit = parse_circuit(deck)
    reference = solve_transient(circuit, end_time=end, max_step=step,
                                engine="reference")
    compiled = solve_transient(circuit, end_time=end, max_step=step,
                               engine="compiled", adaptive=False)
    shared = sorted(set(reference.traces) & set(compiled.traces))
    assert shared, "the two engines produced no common traces"
    for key in shared:
        a = reference.get(key)
        b = compiled.get(key)
        assert a.size == b.size, f"{key}: {a.size} vs {b.size} samples"
        scale = max(float(np.max(np.abs(a))), 1e-12)
        error = float(np.max(np.abs(a - b)))
        assert error / scale < 1e-6, f"{key}: rel error {error / scale:.2e}"


def test_charge_capacitor_agrees_between_engines() -> None:
    table = ChargeTable(PwlTable((0.0, 2.0, 4.0), (0.0, 2e-9, 6e-9)))
    circuit = Circuit(
        title="charge cap",
        elements=(
            Element(name="V1", kind=ElementKind.VSOURCE, nodes=("in", "0"),
                    waveform=Waveform.pwl([(0.0, 0.0), (1e-9, 1.0)])),
            Element(name="R1", kind=ElementKind.RESISTOR, nodes=("in", "out"),
                    value=1e3),
            Element(name="C1", kind=ElementKind.PWL_CAP, nodes=("out", "0"),
                    data=table),
        ),
        models={},
        node_index=NodeIndex.build(("in", "out")),
    )
    reference = solve_transient(circuit, end_time=50e-6, max_step=1e-7,
                                engine="reference")
    compiled = solve_transient(circuit, end_time=50e-6, max_step=1e-7,
                               engine="compiled", adaptive=False)
    for key in ("V(out)", "I(C1)"):
        a, b = reference.get(key), compiled.get(key)
        scale = max(float(np.max(np.abs(a))), 1e-12)
        assert float(np.max(np.abs(a - b))) / scale < 1e-6, key


def test_compiled_error_control_tracks_the_reference_answer() -> None:
    deck, end, step = DECKS["rc"]
    circuit = parse_circuit(deck)
    reference = solve_transient(circuit, end_time=end, max_step=step,
                                engine="reference")
    adaptive = solve_transient(circuit, end_time=end, engine="compiled")
    # Fewer points, same answer: that is what the error control is for.
    assert adaptive.n_points < reference.n_points
    assert float(adaptive.voltage("out")[-1]) == pytest.approx(
        float(reference.voltage("out")[-1]), abs=5e-3)


def test_switch_circuits_stay_on_the_reference_loop() -> None:
    deck = """
* switch
Vdd vdd 0 DC 10
Vg g 0 PWL(0,0 100n,5 200n,5)
S1 vdd out g 0 SW1
.model SW1 SW(Ron=0.1 Roff=1e9 Vt=2 Vh=0)
Rl out 0 100
.tran 0 200n 0 3n
.end
"""
    circuit = parse_circuit(deck)
    result = solve_transient(circuit)            # default: reference
    assert len(result.events) == 1
    # Asking for the compiled engine on a switch circuit routes to the
    # reference loop rather than silently losing the event localisation.
    routed = solve_transient(circuit, engine="compiled")
    assert len(routed.events) == 1
    from engine.solver.fastloop import solve_transient_fast

    with pytest.raises(ValueError, match="switch events"):
        solve_transient_fast(circuit)
