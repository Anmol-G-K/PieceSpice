"""Stage-2 verification: the MNA transient kernel against physics and LTspice.

Three independent references, because agreement with only one of them proves
little:

1. **Closed-form** solutions of the circuits being integrated.  This catches
   sign errors and wrong companion models, which a solver-vs-solver comparison
   can miss if both share the same textbook.
2. **Physical invariants** -- charge delivered, charge stored, energy
   dissipated in the resistor.  These catch formulations that reproduce a
   waveform but not the conservation law behind it.
3. **LTspice on a byte-identical deck.**  This is the reference the speed and
   fidelity claims will be measured against, so the engine must agree with it
   at discretisation noise, not at model-error scale.

The integration order is *measured*, not asserted from the textbook: an
implementation that silently degrades to first order still passes a loose
tolerance.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from engine.envelope import NotYetImplemented, UnsupportedConstruct
from engine.ir.parse import parse_circuit
from engine.solver.result import SolverResult
from engine.solver.transient import solve_op, solve_transient

RC_DECK = """
* rc step, tau = 1 ms
V1 in 0 PWL(0,0 1e-12,1)
R1 in out 1k
C1 out 0 1u
.end
"""

RL_DECK = """
* rl step, tau = 100 us
V1 in 0 PWL(0,0 1e-12,1)
R1 in mid 10
L1 mid 0 1m
.end
"""

RLC_DECK = """
* series rlc, zeta = 0.158
V1 in 0 PWL(0,0 1e-12,1)
L1 in a 1m
R1 a b 10
C1 b 0 1u
.end
"""

LADDER_DECK = """
* linear ladder for the LTspice cross-check
V1 in 0 PWL(0,0 1e-12,1)
R1 in a 1k
C1 a 0 100n
R2 a b 1k
C2 b 0 47n
R3 b 0 10k
.end
"""


def _rc_exact(t: np.ndarray, *, v: float = 1.0, tau: float = 1e-3) -> np.ndarray:
    out = v * (1.0 - np.exp(-t / tau))
    out[t < 1e-12] = 0.0
    return out


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------

def test_solver_result_duck_types_the_spice_result() -> None:
    from simulation.engine import SimResult

    contract = {"traces", "sweep_var", "log_text", "warnings"}
    assert contract <= set(SolverResult.__dataclass_fields__)
    assert contract <= set(SimResult.__dataclass_fields__)


def test_solver_result_rejects_misaligned_traces() -> None:
    with pytest.raises(ValueError, match="inconsistent lengths"):
        SolverResult(traces={"time": np.zeros(3), "V(a)": np.zeros(4)})
    with pytest.raises(ValueError, match="sweep variable"):
        SolverResult(traces={"V(a)": np.zeros(3)}, sweep_var="time")


# ---------------------------------------------------------------------------
# Closed-form references
# ---------------------------------------------------------------------------

def test_rc_step_matches_closed_form() -> None:
    result = solve_transient(parse_circuit(RC_DECK),
                             end_time=5e-3, max_step=1e-5)
    t = result.get("time")
    assert result.steps == 500
    assert result.n_points == 501
    error = np.max(np.abs(result.voltage("out") - _rc_exact(t)))
    assert error < 1e-4, f"RC step error {error:.3e}"


def test_rc_starts_from_the_operating_point_not_from_zero_volts() -> None:
    # The source is 0 V at t=0, so the DC operating point is uncharged; a
    # non-zero v(0) here would offset the whole trace.
    deck = RC_DECK.replace("PWL(0,0 1e-12,1)", "DC 1")
    result = solve_transient(parse_circuit(deck), end_time=1e-3, max_step=1e-5)
    assert result.voltage("out")[0] == pytest.approx(1.0)
    assert np.allclose(result.voltage("out"), 1.0)


def test_rl_step_matches_closed_form_and_sign() -> None:
    result = solve_transient(parse_circuit(RL_DECK),
                             end_time=5e-4, max_step=1e-6)
    t = result.get("time")
    exact = np.where(t < 1e-12, 0.0, (1.0 - np.exp(-t / 1e-4)) / 10.0)
    error = np.max(np.abs(result.current("L1") - exact))
    assert error < 1e-3, f"RL step error {error:.3e}"
    # 5 tau: i = (V/R) * (1 - exp(-5)), not the steady state.
    assert result.current("L1")[-1] == pytest.approx(
        0.1 * (1.0 - np.exp(-5.0)), rel=1e-6)


def test_rlc_underdamped_peak_matches_closed_form() -> None:
    result = solve_transient(parse_circuit(RLC_DECK),
                             end_time=2e-3, max_step=1e-6)
    t = result.get("time")
    w0 = 1.0 / np.sqrt(1e-3 * 1e-6)
    zeta = 10.0 / 2.0 * np.sqrt(1e-6 / 1e-3)
    wd = w0 * np.sqrt(1.0 - zeta ** 2)
    exact = np.where(
        t < 1e-12, 0.0,
        1.0 - np.exp(-zeta * w0 * t) * (
            np.cos(wd * t) + zeta / np.sqrt(1.0 - zeta ** 2) * np.sin(wd * t)))
    v = result.voltage("b")
    assert np.max(np.abs(v - exact)) < 1e-3
    assert v.max() == pytest.approx(exact.max(), rel=1e-3)


# ---------------------------------------------------------------------------
# Physical invariants
# ---------------------------------------------------------------------------

def test_charge_and_energy_balance_in_an_rc_charge() -> None:
    # 1 V step into 1 kOhm + 1 uF: the source supplies Q = C*V = 1 uC, the
    # capacitor stores C*V^2/2 and the resistor dissipates exactly as much.
    result = solve_transient(parse_circuit(RC_DECK),
                             end_time=2e-2, max_step=1e-6)
    t = result.get("time")
    v_in = result.voltage("in")
    v_out = result.voltage("out")
    i_supply = -result.current("V1")           # delivered by the source
    delivered = np.trapezoid(v_in * i_supply, t)
    stored = 0.5 * 1e-6 * v_out[-1] ** 2
    dissipated = np.trapezoid(i_supply ** 2 * 1e3, t)
    assert delivered == pytest.approx(1e-6, rel=2e-3)          # C*V
    assert stored == pytest.approx(0.5e-6, rel=2e-3)           # CV^2/2
    assert dissipated == pytest.approx(0.5e-6, rel=2e-2)       # the other half
    assert delivered == pytest.approx(stored + dissipated, rel=2e-2)


def test_inductor_current_is_continuous_across_a_source_edge() -> None:
    # L current cannot jump; the capacitor voltage cannot either.  A companion
    # model that violates this still looks like a plausible waveform: the first
    # step would carry a large fraction of the steady-state current.
    result = solve_transient(parse_circuit(RLC_DECK),
                             end_time=1e-4, max_step=1e-7)
    i_l = result.current("L1")
    assert i_l[0] == pytest.approx(0.0, abs=1e-12)
    assert abs(i_l[1]) < 0.1 * np.max(np.abs(i_l)), (
        f"inductor current jumped to {i_l[1]:.3e} on the first step")
    assert result.voltage("b")[0] == pytest.approx(0.0, abs=1e-12)
    # The current must climb monotonically over the rise, with no staircase.
    assert np.all(np.diff(i_l[:5]) > 0.0)


# ---------------------------------------------------------------------------
# Integration order, measured
# ---------------------------------------------------------------------------

def _rc_error(step: float, method: str) -> float:
    result = solve_transient(parse_circuit(RC_DECK), end_time=5e-3,
                             max_step=step, method=method)
    t = result.get("time")
    err = np.abs(result.voltage("out") - _rc_exact(t))
    return float(err[t > 10 * step].max())


def test_trapezoidal_is_second_order_and_be_is_first_order() -> None:
    for method, expected in (("be", 2.0), ("trap", 4.0)):
        coarse = _rc_error(5e-5, method)
        fine = _rc_error(2.5e-5, method)
        ratio = coarse / fine
        assert ratio == pytest.approx(expected, rel=0.35), (
            f"{method}: halving the step changed the error {ratio:.2f}x, "
            f"expected ~{expected}x")


def test_trapezoidal_beats_backward_euler_at_the_same_step() -> None:
    assert _rc_error(5e-5, "trap") < 0.1 * _rc_error(5e-5, "be")


# ---------------------------------------------------------------------------
# Deck reading regressions that made a run look plausible and wrong
# ---------------------------------------------------------------------------

def test_tran_suffixes_are_honoured() -> None:
    circuit = parse_circuit(
        "* suffixes\nV1 in 0 DC 1\nR1 in 0 1k\n.tran 0 200u 0 0.1u\n.end")
    directive = circuit.analysis("tran")
    assert directive is not None
    assert directive.end_time == pytest.approx(200e-6)
    assert directive.max_step == pytest.approx(0.1e-6)


def test_pwl_suffixes_are_honoured() -> None:
    circuit = parse_circuit(
        "* pwl suffixes\nV1 in 0 PWL(0 0 1u 5 2u 5)\nR1 in 0 1k\n.end")
    waveform = circuit.element("V1").waveform
    assert waveform is not None
    assert waveform.points == ((0.0, 0.0), (1e-6, 5.0), (2e-6, 5.0))


def test_uic_is_rejected_rather_than_ignored() -> None:
    with pytest.raises(UnsupportedConstruct):
        parse_circuit("* uic\nV1 in 0 DC 1\nR1 in 0 1k\n.tran 0 1m 0 1u uic\n.end")


# ---------------------------------------------------------------------------
# Fail-closed behaviour
# ---------------------------------------------------------------------------

def test_behavioural_expressions_are_still_refused() -> None:
    # A G.. VALUE={..} from a deck is never evaluated: the ROM lowers it to a
    # table instead.  Diode/switch/charge tables are implemented; a raw SPICE
    # expression is not, and must not be mistaken for one.
    deck = """
* behavioural source from a deck
V1 in 0 DC 1
R1 in out 1k
G1 out 0 VALUE = {V(in)*1e-3}
.end
"""
    with pytest.raises(NotYetImplemented):
        solve_transient(parse_circuit(deck), end_time=1e-3, max_step=1e-6)


def test_missing_step_size_fails_instead_of_inventing_one() -> None:
    deck = "* no tran\nV1 in 0 DC 1\nR1 in 0 1k\n.end"
    with pytest.raises(ValueError, match="no \\.tran end time"):
        solve_transient(parse_circuit(deck))
    with pytest.raises(ValueError, match="no step size"):
        solve_transient(parse_circuit(deck), end_time=1e-3)


def test_zero_resistance_is_rejected_not_silently_shorted() -> None:
    deck = "* short\nV1 in 0 DC 1\nR1 in 0 0\n.end"
    with pytest.raises(ValueError, match="zero or negative resistance"):
        solve_transient(parse_circuit(deck), end_time=1e-3, max_step=1e-6)


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="integration method"):
        solve_transient(parse_circuit(RC_DECK), end_time=1e-3,
                        max_step=1e-5, method="gear")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_repeated_runs_are_bit_identical() -> None:
    circuit = parse_circuit(LADDER_DECK)
    first = solve_transient(circuit, end_time=1e-4, max_step=1e-7)
    second = solve_transient(parse_circuit(LADDER_DECK), end_time=1e-4,
                             max_step=1e-7)
    for name in ("V(in)", "V(a)", "V(b)", "I(V1)"):
        assert np.array_equal(first.get(name), second.get(name)), name


def test_operating_point_is_finite_and_matches_the_transient_start() -> None:
    circuit = parse_circuit(LADDER_DECK)
    x0 = solve_op(circuit)
    result = solve_transient(circuit, end_time=1e-4, max_step=1e-7)
    assert np.all(np.isfinite(x0))
    assert result.voltage("a")[0] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# LTspice on a byte-identical deck
# ---------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("PIECESPICE_LIVE", "1") != "1",
                    reason="live LTspice disabled via PIECESPICE_LIVE=0")
def test_agrees_with_ltspice_on_the_same_linear_deck() -> None:
    from simulation.engine import SpiceEngine, TranDirective

    end_time, max_step = 200e-6, 0.1e-6
    deck = LADDER_DECK + ".end\n"
    ours = solve_transient(parse_circuit(deck), end_time=end_time,
                           max_step=max_step)
    theirs = SpiceEngine(temperature=25.0).run_transient(
        deck, TranDirective(step_time=0.0, end_time=end_time,
                            max_step=max_step))
    t_ref = theirs.get("time")
    for node in ("a", "b"):
        reference = np.interp(ours.get("time"), t_ref, theirs.get(f"V({node})"))
        scale = max(float(np.max(np.abs(reference))), 1e-12)
        error = float(np.max(np.abs(ours.voltage(node) - reference)))
        assert error / scale < 1e-4, f"V({node}) rel error {error / scale:.2e}"


@pytest.mark.skipif(os.environ.get("PIECESPICE_LIVE", "1") != "1",
                    reason="live LTspice disabled via PIECESPICE_LIVE=0")
def test_inductor_and_source_current_signs_match_ltspice() -> None:
    from simulation.engine import SpiceEngine, TranDirective

    end_time, max_step = 50e-6, 10e-9
    deck = """
* inductor sign check
V1 in 0 PWL(0,0 1e-12,1)
R1 in a 1k
C1 a 0 100n
L1 a b 10u
R3 b 0 100
.end
"""
    ours = solve_transient(parse_circuit(deck), end_time=end_time,
                           max_step=max_step)
    theirs = SpiceEngine(temperature=25.0).run_transient(
        deck, TranDirective(step_time=0.0, end_time=end_time,
                            max_step=max_step))
    t_ref = theirs.get("time")
    for trace in ("I(L1)", "I(V1)"):
        reference = np.interp(ours.get("time"), t_ref, theirs.get(trace))
        scale = max(float(np.max(np.abs(reference))), 1e-12)
        error = float(np.max(np.abs(ours.get(trace) - reference)))
        assert error / scale < 1e-3, f"{trace} rel error {error / scale:.2e}"
