"""Circuit IR: reader, flattening, and MNA structure.

These tests protect the two properties the solver comparison rests on:
deterministic structure (same deck -> same matrix shape) and faithful reading
of the decks this project actually emits.  No LTspice is needed, so they run in
the LTspice-free CI lane.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pytest

from engine.envelope import UnsupportedConstruct
from engine.ir import mna
from engine.ir.circuit import ElementKind, Waveform, parse_value
from engine.ir.parse import (
    NetlistError,
    parse,
    parse_circuit,
    read_deck,
)

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "reports" / "models"


# ---------------------------------------------------------------------------
# Values and waveforms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0", 0.0),
        ("1e-5", 1e-5),
        ("1.5e+00", 1.5),
        ("5k", 5e3),
        ("2n", 2e-9),
        ("1u", 1e-6),
        ("3p", 3e-12),
        ("1meg", 1e6),      # mega
        ("1MEG", 1e6),
        ("10m", 1e-2),      # milli -- the classic SPICE trap
        ("10kohm", 1e4),    # trailing unit letters are ignored
        ("1mil", 25.4e-6),
    ],
)
def test_parse_value(text: str, expected: float) -> None:
    assert parse_value(text) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("text", ["", "abc", "1x2", "1_000"])
def test_parse_value_rejects_nonsense(text: str) -> None:
    with pytest.raises(ValueError):
        parse_value(text)


def test_pwl_clamps_outside_its_span_and_interpolates_inside() -> None:
    wave = Waveform.pwl([(0.0, 0.0), (2.0, 10.0), (4.0, 10.0)])
    assert wave.value(-1.0) == 0.0
    assert wave.value(0.0) == 0.0
    assert wave.value(1.0) == pytest.approx(5.0)
    assert wave.value(2.0) == pytest.approx(10.0)
    assert wave.value(3.0) == pytest.approx(10.0)
    assert wave.value(99.0) == 10.0


def test_pwl_rejects_non_monotone_times() -> None:
    with pytest.raises(ValueError):
        Waveform.pwl([(0.0, 0.0), (1.0, 1.0), (1.0, 2.0)])


def test_pulse_edges_plateau_and_period() -> None:
    # PULSE(0 5 0 1n 1n 2n 10n): 1 ns rise, 2 ns at 5 V, 1 ns fall, 10 ns period.
    wave = Waveform.pulse_wave(0.0, 5.0, 0.0, 1e-9, 1e-9, 2e-9, 10e-9)
    assert wave.value(-1e-9) == 0.0
    assert wave.value(0.5e-9) == pytest.approx(2.5)
    assert wave.value(1.5e-9) == pytest.approx(5.0)
    assert wave.value(3.5e-9) == pytest.approx(2.5)
    assert wave.value(4.5e-9) == pytest.approx(0.0)
    assert wave.value(10e-9 + 1.5e-9) == pytest.approx(5.0)   # next period


def test_dc_waveform_is_constant_and_time_independent() -> None:
    wave = Waveform.constant(3.3)
    assert wave.value(0.0) == wave.value(1e-3) == 3.3
    assert not wave.is_time_dependent


# ---------------------------------------------------------------------------
# Reading a DPT-shaped deck
# ---------------------------------------------------------------------------

DPT = """
* clamped inductive double pulse
.subckt EPC_ROM d g s
RG g gi 1.5e+00
RD d di 1e-04
RS s ss 1e-04
G_CH di ss VALUE = {(V(gi,ss))*1e-3}
C_QOSS di ss Q = {1e-10*(V(di,ss))}
C_ISS gi ss 1.9e-10
C_RSS gi di 9.8e-12
.model D_BD D(IS=1e-15 N=1.587 RS=3.1e-02)
D_BODY ss di D_BD
.ends EPC_ROM
Vbus vbus 0 DC 11
Lload vbus drain 1e-05
Dclamp drain vbus DCLAMP
.model DCLAMP D(IS=1e-12 N=1 RS=1e-3 CJO=0)
Vgs gate 0 PWL(0,0 1e-06,5 2e-06,5 3e-06,0)
Xdut drain gate 0 EPC_ROM
.tran 0 1.4e-05 0 2e-09
.end
"""


def test_dpt_deck_reads_into_one_flat_circuit() -> None:
    circuit = parse_circuit(DPT)
    assert len(circuit) == 12                       # 8 dut + 4 bench elements
    assert circuit.node_index.names == (
        "vbus", "drain", "gate", "xdut.gi", "xdut.di", "xdut.ss")
    assert set(circuit.models) == {"DCLAMP", "D_BD"}
    assert circuit.models["D_BD"].family == "D"
    assert circuit.models["DCLAMP"].get("IS") == pytest.approx(1e-12)
    assert circuit.models["DCLAMP"].get("CJO") == pytest.approx(0.0)


def test_dut_internals_are_prefixed_and_pins_are_positional() -> None:
    circuit = parse_circuit(DPT)
    gate_resistor = circuit.element("Xdut.RG")
    assert gate_resistor.kind is ElementKind.RESISTOR
    assert gate_resistor.nodes == ("gate", "xdut.gi")
    channel = circuit.element("Xdut.G_CH")
    assert channel.kind is ElementKind.BEHAVIOURAL_SOURCE
    assert channel.nodes == ("xdut.di", "xdut.ss")
    # The expression is kept verbatim as provenance, never evaluated.
    assert channel.expression.startswith("{")
    charge = circuit.element("Xdut.C_QOSS")
    assert charge.kind is ElementKind.CHARGE_CAPACITOR
    assert charge.expression.startswith("{")


def test_transient_directive_is_read_positionally() -> None:
    directive = parse_circuit(DPT).analysis("tran")
    assert directive is not None
    assert directive.end_time == pytest.approx(1.4e-5)
    assert directive.start_time == 0.0
    assert directive.max_step == pytest.approx(2e-9)


def test_two_instances_do_not_collide() -> None:
    deck = DPT.replace(
        "Xdut drain gate 0 EPC_ROM",
        "Xdut drain gate 0 EPC_ROM\nXdut2 drain gate 0 EPC_ROM")
    circuit = parse_circuit(deck)
    assert "xdut.gi" in circuit.node_index
    assert "xdut2.gi" in circuit.node_index
    assert circuit.element("Xdut2.RG").nodes == ("gate", "xdut2.gi")
    # A model declared once per instance must not be a redefinition error.
    assert circuit.models["D_BD"].get("N") == pytest.approx(1.587)


# ---------------------------------------------------------------------------
# MNA structure
# ---------------------------------------------------------------------------

def _pattern(circuit) -> np.ndarray:
    return mna.build(circuit).pattern()


def test_single_resistor_and_source_pattern() -> None:
    circuit = parse_circuit("* series\nV1 a 0 DC 1\nR1 a 0 1k\n.end")
    structure = mna.build(circuit)
    assert structure.unknown_count() == (1, 1)
    assert structure.size == 2
    assert _pattern(circuit).tolist() == [[1, 1], [1, 0]]


def test_divider_pattern_matches_hand_derivation() -> None:
    circuit = parse_circuit(
        "* divider\nV1 in 0 DC 1\nR1 in mid 1k\nR2 mid 0 1k\n.end")
    structure = mna.build(circuit)
    assert structure.node_index.names == ("in", "mid")
    assert structure.branches == (("V1", 2),)
    assert _pattern(circuit).tolist() == [[1, 1, 1], [1, 2, 0], [1, 0, 0]]


def test_rlc_pattern_matches_hand_derivation() -> None:
    circuit = parse_circuit(
        "* rlc\nV1 in 0 PULSE(0 1 0 1n 1n 1u 2u)\n"
        "L1 in mid 1u\nC1 mid 0 1n\nR1 mid 0 10\n.end")
    structure = mna.build(circuit)
    assert structure.branches == (("V1", 2), ("L1", 3))
    # L1 writes both branch equations and its -L di/dt diagonal; C1 and R1
    # both land on (mid, mid).
    assert _pattern(circuit).tolist() == [
        [0, 0, 1, 1],
        [0, 2, 0, 1],
        [1, 0, 0, 0],
        [1, 1, 0, 1],
    ]


def test_dpt_structure_has_one_unknown_per_node_plus_branch() -> None:
    circuit = parse_circuit(DPT)
    structure = mna.build(circuit)
    assert structure.unknown_count() == (6, 3)
    assert structure.size == 9
    assert structure.branches == (("Vbus", 6), ("Lload", 7), ("Vgs", 8))
    # A behavioural source may reference any node: named, not expanded.
    assert structure.dense == ("Xdut.G_CH",)
    assert structure.pattern().max() < 10


def test_structure_is_deterministic_across_parses() -> None:
    first = mna.build(parse_circuit(DPT))
    second = mna.build(parse_circuit(DPT))
    assert first.node_index.names == second.node_index.names
    assert first.branches == second.branches
    assert np.array_equal(first.pattern(), second.pattern())


# ---------------------------------------------------------------------------
# Fail-closed behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "card",
    [".param r=1k", ".ic V(a)=1", ".nodeset V(a)=1", ".global vdd"],
)
def test_dot_cards_that_change_the_solution_are_rejected(card: str) -> None:
    with pytest.raises(UnsupportedConstruct):
        parse_circuit(f"* x\nV1 a 0 DC 1\n{card}\n.end")


@pytest.mark.parametrize("element", ["K1 a b 0.9", "B1 a 0 V=1", "Q1 a b c m"])
def test_unrepresentable_elements_are_rejected(element: str) -> None:
    with pytest.raises(UnsupportedConstruct):
        parse_circuit(f"* x\nV1 a 0 DC 1\n{element}\n.end")


def test_undefined_subcircuit_is_an_error_not_a_silent_open() -> None:
    with pytest.raises(NetlistError, match="undefined subcircuit"):
        parse_circuit("* x\nX1 a b MISSING\n.end")


def test_unclosed_subcircuit_is_an_error() -> None:
    with pytest.raises(NetlistError, match="never closed"):
        parse_circuit("* x\n.subckt s a b\nR1 a b 1\n")


def test_parameterised_subcircuit_header_is_rejected() -> None:
    with pytest.raises(UnsupportedConstruct):
        parse_circuit("* x\n.subckt s a b PARAMS: k=1\nR1 a b 1\n.ends\n.end")


def test_pin_count_mismatch_is_an_error() -> None:
    with pytest.raises(NetlistError, match="pins"):
        parse_circuit("* x\n.subckt s a b\nR1 a b 1\n.ends\nX1 a s\n.end")


# ---------------------------------------------------------------------------
# Every emitted ROM in the repository
# ---------------------------------------------------------------------------

def _subcircuits_of(lib: Path):
    """Read a library the way a deck consumes it.

    SPICE treats line 1 of a *deck* as the title.  An included library is
    textually inserted, so its own first line is never a title -- and some
    emitted libs (the Tier-3 ones) start directly with ``.subckt``.  Prepending
    a title reproduces the real consumption path.
    """
    text = lib.read_text(encoding="utf-8", errors="replace")
    return parse(f"* library {lib.name}\n{text}", source=str(lib)).subcircuits


def _dpt_deck_for(lib: Path) -> str:
    parsed = _subcircuits_of(lib)
    (name, sub), = parsed.items()
    outer = ["drain", "gate", "0"] + [f"aux{i}" for i in range(3, len(sub.pins))]
    lines = [
        "* synthetic DPT around one emitted ROM",
        f".include {lib.resolve()}",
        "Vbus vbus 0 DC 11",
        "Lload vbus drain 1e-05",
        "Dclamp drain vbus DCLAMP",
        ".model DCLAMP D(IS=1e-12 N=1 RS=1e-3 CJO=0)",
        "Vgs gate 0 PWL(0,0 1e-06,5 2e-06,0)",
        f"Xdut {' '.join(outer)} {name}",
    ]
    for i in range(3, len(sub.pins)):
        lines.append(f"Vaux{i} aux{i} 0 DC 25")
    lines += [".tran 0 2e-06 0 2e-09", ".end"]
    return "\n".join(lines)


ALL_LIBS = sorted(glob.glob(str(MODELS / "*.lib")))


@pytest.mark.skipif(not ALL_LIBS, reason="no emitted ROMs present")
@pytest.mark.parametrize("lib", ALL_LIBS, ids=[Path(p).name for p in ALL_LIBS])
def test_every_emitted_rom_parses_inside_a_bench_deck(lib: str, tmp_path: Path) -> None:
    deck = tmp_path / "dpt.net"
    deck.write_text(_dpt_deck_for(Path(lib)), encoding="ascii", errors="replace")
    circuit = read_deck(deck).circuit
    assert len(circuit) >= 6
    assert len(circuit.node_index) >= 3
    assert circuit.models, "every ROM declares at least one .model"
    kinds = {e.kind for e in circuit.elements}
    assert ElementKind.VSOURCE in kinds
    assert ElementKind.INDUCTOR in kinds
    # Structure must assemble for every family without raising.
    structure = mna.build(circuit)
    assert structure.size == len(structure.node_index) + len(structure.branches)


@pytest.mark.skipif(not ALL_LIBS, reason="no emitted ROMs present")
def test_tier4_rom_is_structural_only_no_behavioural_sources(tmp_path: Path) -> None:
    lib = MODELS / "epc2216_PIECESPICE_T4.lib"
    if not lib.is_file():
        pytest.skip("EPC2216 Tier-4 ROM not present")
    deck = tmp_path / "dpt.net"
    deck.write_text(_dpt_deck_for(lib), encoding="ascii", errors="replace")
    circuit = read_deck(deck).circuit
    assert mna.build(circuit).dense == ()
    kinds = {e.kind for e in circuit.elements}
    assert ElementKind.SWITCH in kinds
    assert ElementKind.BEHAVIOURAL_SOURCE not in kinds


@pytest.mark.skipif(not ALL_LIBS, reason="no emitted ROMs present")
def test_tier2_rom_keeps_its_behavioural_and_charge_forms(tmp_path: Path) -> None:
    lib = MODELS / "epc2216_PIECESPICE_T2.lib"
    if not lib.is_file():
        pytest.skip("EPC2216 Tier-2 ROM not present")
    deck = tmp_path / "dpt.net"
    deck.write_text(_dpt_deck_for(lib), encoding="ascii", errors="replace")
    circuit = read_deck(deck).circuit
    kinds = {e.kind for e in circuit.elements}
    assert ElementKind.BEHAVIOURAL_SOURCE in kinds
    assert ElementKind.CHARGE_CAPACITOR in kinds
    assert mna.build(circuit).dense == ("Xdut.G_CH",)


def test_first_line_is_a_title_and_is_not_parsed() -> None:
    # SPICE convention, and LTspice follows it.  Without this the two engines
    # would read a title-less deck as two different circuits.
    circuit = parse_circuit("V1 in 0 DC 1\nR1 in 0 1k\n.end")
    assert [e.name for e in circuit.elements] == ["R1"]
    titled = parse_circuit("* title\nV1 in 0 DC 1\nR1 in 0 1k\n.end")
    assert [e.name for e in titled.elements] == ["V1", "R1"]


def test_a_deck_with_no_readable_content_says_why() -> None:
    with pytest.raises(NetlistError, match="first line as the title"):
        parse_circuit("V1 in 0 DC 1\n")
