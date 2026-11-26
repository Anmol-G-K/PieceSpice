"""Stage-5 verification: the solver-facing ROM contract.

The contract has to hold three ways at once, and each is a separate test:

* **same fit** -- the channel table reproduces the ``PWLModelSpec`` exactly at
  its grid nodes, so the two emitters (SPICE and solver) describe one model;
* **it blocks** -- a channel table without sub-threshold anchors is a model that
  conducts at Vgs = 0, which is how a ROM silently fails to be a switch;
* **it is identified** -- the artifact carries its schema and payload hash, so a
  number measured from it can be re-derived.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

from engine.ir.circuit import ElementKind
from engine.rom import SolverRom, dump, load, manifest, payload_sha256
from engine.rom.bench import DptConditions, dpt
from engine.rom.spec import SCHEMA
from engine.solver.channel import ChannelTable
from engine.solver.transient import solve_transient

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "config" / "outputs"


@functools.lru_cache(maxsize=1)
def _build_spec(device: str):
    from pipeline.fitting import build_pwl_spec

    return build_pwl_spec(device)


def _spec_or_skip(device: str = "epc2216"):
    if not (OUTPUTS / device).is_dir():
        pytest.skip(f"extracted curves for {device} not present")
    return _build_spec(device)


# ---------------------------------------------------------------------------
# The channel table
# ---------------------------------------------------------------------------

def test_channel_table_reproduces_the_spec_at_its_grid_nodes() -> None:
    spec = _spec_or_skip()
    rom = SolverRom.from_spec(spec)
    worst = 0.0
    for vgs, row in zip(spec.vgs_values, spec.breakpoints_id):
        for vds, current in zip(spec.breakpoints_vds, row):
            worst = max(worst, abs(rom.channel.current(vgs, vds) - current))
    assert worst < 1e-9, f"table deviates from the fit by {worst:.3e} A"


def test_channel_grid_carries_the_sub_threshold_anchors() -> None:
    spec = _spec_or_skip()
    rom = SolverRom.from_spec(spec)
    assert rom.channel.vgs[0] == 0.0
    assert rom.channel.vgs[1] == pytest.approx(float(spec.v_th))
    # Both anchor rows must be exactly zero current for every Vds.
    assert set(rom.channel.id[0]) == {0.0}
    assert set(rom.channel.id[1]) == {0.0}


def test_channel_blocks_at_zero_gate() -> None:
    rom = SolverRom.from_spec(_spec_or_skip())
    for vds in (0.0, 5.0, 11.0, 15.0):
        assert rom.channel.current(0.0, vds) == pytest.approx(0.0, abs=1e-15)


def test_channel_conducts_upward_with_gate_and_saturates_with_drain() -> None:
    spec = _spec_or_skip()
    rom = SolverRom.from_spec(spec)
    vgs_on = max(spec.vgs_values)
    assert rom.channel.current(vgs_on, 15.0) > 0.0
    # Above the last Vds point the table saturates instead of extrapolating.
    assert rom.channel.current(vgs_on, 15.0) == pytest.approx(
        rom.channel.current(vgs_on, 100.0))
    # And it is still zero well below threshold, not a downward extrapolation.
    assert rom.channel.current(0.5, 15.0) == pytest.approx(0.0, abs=1e-15)


def test_channel_table_is_planar_within_a_triangle() -> None:
    table = ChannelTable.build([0.0, 1.0], [0.0, 1.0, 2.0],
                               [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    # Triangle 0 (v <= u) is the plane Id = 1*Vgs + 1*Vds.
    assert table.current(0.8, 0.2) == pytest.approx(1.0)
    # Triangle 1 of the first cell (v > u) is the plane Id = 2*Vgs.
    assert table.current(0.2, 0.8) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Artifact identity
# ---------------------------------------------------------------------------

def test_artifact_round_trips_through_disk(tmp_path: Path) -> None:
    rom = SolverRom.from_spec(_spec_or_skip())
    path = dump(rom, tmp_path / "rom.json")
    payload = json.loads(path.read_text(encoding="ascii"))
    assert payload["schema"] == SCHEMA
    assert payload["sha256"] == payload_sha256(payload)

    again = load(path)
    assert again.device == rom.device
    assert again.channel.id == rom.channel.id
    assert again.qoss.table.y == rom.qoss.table.y
    assert again.body_diode.y == rom.body_diode.y


def test_artifact_detects_a_tampered_table(tmp_path: Path) -> None:
    rom = SolverRom.from_spec(_spec_or_skip())
    path = dump(rom, tmp_path / "rom.json")
    payload = json.loads(path.read_text(encoding="ascii"))
    payload["channel"]["id"][-1][-1] *= 2.0
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(ValueError, match="hash mismatch"):
        load(path)


def test_manifest_reports_the_artifact_identity(tmp_path: Path) -> None:
    rom = SolverRom.from_spec(_spec_or_skip())
    path = dump(rom, tmp_path / "rom.json")
    info = manifest(path)
    assert info["schema"] == SCHEMA
    assert info["device"] == rom.device
    assert info["verified"] is True


def test_an_unknown_schema_is_refused() -> None:
    with pytest.raises(ValueError, match="schema"):
        SolverRom.from_dict({"schema": "piecespice.solver_rom/0.1",
                             "device": "x"})


# ---------------------------------------------------------------------------
# The ROM as a circuit
# ---------------------------------------------------------------------------

def test_rom_circuit_mirrors_the_emitted_subcircuit_structure() -> None:
    rom = SolverRom.from_spec(_spec_or_skip())
    circuit = rom.to_circuit()
    kinds = [e.kind for e in circuit.elements]
    assert kinds.count(ElementKind.RESISTOR) == 3          # RG, RD, RS
    assert ElementKind.PWL_CHANNEL in kinds
    assert ElementKind.PWL_CAP in kinds
    assert ElementKind.DIODE in kinds
    assert set(circuit.node_index.names) == {"g", "gi", "d", "di", "s", "ss"}


def _dpt_conditions(max_step: float) -> DptConditions:
    return DptConditions(t1_on=0.2e-6, t1_width=1.0e-6, t2_on=1.4e-6,
                         t2_width=0.4e-6, t_end=2.0e-6, max_step=max_step,
                         edge_time=10e-9)


def _dpt_probe(max_step: float) -> tuple[float, float, float]:
    rom = SolverRom.from_spec(_spec_or_skip())
    result = solve_transient(dpt(rom, _dpt_conditions(max_step)))
    t = result.get("time")
    v_drain = result.voltage("drain")
    i_load = result.current("Lload")
    on = (t > 0.6e-6) & (t < 1.1e-6)
    return (float(v_drain[on].mean()), float(i_load.max()),
            float(v_drain[t < 0.15e-6].max()))


def test_rom_blocks_then_conducts_in_a_double_pulse() -> None:
    v_on, i_peak, v_block = _dpt_probe(5e-9)
    assert v_block > 0.9 * 11.0            # blocks the bus
    assert i_peak > 0.5                    # and conducts
    assert v_on < 0.05                     # on-state is a drop, not a rail


def test_the_rom_answer_is_step_independent() -> None:
    """Same device, two step sizes: the *answer* must not depend on the step.

    This is the check that separates an accuracy limit from a resolution one.
    The run reports region-chatter warnings at the drain transition -- those are
    a performance symptom of a step spanning several table cells, and they must
    not move the converged operating point.
    """
    coarse = _dpt_probe(20e-9)
    fine = _dpt_probe(5e-9)
    assert fine[0] == pytest.approx(coarse[0], rel=0.05)     # on-state Vds
    assert fine[1] == pytest.approx(coarse[1], rel=0.02)     # peak current

# ---------------------------------------------------------------------------
# Basis selection: the solver consumes tables, so the basis is a choice
# ---------------------------------------------------------------------------

def test_ann_basis_builds_a_table_with_the_same_contract() -> None:
    from engine.rom import solver_rom

    rom = solver_rom("epc2216", "ann")
    assert rom.basis == "ann"
    axis = rom.channel.vds
    assert all(b > a for a, b in zip(axis, axis[1:]))
    assert axis[0] == 0.0
    # Dense near zero: the second point must be a fraction of a volt.
    assert axis[1] < 0.01
    # Blocks at zero gate and conducts above threshold.
    assert abs(rom.channel.current(0.0, 11.0)) < 1e-6
    assert rom.channel.current(5.0, 11.0) > 1.0


def test_ann_basis_records_its_basis_in_the_artifact(tmp_path: Path) -> None:
    from engine.rom import dump, load, solver_rom

    rom = solver_rom("epc2216", "ann")
    path = dump(rom, tmp_path / "ann.json")
    payload = json.loads(path.read_text(encoding="ascii"))
    assert payload["basis"] == "ann"
    assert load(path).basis == "ann"


def test_tier2_and_ann_disagree_where_it_matters() -> None:
    """The two bases are not interchangeable, so the choice must be explicit."""
    from engine.rom import solver_rom

    tier2 = solver_rom("epc2216", "tier2")
    ann = solver_rom("epc2216", "ann")
    assert tier2.basis == "tier2"
    # Both saturate to the same current at full drive (-0.2 %) -- that is the
    # shared Curve-1 data.  They separate near threshold, which is exactly the
    # region a switching edge traverses: at Vgs 2 V / Vds 1 V the two bases
    # differ by more than half.
    assert tier2.channel.current(5.0, 11.0) == pytest.approx(
        ann.channel.current(5.0, 11.0), rel=0.05)
    a = tier2.channel.current(2.0, 1.0)
    b = ann.channel.current(2.0, 1.0)
    assert abs(a - b) / max(abs(b), 1e-9) > 0.5
