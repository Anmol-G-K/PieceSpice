"""Unit tests for the SPICE netlist parser (Milestone 1)."""

from pathlib import Path

import pytest

from parser.netlist import NetlistParser

# SpiceModels/* is .gitignore'd (vendor redistribution), so a fresh clone and
# CI do not have it. Tests that need the real vendor file skip instead of
# failing with FileNotFoundError.
_IRF = Path("SpiceModels/irfs7730pbf.spi")
_needs_irf = pytest.mark.skipif(
    not _IRF.exists(),
    reason="SpiceModels/irfs7730pbf.spi is not in the repository "
           "(vendor files are .gitignore'd)")


@_needs_irf
def test_parse_subcircuits():
    spi_path = Path("SpiceModels/irfs7730pbf.spi")
    subcircuits = NetlistParser.parse_subcircuits(spi_path)
    assert len(subcircuits) >= 1
    assert any(s.name == "irfb_s_sl7730pbf" for s in subcircuits)


@_needs_irf
def test_parse_thermal_subckt_too():
    """The vendor file also contains a thermal RC subcircuit; both must be found."""
    subs = NetlistParser.parse_subcircuits(Path("SpiceModels/irfs7730pbf.spi"))
    names = {s.name for s in subs}
    assert "irfb_s_sl7730pbft" in names


@_needs_irf
def test_pin_count_matches_vendor():
    irf = next(
        s for s in NetlistParser.parse_subcircuits(Path("SpiceModels/irfs7730pbf.spi"))
        if s.name == "irfb_s_sl7730pbf"
    )
    assert len(irf.nodes) == 3  # Drain, Gate, Source


def test_subckt_header_params_are_not_pins(tmp_path):
    """PSpice '.SUBCKT X 1 2 3 PARAMS: a=1' - param tokens are not pin nodes."""
    lib = tmp_path / "demo.lib"
    lib.write_text(
        ".SUBCKT DEMO 1 2 3 PARAMS: scale=1.5 temp=25\n"
        "R1 1 2 1k\n"
        ".ENDS DEMO\n"
    )
    subs = NetlistParser.parse_subcircuits(lib)
    assert len(subs) == 1
    assert subs[0].nodes == ["1", "2", "3"]
    assert subs[0].parameters == {"scale": "1.5", "temp": "25"}


def test_multiple_subckts_and_comments(tmp_path):
    lib = tmp_path / "multi.lib"
    lib.write_text(
        "* top comment\n"
        ".SUBCKT A 1 2\nR1 1 2 1k\n.ENDS A\n"
        "* another comment\n.SUBCKT B 3 4 5\nC1 3 4 1n\n.ENDS B\n"
    )
    subs = NetlistParser.parse_subcircuits(lib)
    assert [s.name for s in subs] == ["A", "B"]
    assert subs[1].nodes == ["3", "4", "5"]
