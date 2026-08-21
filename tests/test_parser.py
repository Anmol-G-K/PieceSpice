"""Unit tests for SPICE parser and preprocessor."""

from pathlib import Path
from piecespice.parser.netlist import NetlistParser
from piecespice.parser.preprocessor import normalize_pspice_netlist


def test_parse_subcircuits():
    spi_path = Path("SpiceModels/irfs7730pbf.spi")
    subcircuits = NetlistParser.parse_subcircuits(spi_path)
    assert len(subcircuits) >= 1
    assert any(s.name == "irfb_s_sl7730pbf" for s in subcircuits)


def test_normalize_pspice_param():
    raw = """.SUBCKT TEST 1 2 3
.PARAM pWidth 0.5
.PARAM perim {2*pWidth}
M1 1 2 3 3 NMOS W={pWidth}
.ENDS"""
    normalized = normalize_pspice_netlist(raw)
    assert ".PARAM pWidth=0.5" in normalized
    assert "W=0.5" in normalized
