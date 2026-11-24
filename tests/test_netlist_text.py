"""Unit tests for simulation/netlist_text.py (LTspice netlist builder)."""

import sys
sys.path.insert(0, ".")

from simulation.netlist_text import NetlistBuilder


def test_basic_render_includes_and_end():
    nb = NetlistBuilder("smoke")
    nb.include("SpiceModels/x.lib")
    nb.v_dc("1", "in", "0", 5.0)
    nb.r("1", "in", "out", 10e3)
    text = nb.render()
    assert ".include" in text
    assert "V1 in 0 DC 5" in text
    assert "R1 in out 10000" in text
    assert text.rstrip().endswith(".end")


def test_pulse_source_format():
    nb = NetlistBuilder("pulse")
    nb.v_pulse("gs", "gate", "0", v1=0, v2=5, td=5e-6, tr=2e-8, tf=2e-8,
               pw=9.98e-6, period=2e-5)
    line = [l for l in nb.render().splitlines() if l.startswith("Vgs")][0]
    assert "PULSE(0 5 5e-06 2e-08 2e-08 9.98e-06 2e-05)" in line


def test_x_instance_positional_order():
    nb = NetlistBuilder("xinst")
    nb.x_instance("dut", "EPC2216", ["gate", "drain", "source"])
    text = nb.render()
    assert "Xdut gate drain 0 EPC2216" in text


def test_x_instance_tj_gets_voltage_source():
    nb = NetlistBuilder("tj")
    nb.x_instance("dut", "CPM2-1200-0025B",
                  ["drain", "gate", "source", "tj"], temperature=25.0)
    text = nb.render()
    # V source is emitted BEFORE the X line (builder order), node matches.
    assert "Vtjsrc_dut tj_dut 0 DC 25" in text
    assert "Xdut drain gate 0 tj_dut CPM2-1200-0025B" in text


def test_analysis_card_inserted_before_end():
    nb = NetlistBuilder("card")
    nb.r("1", "a", "b", 1)
    text = nb.render(analysis_card=".op")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    assert lines[-1] == ".end"
    assert lines[-2] == ".op"
