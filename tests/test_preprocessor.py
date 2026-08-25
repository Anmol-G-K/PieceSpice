"""Unit tests for the T1.2 thin preprocessor (LTspice migration)."""

import sys
sys.path.insert(0, ".")

from parser.preprocessor import (
    join_continuations, normalize_pspice_netlist, _strip_comment,
)


def test_continuations_joined():
    src = "* c\nM1 d g s s MOD\n+ W=2u L=0.5u\nR1 a b 1k"
    out = normalize_pspice_netlist(src)
    assert "M1 d g s s MOD W=2u L=0.5u" in out
    assert "\n+" not in out


def test_comments_stripped():
    src = "R1 a b 1k ; load\nC1 b 0 1n $ decoupling"
    out = normalize_pspice_netlist(src)
    assert ";" not in out and "$" not in out
    assert "R1 a b 1k" in out


def test_param_syntax_passes_through_untouched():
    """LTspice evaluates .PARAM natively - we must NOT flatten."""
    src = ".PARAM pwidth={2*len}\nM1 d g s s MOD W={pwidth}"
    out = normalize_pspice_netlist(src)
    assert "{pwidth}" in out and "{2*len}" in out
    assert ".PARAM" in out


def test_table_poly_ddt_pass_through():
    """ddt()/TABLE constructs stay verbatim (LTspice supports them)."""
    src = "B1 out 0 V = ddt(V(in))\nE1 a 0 TABLE {V(x)} = (0,0) (1,1)"
    out = normalize_pspice_netlist(src)
    assert "ddt(V(in))" in out and "TABLE" in out


def test_harden_powers_kwarg_accepted_noop():
    src = "B1 out 0 V = (V(in))**0.5"
    out = normalize_pspice_netlist(src, harden_powers=True)
    assert "**0.5" in out  # untouched


def test_blank_and_star_lines_preserved():
    src = "* header\n\nR1 a b 1k\n* tail"
    out = normalize_pspice_netlist(src)
    assert "* header" in out and "* tail" in out
