"""Unit tests for the PSpice/LTspice dialect normalizer (Milestone 1)."""

from parser.preprocessor import normalize_pspice_netlist


def test_normalize_pspice_param():
    raw = """.SUBCKT TEST 1 2 3
.PARAM pWidth 0.5
.PARAM perim {2*pWidth}
M1 1 2 3 3 NMOS W={pWidth}
.ENDS"""
    normalized = normalize_pspice_netlist(raw)
    assert ".PARAM pWidth=0.5" in normalized
    assert "W=0.5" in normalized


def test_forward_referenced_params_resolve():
    """A param defined in terms of a LATER param must still resolve (fixed point)."""
    raw = ".PARAM a {b*2}\n.PARAM b 3\nR1 1 0 {a}"
    out = normalize_pspice_netlist(raw)
    assert ".PARAM a=6" in out
    assert ".PARAM b=3" in out
    assert "R1 1 0 6" in out


def test_dollar_comments_stripped():
    """The docstring promises $ comment stripping; it must actually happen."""
    raw = "R1 1 0 1k $ bias network\nC1 1 0 1n"
    out = normalize_pspice_netlist(raw)
    assert "$" not in out
    assert "R1 1 0 1k" in out


def test_semicolon_comments_stripped():
    raw = "R1 1 0 1k ; feedback\nC1 1 0 1n"
    out = normalize_pspice_netlist(raw)
    assert ";" not in out


def test_param_expression_functions():
    """Safe function whitelist (min/max/sqrt/...) must evaluate, not pass through."""
    raw = ".PARAM x {sqrt(4)}\nR1 1 0 {x}"
    out = normalize_pspice_netlist(raw)
    assert ".PARAM x=2" in out


def test_unresolvable_param_stays_symbolic():
    """A param referencing an undefined name must not crash or fabricate a value."""
    raw = ".PARAM x {undefined_name}\nR1 1 0 {x}"
    out = normalize_pspice_netlist(raw)
    assert "undefined_name" not in out  # never evaluated into a number
    assert "{x}" in out  # device line passes the reference through untouched
    assert "R1 1 0 {x}" in out


def test_params_hoisted_before_subckt():
    raw = ".SUBCKT T 1 2 3\n.PARAM pA 0.5\nR1 1 2 {pA}\n.ENDS"
    out = normalize_pspice_netlist(raw)
    lines = out.splitlines()
    assert lines[0].startswith(".PARAM pA=0.5")
    assert ".SUBCKT T 1 2 3" in lines[1]
