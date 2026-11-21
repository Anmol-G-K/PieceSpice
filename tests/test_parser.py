"""Extended tests for parser/preprocessor, parser/netlist, and quasi-static
gate-charge reconstruction.

The preprocessor is a text-hygiene pass-through: LTspice resolves .PARAM
hierarchies and TABLE/POLY/fractional powers natively. These tests lock in
that native constructs survive unchanged; dollar comments and continuation
joining remain the only transformations.
"""

import numpy as np
import pytest

from parser.netlist import NetlistParser
from parser.preprocessor import normalize_pspice_netlist


# ---------- preprocessor ----------

def test_normalize_forward_referenced_params():
    """LTspice resolves forward .PARAM references; normalization must not fold them."""
    raw = (
        "M1 d g s s NMOS W={pWidth} PS={perim}\n"
        ".param perim={2*pWidth}\n"
        ".param pWidth=0.6178256\n"
    )
    out = normalize_pspice_netlist(raw)
    assert out == raw.rstrip("\n")
    assert "{pWidth}" in out and "{2*pWidth}" in out
    assert ".param pWidth=0.6178256" in out

def test_normalize_strips_dollar_comments():
    raw = "R1 1 0 1k $ trailing comment"
    out = normalize_pspice_netlist(raw)
    assert "$" not in out
    assert "comment" not in out
    assert out.strip().endswith("1k")


def test_normalize_safe_functions():
    """LTspice-native TABLE/POLY and fractional powers pass through untouched."""
    raw = (
        "G1 d 0 VALUE={TABLE(V(g),0,0,10,1)}\n"
        "R1 a 0 {pow(V(x),1.5)+sqrt(abs(V(y)))}\n"
        "E1 out 0 POLY(2) V(a) V(b) 0 1 2\n"
    )
    out = normalize_pspice_netlist(raw)
    assert out == raw.rstrip("\n")
    assert "TABLE(" in out and "POLY(2)" in out
    assert "pow(V(x),1.5)" in out and "sqrt(abs(V(y)))" in out

def test_parse_subckt_with_params_header(tmp_path):
    """PSpice .SUBCKT ... PARAMS: k=v tokens must go to parameters, not nodes."""
    f = tmp_path / "demo.lib"
    f.write_text(
        ".SUBCKT DEMO 1 2 3 PARAMS: scale=1.5 temp=25\n"
        "R1 1 2 {scale}k\n"
        ".ENDS DEMO\n"
    )
    subs = NetlistParser.parse_subcircuits(f)
    assert len(subs) == 1
    s = subs[0]
    assert s.name == "DEMO"
    assert s.nodes == ["1", "2", "3"]
    assert s.parameters == {"scale": "1.5", "temp": "25"}


def test_parse_plain_subckt_unchanged(tmp_path):
    f = tmp_path / "plain.lib"
    f.write_text(".SUBCKT P d g s\nR1 d g 10\n.ENDS\n")
    s = NetlistParser.parse_subcircuits(f)[0]
    assert s.nodes == ["d", "g", "s"]
    assert s.parameters == {}


# ---------- quasi-static gate charge reconstruction ----------

@pytest.fixture()
def synthetic_curves():
    from common.types import CurveData
    transfer = CurveData(
        name="transfer", x_name="Vgs", x_unit="V", y_name="Id", y_unit="A",
        x_data=[0.0, 4.0, 5.0, 6.0, 10.0],
        y_data=[0.0, 0.001, 10.0, 100.0, 1000.0],
    )
    ciss = CurveData(
        name="ciss", x_name="Vds", x_unit="V", y_name="Ciss", y_unit="F",
        x_data=[0.0, 40.0], y_data=[13e-9, 13e-9],
    )
    crss = CurveData(
        name="crss", x_name="Vds", x_unit="V", y_name="Crss", y_unit="F",
        # descending axis on purpose: exercises _interp_curve reordering
        x_data=[40.0, 0.0], y_data=[0.5e-9, 3e-9],
    )
    return transfer, ciss, crss


def test_reconstruct_monotonic_and_segments(synthetic_curves):
    from simulation.testbenches.gate_charge import reconstruct_gate_charge
    tr, ci, cr = synthetic_curves
    df, bp = reconstruct_gate_charge(tr, ci, cr, v_bus=40.0, i_load=50.0,
                                     v_gs_max=10.0)
    q = df["Qg_nC"].to_numpy()
    assert np.all(np.diff(q) >= -1e-9), "Qg must be monotonically nondecreasing"
    assert bp["v_plateau_V"] == pytest.approx(5.0, abs=0.5)
    assert bp["q_gd_nC"] > 0
    assert bp["q_total_nC"] == pytest.approx(
        bp["q_gs_nC"] + bp["q_gd_nC"] + bp["q_tail_nC"], rel=1e-6
    )
    assert list(df.columns) == ["Qg_nC", "Vgs_V"]


def test_reconstruct_qgd_calibration(synthetic_curves):
    """VM-03 fix: passing a datasheet Qgd target must rescale segment 2."""
    from simulation.testbenches.gate_charge import reconstruct_gate_charge
    tr, ci, cr = synthetic_curves
    _, raw_bp = reconstruct_gate_charge(tr, ci, cr, i_load=50.0)
    target = 79.0  # nC, IRFS7730PBF datasheet typ
    _, cal_bp = reconstruct_gate_charge(tr, ci, cr, i_load=50.0,
                                        qgd_cal_nC=target)
    assert cal_bp["q_gd_nC"] == pytest.approx(target, rel=1e-6)
    assert cal_bp["qgd_cal_scale"] == pytest.approx(target / raw_bp["q_gd_nC"])
    # Qgs and tail are untouched by the Miller rescale
    assert cal_bp["q_gs_nC"] == pytest.approx(raw_bp["q_gs_nC"], rel=1e-9)


def test_reconstruct_plateau_clamped_to_vgs_max(synthetic_curves):
    from simulation.testbenches.gate_charge import reconstruct_gate_charge
    tr, ci, cr = synthetic_curves
    df, bp = reconstruct_gate_charge(tr, ci, cr, i_load=50.0, v_gs_max=4.0)
    assert bp["v_plateau_V"] <= 4.0
    assert df["Vgs_V"].max() <= 4.0


def test_reconstruct_rejects_out_of_range_iload(synthetic_curves):
    from simulation.testbenches.gate_charge import reconstruct_gate_charge
    tr, ci, cr = synthetic_curves
    with pytest.raises(ValueError, match="outside transfer-curve range"):
        reconstruct_gate_charge(tr, ci, cr, i_load=5000.0)


def test_reconstruct_descending_crss_axis(synthetic_curves):
    """Ascending and descending Crss axes must give identical Qgd."""
    from common.types import CurveData
    from simulation.testbenches.gate_charge import reconstruct_gate_charge
    tr, ci, cr = synthetic_curves
    cr_asc = CurveData(**{**cr.model_dump(), "x_data": cr.x_data[::-1],
                          "y_data": cr.y_data[::-1]})
    _, bp_a = reconstruct_gate_charge(tr, ci, cr_asc, i_load=50.0)
    _, bp_d = reconstruct_gate_charge(tr, ci, cr, i_load=50.0)
    assert bp_a["q_gd_nC"] == pytest.approx(bp_d["q_gd_nC"], rel=1e-9)
