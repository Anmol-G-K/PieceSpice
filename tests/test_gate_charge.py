"""Unit tests for the quasi-static gate-charge reconstruction (Milestone 2, Curve 5)."""

import numpy as np
import pandas as pd
import pytest

from common.types import CurveData
from simulation.testbenches.gate_charge import reconstruct_gate_charge


@pytest.fixture
def curves():
    transfer = CurveData(
        name="t", x_name="Vgs", x_unit="V", y_name="Id", y_unit="A",
        x_data=[0, 4, 5, 6, 8, 10],
        y_data=[0, 0.001, 10, 100, 700, 1500],
    )
    ciss = CurveData(
        name="c", x_name="Vds", x_unit="V", y_name="Ciss", y_unit="F",
        x_data=[0, 5, 40], y_data=[13e-9, 13e-9, 13e-9],
    )
    crss = CurveData(
        name="c", x_name="Vds", x_unit="V", y_name="Crss", y_unit="F",
        x_data=[0, 1, 5, 40], y_data=[3e-9, 2.4e-9, 1.4e-9, 0.5e-9],
    )
    return transfer, ciss, crss


def test_three_segment_structure(curves):
    transfer, ciss, crss = curves
    df, bp = reconstruct_gate_charge(transfer, ciss, crss, v_bus=38.0, i_load=100.0)
    # Plateau sits where Id crosses 100 A: at the 5->6 V segment edge (Id=100 @ 6 V)
    assert 5.0 <= bp["v_plateau_V"] <= 6.0
    # All breakpoint charges positive
    for key in ("q_gs_nC", "q_gd_nC", "q_tail_nC", "q_total_nC"):
        assert bp[key] > 0
    # Qgs + Qgd + tail == total
    assert np.isclose(
        bp["q_gs_nC"] + bp["q_gd_nC"] + bp["q_tail_nC"], bp["q_total_nC"], rtol=1e-9
    )


def test_dataframe_format_and_monotonicity(curves):
    transfer, ciss, crss = curves
    df, _ = reconstruct_gate_charge(transfer, ciss, crss)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Qg_nC", "Vgs_V"]
    assert (df["Qg_nC"].diff().dropna() >= -1e-9).all()  # monotonic non-decreasing
    assert df["Vgs_V"].iloc[0] == 0.0
    assert df["Vgs_V"].iloc[-1] == pytest.approx(10.0)


def test_ascending_and_descending_capacitance_axes_agree(curves):
    """Curve 4 may return Vds ascending or log-ascending; results must not flip sign."""
    transfer, ciss, crss = curves
    crss_rev = CurveData(
        name="c", x_name="Vds", x_unit="V", y_name="Crss", y_unit="F",
        x_data=crss.x_data[::-1], y_data=crss.y_data[::-1],
    )
    _, bp1 = reconstruct_gate_charge(transfer, ciss, crss)
    _, bp2 = reconstruct_gate_charge(transfer, ciss, crss_rev)
    assert np.isclose(bp1["q_gd_nC"], bp2["q_gd_nC"], rtol=1e-9)
    assert bp1["q_gd_nC"] > 0 and bp2["q_gd_nC"] > 0


def test_iload_outside_transfer_range_raises(curves):
    transfer, ciss, crss = curves
    with pytest.raises(ValueError, match="outside transfer-curve range"):
        reconstruct_gate_charge(transfer, ciss, crss, i_load=5000.0)


def test_plateau_clamped_to_vgs_max(curves):
    transfer, ciss, crss = curves
    df, bp = reconstruct_gate_charge(transfer, ciss, crss, i_load=100.0, v_gs_max=5.5)
    assert bp["v_plateau_V"] == pytest.approx(5.5)
    assert bp["q_tail_nC"] == pytest.approx(0.0, abs=1e-9)


def test_qgd_datasheet_calibration(curves):
    """VM-03: passing a datasheet Qgd rescales the Miller segment exactly."""
    transfer, ciss, crss = curves
    _, bp_raw = reconstruct_gate_charge(transfer, ciss, crss)
    target = 79.0  # nC, datasheet typ for IRFS7730PBF
    _, bp_cal = reconstruct_gate_charge(transfer, ciss, crss, qgd_cal_nC=target)
    assert bp_cal["q_gd_nC"] == pytest.approx(target, rel=1e-9)
    assert bp_cal["qgd_cal_scale"] == pytest.approx(
        target / (bp_raw["q_gd_nC"] / bp_raw["qgd_cal_scale"]), rel=1e-9
    )
    # Qgs and tail unchanged by Miller calibration
    assert bp_cal["q_gs_nC"] == pytest.approx(bp_raw["q_gs_nC"], rel=1e-9)
    assert bp_cal["q_tail_nC"] == pytest.approx(bp_raw["q_tail_nC"], rel=1e-9)
    # Total shifts by exactly the Qgd delta
    assert bp_cal["q_total_nC"] == pytest.approx(
        bp_raw["q_total_nC"] + (target - bp_raw["q_gd_nC"] / bp_raw["qgd_cal_scale"]),
        rel=1e-6,
    )
