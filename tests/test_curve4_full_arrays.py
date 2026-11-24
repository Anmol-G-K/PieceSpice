"""Offline tests for full Curve-4 C(V) array persistence (no LTspice)."""

import math

import numpy as np
import pytest

from common.types import CurveData
from config.yaml_io import load_output, save_output
from pipeline.curves import _curve4_extracted

CAP_REF = 1.5
VDS = [0.1 * (10.0 ** (i / 2.0)) for i in range(8)]
CISS = [2.0e-9 / (1.0 + 0.05 * v) for v in VDS]
COSS = [5.0e-10 / (1.0 + 0.12 * v) ** 1.5 for v in VDS]
CRSS = [4.0e-11 / (1.0 + 0.3 * v) for v in VDS]

SCALARS = ("ciss_f", "coss_f", "crss_f", "cap_ref_vds")
ARRAYS = {
    "curve_vds": VDS,
    "ciss_curve_f": CISS,
    "coss_curve_f": COSS,
    "crss_curve_f": CRSS,
}


def _curve(name, x, y):
    return CurveData(
        name=name,
        x_name="Drain-Source Voltage (Vds)", x_unit="V",
        y_name="Capacitance", y_unit="F",
        x_data=list(x), y_data=list(y),
    )


def _bench():
    """Synthetic CapacitanceTestbench.run() result."""
    return (_curve("Ciss", VDS, CISS), _curve("Coss", VDS, COSS),
            _curve("Crss", VDS, CRSS))


def _extracted():
    return _curve4_extracted(*_bench(), cap_ref=CAP_REF)


def test_full_arrays_match_bench_order_and_length():
    got = _extracted()
    for key, source in ARRAYS.items():
        assert key in got
        assert isinstance(got[key], list)
        assert len(got[key]) == len(source)
        assert got[key] == list(source)
        assert all(type(v) is float and math.isfinite(v) for v in got[key])


def test_scalar_fields_unchanged():
    ciss, coss, crss = _bench()
    got = _curve4_extracted(ciss, coss, crss, CAP_REF)
    assert set(SCALARS) <= set(got)
    assert got["ciss_f"] == pytest.approx(float(np.interp(CAP_REF, ciss.x_data, ciss.y_data)))
    assert got["coss_f"] == pytest.approx(float(np.interp(CAP_REF, coss.x_data, coss.y_data)))
    assert got["crss_f"] == pytest.approx(float(np.interp(CAP_REF, crss.x_data, crss.y_data)))
    assert got["cap_ref_vds"] == CAP_REF


def test_vds_strictly_ascending():
    vds = _extracted()["curve_vds"]
    assert vds == sorted(vds)
    assert all(b > a for a, b in zip(vds, vds[1:]))


def test_yaml_roundtrip_preserves_finite_floats(tmp_path, monkeypatch):
    monkeypatch.setattr("config.yaml_io.OUTPUT_DIR", tmp_path / "outputs")
    extracted = _extracted()
    save_output("synthetic", "curve4_capacitances", "pass", {}, extracted)
    back = load_output("synthetic", "curve4_capacitances")["extracted"]
    for key, source in ARRAYS.items():
        assert back[key] == list(source)
        assert all(type(v) is float and math.isfinite(v) for v in back[key])
    for key in SCALARS:
        assert back[key] == extracted[key]


@pytest.mark.parametrize("mutation", [
    lambda c, o, r: setattr(c, "x_data", None),
    lambda c, o, r: setattr(o, "y_data", COSS[:-1]),
    lambda c, o, r: setattr(r, "x_data", VDS[:-1]),
    lambda c, o, r: setattr(o, "y_data", [COSS[0], float("nan"), *COSS[2:]]),
    lambda c, o, r: setattr(c, "y_data", [float("inf"), *CISS[1:]]),
    lambda c, o, r: [setattr(cv, "x_data", list(reversed(VDS))) for cv in (c, o, r)],
    lambda c, o, r: [setattr(cv, "x_data", [VDS[0], VDS[0], *VDS[2:]]) for cv in (c, o, r)],
    lambda c, o, r: [setattr(cv, "x_data", VDS[:1]) for cv in (c, o, r)],
    lambda c, o, r: setattr(r, "x_data", [v + 0.01 for v in VDS]),
])
def test_missing_or_short_arrays_fail_closed(mutation):
    ciss, coss, crss = _bench()
    mutation(ciss, coss, crss)
    with pytest.raises(ValueError):
        _curve4_extracted(ciss, coss, crss, CAP_REF)


def test_missing_curve_object_fails_closed():
    ciss, coss, _ = _bench()
    with pytest.raises(ValueError, match="missing"):
        _curve4_extracted(ciss, coss, None, CAP_REF)


def test_run_all_curves_persists_full_arrays_offline(tmp_path, monkeypatch):
    """End-to-end persistence through run_all_curves with synthetic benches."""
    import pipeline.curves as curves

    monkeypatch.setattr("config.yaml_io.OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(curves, "load_input", lambda vendor: _config())
    monkeypatch.setattr("parser.netlist.resolve_pin_connections",
                        lambda *a, **k: ["gate", "drain", "source"])
    monkeypatch.setattr(curves, "StaticSweepTestbench", _FakeStaticSweep)
    monkeypatch.setattr(curves, "CapacitanceTestbench",
                        lambda **kw: _FakeCapBench())
    monkeypatch.setattr(curves, "reconstruct_gate_charge", _fake_gate_charge)

    curves.run_all_curves("synthetic")
    saved = load_output("synthetic", "curve4_capacitances")["extracted"]
    for key, source in ARRAYS.items():
        assert saved[key] == list(source)
    for key in SCALARS:
        assert saved[key] == pytest.approx(_extracted()[key])


def _config():
    return {
        "library": "fake.lib",
        "subcircuit": "FAKE",
        "device": "synthetic",
        "conditions": {
            "temperature": 25.0,
            "Vgs_sweep": [4.0, 6.0], "Vds_sweep_max": 20.0, "Vds_points": 3,
            "Vds_fixed": 10.0, "Vgs_transfer_max": 4.0,
            "Vgs_transfer_points": 3, "Id_load": 1.0,
            "Vgs_fixed": 10.0, "Vds_rds_max": 2.0, "Vds_rds_points": 3,
            "Id_rds_min": 0.1,
            "Vds_cap_min": 0.1, "Vds_cap_max": 60.0, "Cap_points": 8,
            "F_meas": 1e6, "Cap_ref_Vds": CAP_REF,
            "Vbus": 400.0, "Vgs_qg_max": 10.0, "Vds_on": 2.0,
            "Qgd_typ_nC": 20.0,
            "Vsd_max": 3.0, "Vsd_points": 3,
            "Id_th_target": 1e-6, "Vgs_th_max": 4.0,
        },
    }


class _FakeStaticSweep:
    @staticmethod
    def _c(name, x, y):
        return _curve(name, x, y)

    def run_output_characteristics(self, *a, **k):
        return [self._c("Id", [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])]

    def run_transfer_characteristics(self, *a, **k):
        return self._c("Id", [0.0, 2.0, 4.0], [0.0, 1.0, 5.0])

    def run_rds_on(self, *a, **k):
        return self._c("Rds", [0.0, 1.0, 2.0], [1.0, 2.0, 3.0])

    def run_body_diode(self, *a, **k):
        return self._c("Isd", [0.0, 0.5, 1.0], [0.0, 1.0, 2.0])

    def run_threshold_voltage(self, *a, **k):
        return (self._c("Id", [0.0, 2.0, 4.0], [1e-6, 2e-6, 3e-6]), 2.0)


class _FakeCapBench:
    def run(self, *a, **k):
        return _bench()


def _fake_gate_charge(tr, ciss, crss, **kw):
    frame = {"Qg_nC": np.array([0.0, 1.0, 2.0]),
             "Vgs_V": np.array([0.0, 5.0, 10.0])}
    breaks = {"v_plateau_V": 5.0, "q_gs_nC": 10.0, "q_gd_nC": 20.0,
              "q_tail_nC": 5.0, "q_total_nC": 35.0}
    return frame, breaks


def test_check_bias_valid_allows_offstate_ciss_series_rg():
    """Off-state Ciss (Vgs<=0) with series R_g should not trip channel-on guard."""
    from simulation.testbenches.capacitance import (
        CapacitanceTestbench, _BiasMeasurementError)
    # Re=-2.63, Im=-11.3 -> |Re|/|Im|=0.233 > 0.1 (IRFS7730PBF at Vgs=-10V)
    CapacitanceTestbench._check_bias_valid(
        complex(-2.63, -11.3), 0.1, -10.0, "Ciss")
    with pytest.raises(_BiasMeasurementError):
        CapacitanceTestbench._check_bias_valid(
            complex(-2.63, -11.3), 0.1, 2.0, "Ciss")
    with pytest.raises(_BiasMeasurementError):
        CapacitanceTestbench._check_bias_valid(
            complex(-2.63, -11.3), 0.1, -10.0, "Coss")


def test_extract_r_gate_from_ac_probe():
    """extract_r_gate returns -Re(V(gate)) from 1 A AC probe."""
    from types import SimpleNamespace
    from simulation.testbenches.capacitance import CapacitanceTestbench

    class _FakeEngine:
        temperature = 25.0

        def run_ac_sweep(self, text, directive):
            assert "Iprobe gate 0 DC 0 AC 1" in text
            return SimpleNamespace(
                traces={"V(gate)": np.array([-2.6288 - 0.5j])})

    tb = CapacitanceTestbench(engine=_FakeEngine())
    rg = tb.extract_r_gate("fake.lib", "FAKE", vds=30.0,
                           pin_roles=["drain", "gate", "source"])
    assert np.isclose(rg, 2.6288)
