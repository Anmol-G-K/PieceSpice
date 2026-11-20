"""Offline unit tests for Tier-4 loss-LUT extraction (no LTspice runs).

Every input is a real config/output file or a synthetic waveform built in
memory.  The simulator-facing sweep is monkeypatched in every builder test,
so these tests prove extraction/provenance logic without measuring runtime.
"""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from approximation.loss_lut import LossLUT, _node_key, build_lut
from config.yaml_io import CONFIG_ROOT, OUTPUT_DIR, load_input, load_output
from simulation.testbenches import loss_lut_bench as bench
from simulation.testbenches.double_pulse import DoublePulseRun

VENDORS = ("cpm2-1200", "csd16412q5a", "epc2216", "irfs7730pbf")


def _synthetic_dpt(vbus: float = 10.0, current: float = 10.0,
                   t_end: float = 14e-6, truncate: bool = False,
                   gate_off: bool = False) -> DoublePulseRun:
    """In-memory clamped-inductive waveform with known 10/90 % crossings."""
    end = 5e-6 if truncate else t_end
    t = np.arange(0.0, end, 2e-9)
    gate = np.zeros_like(t) if gate_off else np.interp(
        t,
        [0, 1e-6, 1.02e-6, 6e-6, 6.02e-6, 10e-6, 10.02e-6,
         12e-6, 12.02e-6, t_end],
        [0, 0, 10.0, 10.0, 0, 0, 10.0, 10.0, 0, 0])
    vds = np.interp(
        t,
        [0, 10.01e-6, 10.03e-6, 12e-6, 12.02e-6, t_end],
        [vbus, vbus, 0.05, 0.05, vbus, vbus])
    ids = np.interp(
        t,
        [0, 10.01e-6, 10.03e-6, 12e-6, 12.02e-6, t_end],
        [0, 0, current, current, 0, 0])
    return DoublePulseRun(
        time=t, v_ds=vds, i_d=ids, v_gs=gate, v_bus=vbus,
        i_load=current, l_load=vbus * 5e-6 / current, vgs_on=10.0,
        wall_clock_s=0.25, t2_on=10e-6)


def _fake_sweep(vendor: str, **kwargs) -> tuple[bench.SweepResult, ...]:
    """Reuse recorded golden energy only at its own condition; leave gaps."""
    cfg = kwargs.get("config") or load_input(vendor)
    points = kwargs.get("points") or bench.derive_grid(cfg)[0]
    rec = bench.recorded_golden(vendor)
    results = []
    for point in points:
        conditions = bench.dpt_kwargs(cfg, point)
        if (rec["available"] and rec["evaluable"]
                and bench._recorded_matches(rec, point, cfg)):
            results.append(bench.SweepResult(
                point, float(rec["golden"]["e_on_uj"]),
                float(rec["golden"]["e_off_uj"]), "measured",
                "recorded golden DPT", f"{rec['path']}#golden_metrics",
                conditions, {"repeats": rec["repeats"]}))
        else:
            results.append(bench.SweepResult(
                point, math.nan, math.nan, "unmeasured_grid_node",
                "offline test: no simulation", "offline", conditions))
    return tuple(results)


@pytest.mark.parametrize("vendor", VENDORS)
def test_grid_axes_and_inductance_rule_come_only_from_config(vendor):
    cfg = load_input(vendor)
    points, rules = bench.derive_grid(cfg)
    assert {p.vbus_v for p in points} == {
        float(cfg["dpt"]["vbus"]), float(cfg["conditions"]["Vbus"])}
    assert {p.current_a for p in points} == {
        float(cfg["dpt"]["i_load_target"]), float(cfg["conditions"]["Id_load"])}
    for p in points:
        assert p.l_load_h == pytest.approx(
            p.vbus_v * float(cfg["dpt"]["t1_width"]) / p.current_a)
        assert "t1_width" in p.l_load_rule
    crossing = [p for p in points
                if p.vbus_v == float(cfg["dpt"]["vbus"])
                and p.current_a == float(cfg["dpt"]["i_load_target"])]
    assert len(crossing) == 1
    assert crossing[0].l_load_h == pytest.approx(float(cfg["dpt"]["l_load"]))
    assert rules["kind"] == "project_decision"
    assert rules["vbus"].startswith("exact config values")


def test_provenance_locators_point_at_real_config_and_curve_files():
    vendor = "csd16412q5a"
    provenance = bench.config_provenance(vendor)
    path = CONFIG_ROOT / "inputs" / f"{vendor}.yaml"
    assert provenance["file"] == path.relative_to(
        Path(__file__).resolve().parent.parent).as_posix()
    assert provenance["sha256"] == sha256(path.read_bytes()).hexdigest()
    for testbench in ("curve3_rds_on", "curve7_threshold_voltage"):
        meta = bench.curve_provenance(vendor, testbench)
        assert meta["available"]
        assert (OUTPUT_DIR / vendor / f"{testbench}.yaml").is_file()


def test_extract_energy_from_synthetic_golden_run_records_guard():
    run = _synthetic_dpt()
    result = bench.extract_energy_from_run(
        run, t_end=14e-6, source="offline synthetic",
        reference_points=len(run.time), iterations=1234)
    assert result.status == "measured"
    assert result.e_on_uj > 0.0 and result.e_off_uj > 0.0
    assert result.guard["trace_points"] == len(run.time)
    assert result.guard["iterations"] == 1234
    assert result.guard["t_last_s"] == pytest.approx(run.time[-1])
    assert result.guard["t_end_s"] == 14e-6


def test_truncated_and_crossingless_golden_runs_are_not_evaluable():
    truncated = bench.extract_energy_from_run(
        _synthetic_dpt(truncate=True), t_end=14e-6, source="offline synthetic")
    assert truncated.status == "not_evaluable"
    assert "incomplete" in truncated.reason
    assert math.isnan(truncated.e_on_uj) and math.isnan(truncated.e_off_uj)
    assert truncated.guard["t_last_s"] < 0.999 * 14e-6

    no_crossing = bench.extract_energy_from_run(
        _synthetic_dpt(gate_off=True), t_end=14e-6, source="offline synthetic")
    assert no_crossing.status == "not_evaluable"
    assert "crossing" in no_crossing.reason
    assert math.isnan(no_crossing.e_on_uj) and math.isnan(no_crossing.e_off_uj)


def test_negative_loss_is_a_gap_not_a_zero(monkeypatch):
    monkeypatch.setattr(
        bench, "extract_dpt_metrics",
        lambda run: {"e_on_uj": -1.0, "e_off_uj": 2.0})
    result = bench.extract_energy_from_run(
        _synthetic_dpt(), t_end=14e-6, source="offline synthetic")
    assert result.status == "not_evaluable"
    assert "negative" in result.reason
    assert math.isnan(result.e_on_uj) and math.isnan(result.e_off_uj)


@pytest.mark.parametrize("vendor", VENDORS)
def test_offline_builder_sources_recorded_node_and_leaves_explicit_gaps(
    vendor, monkeypatch
):
    monkeypatch.setattr(bench, "run_golden_sweep", _fake_sweep)
    cfg = load_input(vendor)
    lut = build_lut(vendor, simulate=False)
    assert lut.vendor == vendor
    assert lut.temperature_c == float(cfg["conditions"]["temperature"])
    assert lut.vbus_points == tuple(sorted({
        float(cfg["dpt"]["vbus"]), float(cfg["conditions"]["Vbus"])}))
    assert lut.current_points == tuple(sorted({
        float(cfg["dpt"]["i_load_target"]), float(cfg["conditions"]["Id_load"])}))

    key = _node_key(float(cfg["dpt"]["vbus"]), float(cfg["dpt"]["i_load_target"]))
    node = lut.source["nodes"][key]
    assert node["status"] == "measured"
    assert "dpt_metrics.json" in node["source"]
    energy = lut.energy_at(float(cfg["dpt"]["vbus"]),
                           float(cfg["dpt"]["i_load_target"]))
    assert energy["status"] in ("measured", "constant_node", "single_node")

    assert energy["e_on_uj"] == pytest.approx(
        node["e_on_uj"], rel=1e-12)
    assert energy["source"] == node
    assert lut.source["dpt_crossing_evaluable"] is True
    assert lut.source["waveform_gate"] == "retired_by_design"

    gaps = lut.source["gaps"]
    node_gaps = [g for g in gaps if g["metric"] == "e_on_uj,e_off_uj"]
    assert len(node_gaps) == len(lut.vbus_points) * len(lut.current_points) - 1
    assert all(g["status"] == "unmeasured_grid_node" for g in node_gaps)
    assert any(g["metric"] == "r_off_ohms" and g["status"] == "not_measured"
               for g in gaps)
    for v, i in [(v, i) for v in lut.vbus_points for i in lut.current_points
                 if (v, i) != (float(cfg["dpt"]["vbus"]),
                               float(cfg["dpt"]["i_load_target"]))]:
        missing = lut.energy_at(v, i)
        assert missing["status"] != "measured"
        assert math.isnan(missing["e_on_uj"]) and math.isnan(missing["e_off_uj"])

    curve3 = load_output(vendor, "curve3_rds_on")["extracted"]
    curve7 = load_output(vendor, "curve7_threshold_voltage")["extracted"]
    assert len(lut.r_on_ohms) == len(lut.current_points)
    i_ref = float(cfg["conditions"]["Id_load"])
    assert lut.r_on_at(i_ref) == pytest.approx(
        float(curve3["rds_on_mohm_at_Id_load"]) / 1000.0, rel=2e-3)
    assert lut.vth_v == pytest.approx(float(curve7["vth_v"]))
    assert lut.source["config"]["sha256"] == bench.config_provenance(vendor)["sha256"]

    restored = LossLUT.from_dict(json.loads(json.dumps(lut.to_dict())))
    for original, copy in ((lut.e_on_uj, restored.e_on_uj),
                           (lut.e_off_uj, restored.e_off_uj)):
        for row_a, row_b in zip(original, copy):
            for a, b in zip(row_a, row_b):
                assert (math.isnan(a) and math.isnan(b)) or a == b
    assert restored.r_on_ohms == lut.r_on_ohms
    assert restored.vth_v == lut.vth_v


def test_recorded_energy_is_not_reused_at_a_different_condition():
    cfg = load_input("csd16412q5a")
    points, _ = bench.derive_grid(cfg)
    other = dict(bench.recorded_golden("csd16412q5a"))
    other["conditions"] = dict(other["conditions"], vbus=999.0)
    results = bench.run_golden_sweep(
        "csd16412q5a", config=cfg, points=points, recorded=other, simulate=False)
    assert all(r.status == "unmeasured_grid_node" for r in results)
    assert all(math.isnan(r.e_on_uj) for r in results)


def _four_node_lut(e_on, e_off) -> LossLUT:
    return LossLUT(
        vendor="toy", temperature_c=25.0,
        vbus_points=(10.0, 20.0), current_points=(5.0, 10.0),
        e_on_uj=e_on, e_off_uj=e_off,
        r_on_ohms=(0.01, 0.02), vth_v=2.0, vgs_on_v=10.0,
        source={"nodes": {}, "gaps": []})


def test_energy_interpolation_and_nan_corner_policy():
    lut = _four_node_lut(((1.0, 2.0), (3.0, 4.0)),
                         ((10.0, 20.0), (30.0, 40.0)))
    exact = lut.energy_at(10.0, 5.0)
    assert exact["status"] == "measured" and exact["e_on_uj"] == 1.0
    mid = lut.energy_at(15.0, 7.5)
    assert mid["status"] == "interpolated"
    assert mid["e_on_uj"] == pytest.approx(2.5)
    assert mid["e_off_uj"] == pytest.approx(25.0)

    partial = _four_node_lut(((1.0, 2.0), (3.0, math.nan)),
                             ((10.0, 20.0), (30.0, 40.0)))
    missing = partial.energy_at(15.0, 7.5)
    assert missing["status"] == "insufficient_measured_nodes"
    assert math.isnan(missing["e_on_uj"]) and math.isnan(missing["e_off_uj"])
    assert partial.energy_at(10.0, 5.0)["e_on_uj"] == 1.0


def test_no_extrapolation_and_honest_single_node_status():
    lut = _four_node_lut(((1.0, 2.0), (3.0, 4.0)),
                         ((10.0, 20.0), (30.0, 40.0)))
    outside = lut.energy_at(25.0, 7.5)
    assert outside["status"] == "outside_grid"
    assert math.isnan(outside["e_on_uj"])
    with pytest.raises(TypeError):
        lut.energy_at(25.0, 7.5, extrapolate=True)
    assert math.isnan(lut.r_on_at(99.0))

    single = LossLUT(
        vendor="toy", temperature_c=25.0,
        vbus_points=(12.5,), current_points=(10.0,),
        e_on_uj=((1.0,),), e_off_uj=((10.0,),),
        r_on_ohms=(0.01,), vth_v=2.0, vgs_on_v=10.0,
        source={"nodes": {}, "gaps": []})
    assert single.interpolation_capability()["status"] == "constant_node"
    at_node = single.energy_at(12.5, 10.0)
    assert at_node["status"] == "constant_node"
    assert at_node["e_on_uj"] == 1.0
    assert math.isnan(single.energy_at(13.0, 10.0)["e_on_uj"])

    one_finite = _four_node_lut(((1.0, math.nan), (math.nan, math.nan)),
                                ((10.0, math.nan), (math.nan, math.nan)))
    assert one_finite.interpolation_capability()["status"] == "single_node"
    assert one_finite.energy_at(10.0, 5.0)["status"] == "single_node"


def test_efficiency_requires_explicit_output_and_frequency():
    lut = _four_node_lut(((1.0, 2.0), (3.0, 4.0)),
                         ((10.0, 20.0), (30.0, 40.0)))
    no_pout = lut.efficiency_at(10.0, 5.0, f_sw_hz=1000.0)
    assert no_pout["status"] == "output_power_required"
    assert math.isnan(no_pout["efficiency"])
    assert "Vbus*Iload" in no_pout["reason"]
    assert no_pout["p_cond_w"] == pytest.approx(0.25)
    assert no_pout["p_switch_w"] == pytest.approx(0.011)
    assert "thermal" in no_pout["note"].lower()

    explicit = lut.efficiency_at(10.0, 5.0, 100.0, f_sw_hz=1000.0)
    assert explicit["status"] == "computed"
    assert explicit["p_total_w"] == pytest.approx(0.261)
    assert explicit["efficiency"] == pytest.approx(100.0 / 100.261)
    assert explicit["efficiency"] < 1.0

    loss_only = lut.losses_at(10.0, 5.0, f_sw_hz=1000.0)
    assert "efficiency" not in loss_only
    assert loss_only["p_total_w"] == pytest.approx(
        loss_only["p_cond_w"] + loss_only["p_switch_w"])

    no_fsw = lut.efficiency_at(10.0, 5.0, 100.0)
    assert no_fsw["status"] == "frequency_required"
    assert math.isnan(no_fsw["p_switch_w"])
    assert math.isnan(no_fsw["efficiency"])

    zero_fsw = lut.efficiency_at(10.0, 5.0, 100.0, f_sw_hz=0.0)
    assert zero_fsw["p_switch_w"] == 0.0
    assert zero_fsw["status"] == "computed"
    assert zero_fsw["efficiency"] == pytest.approx(100.0 / 100.25)

    gapped = _four_node_lut(((math.nan, 2.0), (3.0, 4.0)),
                            ((10.0, 20.0), (30.0, 40.0)))
    result = gapped.efficiency_at(10.0, 5.0, 100.0, f_sw_hz=1000.0)
    assert math.isnan(result["efficiency"])
    assert result["status"] == "not_evaluable"


def test_negative_energy_is_a_diagnostic_row_not_a_construction_failure():
    lut = _four_node_lut(((-1.0, 2.0), (3.0, 4.0)),
                         ((10.0, 20.0), (30.0, 40.0)))
    assert math.isnan(lut.e_on_uj[0][0])
    diagnostic = [g for g in lut.source["gaps"] if g["status"] == "negative_loss"]
    assert len(diagnostic) == 1
    assert diagnostic[0]["recorded_value"] == -1.0
    assert "cropped" in diagnostic[0]["reason"]
    energy = lut.energy_at(10.0, 5.0)
    assert math.isnan(energy["e_on_uj"])
    assert energy["status"] == "not_evaluable"
    restored = LossLUT.from_dict(json.loads(json.dumps(lut.to_dict())))
    assert math.isnan(restored.e_on_uj[0][0])

    with pytest.raises(ValueError, match="shape"):
        LossLUT("toy", 25.0, (10.0, 20.0), (5.0, 10.0),
                ((1.0, 2.0),), ((10.0, 20.0), (30.0, 40.0)))


@pytest.mark.parametrize("vendor", VENDORS)
def test_every_recorded_condition_is_checked(vendor):
    cfg = load_input(vendor)
    points, _ = bench.derive_grid(cfg)
    rec = bench.recorded_golden(vendor)
    point = next(p for p in points
                 if p.vbus_v == float(rec["conditions"]["vbus"])
                 and p.current_a == float(rec["conditions"]["i_load_target"]))
    check = bench._recorded_condition_check(rec, point, cfg)
    assert check["match"] is True
    assert len(check["matched"]) == len(rec["conditions"])
    assert check["mismatched"] == []
    assert [d["key"] for d in check["not_recorded"]] == ["temperature"]

    for key in rec["conditions"]:
        mutated = dict(rec)
        mutated["conditions"] = dict(rec["conditions"], **{key: 1.234567e9})
        bad = bench._recorded_condition_check(mutated, point, cfg)
        assert bad["match"] is False
        assert any(m["key"] == key for m in bad["mismatched"])
        results = bench.run_golden_sweep(
            vendor, config=cfg, points=(point,), recorded=mutated, simulate=False)
        assert results[0].status == "unmeasured_grid_node"
        assert results[0].guard["condition_check"]["match"] is False


def test_explicit_points_republish_their_own_rules(monkeypatch):
    monkeypatch.setattr(bench, "run_golden_sweep", _fake_sweep)
    vendor = "epc2216"
    cfg = load_input(vendor)
    points, derived = bench.derive_grid(cfg)
    subset = tuple(points[:1])
    lut = build_lut(vendor, config=cfg, simulate=False, points=subset)
    rules = lut.source["grid"]["rules"]
    assert rules["kind"] == "project_decision_explicit_points"
    assert "caller-supplied" in rules["vbus"]
    assert derived["kind"] == "project_decision"
    assert lut.vbus_points == tuple(sorted({p.vbus_v for p in subset}))


def test_lut_publishes_data_limitations_and_device_guard():
    lut = _four_node_lut(((1.0, 2.0), (3.0, 4.0)),
                         ((10.0, 20.0), (30.0, 40.0)))
    record = lut.to_dict()
    assert record["interpolation"]["status"] == "grid_interpolation"
    limit = record["data_limitations"][0]
    assert limit["kind"] == "data_limitation"
    assert "no temperature axis" in limit["rule"]
    assert "not" in limit["rule"] and "electrical power" in limit["rule"]
    assert record["device_values_finite"]["finite"] is True
    assert "not the electrical power" in record["thermal_loss_note"]

    bad = LossLUT("toy", 25.0, (10.0,), (5.0,), ((1.0,),), ((10.0,),),
                  r_on_ohms=(0.01,), vth_v=math.nan, vgs_on_v=10.0)
    assert bad.device_values_finite()["finite"] is False
    assert bad.to_dict()["device_values_finite"]["vth_finite"] is False


def test_representative_current_index_tie_break_is_lower_index():
    lut = LossLUT("toy", 25.0, (10.0,), (10.0, 25.0),
                  ((1.0, 2.0),), ((10.0, 20.0),),
                  r_on_ohms=(0.01, 0.02), vth_v=2.0, vgs_on_v=10.0)
    index, rule = lut.representative_current_index()
    assert index == 0
    assert "lower" in rule
    assert lut.representative_current_index(25.0)[0] == 1
    doc = LossLUT.representative_current_index.__doc__
    assert "even-length" in doc and "NOT itself a measured" in doc


def test_offline_multi_node_grid_reports_single_node_capability(monkeypatch):
    monkeypatch.setattr(bench, "run_golden_sweep", _fake_sweep)
    lut = build_lut("epc2216", simulate=False)
    assert lut.interpolation_capability()["status"] == "single_node"
    assert lut.source["interpolation"]["finite_nodes"] == 1
    assert lut.energy_at(11.0, 10.0)["status"] == "single_node"
    assert math.isnan(lut.energy_at(12.0, 25.0)["e_on_uj"])


@pytest.mark.parametrize("vendor", VENDORS)
def test_coss_f_is_measured_curve4_value_with_full_provenance(vendor, monkeypatch):
    monkeypatch.setattr(bench, "run_golden_sweep", _fake_sweep)
    cfg = load_input(vendor)
    lut = build_lut(vendor, simulate=False)
    extracted = load_output(vendor, "curve4_capacitances")["extracted"]
    assert lut.coss_f == pytest.approx(float(extracted["coss_f"]), rel=1e-12)

    meta = lut.source["coss_f"]
    assert meta["kind"] == "measured"
    assert meta["file"].endswith(f"{vendor}/curve4_capacitances.yaml")
    assert meta["key"] == "extracted.coss_f"
    assert meta["value_f"] == lut.coss_f
    assert meta["ref_voltage_v"] == pytest.approx(float(extracted["cap_ref_vds"]))
    assert meta["ref_voltage_key"] == "extracted.cap_ref_vds"
    assert meta["temperature_c"] == float(cfg["conditions"]["temperature"])
    assert meta["temperature_key"] == "conditions.temperature"
    assert meta["unit"] == "F"
    assert "solver-stabilizing" in meta["rule"]
    assert lut.to_dict()["coss_f"] == lut.coss_f


@pytest.mark.parametrize("vendor", VENDORS)
def test_offline_r_off_is_nan_with_loud_fallback_disclosure(vendor, monkeypatch):
    monkeypatch.setattr(bench, "run_golden_sweep", _fake_sweep)
    lut = build_lut(vendor, simulate=False)
    assert math.isnan(lut.r_off_ohms)

    meta = lut.source["r_off_ohms"]
    assert meta["kind"] == "not_measured"
    assert meta["bench"] == bench.ROFF_BENCH
    assert meta["vgs_v"] == 0.0
    assert meta["vbus_v"] == float(load_input(vendor)["dpt"]["vbus"])
    assert meta["vendor_file"]["sha256"] == sha256(
        (Path(__file__).resolve().parents[1] / load_input(vendor)["library"])
        .read_bytes()).hexdigest()
    assert meta["config_sha256"] == bench.config_provenance(vendor)["sha256"]
    assert meta["fallback"]["source"] == "project_decision"
    assert "disclosure" in meta["fallback"]["note"]
    assert "no live vendor DC sweep" in meta["reason"]

    guard = lut.to_dict()["device_values_finite"]
    assert guard["finite"] is True
    assert guard["r_off_measured"] is False
    assert "falls back" in guard["r_off_note"]
    assert math.isnan(lut.to_dict()["r_off_ohms"])


def test_r_off_from_sweep_uses_final_finite_sample_and_rejects_bad_data():
    meta = {"bench": "fake"}
    value, out = bench.r_off_from_sweep(
        [0.0, 5.0, 10.0, 12.5, 12.5],
        [0.0, 1e-12, 1e-12, 5e-12, float("nan")],
        vbus_v=12.5, meta=meta)
    assert value == pytest.approx(12.5 / 5e-12)
    assert out["kind"] == "measured"
    assert out["value_ohms"] == value
    assert out["ioff_a"] == 5e-12
    assert out["final_sample_vds_v"] == 12.5

    value, negative = bench.r_off_from_sweep(
        [0.0, 1.0, 2.0], [0.0, 1e-12, -1e-9], vbus_v=12.5, meta=meta)
    assert math.isnan(value)
    assert negative["kind"] == "measurement_failed"
    assert "not a positive" in negative["reason"]

    value, empty = bench.r_off_from_sweep(
        [0.0, 1.0], [float("nan"), float("nan")], vbus_v=12.5, meta=meta)
    assert math.isnan(value)
    assert empty["kind"] == "measurement_failed"
    assert "no finite" in empty["reason"]


def test_measure_r_off_ohms_runs_vgs_zero_dc_sweep(monkeypatch):
    vendor = "csd16412q5a"
    cfg = load_input(vendor)
    vbus = float(cfg["dpt"]["vbus"])
    calls: dict[str, Any] = {}
    monkeypatch.setattr("parser.netlist.resolve_pin_connections",
                        lambda lib, subckt, **kw: ["drain", "gate", "source"])

    class _FakeTB:
        def run_output_characteristics(self, lib, subckt, **kwargs):
            calls.update(lib=lib, subckt=subckt, **kwargs)
            return [SimpleNamespace(x_data=[0.0, vbus],
                                    y_data=[0.0, 4e-12])]

    monkeypatch.setattr(
        "simulation.testbenches.static_sweep.StaticSweepTestbench", _FakeTB)
    value, meta = bench.measure_r_off_ohms(vendor, cfg, simulate=True)
    assert value == pytest.approx(vbus / 4e-12)
    assert calls["vgs_list"] == [0.0]
    assert calls["vds_max"] == vbus
    assert calls["vds_points"] == int(cfg["conditions"]["Vds_points"])
    assert calls["pin_roles"] == ["drain", "gate", "source"]
    assert calls["subckt"] == cfg["subcircuit"]
    assert meta["kind"] == "measured"
    assert meta["value_ohms"] == value
    assert meta["ioff_a"] == 4e-12


def test_measure_r_off_ohms_failure_is_nan(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("no vendor dc convergence")

    monkeypatch.setattr("parser.netlist.resolve_pin_connections", boom)
    value, meta = bench.measure_r_off_ohms(
        "csd16412q5a", load_input("csd16412q5a"), simulate=True)
    assert math.isnan(value)
    assert meta["kind"] == "measurement_failed"
    assert "RuntimeError" in meta["reason"]
    assert meta["fallback"]["source"] == "project_decision"


def test_stabilization_fields_survive_json_roundtrip():
    lut = LossLUT("toy", 25.0, (10.0,), (10.0,), ((1.0,),), ((10.0,),),
                  r_on_ohms=(0.01,), vth_v=2.0, vgs_on_v=10.0,
                  coss_f=4.53e-10, r_off_ohms=2.5e12)
    record = json.loads(json.dumps(lut.to_dict()))
    restored = LossLUT.from_dict(record)
    assert restored.coss_f == lut.coss_f
    assert restored.r_off_ohms == lut.r_off_ohms
    guard = lut.device_values_finite()
    assert guard["finite"] is True
    assert guard["coss_f_finite"] is True
    assert guard["r_off_measured"] is True

    missing = LossLUT("toy", 25.0, (10.0,), (10.0,), ((1.0,),), ((10.0,),),
                      r_on_ohms=(0.01,), vth_v=2.0, vgs_on_v=10.0,
                      r_off_ohms=float("nan"))
    missing_guard = missing.device_values_finite()
    assert missing_guard["finite"] is True
    assert missing_guard["r_off_measured"] is False
    assert "falls back" in missing_guard["r_off_note"]
