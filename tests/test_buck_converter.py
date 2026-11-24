"""Tests for the M5 config-driven buck-converter benchmark.

Offline tests use ``tmp_path`` libraries and a synthetic ``SimResult`` through
the bench's ``runner`` hook: no LTspice is required and no efficiency is
measured from a simulator.  The single live test is gated by
``PIECESPICE_LIVE=1`` (the repository convention) and measures one available
vendor/tier end to end.
"""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common.plotting as plotting
from common.plotting import (
    plot_converter_benchmark,
    plot_converter_comparison,
)
from config.yaml_io import load_input
from simulation.engine import SimResult
from simulation.testbenches.buck_converter import (
    CONFIG_PATH,
    ConverterConfigError,
    build_converter_netlist,
    load_converter_config,
    resolve_converter_model,
    run_converter_benchmark,
    run_converter_comparison,
)

SUB = "IDEAL_BUCK"

ROOT = Path(__file__).resolve().parents[1]

# SpiceModels/* is .gitignore'd (vendor models are not redistributed), so a
# fresh clone or CI checkout has no EPC2216 golden library. Tests that compare
# against that file skip; the offline synthetic/non-golden tests still run.
_EPC2216_GOLDEN_LIBRARY = ROOT / str(load_input("epc2216")["library"])
requires_epc2216_golden = pytest.mark.skipif(
    not _EPC2216_GOLDEN_LIBRARY.is_file(),
    reason=("EPC2216 golden vendor library is absent "
            f"({_EPC2216_GOLDEN_LIBRARY}); SpiceModels/ is not redistributed"))


def _library(tmp_path: Path, header: str = "d g s") -> Path:
    path = tmp_path / "ideal_buck.lib"
    path.write_text(
        f".SUBCKT {SUB} {header}\n"
        f"S_SW d s g s SW_BUCK\n"
        ".MODEL SW_BUCK SW(Ron=0.01 Roff=1e12 Vt=2 Vh=0)\n"
        ".ENDS\n",
        encoding="ascii")
    return path


def _config() -> dict:
    return load_converter_config()


def _run(tmp_path: Path, result: SimResult, *, cfg: dict | None = None,
         header: str = "d g s", **overrides) -> tuple[dict, dict]:
    captured: dict = {}

    def runner(netlist, directive):
        captured["netlist"] = netlist
        captured["directive"] = directive
        return result

    kwargs: dict[str, Any] = {
        "config": cfg or _config(),
        "library_path": _library(tmp_path, header=header),
        "subcircuit_name": SUB,
        "runner": runner,
    }
    kwargs.update(overrides)
    out = run_converter_benchmark("epc2216", "tier2", **kwargs)
    return out, captured


def _synthetic_result(*, end_time: float = 1.5e-3, vbus: float = 12.0,
                      duty: float = 0.45, vgs_on: float = 5.0,
                      vout: float = 5.0, load: float = 1.0,
                      icap: float = 0.0,
                      iterations: int | None = 1234,
                      drop: tuple[str, ...] = ()) -> SimResult:
    period = 5.0e-6
    time = np.linspace(0.0, end_time, 30001)
    phase = np.mod(time, period) / period
    hs_on = phase < duty
    ls_on = (phase > duty + 0.01) & (phase < 0.99)
    vsw = np.where(hs_on, vbus, 0.0)
    gate_hs = np.where(hs_on, vgs_on, 0.0)
    traces = {
        "time": time,
        "V(vout)": np.full_like(time, vout),
        "V(sw)": vsw,
        "V(vin)": np.full_like(time, vbus),
        "I(Lout)": load + 0.1 * np.sin(2.0 * np.pi * phase),
        "I(Vbus)": np.full_like(time, -duty * load),
        "I(Rload)": np.full_like(time, load),
        "I(Cout)": np.full_like(time, icap),
        # The high-side drive source is referenced to the switch node.
        "V(ghs)": vsw + gate_hs,
        "V(gls)": np.where(ls_on, vgs_on, 0.0),
    }
    for name in drop:
        traces.pop(name)
    log = "" if iterations is None else f"Total iterations: {iterations}\n"
    return SimResult(traces=traces, sweep_var="time", log_text=log)


# ---------------------------------------------------------------------------
# Offline configuration and emission
# ---------------------------------------------------------------------------

def test_default_config_defines_all_tiers_and_vendors():
    cfg = _config()
    assert cfg["schema"] == "piecespice.converter-bench/1.0"
    tiers = cfg["model_selection"]["tiers"]
    assert set(tiers) == {"tier2", "tier3", "tier3_simplis", "urom", "tier4"}
    assert tiers["tier2"]["tier"] == 2
    assert tiers["tier3"]["variant"] == "v1"
    assert tiers["tier3_simplis"]["variant"] == "simplis"
    assert tiers["urom"]["variant"] == "urom"
    assert tiers["tier4"]["tier"] == 4
    for vendor in ("irfs7730pbf", "csd16412q5a", "epc2216", "cpm2-1200"):
        assert cfg["vendors"][vendor]["vgs_on_v"] > 0.0
    cond = cfg["conditions"]
    assert 0.0 < cond["duty"] < 1.0
    assert cond["simulation"]["settling_cycles"] < cond["simulation"]["cycles"]
    assert cond["output_filter"]["esr_ohm"] >= 0.0


@pytest.mark.parametrize("mutate", [
    lambda c: c["conditions"].__setitem__("duty", 1.2),
    lambda c: c["conditions"]["output_filter"].__setitem__("esr_ohm", -1.0),
    lambda c: c["conditions"]["simulation"].__setitem__("settling_cycles", 999),
    lambda c: c["model_selection"]["tiers"].pop("urom"),
    lambda c: c["freewheel"].__setitem__("mode", "magic"),
])
def test_config_fails_closed_on_malformed_values(tmp_path, mutate):
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "bad_converter.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConverterConfigError):
        load_converter_config(path)


def test_missing_config_file_fails_closed(tmp_path):
    with pytest.raises(ConverterConfigError, match="not found"):
        load_converter_config(tmp_path / "absent.yaml")


def test_netlist_emits_selected_model_and_full_power_stage(tmp_path):
    cfg = _config()
    model = resolve_converter_model(
        cfg, "epc2216", "tier2", library_path=_library(tmp_path),
        subcircuit_name=SUB)
    netlist, schedule = build_converter_netlist(model, config=cfg)
    assert f".include {_library(tmp_path).resolve()}" in netlist
    assert f"Xhs vin ghs sw {SUB}" in netlist
    assert f"Xls sw_ls gls 0 {SUB}" in netlist
    assert "Vhssense bus vin DC 0" in netlist
    assert "Vlssense sw sw_ls DC 0" in netlist
    assert "Vbus bus 0 DC 12" in netlist
    assert "Vghs ghs sw PULSE(" in netlist
    assert "Vgls gls 0 PULSE(" in netlist
    assert "Lout sw lx 2.2e-05" in netlist
    assert "Rldcr lx vout 0.006" in netlist
    assert "Cout vout_c 0 4.700000e-05" in netlist
    assert "Resr vout vout_c 0.004" in netlist
    assert "Rload vout 0 5" in netlist
    assert schedule["freewheel_mode"] == "synchronous"
    assert schedule["t_end_s"] == pytest.approx(300 * 5.0e-6)
    assert schedule["window_start_s"] == pytest.approx(240 * 5.0e-6)
    assert schedule["high_pulse_width_s"] > 0.0
    assert schedule["low_pulse_width_s"] > 0.0


def test_gate_first_header_is_connected_by_role_not_position(tmp_path):
    cfg = _config()
    model = resolve_converter_model(
        cfg, "epc2216", "tier2", library_path=_library(tmp_path, "g d s"),
        subcircuit_name=SUB)
    netlist, _ = build_converter_netlist(model, config=cfg)
    assert f"Xhs ghs vin sw {SUB}" in netlist
    assert f"Xls gls sw_ls 0 {SUB}" in netlist


def test_non_synchronous_mode_emits_freewheel_diode(tmp_path):
    cfg = copy.deepcopy(_config())
    cfg["freewheel"]["mode"] = "diode"
    out, captured = _run(
        tmp_path, _synthetic_result(end_time=1.5e-3), cfg=cfg)
    netlist = captured["netlist"]
    assert "Dfw 0 sw DFW" in netlist
    assert ".model DFW D(" in netlist
    assert "Xls " not in netlist
    assert out["status"] == "measured"


# ---------------------------------------------------------------------------
# Offline extraction, N/A semantics and plotting
# ---------------------------------------------------------------------------

def test_complete_run_extracts_efficiency_and_losses(tmp_path):
    out, captured = _run(
        tmp_path, _synthetic_result(end_time=1.5e-3))
    assert out["status"] == "measured"
    assert out["completed"] is True
    assert out["settled"] is True
    assert out["vout_v"] == pytest.approx(5.0)
    assert out["iout_a"] == pytest.approx(1.0)
    assert out["iin_a"] == pytest.approx(0.45)
    assert out["pout_w"] == pytest.approx(5.0)
    assert out["pin_w"] == pytest.approx(5.4)
    assert out["efficiency"] == pytest.approx(5.0 / 5.4)
    assert 0.0 <= out["vout_error_pct"] <= 1.0
    assert out["i_l_mean_a"] == pytest.approx(1.0, abs=1e-3)
    assert out["v_sw_mean_v"] == pytest.approx(12.0 * 0.45, abs=1e-3)
    assert out["speed"]["iterations"] == 1234
    assert out["speed"]["wall_clock_s"] > 0.0
    assert out["guard"]["completed"] is True
    assert out["settling"]["rel_drift_pct"] == pytest.approx(0.0, abs=1e-9)
    loss = out["loss"]
    assert loss["total_loss_w"] == pytest.approx(0.4)
    assert loss["high_side_conduction_w"] == pytest.approx(0.0, abs=1e-9)
    assert loss["low_side_conduction_w"] == pytest.approx(0.0, abs=1e-9)
    assert loss["switching_residual_w"] == pytest.approx(
        0.4 - loss["inductor_dcr_w"] - loss["output_esr_w"], abs=1e-9)
    assert loss["switching_residual_status"] == "computed"
    assert len(out["netlist_sha256"]) == 64
    assert captured["directive"].end_time == pytest.approx(1.5e-3)


def test_missing_trace_is_reported_not_zero(tmp_path):
    out, _ = _run(
        tmp_path, _synthetic_result(end_time=1.5e-3, drop=("I(Lout)",)))
    assert out["status"] == "missing_trace"
    assert out["completed"] is False
    assert "I(Lout)" in out["reason"]
    assert out["efficiency"] is None and out["pout_w"] is None


@pytest.mark.parametrize("kwargs", [
    {"end_time": 0.75e-3},          # truncated waveform
    {"iterations": 0},              # aborted solver
    {"iterations": None},           # no solver-iteration evidence
])
def test_aborted_runs_return_na_without_fabricated_efficiency(tmp_path, kwargs):
    out, _ = _run(tmp_path, _synthetic_result(**kwargs))
    assert out["status"] == "incomplete"
    assert out["completed"] is False
    assert out["efficiency"] is None
    assert out["pout_w"] is None and out["pin_w"] is None
    assert out["vout_v"] is None and out["iout_a"] is None
    assert "incomplete" in out["reason"]


def test_missing_library_returns_model_unavailable_without_running(tmp_path):
    called = {"runner": False}

    def runner(netlist, directive):
        called["runner"] = True
        return _synthetic_result(end_time=1.5e-3)

    out = run_converter_benchmark(
        "epc2216", "tier2", config=_config(),
        library_path=tmp_path / "absent.lib", runner=runner)
    assert out["status"] == "model_unavailable"
    assert out["completed"] is False
    assert out["efficiency"] is None
    assert "not found" in out["reason"]
    assert called["runner"] is False


def test_runner_exception_is_na(tmp_path):
    def boom(netlist, directive):
        raise RuntimeError("no converter convergence")

    out, _ = _run(tmp_path, _synthetic_result(end_time=1.5e-3), runner=boom)
    assert out["status"] == "error"
    assert out["efficiency"] is None
    assert "no converter convergence" in out["reason"]


def test_plot_helper_saves_four_panel_figure(tmp_path):
    out, _ = _run(tmp_path, _synthetic_result(end_time=1.5e-3))
    target = tmp_path / "buck.png"
    fig, axs = plot_converter_benchmark(out, save_path=target)
    assert target.is_file() and target.stat().st_size > 0
    flat = list(axs.flat)
    assert len(flat) == 4
    # House style: every waveform panel carries an EAST-outside legend.
    for ax in flat[:3]:
        legend = ax.get_legend()
        assert legend is not None
        assert legend.get_bbox_to_anchor()._bbox.x0 >= 1.0
    import matplotlib.pyplot as plt
    plt.close(fig)


def _comparison_runner(captured, *, golden=None, tier=None):
    """Return golden/tier synthetic results keyed by the emitted X line."""
    golden = golden or {}
    tier = tier or {}

    def runner(netlist, directive):
        captured.append((netlist, directive))
        if "Xhs ghs vin sw EPC2216" in netlist:
            return _synthetic_result(**golden)
        return _synthetic_result(**tier)

    return runner


@requires_epc2216_golden
def test_golden_vs_tier_comparison_runs_same_conditions():
    captured: list = []
    runner = _comparison_runner(
        captured,
        golden={"vout": 5.0, "load": 1.0, "icap": 0.0},
        tier={"vout": 5.1, "load": 1.05, "icap": 0.02})
    result = run_converter_comparison(
        "epc2216", "tier2", config=_config(), runner=runner)

    assert result["vendor"] == "epc2216"
    assert result["tier_key"] == "tier2"
    assert result["golden"]["status"] == "measured"
    assert result["tier"]["status"] == "measured"
    assert result["golden"]["completed"] is True
    assert result["tier"]["completed"] is True
    assert result["golden"]["model"]["subcircuit"] == "EPC2216"
    assert result["tier"]["model"]["subcircuit"] == "epc2216_PIECESPICE_T2"

    comparison = result["comparison"]
    assert comparison["status"] == "compared"
    assert comparison["signals"]["vout_v"]["golden"] == pytest.approx(5.0)
    assert comparison["signals"]["vout_v"]["tier"] == pytest.approx(5.1)
    assert comparison["signals"]["i_l_mean_a"]["golden"] == pytest.approx(
        1.0, abs=1e-3)
    assert comparison["signals"]["i_l_mean_a"]["tier"] == pytest.approx(
        1.05, abs=1e-3)
    assert comparison["signals"]["i_cap_mean_a"]["golden"] == pytest.approx(
        0.0, abs=1e-9)
    assert comparison["signals"]["i_cap_mean_a"]["tier"] == pytest.approx(0.02)
    assert comparison["efficiency"]["status"] == "compared"

    golden_netlist, golden_directive = captured[0]
    tier_netlist, tier_directive = captured[1]
    assert "Xhs ghs vin sw EPC2216" in golden_netlist
    assert "Xhs vin ghs sw epc2216_PIECESPICE_T2" in tier_netlist
    assert golden_directive.end_time == pytest.approx(tier_directive.end_time)


@requires_epc2216_golden
def test_comparison_plot_is_one_by_three_with_east_legends(
        tmp_path, monkeypatch):
    captured: list = []
    runner = _comparison_runner(
        captured,
        golden={"vout": 5.0, "load": 1.0, "icap": 0.0},
        tier={"vout": 5.1, "load": 1.05, "icap": 0.02})
    result = run_converter_comparison(
        "epc2216", "tier2", config=_config(), runner=runner)
    monkeypatch.setattr(plotting, "FIGURES_ROOT", tmp_path)
    fig, axes = plot_converter_comparison(result)
    assert axes.shape == (3,)
    assert [ax.get_title() for ax in axes] == [
        "Output voltage", "Inductor current", "Capacitor current"]
    for ax in axes:
        legend = ax.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == [
            "golden", "tier (tier2)"]
        assert legend.get_bbox_to_anchor()._bbox.x0 >= 1.0
    path = tmp_path / "converter" / "epc2216_tier2_buck_compare.png"
    assert path.is_file() and path.stat().st_size > 0
    plt.close(fig)


def test_comparison_plot_labels_na_instead_of_zero(tmp_path, monkeypatch):
    captured: list = []
    runner = _comparison_runner(
        captured,
        golden={"end_time": 0.75e-3},       # truncated -> N/A golden
        tier={"vout": 5.1, "load": 1.05, "icap": 0.02})
    result = run_converter_comparison(
        "epc2216", "tier2", config=_config(), runner=runner)
    assert result["comparison"]["status"] == "golden_unavailable"
    assert result["comparison"]["signals"]["vout_v"]["golden"] is None
    assert result["comparison"]["signals"]["vout_v"]["status"] == "not_available"

    monkeypatch.setattr(plotting, "FIGURES_ROOT", tmp_path)
    fig, axes = plot_converter_comparison(result)
    assert axes.shape == (3,)
    for ax in axes:
        assert len(ax.lines) == 1             # tier only; no golden zero
        labels = " ".join(text.get_text() for text in ax.texts)
        assert "golden: N/A" in labels
    plt.close(fig)


# ---------------------------------------------------------------------------
# Live LTspice example (one vendor/tier), gated by the repo convention
# ---------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("PIECESPICE_LIVE", "1") != "1",
                    reason="live LTspice disabled via PIECESPICE_LIVE=0")
def test_live_buck_benchmark_measures_one_vendor_tier():
    cfg = _config()
    vendor = cfg["model_selection"]["default"]["vendor"]
    tier_key = cfg["model_selection"]["default"]["tier"]
    if not Path(__file__).resolve().parents[1].joinpath(
            "reports", "models").is_dir():
        pytest.skip("emitted model directory is absent")
    out = run_converter_benchmark(vendor, tier_key, config=cfg)
    assert out["status"] == "measured", out["reason"]
    assert out["completed"] is True
    assert out["settled"] is True
    assert out["efficiency"] is not None
    assert math.isfinite(out["efficiency"])
    assert 0.0 < out["efficiency"] <= 1.0
    assert out["vout_v"] is not None and out["vout_v"] > 0.0
    assert out["speed"]["iterations"] > 0
    assert out["speed"]["wall_clock_s"] > 0.0
    assert out["traces"]["time"].size > 100


@requires_epc2216_golden
@pytest.mark.skipif(os.environ.get("PIECESPICE_LIVE", "1") != "1",
                    reason="live LTspice disabled via PIECESPICE_LIVE=0")
def test_live_epc_golden_vs_tier2_comparison(tmp_path, monkeypatch):
    result = run_converter_comparison("epc2216", "tier2")
    golden = result["golden"]
    tier = result["tier"]
    assert golden["status"] == "measured", golden["reason"]
    assert tier["status"] == "measured", tier["reason"]
    assert golden["completed"] is True
    assert tier["completed"] is True
    assert golden["model"]["subcircuit"] == "EPC2216"
    assert tier["model"]["subcircuit"] == "epc2216_PIECESPICE_T2"

    comparison = result["comparison"]
    assert comparison["status"] == "compared", comparison["reason"]
    for key in ("vout_v", "i_l_mean_a", "i_cap_mean_a"):
        entry = comparison["signals"][key]
        assert entry["status"] == "compared"
        assert entry["golden"] is not None and math.isfinite(entry["golden"])
        assert entry["tier"] is not None and math.isfinite(entry["tier"])
    assert math.isfinite(golden["efficiency"])
    assert math.isfinite(tier["efficiency"])

    monkeypatch.setattr(plotting, "FIGURES_ROOT", tmp_path)
    fig, axes = plot_converter_comparison(result)
    assert axes.shape == (3,)
    for ax in axes:
        assert ax.get_legend() is not None
    path = tmp_path / "converter" / "epc2216_tier2_buck_compare.png"
    assert path.is_file() and path.stat().st_size > 0
    plt.close(fig)
