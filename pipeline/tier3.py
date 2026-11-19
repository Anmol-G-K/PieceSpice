"""End-to-end Tier-3 (TPWL terminal model) build and validation.

Writes:
  reports/models/<vendor>_PIECESPICE_T3[_<variant>].lib   emitted subcircuit
  reports/comparison/<vendor>/tier3_validation[_<variant>].json  metrics

Everything reported is measured: function-fit errors on the held-out
second pulse, LTspice DPT figures of the emitted subcircuit, and runtime
medians from the same benchmark that produced the Tier-2 numbers.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from config.yaml_io import load_input
from simulation.engine import SpiceEngine, TranDirective
from simulation.netlist_text import NetlistBuilder
from simulation.testbenches.double_pulse import (
    DoublePulseRun,
    extract_dpt_metrics,
)
from simulation.testbenches.speedup_bench import (
    _pulse_train,
    run_inductive_benchmark,
)
from models.tpwl_netlist_builder import emit_subcircuit, emitted_static_rmse
from approximation.tpwl_deim import build_model
from simulation.testbenches.dpt_trajectory import capture_trajectory
from pipeline.registry import DEFAULT_VARIANT

ROOT = Path(__file__).resolve().parent.parent
METRICS = ("td_on_ns", "tr_ns", "td_off_ns", "tf_ns",
           "dv_dt_max_v_per_ns", "di_dt_max_a_per_ns",
           "e_on_uj", "e_off_uj", "i_peak_a")


def _dpt_conditions(vendor: str) -> dict:
    dpt = load_input(vendor)["dpt"]
    return {k: float(v) for k, v in dpt.items()}


def _run_emitted_dpt(model, lib_path: Path, vendor: str):
    """Simulate the emitted subcircuit in the vendor's DPT bench."""
    dpt = _dpt_conditions(vendor)
    cfg = load_input(vendor)
    temp = float(cfg["conditions"]["temperature"])
    name = f"{model.vendor.upper()}_TPWL_DEIM"
    nb = NetlistBuilder(f"T3_DPT_{vendor}")
    nb.include(lib_path)
    nb.v_dc("bus", "vbus", "0", dpt["vbus"])
    nb.l("load", "vbus", "drain", dpt["l_load"])
    nb.raw_line("Dclamp drain vbus DCLAMP")
    nb.raw_line(".model DCLAMP D(IS=1e-12 N=1 RS=1e-3 CJO=0)")
    nb.raw_line("Vgs gate 0 PWL(" + _pulse_train(
        dpt["t1_on"], dpt["t1_width"], dpt["t2_width"],
        dpt["t2_on"] - dpt["t1_on"], 2, dpt["edge_time"],
        dpt["vgs_on"]) + ")")
    nb.x_instance("dut", name, model.pin_roles, temperature=temp)
    t0 = time.perf_counter()
    res = SpiceEngine(timeout_s=120.0).run_transient(
        nb.render(), TranDirective(step_time=dpt["max_step"],
                                   end_time=dpt["t_end"],
                                   max_step=dpt["max_step"]))
    wall = time.perf_counter() - t0
    run = DoublePulseRun(
        time=np.asarray(res.get("time")),
        v_ds=np.asarray(res.get("V(drain)")),
        i_d=np.abs(np.asarray(res.get("I(Lload)"), dtype=float)),
        v_gs=np.asarray(res.get("V(gate)")),
        v_bus=dpt["vbus"], i_load=dpt["i_load_target"],
        l_load=dpt["l_load"], vgs_on=dpt["vgs_on"],
        wall_clock_s=float(wall), t2_on=dpt["t2_on"])
    return run, res.log_text


def _rel_err(value, ref) -> float:
    if value is None or ref is None or ref == 0:
        return float("nan")
    return float((value - ref) / abs(ref) * 100.0)


def _log_iterations(log: str) -> int | None:
    match = re.search(r"Total iterations:\s*(\d+)", log)
    return int(match.group(1)) if match else None


def _waveform_errors(run, traj) -> dict:
    """Relative RMSE of one completed DPT run against the golden capture."""
    mask = run.time <= traj.time[-1]
    if not np.any(mask):
        return {"vds_rel_rmse_pct": None, "id_rel_rmse_pct": None,
                "compare_points": 0}
    t = run.time[mask]
    vds = np.interp(t, traj.time, traj.input[1])
    ids = np.interp(t, traj.time, np.abs(traj.result.get("I(Lload)")))
    return {
        "vds_rel_rmse_pct": float(100 * np.sqrt(np.mean(
            (run.v_ds[mask] - vds) ** 2)) / (np.ptp(traj.input[1]) or 1)),
        "id_rel_rmse_pct": float(100 * np.sqrt(np.mean(
            (run.i_d[mask] - ids) ** 2)) / (
                np.ptp(np.abs(traj.result.get("I(Lload)"))) or 1)),
        "compare_points": int(mask.sum()),
    }


def run_vendor(vendor: str, variant: str = DEFAULT_VARIANT) -> dict:
    """Build, validate, emit and persist one vendor's Tier-3 model."""
    suffix = "" if variant == DEFAULT_VARIANT else f"_{variant}"
    t0 = time.perf_counter()
    traj = capture_trajectory(vendor)
    capture_wall = time.perf_counter() - t0

    # AC-derived Coss(Vds): base drain charge (the DPT never moves Vds at
    # Vgs = 0, so this table is what keeps the blocking state physical).
    from config.yaml_io import load_output
    cfg = load_input(vendor)
    cond = cfg["conditions"]
    v_block = None
    try:
        v_block = float(load_output(
            vendor, "curve7_threshold_voltage")["extracted"]["vth_v"])
    except (FileNotFoundError, KeyError, ValueError):
        pass
    v_width = float(cond["Vgs_transfer_max"]) / \
        float(cond["Vgs_transfer_points"])
    coss_curve = None
    try:
        from simulation.testbenches.capacitance import CapacitanceTestbench
        _, coss, _ = CapacitanceTestbench(
            f_meas=cond.get("F_meas")).run(
            Path(cfg["library"]).resolve(), cfg["subcircuit"],
            vds_min=cond["Vds_cap_min"], vds_max=cond["Vds_cap_max"],
            points=cond["Cap_points"],
            pin_roles=cfg.get("pins", {}).get("pin_order"))
        coss_curve = (np.asarray(coss.x_data), np.asarray(coss.y_data))
    except Exception as exc:  # reported, never fabricated
        print(f"[tier3] {vendor}: Coss base unavailable ({exc})", flush=True)
    model = build_model(traj, coss_curve=coss_curve,
                        v_block=v_block, v_width=v_width)
    fit_wall = time.perf_counter() - t0 - capture_wall

    lib = ROOT / "reports" / "models" / f"{vendor}_PIECESPICE_T3{suffix}.lib"
    lib.parent.mkdir(parents=True, exist_ok=True)
    lib.write_text(emit_subcircuit(model), encoding="ascii")
    static_emitted = emitted_static_rmse(model)
    dpt = _dpt_conditions(vendor)
    golden_run = DoublePulseRun(
        time=traj.time, v_ds=traj.input[1],
        i_d=np.abs(traj.result.get("I(Lload)")),
        v_gs=traj.input[0], v_bus=dpt["vbus"],
        i_load=dpt["i_load_target"], l_load=dpt["l_load"],
        vgs_on=dpt["vgs_on"], wall_clock_s=capture_wall,
        t2_on=dpt["t2_on"])
    golden = extract_dpt_metrics(golden_run)

    tier3_run, tier3, log = None, {}, ""
    dpt_info = {"completed": False, "trace_points": 0,
                "golden_trace_points": int(len(traj.time)),
                "iterations": None, "t_last_s": None,
                "error": None, "waveform": {}}
    try:
        tier3_run, log = _run_emitted_dpt(model, lib, vendor)
        n3 = int(len(tier3_run.time))
        it3 = _log_iterations(log)
        t_last = float(tier3_run.time[-1]) if n3 else 0.0
        # Anti-fake guard: an aborted solver run is never a completed DPT.
        complete = (n3 >= 0.5 * len(traj.time)
                    and t_last >= 0.999 * dpt["t_end"])
        dpt_info.update(trace_points=n3, iterations=it3, t_last_s=t_last)
        if complete:
            dpt_info["completed"] = True
            dpt_info["waveform"] = _waveform_errors(tier3_run, traj)
            tier3 = extract_dpt_metrics(tier3_run)
        else:
            dpt_info["error"] = (
                f"incomplete: {n3} trace points, t_last={t_last:.3g} s")
            print(f"[tier3] {vendor}: emitted DPT incomplete "
                  f"({n3} points, t_last={t_last:.3g} s)", flush=True)
    except Exception as exc:  # solver failures are reported, never fabricated
        log = f"{type(exc).__name__}: {exc}"
        dpt_info["error"] = log[:200]
        print(f"[tier3] {vendor}: emitted DPT failed: {type(exc).__name__}",
              flush=True)
    recorded = ROOT / "reports" / "comparison" / vendor / "dpt_metrics.json"
    tier2 = json.loads(recorded.read_text(encoding="utf-8"))["tier2_metrics"] \
        if recorded.exists() else {}

    comparison = {}
    for m in METRICS:
        g, t3 = golden.get(m), tier3.get(m)
        row = {"golden": g, "tier3": t3,
               "tier3_rel_err_pct": _rel_err(t3, g)}
        if m in tier2:
            row["tier2"] = tier2[m]
            row["tier2_rel_err_pct"] = _rel_err(tier2[m], g)
        comparison[m] = row

    # runtime medians from the same benchmark that produced the Tier-2 table
    cfg = load_input(vendor)
    bench_kw = {
        "vbus": dpt["vbus"], "vgs_on": dpt["vgs_on"],
        "i_load": dpt["i_load_target"], "l_load": dpt["l_load"],
        "first_pulse_width": dpt["t1_width"],
        "pulse_width": dpt["t2_width"],
        "pulse_period": dpt["t2_on"] - dpt["t1_on"],
        "n_pulses": 2, "edge_time": dpt["edge_time"],
        "max_step": dpt["max_step"], "repeats": 3, "timeout_s": 120.0,
    }
    t1 = time.perf_counter()
    golden_time = run_inductive_benchmark(
        Path(cfg["library"]).resolve(), cfg["subcircuit"],
        model.pin_roles, **bench_kw)
    bench_wall_golden = time.perf_counter() - t1
    t1 = time.perf_counter()
    try:
        rom_time = run_inductive_benchmark(
            lib, f"{model.vendor.upper()}_TPWL_DEIM",
            model.pin_roles, **bench_kw)
    except Exception as exc:  # reported, never fabricated
        rom_time = {"wall_clock_s": None, "solver_elapsed_s": None,
                    "iterations": None, "trace_points": 0,
                    "error": str(exc)[:200]}
    bench_wall_rom = time.perf_counter() - t1
    # An aborted bench run has few iterations; call it incomplete, not fast.
    bench_complete = ((rom_time.get("iterations") or 0) >=
                      0.5 * (golden_time["iterations"] or 0)
                      and (rom_time.get("trace_points") or 0) >=
                      0.5 * (golden_time["trace_points"] or 0))

    record = {
        "schema": "piecespice.tier3/2.0",
        "vendor": vendor,
        "variant": variant,
        "methods": ["TPWL (Rewienski & White 2003)",
                    "POD-DEIM anchors (Chaturantabut & Sorensen 2010)"],
        "algorithm": {
            "weight_exponent": 25.0,
            "cells": model.meta["cells"],
            "pod_energy": model.meta["pod_energy"],
            "deim_seed_count": model.meta["deim_seed_count"],
            "edge_smoothing_window": model.edge_window,
            "anchors_scaled": model.anchors.tolist(),
            "v_block_vth": model.v_block,
            "v_width_v": model.v_width,
        },
        "function_fit": {
            "static_curve_rmse_rel": model.meta.get("static_rmse_rel"),
            "emitted_static_curve_rmse_rel": static_emitted,
            "static_samples": model.meta.get("static_samples"),
            "heldout_id_rmse_rel": model.meta["heldout_id_rmse"],
            "heldout_qg_rmse_rel": model.meta["heldout_qg_rmse"],
        },
        "dpt_run": dpt_info,
        "dpt_comparison": comparison,
        "runtime": {
            "golden": {k: golden_time.get(k) for k in
                       ("wall_clock_s", "solver_elapsed_s", "iterations",
                        "trace_points")},
            "tier3": {k: rom_time.get(k) for k in
                      ("wall_clock_s", "solver_elapsed_s", "iterations",
                       "trace_points")},
            "tier3_bench_completed": bench_complete,
            "solver_speedup": (
                golden_time["solver_elapsed_s"] / rom_time["solver_elapsed_s"]
                if bench_complete and rom_time.get("solver_elapsed_s")
                else None),
            "bench_wall_s": {"golden": bench_wall_golden,
                             "tier3": bench_wall_rom},
        },
        "build_seconds": {"capture": capture_wall, "fit": fit_wall},
        "spice_log_tail": "\n".join(log.splitlines()[-6:]),
        "emitted_lib": str(lib.relative_to(ROOT)),
    }
    out = ROOT / "reports" / "comparison" / vendor / f"tier3_validation{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
