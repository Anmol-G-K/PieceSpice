"""Tier-3 ``simplis`` variant: SIMPLIS-style Level-2 charge tables.

Assembles the three (V,Q) charge tables from the ONE existing AC sweep
(Ciss/Coss/Crss vs Vds, re-extracted with the same capacitance bench that
``pipeline.tier3`` uses), keeps the closed Tier-2 channel/diode/parasitics,
emits ``<vendor>_PIECESPICE_T3_simplis.lib``, and validates it with the same
DPT / waveform / speed benches as the v1 Tier-3 record.

All reported numbers are measured; incomplete simulations stay incomplete.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from common.types import PWLModelSpec, trim_trailing_invalid
from config.yaml_io import load_input, load_output
from approximation.charge_model import integrate_charge, pwl_charge_segments
from metrics.static import compute_curve_error
from models.simplis_netlist_builder import (
    CDS_NSEG,
    CDG_NSEG,
    MAX_CGS_NSEG,
    RELTOL,
    build_simplis_subcircuit,
    cgs_linearity_decision,
)
from pipeline import fitting
from pipeline.curves import _cap_ref_vds
from pipeline.registry import ROOT
from pipeline.tier3 import (
    METRICS,
    _dpt_conditions,
    _log_iterations,
    _rel_err,
    _waveform_errors,
)
from simulation.engine import SpiceEngine, TranDirective
from simulation.netlist_text import NetlistBuilder
from simulation.testbenches.capacitance import CapacitanceTestbench, F_MEAS_DEFAULT
from simulation.testbenches.double_pulse import DoublePulseRun, extract_dpt_metrics
from simulation.testbenches.dpt_trajectory import capture_trajectory
from simulation.testbenches.speedup_bench import _pulse_train, run_inductive_benchmark


# 1.5 Ohm / 0.1 mOhm / 0.1 mOhm: the Tier-2 emitter's documented metadata
# defaults (models/netlist_builder.py::_META_DEFAULTS). Reused, not retuned.
_PARASITICS = {"r_gate": 1.5, "r_source": 1e-4, "r_drain": 1e-4}

# The emitted simplis subckt header is ALWAYS `.SUBCKT <name> d g s`
# (models/simplis_netlist_builder.build_simplis_subcircuit). Its port order is
# its own, NOT the golden subcircuit's: EPC's golden is gate-first and CPM2's
# golden carries a 4th Tj port. Every X-instance of the emitted model binds
# these roles; the golden side keeps traj.pin_roles.
_EMITTED_PIN_ROLES = ["drain", "gate", "source"]


def _build_spec(vendor: str) -> tuple[PWLModelSpec, dict]:
    """Build the spec from extracted curves plus one live AC capacitance sweep.

    Curve-1/3/6/7 assembly follows ``pipeline.fitting.fit_pwl_model`` exactly;
    the charge side replaces Tier-2's heuristic C_ISS/C_RSS scalars with the
    documented three-cap decomposition of the same AC sweep.
    """
    cfg = load_input(vendor)
    cond = cfg["conditions"]
    device = cfg["device"]

    # ---- Channel (Curve 1), Rds(on) (Curve 3), Vth (Curve 7), diode (6) ----
    c1 = fitting._curve(vendor, "curve1_output_characteristics")["extracted"]
    prefix = "curve_id_vgs_"
    vgs_values = sorted(
        float(k[len(prefix):-2]) for k in c1 if k.startswith(prefix))
    rows = []
    ref_len = None
    for v in vgs_values:
        key = next(k for k in c1 if k.startswith(prefix)
                   and abs(float(k[len(prefix):-2]) - v) < 1e-9)
        y = fitting._arr(c1[key])
        xs = np.linspace(0.0, float(cond["Vds_sweep_max"]), len(y))
        xs, y = trim_trailing_invalid(xs, y)
        if ref_len is None:
            ref_len = len(y)
        rows.append(np.interp(
            np.linspace(0, float(cond["Vds_sweep_max"]), ref_len), xs, y))
    breakpoints_vds = list(np.linspace(
        0.0, float(cond["Vds_sweep_max"]), ref_len))
    breakpoints_id = [list(map(float, r)) for r in rows]

    rds_on = 0.005  # fitting.py fallback when curve3 is absent
    try:
        c3 = fitting._curve(vendor, "curve3_rds_on")["extracted"]
        rx = fitting._arr(c3["curve_id"])
        ry = fitting._arr(c3["curve_rds_mohm"]) * 1e-3
        if len(rx):
            rds_on = float(np.median(ry[ry > 0]))
    except FileNotFoundError:
        pass

    v_th = 0.0  # fitting.py fallback when curve7 is absent
    try:
        v_th = float(load_output(vendor, "curve7_threshold_voltage")
                     ["extracted"]["vth_v"])
    except (FileNotFoundError, KeyError, ValueError):
        pass

    b_vsd: list = []
    b_isd: list = []
    try:
        c6 = load_output(vendor, "curve6_body_diode")["extracted"]
        xv, yv = trim_trailing_invalid(fitting._arr(c6["curve_vsd"]),
                                       np.abs(fitting._arr(c6["curve_isd"])))
        b_vsd, b_isd = xv.tolist(), yv.tolist()
    except FileNotFoundError:
        pass

    # ---- ONE AC sweep: Ciss, Coss, Crss on the same Vds grid ---------------
    ciss_c, coss_c, crss_c = CapacitanceTestbench(
        f_meas=cond.get("F_meas")).run(
        Path(cfg["library"]).resolve(), cfg["subcircuit"],
        vds_min=cond["Vds_cap_min"], vds_max=cond["Vds_cap_max"],
        points=cond["Cap_points"],
        pin_roles=cfg.get("pins", {}).get("pin_order"))
    v = np.asarray(ciss_c.x_data, dtype=float)
    ciss = np.asarray(ciss_c.y_data, dtype=float)
    coss = np.asarray(coss_c.y_data, dtype=float)
    crss = np.asarray(crss_c.y_data, dtype=float)
    if v.size < 2 or not np.all(np.diff(v) > 0):
        raise RuntimeError(f"{vendor}: AC capacitance sweep has no valid grid")

    # Standard three-cap identity; tiny negative deltas are extraction noise,
    # clipped to zero and counted (never hidden).
    cgs_c = np.maximum(ciss - crss, 0.0)
    cds_c = np.maximum(coss - crss, 0.0)
    cdg_c = crss.copy()
    n_clipped = {
        "cgs": int(np.count_nonzero(ciss - crss < 0.0)),
        "cds": int(np.count_nonzero(coss - crss < 0.0)),
    }

    # ---- Documented SIMPLIS Level-2 charge tables -------------------------
    # Interior breakpoints: OUR greedy pwl_charge_segments rule (SIMPLIS does
    # not document its interior-breakpoint selection), capped at NSEG+1 pairs.
    cgs_full = integrate_charge(v, cgs_c)
    decision = cgs_linearity_decision(cgs_full.v, cgs_full.q, RELTOL)
    if decision.is_linear:
        # CGS_NSEG=1 (linear) [S-MOS]; the emitted C is the endpoint chord.
        cgs_pairs = [(float(cgs_full.v[0]), float(cgs_full.q[0])),
                     (float(cgs_full.v[-1]), float(cgs_full.q[-1]))]
    else:
        cgs_pairs = pwl_charge_segments(
            v, cgs_c, rel_tolerance=RELTOL,
            max_points=MAX_CGS_NSEG + 1).spice_pairs()
    cds_table = pwl_charge_segments(
        v, cds_c, rel_tolerance=RELTOL, max_points=CDS_NSEG + 1)
    cdg_table = pwl_charge_segments(
        v, cdg_c, rel_tolerance=RELTOL, max_points=CDG_NSEG + 1)
    cds_pairs = cds_table.spice_pairs()
    cdg_pairs = cdg_table.spice_pairs()

    cap_ref = _cap_ref_vds(cond)
    c_eq = {
        name: integrate_charge(v, curve, v_ref=cap_ref).c_eq_charge
        for name, curve in (("ciss", ciss), ("coss", coss), ("crss", crss))
    }
    spec = PWLModelSpec(
        device_name=device,
        tier=3,
        breakpoints_vds=breakpoints_vds,
        breakpoints_id=breakpoints_id,
        vgs_values=[float(x) for x in vgs_values],
        q_oss_vs_v=[a for a, _ in cds_pairs],
        q_oss_charge_c=[b for _, b in cds_pairs],
        q_iss_vs_v=[a for a, _ in cgs_pairs],
        q_iss_charge_c=[b for _, b in cgs_pairs],
        q_rss_vs_v=[a for a, _ in cdg_pairs],
        q_rss_charge_c=[b for _, b in cdg_pairs],
        rds_on=rds_on,
        v_th=v_th,
        body_diode_vsd=b_vsd,
        body_diode_isd=b_isd,
        metadata={
            **_PARASITICS,
            "c_iss_lin": c_eq["ciss"],
            "c_oss_lin": c_eq["coss"],
            "c_rss_lin": c_eq["crss"],
        },
    )

    # Extracted scalar cross-check from the already-recorded curve4 file.
    try:
        c4 = load_output(vendor, "curve4_capacitances")["extracted"]
        curve4_scalars = {k: float(c4[k])
                          for k in ("ciss_f", "coss_f", "crss_f",
                                    "cap_ref_vds")}
    except (FileNotFoundError, KeyError, ValueError):
        curve4_scalars = {}

    details = {
        "cgs_decision": decision.as_dict(),
        "nseg": {"cgs": len(cgs_pairs) - 1,
                 "cds": len(cds_pairs) - 1,
                 "cdg": len(cdg_pairs) - 1},
        "table_points": {"cgs": len(cgs_pairs),
                         "cds": len(cds_pairs),
                         "cdg": len(cdg_pairs)},
        "max_abs_charge_error_c": {
            "cgs": float(np.max(np.abs(
                np.interp(cgs_full.v, [p[0] for p in cgs_pairs],
                          [p[1] for p in cgs_pairs]) - cgs_full.q))),
            "cds": cds_table.max_abs_error_c,
            "cdg": cdg_table.max_abs_error_c,
        },
        "rel_tolerance": RELTOL,
        "ac_sweep": {"v_min_v": float(v[0]), "v_max_v": float(v[-1]),
                     "points": int(v.size),
                     "f_meas_hz": float(cond.get("F_meas", F_MEAS_DEFAULT))},
        "clipped_delta_points": n_clipped,
        "cap_ref_vds": cap_ref,
        "curve4_recorded_scalars": curve4_scalars,
        "breakpoint_formula": "greedy pwl_charge_segments (our rule; "
                              "SIMPLIS interior rule undocumented)",
    }
    return spec, details


def _run_emitted_dpt(lib_path: Path, vendor: str, traj):
    """Run the emitted simplis subcircuit in the vendor's DPT bench.

    Mirrors ``pipeline.tier3._run_emitted_dpt`` (same bench, solver options
    and metrics) but binds the variant-specific subcircuit name.
    """
    dpt = _dpt_conditions(vendor)
    cfg = load_input(vendor)
    temp = float(cfg["conditions"]["temperature"])
    subckt = f"{cfg['device']}_PIECESPICE_T3_simplis"
    nb = NetlistBuilder(f"T3_DPT_{vendor}_simplis")
    nb.include(lib_path)
    nb.v_dc("bus", "vbus", "0", dpt["vbus"])
    nb.l("load", "vbus", "drain", dpt["l_load"])
    nb.raw_line("Dclamp drain vbus DCLAMP")
    nb.raw_line(".model DCLAMP D(IS=1e-12 N=1 RS=1e-3 CJO=0)")
    nb.raw_line("Vgs gate 0 PWL(" + _pulse_train(
        dpt["t1_on"], dpt["t1_width"], dpt["t2_width"],
        dpt["t2_on"] - dpt["t1_on"], 2, dpt["edge_time"],
        dpt["vgs_on"]) + ")")
    nb.x_instance("dut", subckt, _EMITTED_PIN_ROLES, temperature=temp)
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


def _emitted_static_rmse(vendor: str, lib: Path):
    """Measured Curve-1 full-sweep rel-RMSE of the emitted subcircuit."""
    rows = fitting.compare_tier2_vs_golden_full(vendor, lib_path=lib)
    per_curve = [
        float(compute_curve_error(np.asarray(golden), np.asarray(cand))
              .relative_rmse_pct)
        for _, _, golden, cand in rows
    ]
    return {
        "mean_rel_rmse_pct": float(np.mean(per_curve)),
        "per_curve_rel_rmse_pct": per_curve,
        "n_curves": len(per_curve),
        "n_points_per_curve": int(rows[0][1].size) if rows else 0,
    }


def _recorded_tier2_static(vendor: str) -> float | None:
    """Tier-2's recorded mean full-sweep rel-RMSE (never recomputed/faked)."""
    path = ROOT / "reports" / "comparison" / vendor / \
        "full_sweep_comparison.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    return float(np.mean([r["rel_rmse_pct"] for r in rows]))


def run_vendor(vendor: str) -> dict:
    """Build, validate and persist one vendor's ``simplis`` Tier-3 model."""
    t0 = time.perf_counter()
    traj = capture_trajectory(vendor)
    capture_wall = time.perf_counter() - t0

    spec, details = _build_spec(vendor)
    fit_wall = time.perf_counter() - t0 - capture_wall

    lib = ROOT / "reports" / "models" / \
        f"{vendor}_PIECESPICE_T3_simplis.lib"
    lib.parent.mkdir(parents=True, exist_ok=True)
    text = build_simplis_subcircuit(spec)
    lib.write_text(text, encoding="ascii")
    if not text.isascii():
        raise RuntimeError(f"{vendor}: emitted netlist is not ASCII")

    t1 = time.perf_counter()
    static = _emitted_static_rmse(vendor, lib)
    static_wall = time.perf_counter() - t1
    t2_static = _recorded_tier2_static(vendor)
    static_delta = (static["mean_rel_rmse_pct"] - t2_static
                    if t2_static is not None else None)

    dpt = _dpt_conditions(vendor)
    golden_run = DoublePulseRun(
        time=traj.time, v_ds=traj.input[1],
        i_d=np.abs(traj.result.get("I(Lload)")),
        v_gs=traj.input[0], v_bus=dpt["vbus"],
        i_load=dpt["i_load_target"], l_load=dpt["l_load"],
        vgs_on=dpt["vgs_on"], wall_clock_s=capture_wall,
        t2_on=dpt["t2_on"])
    golden = extract_dpt_metrics(golden_run)

    rom, rom_metrics, log = None, {}, ""
    dpt_info = {"completed": False, "trace_points": 0,
                "golden_trace_points": int(len(traj.time)),
                "iterations": None, "t_last_s": None,
                "error": None, "waveform": {}}
    try:
        rom, log = _run_emitted_dpt(lib, vendor, traj)
        n3 = int(len(rom.time))
        it3 = _log_iterations(log)
        t_last = float(rom.time[-1]) if n3 else 0.0
        # Anti-fake guard, identical to pipeline.tier3: >=50% of golden
        # points AND the full configured stop time.
        complete = (n3 >= 0.5 * len(traj.time)
                    and t_last >= 0.999 * dpt["t_end"])
        dpt_info.update(trace_points=n3, iterations=it3, t_last_s=t_last)
        if complete:
            dpt_info["completed"] = True
            dpt_info["waveform"] = _waveform_errors(rom, traj)
            rom_metrics = extract_dpt_metrics(rom)
        else:
            dpt_info["error"] = (
                f"incomplete: {n3} trace points, t_last={t_last:.3g} s")
            print(f"[tier3/simplis] {vendor}: emitted DPT incomplete "
                  f"({n3} points, t_last={t_last:.3g} s)", flush=True)
    except Exception as exc:  # solver failures reported, never fabricated
        log = f"{type(exc).__name__}: {exc}"
        dpt_info["error"] = log[:200]
        print(f"[tier3/simplis] {vendor}: emitted DPT failed: "
              f"{type(exc).__name__}", flush=True)

    recorded = ROOT / "reports" / "comparison" / vendor / "dpt_metrics.json"
    tier2 = json.loads(recorded.read_text(encoding="utf-8"))["tier2_metrics"] \
        if recorded.exists() else {}
    comparison = {}
    for m in METRICS:
        g, t3 = golden.get(m), rom_metrics.get(m)
        row = {"golden": g, "tier3": t3, "tier3_rel_err_pct": _rel_err(t3, g)}
        if m in tier2:
            row["tier2"] = tier2[m]
            row["tier2_rel_err_pct"] = _rel_err(tier2[m], g)
        comparison[m] = row

    # Runtime medians from the same benchmark as the Tier-2/v1 records.
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
        traj.pin_roles, **bench_kw)
    bench_wall_golden = time.perf_counter() - t1
    t1 = time.perf_counter()
    try:
        rom_time = run_inductive_benchmark(
            lib, f"{cfg['device']}_PIECESPICE_T3_simplis",
            _EMITTED_PIN_ROLES, **bench_kw)
    except Exception as exc:  # reported, never fabricated
        rom_time = {"wall_clock_s": None, "solver_elapsed_s": None,
                    "iterations": None, "trace_points": 0,
                    "error": str(exc)[:200]}
    bench_wall_rom = time.perf_counter() - t1
    bench_complete = ((rom_time.get("iterations") or 0) >=
                      0.5 * (golden_time["iterations"] or 0)
                      and (rom_time.get("trace_points") or 0) >=
                      0.5 * (golden_time["trace_points"] or 0))

    record = {
        "schema": "piecespice.tier3/2.0",
        "vendor": vendor,
        "variant": "simplis",
        "methods": [
            "SIMPLIS Level-2 (CGS/CDS/CDG V-Q tables; S-MOS, S-APPB)",
            "Tier-2 CPWL channel + body diode + parasitics (unchanged)",
        ],
        "algorithm": {
            **details,
            "channel": "reused models.netlist_builder._channel_expression",
        },
        "function_fit": {
            "static_curve_rmse_rel": static["mean_rel_rmse_pct"],
            "emitted_static_curve_rmse_rel": static["mean_rel_rmse_pct"],
            "emitted_static_per_curve_pct": static["per_curve_rel_rmse_pct"],
            "emitted_static_samples": static["n_points_per_curve"],
            "tier2_recorded_mean_rel_rmse_pct": t2_static,
            "static_delta_pp_vs_tier2": static_delta,
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
        "build_seconds": {"capture": capture_wall, "fit": fit_wall,
                          "static_sweep": static_wall},
        "spice_log_tail": "\n".join(log.splitlines()[-6:]),
        "emitted_lib": str(lib.relative_to(ROOT)),
    }
    out = ROOT / "reports" / "comparison" / vendor / \
        "tier3_validation_simplis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
