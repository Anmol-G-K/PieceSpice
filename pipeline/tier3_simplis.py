"""Tier-3 ``simplis`` variant: SIMPLIS-style Level-2 charge tables.

Assembles the three (V,Q) charge tables from the ONE existing AC sweep
(Ciss/Coss/Crss vs Vds, re-extracted with the same capacitance bench that
``pipeline.tier3`` uses), keeps the closed Tier-2 channel/diode/parasitics,
emits ``<vendor>_PIECESPICE_T3_simplis.lib``, and validates it with the same
DPT / waveform / speed benches as the v1 Tier-3 record.

All reported numbers are measured; incomplete simulations stay incomplete.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np

from common.types import PWLModelSpec, trim_trailing_invalid
from config.yaml_io import load_input, load_output
from approximation.charge_model import integrate_charge
from metrics.static import compute_curve_error
from models.simplis_netlist_builder import (
    RELTOL,
    build_simplis_subcircuit,
    cgs_linearity_decision,
)
from models.hybrid_channel import HybridChannelFit, fit_hybrid_channel
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


@dataclass(frozen=True)
class _ChargeFit:
    """Greedy PWL fit of an anchored Q(V) curve."""

    pairs: list[Tuple[float, float]]
    v_full: np.ndarray
    q_full: np.ndarray
    max_abs_error_c: float
    error_budget_c: float


def _fit_anchored_charge(
    v, c, rel_tol: float, *, forced_v: float | None = None,
    forbid_below: float | None = None,
) -> _ChargeFit:
    """Fit Q(V) with a (0, 0) anchor and NO max_points cap.

    Same farthest-point rule as ``approximation.charge_model.pwl_charge_segments``
    with three measured-data extensions: Q(0)=0, Q(v_min)=s0*v_min where s0
    is the FIRST MEASURED segment slope (no invented capacitance), and an
    optional DPT-derived breakpoint that is forced into the table while
    interior points below it are forbidden. The loop stops only when the
    repo's absolute charge budget is met, so the achieved error/segments
    are measured rather than truncated.
    """
    measured = integrate_charge(np.asarray(v, dtype=float),
                                np.asarray(c, dtype=float))
    vv, qq = measured.v, measured.q
    s0 = float((qq[1] - qq[0]) / (vv[1] - vv[0]))
    va = np.concatenate(([0.0], vv))
    qa = np.concatenate(([0.0], qq + s0 * vv[0]))

    forced_idx = None
    if forced_v is not None and va[0] < forced_v < va[-1]:
        j = int(np.searchsorted(va, forced_v))
        if j >= len(va) or not np.isclose(va[j], forced_v):
            q_forced = float(np.interp(forced_v, va, qa))
            va = np.insert(va, j, forced_v)
            qa = np.insert(qa, j, q_forced)
        forced_idx = int(np.searchsorted(va, forced_v))

    budget = float(rel_tol) * float(qa[-1])
    idx = {0, len(va) - 1}
    if forced_idx is not None:
        idx.add(forced_idx)
    while True:
        chosen = sorted(idx)
        err = np.abs(qa - np.interp(va, va[chosen], qa[chosen]))
        err[chosen] = 0.0
        if forbid_below is not None:
            err[va < forbid_below] = 0.0
        worst = int(np.argmax(err))
        if err[worst] <= budget:
            break
        idx.add(worst)

    chosen = sorted(idx)
    vb, qb = va[chosen], qa[chosen]
    full_err = float(np.max(np.abs(qa - np.interp(va, vb, qb))))
    return _ChargeFit(
        pairs=[(float(x), float(y)) for x, y in zip(vb, qb)],
        v_full=va, q_full=qa,
        max_abs_error_c=full_err, error_budget_c=budget)


def _cap_curve_to_vbus(v, curves, vbus: float):
    """Trim the AC sweep at the DPT bus voltage (interpolated endpoint)."""
    if vbus is None or vbus >= v[-1]:
        return v, curves
    grid = np.concatenate([v[v < vbus], [vbus]])
    return grid, tuple(np.interp(grid, v, c) for c in curves)


def _on_state_breakpoint(vgs, vds, vgs_on: float, vbus: float) -> dict:
    """DPT-derived CDS/CDG breakpoint: max Vds at >=95% of configured Vgs_on."""
    gate = np.asarray(vgs, dtype=float)
    drain = np.asarray(vds, dtype=float)
    mask = gate >= 0.95 * float(vgs_on)
    if not np.any(mask):
        raise RuntimeError(
            "golden DPT has no samples at >= 95% of configured Vgs_on")
    measured = float(np.max(drain[mask]))
    used = float(min(measured, float(vbus)))
    return {
        "measured_v": measured,
        "used_v": used,
        "clamped_to_vbus": measured > float(vbus),
        "rule": "max Vds where golden input[0] >= 0.95*vgs_on (from YAML), "
                "clamped to vbus so no breakpoint exceeds the bus",
    }


def _table_error(fit: _ChargeFit, pairs) -> float:
    return float(np.max(np.abs(
        np.interp(fit.v_full, [p[0] for p in pairs],
                  [p[1] for p in pairs]) - fit.q_full)))


def _fit_record(fit: _ChargeFit, pairs) -> dict:
    err = _table_error(fit, pairs)
    budget = fit.error_budget_c
    q_max = float(fit.q_full[-1])
    return {
        "n_points": len(pairs),
        "n_segments": len(pairs) - 1,
        "max_abs_error_c": err,
        "error_budget_c": budget,
        "error_ratio": err / budget if budget > 0 else None,
        "tolerance_met": bool(err <= budget),
        "achieved_rel_tolerance": err / q_max if q_max > 0 else None,
        "first_breakpoint_v": float(pairs[0][0]),
        "last_breakpoint_v": float(pairs[-1][0]),
    }


def _finite(value) -> bool:
    return value is not None and math.isfinite(float(value))


def _fidelity_metrics(waveform: dict, comparison: dict) -> dict:
    keys = ("tr_ns", "tf_ns", "e_on_uj", "e_off_uj")
    rel = {}
    for key in keys:
        value = comparison.get(key, {}).get("tier3_rel_err_pct")
        rel[key] = abs(float(value)) if _finite(value) else None
    return {
        "vds_rmse_pct": waveform.get("vds_rel_rmse_pct"),
        "id_rmse_pct": waveform.get("id_rel_rmse_pct"),
        "abs_rel_err_pct": rel,
    }


def _select_charge_candidate(candidates: dict) -> tuple[str, str]:
    """Select tier2 only on a strict, all-metric improvement over tables."""
    a = candidates["tables"]["fidelity"]
    b = candidates["tier2"]["fidelity"]
    complete = lambda f: (_finite(f["vds_rmse_pct"])
                          and _finite(f["id_rmse_pct"])
                          and all(_finite(x)
                                  for x in f["abs_rel_err_pct"].values()))
    if not complete(a) or not complete(b):
        return ("tables",
                "at least one candidate has an incomplete guard/failed DPT; "
                "tables retained")
    better = (b["vds_rmse_pct"] < a["vds_rmse_pct"]
              and b["id_rmse_pct"] < a["id_rmse_pct"]
              and all(b["abs_rel_err_pct"][k] < a["abs_rel_err_pct"][k]
                      for k in b["abs_rel_err_pct"]))
    if better:
        return ("tier2",
                "tier2 charge strictly improves Vds RMSE, Id RMSE and "
                "abs(rel err) of tr/tf/e_on/e_off")
    return ("tables",
            "tier2 charge does not strictly improve every recorded "
            "fidelity metric; tables retained")


_CHANNEL_RULE = (
    "hybrid selected ONLY IF, for both charge modes, it strictly improves "
    "Vds RMSE, Id RMSE and abs(rel err) of tr/tf/e_on/e_off over the "
    "corresponding Tier-2-channel candidate on the same forced-2ns DPT "
    "guard, AND its measured adaptive solver/wall speedup is not worse "
    "(not measured => rejected, never fabricated). Else Tier-2 channel."
)


def _select_channel_candidate(
    candidates: dict, *, adaptive_results: dict[str, dict] | None = None
) -> tuple[str, str, dict]:
    """Strict, all-metric hybrid gate; the Tier-2 channel is the default."""
    speedups = adaptive_results or {}
    pairs = (("tables", "hybrid_tables"), ("tier2", "hybrid_tier2"))
    metrics = ("vds_rmse_pct", "id_rmse_pct")
    fidelity_keys = ("tr_ns", "tf_ns", "e_on_uj", "e_off_uj")
    evidence: dict = {"pairs": {}}
    fidelity_ok = True
    for base, hybrid in pairs:
        if base not in candidates or hybrid not in candidates:
            evidence["pairs"][hybrid] = {
                "compared_with": base, "strict_gain": False,
                "reason": "candidate missing (hybrid fit failed or not measured)",
            }
            fidelity_ok = False
            continue
        a, b = candidates[base]["fidelity"], candidates[hybrid]["fidelity"]
        complete = lambda f: (_finite(f["vds_rmse_pct"])
                              and _finite(f["id_rmse_pct"])
                              and all(_finite(f["abs_rel_err_pct"].get(k))
                                      for k in fidelity_keys))
        if not complete(a) or not complete(b):
            evidence["pairs"][hybrid] = {
                "compared_with": base, "strict_gain": False,
                "reason": "incomplete DPT guard on at least one candidate",
                "tier2_channel_completed": candidates[base].get(
                    "dpt_run", {}).get("completed"),
                "hybrid_completed": candidates[hybrid].get(
                    "dpt_run", {}).get("completed"),
            }
            fidelity_ok = False
            continue
        rows = {}
        for key in metrics:
            rows[key] = {"tier2_channel": a[key], "hybrid": b[key],
                         "strict_gain": b[key] < a[key]}
        for key in fidelity_keys:
            av, bv = a["abs_rel_err_pct"][key], b["abs_rel_err_pct"][key]
            rows[key] = {"tier2_channel": av, "hybrid": bv,
                         "strict_gain": bv < av}
        strict = all(row["strict_gain"] for row in rows.values())
        evidence["pairs"][hybrid] = {
            "compared_with": base, "metrics": rows, "strict_gain": strict,
            "tier2_channel_completed": candidates[base].get(
                "dpt_run", {}).get("completed"),
            "hybrid_completed": candidates[hybrid].get(
                "dpt_run", {}).get("completed"),
        }
        fidelity_ok = fidelity_ok and strict
    if not fidelity_ok:
        failed = []
        for hybrid, row in evidence["pairs"].items():
            if row["strict_gain"]:
                continue
            if "metrics" in row:
                keys = [k for k, v in row["metrics"].items()
                        if not v["strict_gain"]]
                failed.append(f"{hybrid} vs {row['compared_with']}: "
                              f"{','.join(keys)}")
            else:
                failed.append(f"{hybrid}: {row['reason']}")
        return ("tier2",
                "hybrid rejected: strict all-metric DPT fidelity gain not "
                "proven (" + "; ".join(failed) + "); Tier-2 channel retained",
                evidence)
    adaptive_ok = True
    adaptive_rows = {}
    for base, hybrid in pairs:
        for combo in (base, hybrid):
            row = speedups.get(combo)
            if not row or not row.get("completed") or not all(
                    _finite(row.get(k)) for k in
                    ("solver_speedup", "wall_speedup")):
                adaptive_ok = False
                adaptive_rows[combo] = {
                    "status": "adaptive not measured / incomplete"}
        if adaptive_ok:
            not_worse = all(
                speedups[hybrid][k] >= speedups[base][k]
                for k in ("solver_speedup", "wall_speedup"))
            adaptive_rows[hybrid] = {
                "compared_with": base,
                "tier2_channel": speedups[base],
                "hybrid": speedups[hybrid],
                "not_worse": not_worse,
            }
            adaptive_ok = not_worse
    evidence["adaptive"] = adaptive_rows
    if not adaptive_ok:
        return ("tier2",
                "hybrid strict DPT gain measured, but adaptive speedup is "
                "not measured or not no-worse for both hybrid charge modes; "
                "Tier-2 channel retained",
                evidence)
    return ("hybrid",
            "hybrid strictly improves every DPT fidelity metric in both "
            "charge modes and its measured adaptive speedup is not worse",
            evidence)

def _adaptive_guard_run(lib, subckt, pins, kw: dict, end: float,
                        temp: float, max_step: float | None) -> dict:
    """One netlist-identical run to read t_last (the bench returns medians)."""
    nb = NetlistBuilder(f"SPEEDUP_DPT_{subckt}")
    nb.include(lib)
    nb.v_dc("bus", "vbus", "0", kw["vbus"])
    nb.l("load", "vbus", "drain", kw["l_load"])
    nb.raw_line("Dclamp drain vbus DCLAMP")
    nb.raw_line(".model DCLAMP D(IS=1e-12 N=1 RS=1e-3 CJO=0)")
    nb.raw_line("Vgs gate 0 PWL(" + _pulse_train(
        1e-6, kw["first_pulse_width"], kw["pulse_width"],
        kw["pulse_period"], kw["n_pulses"], kw["edge_time"],
        kw["vgs_on"]) + ")")
    nb.x_instance("dut", subckt, pins, temperature=temp)
    directive = TranDirective(step_time=max_step or kw["edge_time"],
                              end_time=end, max_step=max_step)
    result = SpiceEngine(temperature=temp,
                         timeout_s=kw["timeout_s"]).run_transient(
        nb.render(), directive)
    t = np.asarray(result.get("time"), dtype=float)
    found = re.search(r"Total elapsed time:\s*([\d.]+)", result.log_text)
    return {
        "t_last_s": float(t[-1]) if t.size else None,
        "iterations": _log_iterations(result.log_text),
        "trace_points": int(t.size),
        "solver_elapsed_s": float(found.group(1)) if found else None,
    }


def _run_adaptive_benchmark(vendor: str, cfg: dict, dpt: dict, lib: Path,
                            temp: float) -> dict:
    """20-pulse adaptive bench with a 3-repeat + points/t_last rejection guard."""
    kw = {
        "vbus": dpt["vbus"], "vgs_on": dpt["vgs_on"],
        "i_load": dpt["i_load_target"], "l_load": dpt["l_load"],
        "first_pulse_width": dpt["t1_width"],
        "pulse_width": dpt["t2_width"],
        "pulse_period": dpt["t2_on"] - dpt["t1_on"],
        "n_pulses": 20, "edge_time": dpt["edge_time"],
        "max_step": None, "temperature": temp,
        "repeats": 3, "timeout_s": 90.0,
    }
    end = (1.0e-6 + (kw["n_pulses"] - 1) * kw["pulse_period"]
           + kw["pulse_width"] + 2e-6)
    golden_step = 100e-9 if vendor == "epc2216" else None
    sides = (
        ("golden", Path(cfg["library"]).resolve(), cfg["subcircuit"],
         None, golden_step),
        ("tier3_simplis", lib,
         f"{cfg['device']}_PIECESPICE_T3_simplis",
         _EMITTED_PIN_ROLES, None),
    )
    guard_text = (
        "3/3 repeats must return; median solver/wall/iterations/points > 0; "
        "identical-netlist guard run must reach t_last >= 0.999*end_time_s "
        "with points >= 0.5*median points; timeouts/exceptions fail the side")
    results = {}
    for label, library, subckt, pins, max_step in sides:
        pins = list(cfg["pins"]["pin_order"]) if pins is None else pins
        bench, guard = {}, {}
        error = None
        t_wall = time.perf_counter()
        try:
            bench = run_inductive_benchmark(library, subckt, pins,
                                            **{**kw, "max_step": max_step})
            guard = _adaptive_guard_run(library, subckt, pins, kw, end,
                                        temp, max_step)
        except Exception as exc:  # failed repeats => no speedup, never faked
            error = f"{type(exc).__name__}: {exc}"[:200]
        median_ok = all(_finite(bench.get(k)) and bench[k] > 0
                        for k in ("wall_clock_s", "solver_elapsed_s",
                                  "iterations", "trace_points"))
        t_last = guard.get("t_last_s")
        guard_ok = (
            _finite(t_last) and t_last >= 0.999 * end
            and (guard.get("trace_points") or 0)
            >= 0.5 * (bench.get("trace_points") or 0)
            and (guard.get("iterations") or 0) > 0
            and bench.get("end_time_s") == end)
        row = {
            "library": str(library),
            "subcircuit": subckt,
            "max_step": max_step,
            "end_time_s": end,
            "repeats": kw["repeats"],
            "bench_wall_s": time.perf_counter() - t_wall,
            "guard_run": guard,
            "completed": bool(error is None and median_ok and guard_ok),
            "anti_fake_guard": guard_text,
        }
        for key in ("wall_clock_s", "solver_elapsed_s", "iterations",
                    "trace_points", "trace_names"):
            row[key] = bench.get(key)
        if error:
            row["error"] = error
        results[label] = row

    both = all(row["completed"] for row in results.values())
    notes = [
        "All timing rows are medians of completed runs only; failed repeats "
        "or guards carry completed=false and no speedup.",
    ]
    if golden_step is not None:
        notes.append(
            f"EPC golden uses a documented {golden_step:.0e} s max_step cap "
            "(fully adaptive golden timed out out-of-band); the ROM stays "
            "fully adaptive (max_step=None).")
    return {
        "conditions": {
            "n_pulses": kw["n_pulses"], "max_step_golden_s": golden_step,
            "max_step_simplis_s": None, "end_time_s": end,
            "repeats": kw["repeats"], "timeout_s": kw["timeout_s"],
            "topology": "clamped-inductive DPT, per-vendor config YAML",
        },
        **results,
        "solver_speedup": (
            results["golden"]["solver_elapsed_s"]
            / results["tier3_simplis"]["solver_elapsed_s"]
            if both else None),
        "wall_speedup": (
            results["golden"]["wall_clock_s"]
            / results["tier3_simplis"]["wall_clock_s"]
            if both else None),
        "epc_golden_fully_adaptive_attempt": None,
        "notes": notes,
    }


def _build_spec(
    vendor: str, *, dpt_vds=None, dpt_vgs=None,
    vgs_on: float | None = None, vbus: float | None = None,
) -> tuple[PWLModelSpec, dict, HybridChannelFit | None]:
    """Build the spec from extracted curves plus one live AC capacitance sweep.

    Curve-1/3/6/7 assembly follows ``pipeline.fitting.fit_pwl_model`` exactly;
    the charge side replaces Tier-2's heuristic C_ISS/C_RSS scalars with the
    documented three-cap decomposition of the same AC sweep. The golden DPT
    trace (already captured by ``run_vendor``) supplies the data-derived
    on-state breakpoint and the bus cap; no extra simulation is needed.
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
    # Interior breakpoints: OUR greedy farthest-point rule (SIMPLIS does not
    # document its selection rule). No max_points cap: fit until the repo's
    # 0.5% charge budget is actually met and record what was achieved.
    v_fit, (cgs_fit_c, cds_fit_c, cdg_fit_c) = _cap_curve_to_vbus(
        v, (cgs_c, cds_c, cdg_c), vbus)
    on_bp = (_on_state_breakpoint(dpt_vgs, dpt_vds, vgs_on, vbus)
             if dpt_vds is not None and dpt_vgs is not None
             and vgs_on is not None and vbus is not None else None)
    forced = on_bp["used_v"] if on_bp else None
    forbid = forced if on_bp and not on_bp["clamped_to_vbus"] else None
    cgs_fit = _fit_anchored_charge(v_fit, cgs_fit_c, RELTOL)
    cds_fit = _fit_anchored_charge(
        v_fit, cds_fit_c, RELTOL, forced_v=forced, forbid_below=forbid)
    cdg_fit = _fit_anchored_charge(
        v_fit, cdg_fit_c, RELTOL, forced_v=forced, forbid_below=forbid)
    decision = cgs_linearity_decision(cgs_fit.v_full, cgs_fit.q_full, RELTOL)
    if decision.is_linear:
        # CGS_NSEG=1 (linear) [S-MOS]; the emitted C is the endpoint chord.
        cgs_pairs = [cgs_fit.pairs[0], cgs_fit.pairs[-1]]
    else:
        cgs_pairs = cgs_fit.pairs
    cds_pairs, cdg_pairs = cds_fit.pairs, cdg_fit.pairs

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

    # Fit the measured hinge_c channel to the SAME Curve-1 rows. Every vendor
    # goes through _build_spec; the fit is a candidate only, never a default.
    try:
        hybrid_fit = fit_hybrid_channel(
            spec.vgs_values, spec.breakpoints_vds, spec.breakpoints_id,
            rel_tolerance=RELTOL)
        # vds_knees is one identical shared-knee row per Vgs family; record
        # that list once and assert the rows agree (no silent divergence).
        knees_v = list(hybrid_fit.vds_knees[0])
        assert all(list(row) == knees_v for row in hybrid_fit.vds_knees), (
            f"{vendor}: hybrid vds_knees rows are not identical")
        hybrid_provenance = {
            "status": "fitted",
            "fit_kind": hybrid_fit.meta.get("kind"),
            "n_knees": hybrid_fit.meta.get("n_knees"),
            "rel_tolerance": hybrid_fit.meta.get("rel_tolerance"),
            "achieved_rel_rmse": hybrid_fit.meta.get("achieved_rel_rmse"),
            "max_abs_fit_error_a": hybrid_fit.max_abs_fit_error_a,
            "stop_criterion": hybrid_fit.meta.get("stop_criterion"),
            "smoothing_width_derivation": hybrid_fit.meta.get(
                "smoothing_width_derivation"),
            "knee_selection": hybrid_fit.meta.get("knee_selection"),
            "knee_scope": hybrid_fit.meta.get("knee_scope"),
            "knees_v": knees_v,
            "smoothing_width_v": hybrid_fit.smoothing_width_v,
            "source_rows": {
                "vgs_values": list(spec.vgs_values),
                "n_vds": len(spec.breakpoints_vds),
                "target": "Curve-1 breakpoints_id (same spec as Tier-2 channel)",
            },
            "expression_ascii": hybrid_fit.expression.isascii(),
            "expression_chars": len(hybrid_fit.expression),
            "known_limitation": hybrid_fit.meta.get("known_limitation"),
        }
    except Exception as exc:
        hybrid_fit = None
        hybrid_provenance = {
            "status": "fit_failed",
            "error": f"{type(exc).__name__}: {exc}"[:200],
            "rel_tolerance": RELTOL,
            "source_rows": {
                "n_vgs": len(spec.vgs_values),
                "n_vds": len(spec.breakpoints_vds),
            },
        }

    # Extracted scalar cross-check from the already-recorded curve4 file.
    try:
        c4 = load_output(vendor, "curve4_capacitances")["extracted"]
        curve4_scalars = {k: float(c4[k])
                          for k in ("ciss_f", "coss_f", "crss_f",
                                    "cap_ref_vds")}
    except (FileNotFoundError, KeyError, ValueError):
        curve4_scalars = {}

    fits = {"cgs": cgs_fit, "cds": cds_fit, "cdg": cdg_fit}
    emitted_pairs = {"cgs": cgs_pairs, "cds": cds_pairs, "cdg": cdg_pairs}
    details = {
        "cgs_decision": decision.as_dict(),
        "nseg": {k: len(p) - 1 for k, p in emitted_pairs.items()},
        "table_points": {k: len(p) for k, p in emitted_pairs.items()},
        "max_abs_charge_error_c": {
            k: _table_error(fits[k], emitted_pairs[k]) for k in fits},
        "charge_fit": {
            k: _fit_record(fits[k], emitted_pairs[k]) for k in fits},
        "anchor": {
            "v0_v": 0.0, "q0_c": 0.0,
            "v_min_measured_v": float(v[0]),
            "first_segment_slope_f": {
                k: float((f.q_full[1] - f.q_full[0]) /
                         (f.v_full[1] - f.v_full[0]))
                for k, f in fits.items()},
        },
        "on_state_breakpoint": on_bp,
        "vbus_cap_v": float(vbus) if vbus is not None else None,
        "rel_tolerance": RELTOL,
        "max_points_cap": None,
        "ac_sweep": {"v_min_v": float(v[0]), "v_max_v": float(v[-1]),
                     "points": int(v.size),
                     "f_meas_hz": float(cond.get("F_meas", F_MEAS_DEFAULT))},
        "clipped_delta_points": n_clipped,
        "cap_ref_vds": cap_ref,
        "curve4_recorded_scalars": curve4_scalars,
        "breakpoint_formula": "greedy pwl_charge_segments farthest-point rule "
        "on anchored Q(V); no point cap; CDS/CDG forced DPT on-state "
        "breakpoint; no breakpoint above vbus",
        "hybrid_channel": hybrid_provenance,
    }
    return spec, details, hybrid_fit


def _run_emitted_dpt(lib_path: Path, vendor: str):
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
    res = SpiceEngine(temperature=temp, timeout_s=120.0).run_transient(
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
    """Build and measure all channel/charge candidates; persist the winner."""
    t0 = time.perf_counter()
    traj = capture_trajectory(vendor)
    capture_wall = time.perf_counter() - t0
    cfg = load_input(vendor)
    dpt = _dpt_conditions(vendor)
    temp = float(cfg["conditions"]["temperature"])

    spec, details, hybrid_fit = _build_spec(
        vendor, dpt_vds=traj.input[1], dpt_vgs=traj.input[0],
        vgs_on=dpt["vgs_on"], vbus=dpt["vbus"])
    fit_wall = time.perf_counter() - t0 - capture_wall

    lib = ROOT / "reports" / "models" / \
        f"{vendor}_PIECESPICE_T3_simplis.lib"
    lib.parent.mkdir(parents=True, exist_ok=True)
    modes = ("tables", "tier2")
    if hybrid_fit is not None:
        modes += ("hybrid_tables", "hybrid_tier2")
    texts = {
        mode: build_simplis_subcircuit(
            spec, charge_mode=mode.removeprefix("hybrid_"),
            channel_fit=hybrid_fit if mode.startswith("hybrid_") else None)
        for mode in modes}
    for mode, text in texts.items():
        if not text.isascii():
            raise RuntimeError(f"{vendor}: {mode} netlist is not ASCII")
    candidate_libs = {
        mode: lib.with_name(f"{lib.stem}_{mode}_charge.lib")
        for mode in texts}
    for mode, text in texts.items():
        candidate_libs[mode].write_text(text, encoding="ascii")

    golden_run = DoublePulseRun(
        time=traj.time, v_ds=traj.input[1],
        i_d=np.abs(traj.result.get("I(Lload)")),
        v_gs=traj.input[0], v_bus=dpt["vbus"],
        i_load=dpt["i_load_target"], l_load=dpt["l_load"],
        vgs_on=dpt["vgs_on"], wall_clock_s=capture_wall,
        t2_on=dpt["t2_on"])
    golden = extract_dpt_metrics(golden_run)

    # Same forced-2ns DPT bench and completion guard for EVERY candidate.
    candidates = {}
    for mode, candidate_lib in candidate_libs.items():
        info = {"completed": False, "trace_points": 0,
                "golden_trace_points": int(len(traj.time)),
                "iterations": None, "t_last_s": None,
                "forced_2ns_wall_s": None,
                "error": None, "waveform": {}}
        wave, rom_metrics, log = {}, {}, ""
        try:
            run, log = _run_emitted_dpt(candidate_lib, vendor)
            info["forced_2ns_wall_s"] = float(run.wall_clock_s)
            n3 = int(len(run.time))
            it3 = _log_iterations(log)
            t_last = float(run.time[-1]) if n3 else 0.0
            complete = (n3 >= 0.5 * len(traj.time)
                        and t_last >= 0.999 * dpt["t_end"])
            info.update(trace_points=n3, iterations=it3, t_last_s=t_last)
            if complete:
                info["completed"] = True
                wave = _waveform_errors(run, traj)
                info["waveform"] = wave
                rom_metrics = extract_dpt_metrics(run)
            else:
                info["error"] = (
                    f"incomplete: {n3} trace points, t_last={t_last:.3g} s")
                print(f"[tier3/simplis] {vendor}/{mode}: DPT incomplete "
                      f"({n3} points, t_last={t_last:.3g} s)", flush=True)
        except Exception as exc:  # solver failures reported, never fabricated
            log = f"{type(exc).__name__}: {exc}"
            info["error"] = log[:200]
            print(f"[tier3/simplis] {vendor}/{mode}: DPT failed: "
                  f"{type(exc).__name__}", flush=True)

        recorded = ROOT / "reports" / "comparison" / vendor / \
            "dpt_metrics.json"
        tier2 = json.loads(recorded.read_text(encoding="utf-8"))[
            "tier2_metrics"] if recorded.exists() else {}
        comparison = {}
        for m in METRICS:
            g, t3 = golden.get(m), rom_metrics.get(m)
            row = {"golden": g, "tier3": t3,
                   "tier3_rel_err_pct": _rel_err(t3, g)}
            if m in tier2:
                row["tier2"] = tier2[m]
                row["tier2_rel_err_pct"] = _rel_err(tier2[m], g)
            comparison[m] = row
        candidates[mode] = {
            "library": str(candidate_lib.relative_to(ROOT)),
            "dpt_run": info,
            "dpt_comparison": comparison,
            "waveform": wave,
            "fidelity": _fidelity_metrics(wave, comparison),
            "spice_log_tail": "\n".join(log.splitlines()[-6:]),
        }

    selection, reason = _select_charge_candidate(candidates)
    channel_selection, channel_reason, channel_evidence = (
        _select_channel_candidate(candidates))
    final = (("hybrid_" if channel_selection == "hybrid" else "")
             + selection)
    if final not in candidates:  # never select an unmeasured candidate
        channel_selection = "tier2"
        channel_reason = "hybrid candidate missing; Tier-2 channel retained"
        final = selection
    lib.write_text(texts[final], encoding="ascii")

    t1 = time.perf_counter()
    static = _emitted_static_rmse(vendor, lib)
    static_wall = time.perf_counter() - t1
    t2_static = _recorded_tier2_static(vendor)
    static_delta = (static["mean_rel_rmse_pct"] - t2_static
                    if t2_static is not None else None)

    # Forced-2ns runtime: fidelity control, NOT the adaptive speed claim.
    bench_kw = {
        "vbus": dpt["vbus"], "vgs_on": dpt["vgs_on"],
        "i_load": dpt["i_load_target"], "l_load": dpt["l_load"],
        "first_pulse_width": dpt["t1_width"],
        "pulse_width": dpt["t2_width"],
        "pulse_period": dpt["t2_on"] - dpt["t1_on"],
        "n_pulses": 2, "edge_time": dpt["edge_time"],
        "max_step": dpt["max_step"], "temperature": temp,
        "repeats": 3, "timeout_s": 120.0,
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

    adaptive = _run_adaptive_benchmark(vendor, cfg, dpt, lib, temp)
    adaptive["selected_candidate"] = final
    adaptive["scope"] = "final selected candidate only; no adaptive runs for non-selected candidates"

    record = {
        "schema": "piecespice.tier3/2.0",
        "vendor": vendor,
        "variant": "simplis",
        "methods": [
            "SIMPLIS Level-2 (CGS/CDS/CDG V-Q tables; S-MOS, S-APPB)",
            "Tier-2 CPWL channel + body diode + parasitics (unchanged)",
        ] + (["Gated hybrid hinge_c channel (measured candidates; default off)"]
             if any(m.startswith("hybrid_") for m in candidates) else []),
        "algorithm": {
            **details,
            "channel": (
                "hybrid hinge_c (models.hybrid_channel.fit_hybrid_channel); "
                "strictly gated by measured DPT fidelity and adaptive speed"
                if channel_selection == "hybrid" else
                "reused models.netlist_builder._channel_expression "
                "(hybrid rejected by strict gate)"),
            "charge_candidate_selected": selection,
            "channel_candidate_selected": channel_selection,
        },
        "function_fit": {
            "static_curve_rmse_rel": static["mean_rel_rmse_pct"],
            "emitted_static_curve_rmse_rel": static["mean_rel_rmse_pct"],
            "emitted_static_per_curve_pct": static["per_curve_rel_rmse_pct"],
            "emitted_static_samples": static["n_points_per_curve"],
            "tier2_recorded_mean_rel_rmse_pct": t2_static,
            "static_delta_pp_vs_tier2": static_delta,
            "static_rmse_scope": "channel_only_dc_capacitors_are_open",
        },
        "dpt_run": candidates[final]["dpt_run"],
        "dpt_waveform": candidates[final]["waveform"],
        "dpt_comparison": candidates[final]["dpt_comparison"],
        "charge_candidates": {
            "modes": list(candidates),
            **candidates,
            "selection": selection,
            "selection_reason": reason,
            "selection_rule": "tier2 only if it strictly improves Vds RMSE, "
            "Id RMSE and abs(rel err) of tr/tf/e_on/e_off; else tables",
        },
        "channel_candidates": {
            "modes": list(candidates),
            **candidates,
            "selection": channel_selection,
            "selection_reason": channel_reason,
            "selection_rule": _CHANNEL_RULE,
            "evidence": channel_evidence,
        },
        "channel_selection": channel_selection,
        "channel_selection_reason": channel_reason,
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
            "bench": "forced 2ns fidelity control; not the speed claim",
        },
        "speed_adaptive_20": adaptive,
        "build_seconds": {"capture": capture_wall, "fit": fit_wall,
                          "static_sweep": static_wall},
        "spice_log_tail": candidates[final]["spice_log_tail"],
        "emitted_lib": str(lib.relative_to(ROOT)),
    }
    out = ROOT / "reports" / "comparison" / vendor / \
        "tier3_validation_simplis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    prior = {}
    if out.exists():
        try:
            prior = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prior = {}
    if isinstance(prior, dict) and "speed_adaptive_20" in prior:
        record["speed_adaptive_20_source"] = prior["speed_adaptive_20"]
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
