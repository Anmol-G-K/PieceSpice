"""End-to-end Tier-3 (TPWL + POD-DEIM) build and validation for one vendor.

Writes:
  reports/models/<vendor>_PIECESPICE_T3.lib       emitted ASCII subcircuit
  reports/comparison/<vendor>/tier3_validation.json   measured metrics

No number is fabricated: every reported value comes from the captured
trajectory, the fit, or an LTspice run of the emitted subcircuit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config.yaml_io import load_input
from simulation.engine import SpiceEngine, TranDirective
from simulation.netlist_text import NetlistBuilder
from simulation.testbenches.speedup_bench import _pulse_train
from tier3.emitter import emit_subcircuit
from tier3.tpwl_deim import TPWLModel, build_model, validate_dynamics
from tier3.trajectory import Trajectory, capture_trajectory

ROOT = Path(__file__).resolve().parent.parent


def simulate_emitted(model: TPWLModel, lib_path: Path,
                     traj: Trajectory) -> dict:
    """Run the emitted subcircuit in the same DPT bench as the golden model."""
    cfg = load_input(model.vendor)
    dpt = cfg["dpt"]
    nb = NetlistBuilder(f"T3_CHECK_{model.vendor}")
    nb.include(lib_path)
    nb.v_dc("bus", "vbus", "0", float(dpt["vbus"]))
    nb.l("load", "vbus", "drain", float(dpt["l_load"]))
    nb.raw_line("Dclamp drain vbus DCLAMP")
    nb.raw_line(".model DCLAMP D(IS=1e-12 N=1 RS=1e-3 CJO=0)")
    nb.raw_line("Vgs gate 0 PWL(" + _pulse_train(
        float(dpt["t1_on"]), float(dpt["t1_width"]), float(dpt["t2_width"]),
        float(dpt["t2_on"]) - float(dpt["t1_on"]), 2,
        float(dpt["edge_time"]), float(dpt["vgs_on"])) + ")")
    name = f"{model.vendor.upper()}_TPWL_DEIM"
    nb.x_instance("dut", name, model.pin_roles,
                  temperature=cfg["conditions"]["temperature"])
    step = float(dpt["max_step"])
    try:
        result = SpiceEngine(timeout_s=120.0).run_transient(
            nb.render(), TranDirective(step_time=step, end_time=float(dpt["t_end"]),
                                       max_step=step))
        t = np.asarray(result.get("time"))
        id_sim = np.asarray(result.get("I(Lload)")) - np.asarray(result.get("I(Dclamp)"))
        ig_sim = -np.asarray(result.get("I(Vgs)"))
    except Exception as exc:  # solver failures are reported, never fabricated
        return {"evaluable": False, "error": f"{type(exc).__name__}: {exc}"[:220]}
    if len(t) < 0.5 * len(traj.time):
        return {"evaluable": False,
                "error": f"solver truncated at {len(t)} of {len(traj.time)} samples"}
    t_ref = traj.time
    id_ref = np.interp(t, t_ref, traj.output[0])
    ig_ref = np.interp(t, t_ref, traj.output[1])
    span_id = float(np.ptp(id_ref)) or 1.0
    span_ig = float(np.ptp(ig_ref)) or 1.0
    return {"evaluable": True, "points": len(t), "rmse_id": float(np.sqrt(np.mean((id_sim - id_ref) ** 2))),
            "rmse_id_rel": float(np.sqrt(np.mean((id_sim - id_ref) ** 2)) / span_id),
            "rmse_ig": float(np.sqrt(np.mean((ig_sim - ig_ref) ** 2))),
            "rmse_ig_rel": float(np.sqrt(np.mean((ig_sim - ig_ref) ** 2)) / span_ig)}


def run_vendor(vendor: str) -> dict:
    """Build, validate, emit and persist one vendor's Tier-3 model."""
    traj = capture_trajectory(vendor)
    model = build_model(traj)
    python_validation = validate_dynamics(model, traj)

    lib = ROOT / "reports" / "models" / f"{vendor}_PIECESPICE_T3.lib"
    lib.parent.mkdir(parents=True, exist_ok=True)
    text = emit_subcircuit(model)
    lib.write_text(text, encoding="ascii")

    spice = simulate_emitted(model, lib, traj)
    record = {
        "schema": "piecespice.tier3/1.0",
        "vendor": vendor,
        "methods": ["TPWL (Rewienski & White 2003)", "POD-DEIM (Chaturantabut & Sorensen 2010)"],
        "algorithm": {
            "weight_exponent": model.meta["weight_exponent"],
            "pod_rank": model.r,
            "linearization_points": model.n_points,
            "linearization_eps": model.eps,
            "deim_indices": model.meta["deim_index_names"],
            "smoothing_window": model.meta["smoothing_window"],
            "stability_projection": model.meta["stability_projection"],
            "max_reduced_eigenvalue": model.meta["max_reduced_eigenvalue"],
            "stiffness_bound": model.meta["stiffness_bound"],
            "time_scale_s": model.time_scale,
        },
        "python_validation": python_validation,
        "spice_validation": spice,
        "emitted_lib": str(lib.relative_to(ROOT)),
    }
    out = ROOT / "reports" / "comparison" / vendor / "tier3_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
