"""Regenerate ``dpt_metrics.json`` with the device-current identity.

For each configured vendor the script runs the golden and emitted Tier-2 DPT
on the shared clamped-inductive bench and records:

* ``golden_metrics`` / ``tier2_metrics`` from ``extract_dpt_metrics`` using
  the bench's ``I(Lload)-I(Dclamp)`` current;
* the energy-current definition of each arm (signed integration; timing/G3
  stay on the absolute magnitude), so a consumer never has to guess;
* the measured wall clocks and the speed ratio;
* an explicit failure reason if either arm aborts.

The existing completion/NaN semantics are preserved: a missing crossing is
``null`` (JSON) / NaN (in memory), never zero.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path


from config.yaml_io import load_input
from parser.netlist import resolve_pin_connections
from pipeline.registry import ROOT, lib_path
from pipeline.registry import VENDORS
from simulation.testbenches.double_pulse import (
    extract_dpt_metrics,
    run_double_pulse,
)

_METRIC_KEYS = (
    "td_on_ns", "tr_ns", "td_off_ns", "tf_ns",
    "dv_dt_max_v_per_ns", "di_dt_max_a_per_ns",
    "e_on_uj", "e_off_uj", "e_oss_uj", "i_peak_a",
)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _one(lib: Path, subckt: str, roles: list[str], cfg: dict) -> dict:
    dpt = cfg["dpt"]
    t0 = time.perf_counter()
    run = run_double_pulse(
        lib, subckt, roles,
        vbus=float(dpt["vbus"]), vgs_on=float(dpt["vgs_on"]),
        i_load_target=float(dpt["i_load_target"]),
        l_load=float(dpt["l_load"]),
        t1_on=float(dpt["t1_on"]), t1_width=float(dpt["t1_width"]),
        t2_on=float(dpt["t2_on"]), t2_width=float(dpt["t2_width"]),
        edge_time=float(dpt["edge_time"]), t_end=float(dpt["t_end"]),
        max_step=float(dpt["max_step"]),
        temperature=float(cfg["conditions"].get("temperature", 25.0)),
    )
    wall = time.perf_counter() - t0
    metrics = extract_dpt_metrics(run)
    return {
        "metrics": metrics,
        "energy_current_definition": metrics.get(
            "energy_current_definition"),
        "wall_clock_s": float(wall),
        "current_definition": run.current_definition,
        "trace_points": int(len(run.time)),
    }


def regenerate(vendor: str) -> dict:
    cfg = load_input(vendor)
    dpt = cfg["dpt"]
    golden_lib = Path(cfg["library"]).resolve()
    golden_sub = cfg["subcircuit"]
    golden_roles = resolve_pin_connections(
        golden_lib, golden_sub,
        pin_order_semantic=cfg.get("pins", {}).get("pin_order_semantic"))
    tier2_lib = lib_path(vendor, 2)
    if not tier2_lib.exists():
        raise FileNotFoundError(
            f"{vendor}: Tier-2 lib missing at {tier2_lib}")

    record: dict = {
        "schema": "piecespice.dpt/1.0",
        "vendor": vendor,
        "conditions": {
            "vbus": float(dpt["vbus"]),
            "vgs_on": float(dpt["vgs_on"]),
            "i_load_target": float(dpt["i_load_target"]),
            "l_load": float(dpt["l_load"]),
            "t1_on": float(dpt["t1_on"]),
            "t1_width": float(dpt["t1_width"]),
            "t2_on": float(dpt["t2_on"]),
            "t2_width": float(dpt["t2_width"]),
            "edge_time": float(dpt["edge_time"]),
            "t_end": float(dpt["t_end"]),
            "max_step": float(dpt["max_step"]),
        },
        "repeats": 1,
        "current_definition": "I(Lload)-I(Dclamp)",
        "energy_current_definition": {
            "golden": None, "tier2": None,
            "rule": ("switching energy integrates the signed device current "
                     "I(Lload)-I(Dclamp); timing/G3 remain on its absolute "
                     "magnitude"),
        },
        "golden_failure": None,
        "golden_metrics": {},
        "tier2_metrics": {},
        "wall_clock_s": {
            "golden_median": None, "tier2_median": None, "speedup": None,
        },
    }
    try:
        golden = _one(golden_lib, golden_sub, golden_roles, cfg)
        record["golden_metrics"] = {k: golden["metrics"][k]
                                    for k in _METRIC_KEYS}
        record["energy_current_definition"]["golden"] = \
            golden["energy_current_definition"]
        record["wall_clock_s"]["golden_median"] = golden["wall_clock_s"]
        record["current_definition"] = golden["current_definition"]
    except Exception as exc:  # a vendor abort is reported, never fabricated
        record["golden_failure"] = f"{type(exc).__name__}: {exc}"[:300]

    try:
        tier2 = _one(tier2_lib, tier2_lib.stem,
                     ["drain", "gate", "source"], cfg)
        record["tier2_metrics"] = {k: tier2["metrics"][k]
                                   for k in _METRIC_KEYS}
        record["energy_current_definition"]["tier2"] = \
            tier2["energy_current_definition"]
        record["wall_clock_s"]["tier2_median"] = tier2["wall_clock_s"]
    except Exception as exc:
        record["tier2_failure"] = f"{type(exc).__name__}: {exc}"[:300]

    gold_wall = record["wall_clock_s"]["golden_median"]
    tier2_wall = record["wall_clock_s"]["tier2_median"]
    if gold_wall and tier2_wall:
        record["wall_clock_s"]["speedup"] = float(gold_wall / tier2_wall)

    out = ROOT / "reports" / "comparison" / vendor / "dpt_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(_json_safe(record), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"vendor": vendor, "path": str(out.relative_to(ROOT)),
            "golden_failure": record["golden_failure"]}


def main() -> None:
    for vendor in VENDORS:
        print(json.dumps(regenerate(vendor), sort_keys=True))


if __name__ == "__main__":
    main()
