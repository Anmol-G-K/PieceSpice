"""Stage-6 harness: three arms, one bench, one record.

The measurement that matters is a difference of differences:

```text
A  vendor golden .lib   --LTspice-->        the reference
B  emitted ROM .lib     --LTspice-->        same simulator, reduced model
C  solver ROM tables    --our solver-->     same model, different engine

B - A  = model error    (what the ROM costs; already recorded elsewhere)
C - B  = solver error   (what the engine costs -- the only new number)
C - A  = total          (what a user actually gets)
```

Keeping A, B and C apart is the whole point. A single "solver vs golden" number
would fold the ROM's approximation into the engine's and neither could be
attributed or improved.

Two honesty rules are enforced in the record rather than in prose:

* every waveform difference is reported **peak-normalised** (a near-zero
  current trace has a tiny span and a raw relative RMSE would be meaningless);
* every speed row carries a completion guard and the measured LTspice floor,
  because a short run's speedup is mostly LTspice's fixed startup.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from engine.rom import SolverRom
from engine.rom.artifact import dump as dump_rom
from engine.rom.bench import DptConditions, dpt
from engine.solver.transient import solve_transient
from simulation.testbenches.double_pulse import (
    DoublePulseRun,
    extract_dpt_metrics,
    run_double_pulse,
)

SCHEMA = "piecespice.solver_validation/1.0"

#: LTspice's reported wall time on a trivial deck.  Measured on this machine at
#: 0.055 s; the repository constant (0.075 s) is used so the number is
#: comparable with the other records rather than flattering this one.
LTSPICE_FLOOR_S = 0.075

ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class Arm:
    """One measurable arm: waveforms, cost, and its own metric block."""

    name: str
    time: np.ndarray
    v_ds: np.ndarray
    i_d: np.ndarray
    v_gs: np.ndarray
    wall_s: float
    elapsed_s: float | None
    points: int
    steps: int | None = None
    events: int | None = None
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _load_config(vendor: str) -> dict:
    path = ROOT / "config" / "inputs" / f"{vendor}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _header(lib: Path) -> tuple[str, list[str]]:
    """Subcircuit name and pin tokens from the file's ``.subckt`` header.

    A regex, not the deck reader: SPICE treats line 1 of a *file* as a title and
    some emitted libs (the Tier-3 ones) start directly with ``.subckt``, so
    reading them as decks would drop the header.
    """
    text = lib.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^\s*\.subckt\s+(\S+)\s+(.+)$", text,
                      re.IGNORECASE | re.MULTILINE)
    if match is None:
        raise ValueError(f"{lib}: no .subckt header")
    tokens = [t for t in match.group(2).split() if "=" not in t
              and not t.lower().startswith("params:")]
    return match.group(1), [t.lower() for t in tokens]


_DEFAULT_ROLE = {"d": "drain", "g": "gate", "s": "source", "tj": "tj"}


def _pin_roles(lib: Path, subcircuit: str,
               semantic: dict[str, str] | None) -> list[str]:
    """Roles in header order, derived from the file itself.

    The config's ``pin_order`` lists the ports a bench *drives*, which is not
    the same as the ports the subcircuit *has*: CPM2-1200's fourth ``Tj`` port
    is absent from ``pin_order``, so a bench built from the config alone leaves
    it floating and LTspice aborts (the catalogued VM-10).  Reading the header
    is what keeps the fourth port driven.
    """
    name, tokens = _header(lib)
    if name.lower() != subcircuit.lower():
        raise ValueError(
            f"{lib}: header declares {name!r}, config names {subcircuit!r}")
    by_token = {token.lower(): role.lower()
                for role, token in (semantic or {}).items()}
    roles: list[str] = []
    for token in tokens:
        role = by_token.get(token) or _DEFAULT_ROLE.get(token)
        if role is None:
            raise ValueError(f"{lib}: cannot map pin {token!r} to a role")
        roles.append(role)
    return roles


def _rom_subcircuit(lib: Path) -> tuple[str, list[str]]:
    """Subcircuit name and pin roles straight from the emitted header.

    A regex, not the deck reader: SPICE treats line 1 of a *file* as a title and
    some emitted libs (the Tier-3 ones) start directly with ``.subckt``, so
    reading them as decks would drop the header.
    """
    text = lib.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^\.subckt\s+(\S+)\s+(.+)$", text, re.IGNORECASE | re.MULTILINE)
    if match is None:
        raise ValueError(f"{lib}: no .subckt header")
    roles = {"d": "drain", "g": "gate", "s": "source"}
    pins = [p.lower() for p in match.group(2).split()]
    try:
        return match.group(1), [roles[p] for p in pins]
    except KeyError as exc:
        raise ValueError(f"{lib}: unexpected pin {exc}") from None


def _conditions(cfg: dict) -> dict:
    dpt = cfg["dpt"]
    return {
        "vbus": float(dpt["vbus"]),
        "vgs_on": float(dpt["vgs_on"]),
        "i_load_target": float(dpt.get("i_load_target", 10.0)),
        "l_load": float(dpt["l_load"]),
        "t1_on": float(dpt["t1_on"]),
        "t1_width": float(dpt["t1_width"]),
        "t2_on": float(dpt["t2_on"]),
        "t2_width": float(dpt["t2_width"]),
        "edge_time": float(dpt["edge_time"]),
        "t_end": float(dpt["t_end"]),
        "max_step": float(dpt["max_step"]),
        "temperature": float(cfg.get("conditions", {}).get("temperature", 25.0)),
    }


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def _ltspice_arm(name: str, library: Path, subcircuit: str,
                 pin_order: list[str], cond: dict,
                 repeats: int = 3) -> Arm:
    runs: list[DoublePulseRun] = []
    for _ in range(repeats):
        runs.append(run_double_pulse(library, subcircuit, pin_order, **cond))
    walls = [r.wall_clock_s for r in runs]
    elapsed = [r.solver_elapsed_s for r in runs if r.solver_elapsed_s]
    # Keep the run that got furthest: repeats exist to smooth timing noise, not
    # to average over a truncated trace.  A truncated run must never be
    # averaged in -- that is how an aborted simulation becomes a waveform.
    best = max(runs, key=lambda run: run.time[-1])
    complete = (np.all(np.isfinite(best.v_ds))
                and best.time[-1] >= cond["t_end"] * 0.999)
    if not complete:
        raise RuntimeError(
            f"{name}: incomplete run (t_last {best.time[-1]:g} s of "
            f"{cond['t_end']:g} s, finite={bool(np.all(np.isfinite(best.v_ds)))})")
    runs_complete = sum(
        1 for run in runs
        if np.all(np.isfinite(run.v_ds)) and run.time[-1] >= cond["t_end"] * 0.999)
    return Arm(
        name=name,
        time=best.time,
        v_ds=best.v_ds,
        i_d=best.i_d,
        v_gs=best.v_gs,
        wall_s=float(statistics.median(walls)),
        elapsed_s=float(statistics.median(elapsed)) if elapsed else None,
        points=int(best.time.size),
        warnings=([f"{len(runs) - runs_complete} of {len(runs)} repeats were "
                   "incomplete and were discarded"] if runs_complete < len(runs)
                  else []),
        metrics=extract_dpt_metrics(best),
    )


def _solver_arm(rom: SolverRom, cond: dict, repeats: int = 3,
                *, adaptive: bool = False, label: str = "solver",
                honour_ceiling: bool = False) -> Arm:
    conditions = DptConditions(
        vbus=cond["vbus"], vgs_on=cond["vgs_on"],
        i_load_target=cond["i_load_target"], l_load=cond["l_load"],
        t1_on=cond["t1_on"], t1_width=cond["t1_width"],
        t2_on=cond["t2_on"], t2_width=cond["t2_width"],
        edge_time=cond["edge_time"], t_end=cond["t_end"],
        max_step=cond["max_step"], temperature_c=cond["temperature"],
    )
    circuit = dpt(rom, conditions)
    kwargs = dict(engine="compiled", adaptive=adaptive)
    if not adaptive or honour_ceiling:
        kwargs["max_step"] = cond["max_step"]
    solve_transient(circuit, **kwargs)                  # warm the JIT
    results = [solve_transient(circuit, **kwargs) for _ in range(repeats)]
    walls = [r.solver_time_s for r in results]
    result = results[-1]
    t = result.get("time")
    if not np.all(np.isfinite(result.voltage("drain"))):
        raise RuntimeError("solver arm produced a non-finite drain voltage")
    i_load = result.current("Lload")
    i_clamp = result.current("Dclamp")
    signed = i_load - i_clamp
    run = DoublePulseRun(
        time=t, v_ds=result.voltage("drain"), i_d=np.abs(signed),
        v_gs=result.voltage("gate"), v_bus=cond["vbus"],
        i_load=cond["i_load_target"], l_load=cond["l_load"],
        vgs_on=cond["vgs_on"], t2_on=cond["t2_on"],
        wall_clock_s=float(statistics.median(walls)),
        current_definition="I(Lload)-I(Dclamp)", i_d_signed=signed,
    )
    return Arm(
        name=label,
        time=t,
        v_ds=run.v_ds,
        i_d=run.i_d,
        v_gs=run.v_gs,
        wall_s=float(statistics.median(walls)),
        elapsed_s=None,
        points=int(t.size),
        steps=result.steps,
        events=len(result.events),
        warnings=list(result.warnings),
        metrics=extract_dpt_metrics(run),
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _on(reference: Arm, other: Arm, field: str) -> tuple[np.ndarray, np.ndarray]:
    """Sample *other* onto the reference time grid, inside the shared span."""
    ref_t = reference.time
    ref_y = getattr(reference, field)
    lo = max(ref_t[0], other.time[0])
    hi = min(ref_t[-1], other.time[-1])
    mask = (ref_t >= lo) & (ref_t <= hi)
    return ref_y[mask], np.interp(ref_t[mask], other.time, getattr(other, field))


def _edge_mask(arm: Arm, half_width: float = 50e-9) -> np.ndarray:
    """True within +/- *half_width* of a gate transition.

    The gate waveform is the clock of a switching bench: its steepest slopes
    mark where the device is actually switching.  Splitting the error into
    edge and non-edge parts separates "the edges are mis-timed" (a timing
    finding) from "the model is wrong" (a fidelity finding) -- they are
    different problems with different fixes.
    """
    slope = np.abs(np.gradient(arm.v_gs, arm.time))
    peak = float(np.max(slope)) if slope.size else 0.0
    if peak <= 0.0:
        return np.zeros(arm.time.size, dtype=bool)
    mask = slope > 0.1 * peak
    out = np.zeros(arm.time.size, dtype=bool)
    for index in np.flatnonzero(mask):
        t0 = arm.time[index]
        out |= np.abs(arm.time - t0) <= half_width
    return out


def _waveform_difference(reference: Arm, other: Arm, field: str) -> dict[str, Any]:
    ref_y, test_y = _on(reference, other, field)
    error = test_y - ref_y
    peak = float(np.max(np.abs(ref_y)))
    scale = max(peak, 1e-12)
    edge = _edge_mask(reference)[:ref_y.size] if ref_y.size else np.zeros(0, bool)
    total_sq = float(np.sum(error ** 2))
    edge_sq = float(np.sum(error[edge] ** 2)) if edge.any() else 0.0
    return {
        "points": int(ref_y.size),
        "peak": peak,
        "max_abs": float(np.max(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "rel_rmse_pct": float(np.sqrt(np.mean(error ** 2)) / scale * 100.0),
        "max_rel_pct": float(np.max(np.abs(error)) / scale * 100.0),
        # Pearson r answers "is this the same shape?" separately from "how far
        # off is it?" -- a gain error and a phase shift score very differently.
        "pearson_r": float(np.corrcoef(ref_y, test_y)[0, 1])
        if ref_y.size > 2 and np.std(ref_y) > 0 and np.std(test_y) > 0 else float("nan"),
        # Where the error lives: inside the switching windows or everywhere.
        "edge_time_fraction": float(np.mean(edge)) if edge.size else 0.0,
        "error_in_edges_fraction": (edge_sq / total_sq) if total_sq > 0 else 0.0,
        "ref_min": float(np.min(ref_y)),
        "ref_max": float(np.max(ref_y)),
        "test_min": float(np.min(test_y)),
        "test_max": float(np.max(test_y)),
    }


def _compare(reference: Arm, other: Arm) -> dict[str, Any]:
    return {
        "reference": reference.name,
        "measured": other.name,
        "vds": _waveform_difference(reference, other, "v_ds"),
        "id": _waveform_difference(reference, other, "i_d"),
    }


def _metric_delta(reference: dict, other: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("td_on_ns", "tr_ns", "td_off_ns", "tf_ns",
                "e_on_uj", "e_off_uj", "i_peak_a"):
        a = reference.get(key)
        b = other.get(key)
        if a is None or b is None:
            out[key] = {"reference": a, "measured": b, "delta": None,
                        "rel_pct": None}
            continue
        a, b = float(a), float(b)
        out[key] = {
            "reference": a,
            "measured": b,
            "delta": b - a,
            "rel_pct": (b - a) / a * 100.0 if a else None,
        }
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def validate_vendor(vendor: str, *, repeats: int = 3,
                    out_dir: str | Path = "reports/comparison",
                    run_rom_ltspice: bool = True) -> dict[str, Any]:
    """Run the three arms for *vendor* and write the validation record."""
    cfg = _load_config(vendor)
    cond = _conditions(cfg)
    name = str(cfg["device"])

    semantic = dict(cfg["pins"].get("pin_order_semantic", {}))
    golden_lib = ROOT / cfg["library"]
    golden_pins = _pin_roles(golden_lib, cfg["subcircuit"], semantic)
    golden = _ltspice_arm("golden", golden_lib, cfg["subcircuit"],
                          golden_pins, cond, repeats)

    rom_lib = ROOT / "reports" / "models" / f"{vendor}_PIECESPICE_T2.lib"
    if not rom_lib.is_file():
        raise FileNotFoundError(
            f"{rom_lib} is missing; run the Tier-2 fit for {vendor} first")
    rom_subckt, rom_pins = _rom_subcircuit(rom_lib)

    arms = {"golden": golden}
    if run_rom_ltspice:
        arms["rom_ltspice"] = _ltspice_arm(
            "rom_ltspice", rom_lib, rom_subckt, rom_pins, cond, repeats)

    from pipeline.fitting import build_pwl_spec

    rom = SolverRom.from_spec(build_pwl_spec(vendor))
    # Three step policies on the same model: a forced step, error control under
    # SPICE's max-step ceiling (what LTspice actually does with this deck), and
    # error control alone.
    arms["solver"] = _solver_arm(rom, cond, repeats, adaptive=False,
                                 label="solver")
    arms["solver_capped"] = _solver_arm(rom, cond, repeats, adaptive=True,
                                        honour_ceiling=True,
                                        label="solver_capped")
    arms["solver_adaptive"] = _solver_arm(rom, cond, repeats, adaptive=True,
                                          label="solver_adaptive")

    out_path = ROOT / out_dir / vendor / "solver_validation.json"
    artifacts = {"rom": str(dump_rom(rom, out_path.with_suffix(".rom.json")))}

    record: dict[str, Any] = {
        "schema": SCHEMA,
        "vendor": vendor,
        "device": name,
        "conditions": cond,
        "rom_artifact": artifacts,
        "arms": {
            key: {
                "wall_s": arm.wall_s,
                "ltspice_elapsed_s": arm.elapsed_s,
                "points": arm.points,
                "steps": arm.steps,
                "events": arm.events,
                "warnings": arm.warnings,
                "metrics": arm.metrics,
            }
            for key, arm in arms.items()
        },
        "comparison": {},
        "speed": {},
    }

    if "rom_ltspice" in arms:
        reference = arms["rom_ltspice"]
        record["comparison"]["model_error_B_minus_A"] = _compare(golden, reference)
        record["comparison"]["timing_B_minus_A"] = _metric_delta(
            golden.metrics, reference.metrics)
        for key in ("solver", "solver_capped", "solver_adaptive"):
            if key not in arms:
                continue
            suffix = "C_minus_B" if key == "solver" else f"C_minus_B_{key[7:]}"
            record["comparison"][f"solver_error_{suffix}"] = _compare(
                reference, arms[key])
            record["comparison"][f"timing_{suffix}"] = _metric_delta(
                reference.metrics, arms[key].metrics)
    for key in ("solver", "solver_capped", "solver_adaptive"):
        if key not in arms:
            continue
        suffix = "C_minus_A" if key == "solver" else f"C_minus_A_{key[7:]}"
        record["comparison"][f"total_{suffix}"] = _compare(golden, arms[key])
        record["comparison"][f"timing_{suffix}"] = _metric_delta(
            golden.metrics, arms[key].metrics)

    reference_solver = (arms["rom_ltspice"].elapsed_s
                        if "rom_ltspice" in arms else golden.elapsed_s)
    solver_arm = arms["solver"]
    adaptive_arm = arms.get("solver_adaptive")
    for key, block_name in (("solver_capped", "speed_capped"),
                            ("solver_adaptive", "speed_adaptive")):
        arm = arms.get(key)
        if arm is None:
            continue
        record[block_name] = {
            "solver_wall_s": arm.wall_s,
            "solver_steps": arm.steps,
            "us_per_step": (arm.wall_s / arm.steps * 1e6 if arm.steps else None),
            "reference_ltspice_elapsed_s": reference_solver,
            "solver_elapsed_speedup": (
                reference_solver / arm.wall_s
                if reference_solver and arm.wall_s else None),
            "net_solver_elapsed_speedup": (
                max((reference_solver or 0.0) - LTSPICE_FLOOR_S, 1e-4) / arm.wall_s
                if reference_solver else None),
            "relative_steps_vs_fixed": (arm.steps / solver_arm.steps
                                        if solver_arm.steps else None),
        }
    if False:
        record["speed_adaptive"] = {
            "solver_wall_s": adaptive_arm.wall_s,
            "solver_steps": adaptive_arm.steps,
            "us_per_step": (adaptive_arm.wall_s / adaptive_arm.steps * 1e6
                            if adaptive_arm.steps else None),
            "reference_ltspice_elapsed_s": reference_solver,
            "solver_elapsed_speedup": (
                reference_solver / adaptive_arm.wall_s
                if reference_solver and adaptive_arm.wall_s else None),
            "net_solver_elapsed_speedup": (
                max((reference_solver or 0.0) - LTSPICE_FLOOR_S, 1e-4)
                / adaptive_arm.wall_s if reference_solver else None),
            "relative_steps_vs_fixed": (
                adaptive_arm.steps / solver_arm.steps
                if solver_arm.steps else None),
        }
    reference_wall = arms["rom_ltspice"].wall_s if "rom_ltspice" in arms \
        else golden.wall_s
    record["speed"] = {
        "solver_wall_s": solver_arm.wall_s,
        "solver_steps": solver_arm.steps,
        "solver_points": solver_arm.points,
        "us_per_step": (solver_arm.wall_s / solver_arm.steps * 1e6
                        if solver_arm.steps else None),
        "reference_ltspice_wall_s": reference_wall,
        "reference_ltspice_elapsed_s": reference_solver,
        "wall_speedup": (reference_wall / solver_arm.wall_s
                         if solver_arm.wall_s else None),
        "solver_elapsed_speedup": (reference_solver / solver_arm.wall_s
                                   if reference_solver and solver_arm.wall_s
                                   else None),
        "net_solver_elapsed_speedup": (
            max((reference_solver or 0.0) - LTSPICE_FLOOR_S, 1e-4)
            / solver_arm.wall_s if reference_solver else None),
        "ltspice_floor_s": LTSPICE_FLOOR_S,
        "note": ("reference is the emitted-ROM LTspice arm when available, so "
                 "the speed ratio is engine-vs-engine on one model"),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True),
                        encoding="ascii")
    return record


def summarise(record: dict[str, Any]) -> str:
    """One-line-per-number digest, for the terminal."""
    lines = [f"{record['vendor']} ({record['device']})"]
    for key in ("model_error_B_minus_A", "solver_error_C_minus_B",
                "total_C_minus_A"):
        block = record["comparison"].get(key)
        if block is None:
            continue
        vds = block["vds"]
        id_ = block["id"]
        lines.append(
            f"  {key:22s} Vds {vds['rel_rmse_pct']:6.2f}% (r={vds['pearson_r']:.4f},"
            f" edges {vds['error_in_edges_fraction'] * 100:3.0f}% of err)"
            f"   Id {id_['rel_rmse_pct']:6.2f}% (r={id_['pearson_r']:.4f},"
            f" edges {id_['error_in_edges_fraction'] * 100:3.0f}% of err)")
    speed = record["speed"]
    if speed.get("solver_elapsed_speedup"):
        lines.append(
            f"  speed {speed['solver_elapsed_speedup']:5.2f}x "
            f"(net {speed['net_solver_elapsed_speedup']:5.2f}x)  "
            f"{speed['us_per_step']:.1f} us/step, {speed['solver_steps']} steps")
    return "\n".join(lines)
