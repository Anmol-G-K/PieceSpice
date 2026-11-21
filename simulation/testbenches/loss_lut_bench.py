"""Golden-DPT sweeps that feed the Tier-4 conduction/switching loss LUT.

Every numeric input is read from ``config/inputs/<vendor>.yaml``.  The only
derived number is the inductor needed to reach a grid current at the end of
the configured first pulse:  L = Vbus * t1_width / Id  (the same rule that
reproduces every recorded ``dpt.l_load`` value).  No datasheet point, no
fitted loss and no simulator-time constant is introduced here.

Reuse of a recorded golden energy requires exact equality on **every** DPT
condition recorded in ``piecespice.dpt/1.0#conditions``; keys the artifact
does not record are listed explicitly rather than assumed to match.

A grid point is ``not_evaluable`` when the golden run aborts, misses a level
crossing, or yields a negative switching energy (the documented PLECS
negative-loss diagnostic).  NaN is never replaced by a recorded number from a
different condition.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from config.yaml_io import CONFIG_ROOT, OUTPUT_DIR, load_input, load_output
from simulation.testbenches.double_pulse import (
    DoublePulseRun,
    extract_dpt_metrics,
    run_double_pulse,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "comparison"

# Reused project guard from pipeline/tier3.py::run_vendor (see the completion
# guard in docs/fidelity_speed_research.md).  Kept named so provenance can
# label them as reused project decisions, not new tolerances.
T_LAST_FRACTION = 0.999
TRACE_POINT_FRACTION = 0.5
# PLECS P-TC: negative loss is diagnosed/cropped, never reported as a loss.
NEGATIVE_LOSS_DIAGNOSTIC = 0.0

ENERGY_KEYS = ("e_on_uj", "e_off_uj")


@dataclass(frozen=True)
class GridPoint:
    """One (Vbus, load-current, temperature) golden DPT condition."""

    vbus_v: float
    current_a: float
    l_load_h: float
    vbus_rule: str
    current_rule: str
    l_load_rule: str
    temperature_c: float = float("nan")
    temperature_rule: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "vbus_v": self.vbus_v,
            "current_a": self.current_a,
            "l_load_h": self.l_load_h,
            "temperature_c": self.temperature_c,
            "vbus_rule": self.vbus_rule,
            "current_rule": self.current_rule,
            "l_load_rule": self.l_load_rule,
            "temperature_rule": self.temperature_rule,
        }


def _unique_sorted(specs: tuple[tuple[float, str], ...]) -> tuple[tuple[float, str], ...]:
    """Sort config-derived axis values, merging duplicate provenance rules."""
    merged: dict[float, str] = {}
    for raw, rule in specs:
        value = float(raw)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"config-derived grid value must be positive: {rule}={raw!r}")
        if value in merged:
            merged[value] = f"{merged[value]} == {rule}"
        else:
            merged[value] = rule
    return tuple(sorted(merged.items()))


def _unique_sorted_temperatures(
    specs: tuple[tuple[float, str], ...],
) -> tuple[tuple[float, str], ...]:
    """Sort config-derived temperatures, merging duplicate provenance rules.

    Temperatures may be negative; only non-finite values are rejected.
    """
    merged: dict[float, str] = {}
    for raw, rule in specs:
        value = float(raw)
        if not np.isfinite(value):
            raise ValueError(
                f"config-derived temperature must be finite: {rule}={raw!r}")
        if value in merged:
            merged[value] = f"{merged[value]} == {rule}"
        else:
            merged[value] = rule
    return tuple(sorted(merged.items()))


def _loss_grid_axis(config: Mapping[str, Any], key: str, *,
                    positive: bool) -> tuple[tuple[float, str], ...]:
    """Validate one ``config.loss_grid`` axis and pair values with its rule."""
    raw = config["loss_grid"].get(key)
    if raw is None:
        raise ValueError(f"config loss_grid is missing required key {key!r}")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        raise ValueError(f"config loss_grid.{key} must be a list")
    specs = tuple((float(value), f"loss_grid.{key}[{index}]")
                  for index, value in enumerate(raw))
    if not specs:
        raise ValueError(f"config loss_grid.{key} must be non-empty")
    if positive:
        return _unique_sorted(specs)
    return _unique_sorted_temperatures(specs)


def derive_grid(config: Mapping[str, Any]) -> tuple[tuple[GridPoint, ...], dict[str, Any]]:
    """Derive the DPT grid from config; returns points and their rules.

    With ``config.loss_grid`` present, the full Cartesian product of the
    explicit ``vbus_points`` x ``current_points`` x ``temperature_points`` is
    measured.  Without it, the historical config-derived 2-point Vbus/current
    grid at ``conditions.temperature`` is preserved and labelled as the
    default source.
    """
    dpt = config["dpt"]
    cond = config["conditions"]
    l_rule = (
        "L = Vbus * dpt.t1_width / Id "
        "(identical to the recorded dpt.l_load for the configured DPT crossing)"
    )
    loss_grid = config.get("loss_grid")
    if loss_grid is not None:
        if not isinstance(loss_grid, Mapping):
            raise ValueError("config loss_grid must be a mapping")
        v_axis = _loss_grid_axis(config, "vbus_points", positive=True)
        i_axis = _loss_grid_axis(config, "current_points", positive=True)
        t_axis = _loss_grid_axis(config, "temperature_points", positive=False)
        points = tuple(
            GridPoint(
                vbus_v=v, current_a=i, l_load_h=v * float(dpt["t1_width"]) / i,
                vbus_rule=rv, current_rule=ri, l_load_rule=l_rule,
                temperature_c=t, temperature_rule=rt,
            )
            for v, rv in v_axis for i, ri in i_axis for t, rt in t_axis
        )
        return points, {
            "vbus": "exact config values loss_grid.vbus_points, sorted unique",
            "current": ("exact config values loss_grid.current_points, "
                        "sorted unique"),
            "temperature": ("exact config values loss_grid.temperature_points, "
                            "sorted unique"),
            "l_load": l_rule,
            "vgs_on": "dpt.vgs_on (DPT gate drive)",
            "source": "loss_grid",
            "kind": "config_loss_grid",
            "rationale": ("the caller supplied the full measured grid; no new "
                          "numeric grid value is introduced and unmeasured "
                          "crossings stay NaN"),
        }

    v_axis = _unique_sorted((
        (float(dpt["vbus"]), "dpt.vbus"),
        (float(cond["Vbus"]), "conditions.Vbus"),
    ))
    i_axis = _unique_sorted((
        (float(dpt["i_load_target"]), "dpt.i_load_target"),
        (float(cond["Id_load"]), "conditions.Id_load"),
    ))
    temperature = float(cond["temperature"])
    if not np.isfinite(temperature):
        raise ValueError("config conditions.temperature must be finite")
    points = tuple(
        GridPoint(
            vbus_v=v, current_a=i, l_load_h=v * float(dpt["t1_width"]) / i,
            vbus_rule=rv, current_rule=ri, l_load_rule=l_rule,
            temperature_c=temperature,
            temperature_rule="conditions.temperature",
        )
        for v, rv in v_axis for i, ri in i_axis
    )
    rules = {
        "vbus": "exact config values dpt.vbus and conditions.Vbus, sorted unique",
        "current": "exact config values dpt.i_load_target and conditions.Id_load, "
                   "sorted unique",
        "l_load": l_rule,
        "temperature": "conditions.temperature",
        "vgs_on": "dpt.vgs_on (DPT gate drive)",
        "source": "config_derived_default",
        "kind": "project_decision",
        "rationale": "no new numeric grid value is introduced; unmeasured crossings "
                     "stay NaN instead of being interpolated into a measurement",
    }
    return points, rules


def dpt_kwargs(config: Mapping[str, Any], point: GridPoint) -> dict[str, Any]:
    """Build run_double_pulse() inputs solely from config plus the grid point."""
    dpt = config["dpt"]
    keys = ("vgs_on", "t1_on", "t1_width", "t2_on", "t2_width",
            "edge_time", "t_end", "max_step")
    kw = {k: float(dpt[k]) for k in keys}
    temperature = float(point.temperature_c)
    if not math.isfinite(temperature):
        # A legacy/caller-built GridPoint without a temperature uses the
        # config condition rather than passing NaN into the simulator.
        temperature = float(config["conditions"]["temperature"])
    kw.update(vbus=point.vbus_v, i_load_target=point.current_a,
              l_load=point.l_load_h, temperature=temperature)
    return kw


def config_provenance(vendor: str) -> dict[str, Any]:
    """Fingerprint the config file that supplied every numeric input."""
    path = CONFIG_ROOT / "inputs" / f"{vendor}.yaml"
    payload = path.read_bytes()
    return {
        "file": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def curve_provenance(vendor: str, testbench: str) -> dict[str, Any]:
    """Locate a measured output curve without loading its (large) arrays."""
    path = OUTPUT_DIR / vendor / f"{testbench}.yaml"
    return {
        "file": path.relative_to(ROOT).as_posix(),
        "available": path.exists(),
    }


# --- measured off-state resistance (Tier-4 solver stabilization) ------------
# The emitted Tier-4 switch used a hardcoded ideal Roff.  The stabilization
# contract replaces it with a vendor-measured value wherever a Vgs=0 DC sweep
# is measurable.  A failed sweep is NaN with full provenance: the emitter then
# falls back to a project-decision Roff with a loud comment and JSON
# disclosure.  No fallback number is invented in this bench or in the LUT.
ROFF_BENCH = ("simulation/testbenches/static_sweep.StaticSweepTestbench"
              ".run_output_characteristics")
ROFF_FALLBACK = {
    "source": "project_decision",
    "note": ("r_off_ohms is NaN: no measurable vendor Vgs=0 sweep. The emitter "
             "falls back to a project-decision Roff with a loud comment and "
             "JSON disclosure; no number is invented in the LUT."),
}
ROFF_METHOD = ("Roff = Vbus / Ioff; Ioff is the final finite I(Vds) sample of "
               "a Vgs=0 vendor DC sweep over Vds 0..config dpt.vbus")


def _under_root(path: Path) -> bool:
    try:
        path.relative_to(ROOT)
        return True
    except ValueError:
        return False


def vendor_library_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fingerprint the vendor library file that supplies the measured sweep."""
    raw = str(config["library"])
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    out: dict[str, Any] = {
        "file": raw, "available": path.is_file(), "sha256": None, "bytes": None,
    }
    if out["available"]:
        payload = path.read_bytes()
        out.update(
            file=path.relative_to(ROOT).as_posix() if _under_root(path) else str(path),
            sha256=sha256(payload).hexdigest(), bytes=len(payload))
    return out


def r_off_from_sweep(
    x_data: Any, y_data: Any, *, vbus_v: float,
    meta: Mapping[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Derive Roff from a Vgs=0 DC sweep; NaN with reason on bad data.

    ``Ioff`` is the final finite I(Vds) sample, so a non-converged trailing
    point cannot silently replace the measured off-state current.  A
    non-positive current (no measurable off state) is a failure, never a
    fabricated large resistance.
    """
    base = dict(meta)
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    finite = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if finite.size == 0:
        return float("nan"), {
            **base, "kind": "measurement_failed", "value_ohms": float("nan"),
            "ioff_a": float("nan"),
            "reason": "Vgs=0 DC sweep has no finite I(Vds) sample",
        }
    k = int(finite[-1])
    ioff = float(y[k])
    sample_vds = float(x[k])
    if not math.isfinite(ioff) or ioff <= 0.0:
        return float("nan"), {
            **base, "kind": "measurement_failed", "value_ohms": float("nan"),
            "ioff_a": ioff, "final_sample_vds_v": sample_vds,
            "reason": ("final finite Vgs=0 sample is not a positive "
                       "off-state current"),
        }
    roff = float(vbus_v) / ioff
    if not math.isfinite(roff):
        return float("nan"), {
            **base, "kind": "measurement_failed", "value_ohms": float("nan"),
            "ioff_a": ioff, "final_sample_vds_v": sample_vds,
            "reason": "Vbus / Ioff is not finite",
        }
    return roff, {
        **base, "kind": "measured", "value_ohms": roff, "ioff_a": ioff,
        "final_sample_vds_v": sample_vds, "sweep_points": int(y.size),
    }


def measure_r_off_ohms(vendor: str, config: Mapping[str, Any], *,
                       simulate: bool = True) -> tuple[float, dict[str, Any]]:
    """Measure Roff from a fresh Vgs=0 vendor DC sweep; NaN on any failure.

    ``simulate=False`` is the offline path: no LTspice run is attempted and the
    result is an explicit NaN with a ``not_measured`` provenance record.  Pin
    order is resolved semantically (never by position) as everywhere else.
    """
    cond = config["conditions"]
    vbus = float(config["dpt"]["vbus"])
    vds_points = int(cond["Vds_points"])
    base: dict[str, Any] = {
        "bench": ROFF_BENCH,
        "vgs_v": 0.0,
        "vgs_rule": "Vgs=0 off-state vendor DC sweep (Tier-4 stabilization contract)",
        "vbus_v": vbus,
        "vbus_rule": "config dpt.vbus",
        "vds_sweep_max_v": vbus,
        "vds_points": vds_points,
        "vendor_file": vendor_library_provenance(config),
        "config_sha256": config_provenance(vendor)["sha256"],
        "method": ROFF_METHOD,
        "rule": ("measured off-state resistance replaces the hardcoded ideal "
                 "Roff; NaN means emitter fallback with disclosure"),
        "fallback": dict(ROFF_FALLBACK),
    }
    if not simulate:
        return float("nan"), {
            **base, "kind": "not_measured", "value_ohms": float("nan"),
            "ioff_a": float("nan"),
            "reason": ("offline build: no live vendor DC sweep; Roff is NaN "
                       "and the emitter falls back with disclosure"),
        }
    try:
        from parser.netlist import resolve_pin_connections
        from simulation.testbenches.static_sweep import StaticSweepTestbench

        lib = ROOT / str(config["library"])
        subckt = str(config["subcircuit"])
        roles = resolve_pin_connections(
            lib, subckt,
            pin_order_semantic=config.get("pins", {}).get("pin_order_semantic"))
        curves = StaticSweepTestbench().run_output_characteristics(
            lib, subckt, vgs_list=[0.0], vds_max=vbus,
            vds_points=vds_points, pin_roles=roles)
        curve = curves[0]
        return r_off_from_sweep(curve.x_data, curve.y_data,
                                vbus_v=vbus, meta=base)
    except Exception as exc:  # solver/parser failures are reported, never faked
        return float("nan"), {
            **base, "kind": "measurement_failed", "value_ohms": float("nan"),
            "ioff_a": float("nan"), "reason": f"{type(exc).__name__}: {exc}",
        }


def recorded_golden(vendor: str) -> dict[str, Any]:
    """Read the recorded golden DPT metrics, if present, with their conditions."""
    path = REPORT_DIR / vendor / "dpt_metrics.json"
    validation = REPORT_DIR / vendor / "tier3_validation.json"
    golden_points = None
    if validation.exists():
        golden_points = (json.loads(validation.read_text(encoding="utf-8"))
                         .get("dpt_run") or {}).get("golden_trace_points")
    if not path.exists():
        return {"path": path.relative_to(ROOT).as_posix(),
                "available": False, "evaluable": False,
                "golden_trace_points": golden_points,
                "failure": "no recorded piecespice.dpt/1.0 file"}
    data = json.loads(path.read_text(encoding="utf-8"))
    golden = data.get("golden_metrics") or {}
    finite = all(
        isinstance(golden.get(k), (int, float))
        and np.isfinite(float(golden[k])) for k in ENERGY_KEYS
    )
    failure = str(data.get("golden_failure") or "")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "available": True,
        "schema": data.get("schema"),
        "conditions": data.get("conditions") or {},
        "golden": golden,
        "repeats": data.get("repeats"),
        "wall_clock_s": (data.get("wall_clock_s") or {}).get("golden_median"),
        "failure": failure,
        "evaluable": finite and not failure,
        "golden_trace_points": golden_points,
    }


def _expected_conditions(point: GridPoint,
                         config: Mapping[str, Any]) -> dict[str, Any]:
    """Every DPT condition the current config would pass to run_double_pulse."""
    expected = dict(dpt_kwargs(config, point))
    expected["temperature"] = float(config["conditions"]["temperature"])
    return expected


def _recorded_condition_check(recorded: Mapping[str, Any], point: GridPoint,
                              config: Mapping[str, Any]) -> dict[str, Any]:
    """Compare the recorded condition dict with the current config conditions.

    Explicit rule (D4): a recorded golden energy is reused only when every key
    present in ``piecespice.dpt/1.0#conditions`` is present in the current
    config-derived conditions and numerically equal.  Keys the config supplies
    but the artifact does not record (e.g. ``temperature``) are listed under
    ``not_recorded`` rather than silently assumed to match.  A mismatch or an
    unexpected recorded condition vetoes reuse.
    """
    cond = recorded.get("conditions") or {}
    expected = _expected_conditions(point, config)
    check: dict[str, Any] = {
        "rule": ("exact equality on every recorded condition key; keys absent "
                 "from the recorded artifact are reported, not assumed to match"),
        "matched": [], "mismatched": [], "not_recorded": [], "match": False,
    }
    if not recorded.get("available") or not cond:
        check["reason"] = "recorded artifact unavailable or has no conditions"
        return check
    for key in sorted(cond):
        if key not in expected:
            check["mismatched"].append({
                "key": key, "recorded": cond[key], "expected": None,
                "reason": "recorded condition is not a current config condition",
            })
            continue
        try:
            recorded_value = float(cond[key])
            expected_value = float(expected[key])
        except (TypeError, ValueError):
            check["mismatched"].append({
                "key": key, "recorded": cond[key], "expected": expected[key],
                "reason": "non-numeric condition value",
            })
            continue
        entry = {"key": key, "recorded": recorded_value,
                 "expected": expected_value}
        if recorded_value == expected_value:
            check["matched"].append(entry)
        else:
            check["mismatched"].append(entry)
    # The recorded piecespice.dpt/1.0 artifact has no temperature key.  It may
    # only be reused at the config reference temperature; a multi-temperature
    # grid point must be measured, never filled with cross-temperature data.
    if "temperature" not in cond:
        config_temperature = float(config["conditions"]["temperature"])
        point_temperature = float(getattr(point, "temperature_c",
                                          config_temperature))
        if (math.isfinite(point_temperature)
                and point_temperature != config_temperature):
            check["mismatched"].append({
                "key": "temperature",
                "recorded": None,
                "expected": point_temperature,
                "reason": ("recorded artifact has no temperature condition; "
                           "it cannot be reused at a different temperature"),
            })
    for key in sorted(expected):
        if key not in cond:
            check["not_recorded"].append({"key": key, "expected": expected[key]})
    check["match"] = not check["mismatched"]
    return check


def _recorded_matches(recorded: Mapping[str, Any], point: GridPoint,
                      config: Mapping[str, Any]) -> bool:
    """Thin boolean view of the full recorded-condition check."""
    return _recorded_condition_check(recorded, point, config)["match"]


@dataclass
class SweepResult:
    """One measured (or explicitly non-evaluable) golden DPT grid point."""

    point: GridPoint
    e_on_uj: float
    e_off_uj: float
    status: str
    reason: str
    source: str
    conditions: dict[str, Any] = field(default_factory=dict)
    guard: dict[str, Any] = field(default_factory=dict)
    wall_clock_s: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "point": self.point.as_dict(),
            "e_on_uj": self.e_on_uj,
            "e_off_uj": self.e_off_uj,
            "status": self.status,
            "reason": self.reason,
            "source": self.source,
            "conditions": self.conditions,
            "guard": self.guard,
            "wall_clock_s": self.wall_clock_s,
        }


def extract_energy_from_run(
    run: DoublePulseRun,
    *,
    t_end: float,
    source: str,
    reference_points: int | None = None,
    iterations: int | None = None,
    temperature_c: float = float("nan"),
) -> SweepResult:
    """Extract golden switching energies with the Tier-3 completion guard."""
    t = np.asarray(run.time, dtype=float)
    point = GridPoint(
        vbus_v=float(run.v_bus), current_a=float(run.i_load),
        l_load_h=float(run.l_load), vbus_rule="run.v_bus",
        current_rule="run.i_load", l_load_rule="run.l_load",
        temperature_c=float(temperature_c),
        temperature_rule=("run temperature (passed from dpt_kwargs)"
                          if math.isfinite(float(temperature_c)) else
                          "run object exposes no temperature"),
    )
    conditions = {
        "vbus_v": float(run.v_bus), "current_a": float(run.i_load),
        "vgs_on_v": float(run.vgs_on), "l_load_h": float(run.l_load),
        "t2_on_s": float(run.t2_on),
    }
    if math.isfinite(float(temperature_c)):
        conditions["temperature_c"] = float(temperature_c)
    if t.size == 0 or not np.all(np.isfinite(t)):
        return SweepResult(point, float("nan"), float("nan"), "not_evaluable",
                           "empty or non-finite golden time vector", source,
                           conditions)
    t_last = float(t[-1])
    guard = {
        "trace_points": int(t.size),
        "reference_points": None if reference_points is None else int(reference_points),
        "t_last_s": t_last,
        "t_end_s": float(t_end),
        "iterations": iterations,
        "iterations_note": (
            None if iterations is not None else
            "DoublePulseRun does not expose the solver log; iterations N/A"),
        "rule": (f"t_last >= {T_LAST_FRACTION:g}*t_end and trace_points >= "
                 f"{TRACE_POINT_FRACTION:g}*reference_points"),
    }
    complete = (
        t_last >= T_LAST_FRACTION * float(t_end)
        and (reference_points is None or t.size >= TRACE_POINT_FRACTION * reference_points)
        and float(run.t2_on) <= t_last
    )
    if not complete:
        return SweepResult(point, float("nan"), float("nan"), "not_evaluable",
                           "incomplete golden DPT run", source, conditions, guard,
                           float(run.wall_clock_s))

    metrics = extract_dpt_metrics(run)
    values = {k: float(metrics[k]) for k in ENERGY_KEYS}
    guard["golden_metrics"] = {k: values[k] for k in ENERGY_KEYS}
    missing = [k for k, v in values.items() if not np.isfinite(v)]
    if missing:
        return SweepResult(point, values[ENERGY_KEYS[0]], values[ENERGY_KEYS[1]],
                           "not_evaluable",
                           "missing golden DPT crossing for " + ",".join(missing),
                           source, conditions, guard, float(run.wall_clock_s))
    negative = [k for k, v in values.items() if v < NEGATIVE_LOSS_DIAGNOSTIC]
    if negative:
        return SweepResult(point, float("nan"), float("nan"), "not_evaluable",
                           "negative switching loss for " + ",".join(negative),
                           source, conditions, guard, float(run.wall_clock_s))
    return SweepResult(point, values[ENERGY_KEYS[0]], values[ENERGY_KEYS[1]],
                       "measured", "golden DPT switching energy from "
                       "metrics.loss.compute_switching_energy", source,
                       conditions, guard, float(run.wall_clock_s))


def run_golden_sweep(
    vendor: str,
    *,
    config: Mapping[str, Any] | None = None,
    points: tuple[GridPoint, ...] | None = None,
    recorded: Mapping[str, Any] | None = None,
    simulate: bool = True,
    reproduce_recorded: bool = False,
) -> tuple[SweepResult, ...]:
    """Measure E_on/E_off at every grid point; NaN where golden data is absent.

    The recorded golden DPT crossing is reused only at its exact measured
    conditions.  Every other point is simulated with the unchanged
    ``run_double_pulse`` bench and the Tier-3 completion guard.  Failures are
    returned as explicit gaps, never imputed.
    """
    cfg = load_input(vendor) if config is None else config
    grid = points if points is not None else derive_grid(cfg)[0]
    rec = recorded_golden(vendor) if recorded is None else recorded
    results: list[SweepResult] = []
    for point in grid:
        kw = dpt_kwargs(cfg, point)
        conditions = dict(kw)
        conditions.update(point.as_dict())
        source = "simulation/testbenches/double_pulse.run_double_pulse"
        condition_check = _recorded_condition_check(rec, point, cfg)
        if (not reproduce_recorded and rec.get("available")
                and condition_check["match"]):
            source = f"{rec['path']}#golden_metrics"
            guard = {
                "repeats": rec.get("repeats"),
                "iterations": None,
                "trace_points": None,
                "t_last_s": None,
                "golden_trace_points": rec.get("golden_trace_points"),
                "note": "piecespice.dpt/1.0 records repeats and energies; "
                        "per-run points are in tier3_validation.json, "
                        "iterations N/A for the recorded golden run",
                "condition_check": condition_check,
            }
            if rec["evaluable"]:
                results.append(SweepResult(
                    point, float(rec["golden"][ENERGY_KEYS[0]]),
                    float(rec["golden"][ENERGY_KEYS[1]]), "measured",
                    "recorded vendor golden DPT metrics at identical config "
                    "conditions", source, conditions, guard,
                    None if rec.get("wall_clock_s") is None
                    else float(rec["wall_clock_s"])))
            else:
                results.append(SweepResult(
                    point, float("nan"), float("nan"), "not_evaluable",
                    f"recorded golden not evaluable: {rec.get('failure')}",
                    source, conditions, guard, None))
            continue
        if not simulate:
            results.append(SweepResult(
                point, float("nan"), float("nan"), "unmeasured_grid_node",
                "no golden measurement at this grid point and simulation "
                "disabled; no value invented", source, conditions,
                {"condition_check": condition_check}))
            continue
        try:
            run = run_double_pulse(
                Path(cfg["library"]).resolve(), cfg["subcircuit"],
                list(cfg["pins"]["pin_order"]), **kw)
        except Exception as exc:  # solver/backend failures are reported, never faked
            results.append(SweepResult(
                point, float("nan"), float("nan"), "not_evaluable",
                f"{type(exc).__name__}: {exc}", source, conditions))
            continue
        results.append(extract_energy_from_run(
            run, t_end=kw["t_end"], source=source,
            reference_points=(None if not rec.get("available") else
                              rec.get("golden_trace_points")),
            temperature_c=float(kw["temperature"]),
        ))
    return tuple(results)
