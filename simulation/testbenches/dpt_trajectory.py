"""Capture one vendor DPT trajectory for TPWL training.

State, input and output channels are taken from the LTspice raw file:

* state x(t) = all internal subcircuit node voltages, exposed with
  ``.save V(xdut:*)`` (verified working under LTspice 26 batch mode);
* input u(t) = [V(gate), V(drain)] — the two terminal voltages that drive
  the device from the outside;
* output y(t) = [I_d, I_g] reconstructed from *external* bench currents:
  I_d = I(Lload) - I(Dclamp), I_g = -I(Vgs). This identity was verified
  against Ix(xdut:1)/Ix(xdut:2) on IRFS7730PBF to machine precision.

All bias/timing values come from config/inputs/<vendor>.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import math

from config.yaml_io import load_input
from parser.netlist import resolve_pin_connections
from simulation.engine import SimResult, SpiceEngine, TranDirective
from simulation.netlist_text import NetlistBuilder
from simulation.testbenches.speedup_bench import (
    _iterations as _log_iterations, _pulse_train)

# Completion fractions used by the golden reference guard.  These mirror the
# unified-ROM DPT guard constants; no new tolerance is introduced here.
DPT_T_LAST_FRACTION = 0.999
DPT_TRACE_POINT_FRACTION = 0.5


@dataclass
class Trajectory:
    """One DPT training trajectory from the golden vendor model."""
    vendor: str
    time: np.ndarray                 # (K,)
    state: np.ndarray                # (N, K) internal node voltages
    state_names: list[str]
    input: np.ndarray                # (p, K) = [Vgs, Vds]
    output: np.ndarray               # (q, K) = [Id, Ig]
    pin_roles: list[str]
    subcircuit: str
    result: SimResult                # full engine result (log, raw path)
    edge_time: float = 0.0                 # configured Vgs edge time (s)
    completion: dict | None = None                 # captured reference guard evidence


def _state_names(result: SimResult) -> list[str]:
    """Internal X-instance node voltages, in deterministic sorted order."""
    return sorted(k for k in result.traces if k.startswith("V(xdut:"))

def trajectory_completion(time, log_text: str, *, t_end: float,
                          max_step, t_last_fraction: float = DPT_T_LAST_FRACTION,
                          point_fraction: float = DPT_TRACE_POINT_FRACTION
                          ) -> dict:
    """Fail-closed guard evidence for the golden reference trajectory.

    The reference is a third capture with its own ``.save`` card; a short
    capture must never become the denominator of a DPT fidelity comparison.
    ``min_points`` is derived from the configured ``max_step`` (no vendor
    constant) and the existing 50 % trace-point fraction.  No new tolerance
    is introduced.
    """
    t = np.asarray(time, dtype=float)
    points = int(t.size)
    iterations = _log_iterations(log_text or "") if log_text else None
    t_last = float(t[-1]) if points else None
    end = float(t_end)
    try:
        step = float(max_step)
    except (TypeError, ValueError):
        step = None
    min_points = None
    if step is not None and math.isfinite(step) and step > 0.0:
        expected = math.ceil(end / step) + 1
        min_points = math.ceil(point_fraction * expected)
    reasons: list[str] = []
    if iterations is None or iterations <= 0:
        reasons.append("non_positive_iterations")
    if points == 0:
        reasons.append("no_trace_points")
    if t_last is None or t_last < t_last_fraction * end:
        reasons.append("t_last_below_guard")
    if min_points is not None and points < min_points:
        reasons.append("insufficient_trace_points")
    return {
        "completed": not reasons,
        "t_last": t_last, "iterations": iterations, "points": points,
        "min_points": min_points, "t_end": end,
        "max_step": step if step is not None and math.isfinite(step) else None,
        "reasons": reasons,
    }


def capture_trajectory(vendor: str, *, timeout_s: float = 120.0,
                       temperature: float | None = None,
                       require_complete: bool = False) -> Trajectory:
    """Run the configured two-pulse DPT and return its state/input/output.

    ``require_complete`` makes a short/zero-iteration/too-sparse reference
    raise instead of returning a trajectory usable as a golden denominator.
    """
    cfg = load_input(vendor)
    dpt = cfg["dpt"]
    lib = Path(cfg["library"]).resolve()
    pins = resolve_pin_connections(
        lib, cfg["subcircuit"], cfg["pins"]["pin_order_semantic"])
    temp = cfg["conditions"]["temperature"] if temperature is None else temperature

    nb = NetlistBuilder(f"TPWL_TRAIN_{vendor}")
    nb.include(lib)
    nb.v_dc("bus", "vbus", "0", float(dpt["vbus"]))
    nb.l("load", "vbus", "drain", float(dpt["l_load"]))
    nb.raw_line("Dclamp drain vbus DCLAMP")
    nb.raw_line(".model DCLAMP D(IS=1e-12 N=1 RS=1e-3 CJO=0)")
    nb.raw_line("Vgs gate 0 PWL(" + _pulse_train(
        float(dpt["t1_on"]), float(dpt["t1_width"]), float(dpt["t2_width"]),
        float(dpt["t2_on"]) - float(dpt["t1_on"]), 2,
        float(dpt["edge_time"]), float(dpt["vgs_on"])) + ")")
    nb.x_instance("dut", cfg["subcircuit"], pins, temperature=temp)
    nb.raw_line(".save V(xdut:*) V(drain) V(gate) I(Lload) I(Dclamp) I(Vgs)")

    step = dpt.get("max_step")
    directive = TranDirective(
        step_time=float(step) if step is not None else float(dpt["edge_time"]),
        end_time=float(dpt["t_end"]),
        max_step=float(step) if step is not None else None)
    result = SpiceEngine(temperature=temp, timeout_s=timeout_s).run_transient(
        nb.render(), directive)

    names = _state_names(result)
    if not names:
        raise RuntimeError(f"{vendor}: no internal state traces captured")
    t = np.asarray(result.get("time"), dtype=float)
    x = np.vstack([np.asarray(result.get(n), dtype=float) for n in names])
    u = np.vstack([np.asarray(result.get("V(gate)"), dtype=float),
                   np.asarray(result.get("V(drain)"), dtype=float)])
    y = np.vstack([
        np.asarray(result.get("I(Lload)"), dtype=float)
        - np.asarray(result.get("I(Dclamp)"), dtype=float),
        -np.asarray(result.get("I(Vgs)"), dtype=float)])
    if not (x.shape[1] == u.shape[1] == y.shape[1] == t.shape[0]):
        raise RuntimeError(f"{vendor}: trace length mismatch")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise RuntimeError(f"{vendor}: non-finite trajectory samples")
    completion = trajectory_completion(
        t, getattr(result, "log_text", ""), t_end=float(dpt["t_end"]),
        max_step=step)
    if require_complete and not completion["completed"]:
        raise RuntimeError(
            f"{vendor}: incomplete golden DPT reference "
            f"({'; '.join(completion['reasons'])})")
    return Trajectory(vendor=vendor, time=t, state=x, state_names=names,
                      input=u, output=y, pin_roles=pins,
                      subcircuit=cfg["subcircuit"], result=result,
                      edge_time=float(dpt["edge_time"]),
                      completion=completion)
