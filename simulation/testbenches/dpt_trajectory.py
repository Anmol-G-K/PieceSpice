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

from config.yaml_io import load_input
from parser.netlist import resolve_pin_connections
from simulation.engine import SimResult, SpiceEngine, TranDirective
from simulation.netlist_text import NetlistBuilder
from simulation.testbenches.speedup_bench import _pulse_train


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


def _state_names(result: SimResult) -> list[str]:
    """Internal X-instance node voltages, in deterministic sorted order."""
    return sorted(k for k in result.traces if k.startswith("V(xdut:"))


def capture_trajectory(vendor: str, *, timeout_s: float = 120.0,
                       temperature: float | None = None) -> Trajectory:
    """Run the configured two-pulse DPT and return its state/input/output."""
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
    return Trajectory(vendor=vendor, time=t, state=x, state_names=names,
                      input=u, output=y, pin_roles=pins,
                      subcircuit=cfg["subcircuit"], result=result,
                      edge_time=float(dpt["edge_time"]))
