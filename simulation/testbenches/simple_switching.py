"""M4-subset switching testbench — LTspice backend.

Circuit (simplest meaningful switching stress):
    Vbus -- Rload -- drain ... source -- gnd
    Vgs pulse source on gate: 50 kHz, 50% duty, 2 ns edges.

Rload = Vbus / Id_target so on-state current is the configured load.
All bias values come from arguments; nothing device-specific here.

Backend: netlist text -> simulation.engine.SpiceEngine.run_transient ->
SimResult traces. The SwitchingRun contract is unchanged, so
metrics/fitness.py and notebooks/simple_plotting.py keep working.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from simulation.engine import SpiceEngine, TranDirective
from simulation.netlist_text import NetlistBuilder

_LOGGER = logging.getLogger(__name__)

F_SW_HZ = 50_000.0          # user spec
DUTY = 0.5                  # user spec
N_PERIODS_STEADY = 3        # simulate this many periods


@dataclass
class SwitchingRun:
    """Raw transient waveforms of one switching run."""

    time: np.ndarray
    v_ds: np.ndarray
    i_d: np.ndarray
    v_gs: np.ndarray
    v_bus: float
    i_load: float
    r_load: float
    vgs_on: float


def _r_for_current(vbus: float, i_target: float) -> float:
    """Load resistance that draws i_target at vbus."""
    return float(vbus) / float(i_target)


def run_simple_switching(
    library_path,
    subcircuit_name: str,
    pin_order: list[str],
    *,
    vbus: float,
    vgs_on: float,
    i_load_target: float = 1.0,
    temperature: float = 25.0,
) -> SwitchingRun:
    """Simulate one 50 kHz / 50% duty resistive-load switching transient.

    Args mirror config/inputs/<vendor>.yaml conditions - nothing
    device-specific lives here. pin_order is positional role order from
    the yaml semantic pin map; a 'tj' role gets its thermal source
    automatically.
    """
    r_load = _r_for_current(vbus, i_load_target)
    period = 1.0 / F_SW_HZ
    end_time = N_PERIODS_STEADY * period
    step_time = 2e-9                      # resolve ~20 ns edges
    pw = DUTY * period - 20e-9            # gate ON width

    engine = SpiceEngine(temperature=temperature)
    nb = NetlistBuilder(f"M4SUBSET_{subcircuit_name}")
    nb.include(library_path)
    nb.v_dc("bus", "vbus_node", "0", vbus)
    nb.r("load", "vbus_node", "drain", r_load)
    nb.v_pulse("gs", "gate", "0", v1=0.0, v2=vgs_on,
               td=5e-06, tr=2e-08, tf=2e-08, pw=pw, period=period)
    nb.x_instance("dut", subcircuit_name, pin_order, temperature=temperature)

    directive = TranDirective(step_time=step_time, end_time=end_time,
                              max_step=step_time)
    result = engine.run_transient(nb.render(), directive)

    time_s = np.asarray(result.get("time"), dtype=float)

    # Vds across the device: drain node voltage relative to ground
    # (source tied to 0 for 3-pin devices).
    if "V(drain)" in result.traces:
        v_ds = np.asarray(result.get("V(drain)"), dtype=float)
    else:
        raise KeyError(
            f"V(drain) trace missing; available: {sorted(result.traces)}")

    if "I(Rload)" in result.traces:
        i_d = np.abs(np.asarray(result.get("I(Rload)"), dtype=float))
    elif "I(Vbus)" in result.traces:
        i_d = np.abs(np.asarray(result.get("I(Vbus)"), dtype=float))
    else:
        i_d = (vbus - v_ds) / r_load      # analytic fallback (resistive)

    v_gs = (np.asarray(result.get("V(gate)"), dtype=float)
            if "V(gate)" in result.traces
            else np.full_like(time_s, vgs_on))

    return SwitchingRun(
        time=time_s, v_ds=v_ds, i_d=i_d, v_gs=v_gs,
        v_bus=float(vbus), i_load=float(i_load_target),
        r_load=r_load, vgs_on=float(vgs_on))


def steady_state_slice(run: SwitchingRun) -> SwitchingRun:
    """Return only the final full period (transients settled)."""
    period = 1.0 / F_SW_HZ
    t_start = run.time[-1] - period
    mask = run.time >= t_start
    return SwitchingRun(
        time=run.time[mask] - t_start,
        v_ds=run.v_ds[mask],
        i_d=run.i_d[mask],
        v_gs=run.v_gs[mask],
        v_bus=run.v_bus, i_load=run.i_load,
        r_load=run.r_load, vgs_on=run.vgs_on,
    )
