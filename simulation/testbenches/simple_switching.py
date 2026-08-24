"""M4-subset: minimal 50 kHz / 50% duty resistive switching testbench.

Circuit (deliberately the simplest meaningful switching stress):
    Vbus ── R_load ── drain ... source ── gnd
    Vgs pulse source on gate (0 -> Vgs_on), f=50kHz, duty=50%.

R_load is chosen from the YAML so ON-state current = Id_switch_target (1 A):
R = Vbus / I. All bias values come from config; nothing hardcoded.

The testbench runs IDENTICALLY against the vendor .lib and the synthesized
Tier-2 .lib so metrics are directly comparable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.diagnostics import install_notebook_quiet_console
from simulation.engine import SpiceEngine

# Notebook-safe: suppress ngspice console echo at the callback level
# (under Jupyter each message costs a captured-pipe write -> 10x+ slowdown).
install_notebook_quiet_console()

_LOGGER = logging.getLogger(__name__)

F_SW_HZ = 50_000.0          # user spec
DUTY = 0.5                  # user spec
N_PERIODS_STEADY = 3        # simulate this many periods; analyze the LAST


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
    """Load resistance that draws i_target at vbus (on-state)."""
    return float(vbus) / float(i_target)


def run_simple_switching(
    library_path: Path,
    subcircuit_name: str,
    pin_order: list[str],
    *,
    vbus: float,
    vgs_on: float,
    i_load_target: float = 1.0,
    temperature: float = 25.0,
) -> SwitchingRun:
    """Simulate one 50 kHz / 50% duty switching transient.

    Args mirror config/inputs/<vendor>.yaml conditions - no device-specific
    values live here.
    """
    r_load = _r_for_current(vbus, i_load_target)
    period = 1.0 / F_SW_HZ
    end_time = N_PERIODS_STEADY * period
    # resolve edges in ~2 ns to capture tens-of-ns transitions
    step_time = 1e-9 * 2

    engine = SpiceEngine(temperature=temperature)
    circuit = engine.create_circuit(f"M4SUBSET_{subcircuit_name}")

    circuit.V("bus", "vbus_node", circuit.gnd, float(vbus))
    circuit.R("load", "vbus_node", "drain", r_load)

    # Pulse gate drive: rise/fall 20 ns (typical logic driver edge)
    t_rise = t_fall = 20e-9
    pw = DUTY * period - t_rise
    circuit.PulseVoltageSource(
        "gs", "gate", circuit.gnd,
        initial_value=0.0, pulsed_value=float(vgs_on),
        delay_time=period * 0.25,      # let supply settle first quarter
        rise_time=t_rise, fall_time=t_fall,
        pulse_width=pw, period=period,
    )

    pins = dict(zip(("drain", "gate", "source"),
                    ("drain", "gate", circuit.gnd)))
    order = [pins.get(p, p) for p in pin_order]

    circuit.include(str(Path(library_path)))

    circuit.X("DUT", subcircuit_name, *order)

    analysis = engine.run_transient(circuit, step_time=step_time,
                                    end_time=end_time)
    time = np.asarray(analysis.time, dtype=float)
    v_drain = np.asarray(analysis.nodes["drain"], dtype=float)
    v_gate = np.asarray(analysis.nodes["gate"], dtype=float)

    # Id through the load = (Vbus - Vds)/R (sign-safe for both models)
    i_d = (float(vbus) - v_drain) / r_load

    return SwitchingRun(
        time=time,
        v_ds=v_drain,
        i_d=i_d,
        v_gs=v_gate,
        v_bus=float(vbus),
        i_load=float(i_load_target),
        r_load=r_load,
        vgs_on=float(vgs_on),
    )


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
