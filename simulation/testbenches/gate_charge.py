"""Gate charge testbench: Vgs vs Qg under constant-current gate drive.

Datasheet conditions reproduced: drain held at Vbus by an ideal source,
gate charged with a constant current starting at t = 0, so Qg(t) = Ig * t.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

from parser.preprocessor import normalize_pspice_netlist
from simulation.engine import SpiceEngine


def run_gate_charge(
    lib_path: Path,
    subcircuit_name: str,
    v_bus: float = 40.0,
    i_load: float = 50.0,
    i_gate: float = 5e-3,
    v_gs_max: float = 10.0,
    t_end: float = 120e-6,
    step_time: float = 20e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the gate-charge transient under datasheet conditions.

    A constant-current load (i_load into the drain) plus a rail-clamp diode
    hold the device at its rated load current: an ideal voltage source on the
    drain would let the macromodel conduct thousands of amps, driving its
    internal capacitance-sense network unstable.
    """
    """Run the gate-charge transient.

    Returns:
        q_g: Gate charge array [C]
        v_gs: Gate-source voltage array [V]
        v_ds: Drain-source voltage array [V]
    """
    raw = lib_path.read_text(encoding='utf-8', errors='ignore')
    temp_lib = lib_path.parent / f"_temp_norm_{lib_path.name}"
    temp_lib.write_text(normalize_pspice_netlist(raw))

    try:
        circuit = Circuit("Gate_Charge")
        circuit.include(str(temp_lib.resolve()))
        circuit.X("M1", subcircuit_name, "d", "g", circuit.gnd)
        # Drain rail + clamp: before turn-on the load current charges the
        # drain until the clamp returns it to the rail (~Vbus); after turn-on
        # it flows through the channel, holding Id = i_load during the Miller
        # plateau.
        circuit.V("bus", "vbus", circuit.gnd, float(v_bus)@u_V)
        circuit.model('DCLAMP', 'D', IS=1e-10)
        circuit.D("clamp", "d", "vbus", model='DCLAMP')
        circuit.I("load", circuit.gnd, "d", float(i_load)@u_A)
        circuit.PieceWiseLinearCurrentSource(
            "drive", "g", circuit.gnd,
            values=[
                (0.0@u_s, 0.0@u_A),
                (1e-9@u_s, float(i_gate)@u_A),
                (float(t_end)@u_s, float(i_gate)@u_A),
            ],
        )

        engine = SpiceEngine()
        analysis = engine.run_transient(circuit, step_time=step_time, end_time=t_end)

        time = np.array(analysis.time)
        v_gs = np.array(analysis.nodes['g'])
        v_ds = np.array(analysis.nodes['d'])
        q_g = i_gate * time

        # Truncate at first crossing of v_gs_max
        idx = np.argmax(v_gs >= v_gs_max)
        if v_gs[idx] >= v_gs_max and idx > 0:
            # Linear interpolation onto the exact crossing
            t0, t1 = time[idx - 1], time[idx]
            v0, v1 = v_gs[idx - 1], v_gs[idx]
            frac = (v_gs_max - v0) / max(v1 - v0, 1e-15)
            t_cross = t0 + frac * (t1 - t0)
            keep = idx + 1
            time, v_gs, v_ds, q_g = time[:keep], v_gs[:keep], v_ds[:keep], q_g[:keep]
            q_g[-1] = i_gate * t_cross
            v_gs[-1] = v_gs_max

        return q_g, v_gs, v_ds
    finally:
        if temp_lib.exists():
            temp_lib.unlink()
