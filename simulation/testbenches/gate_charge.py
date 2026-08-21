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
    i_gate: float = 5e-3,
    v_gs_max: float = 10.0,
    t_end: float = 40e-6,
    step_time: float = 50e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the gate-charge transient.

    Returns:
        q_g: Gate charge array [C]
        v_gs: Gate-source voltage array [V]
        i_d: Drain current array [A]
    """
    raw = lib_path.read_text(encoding='utf-8', errors='ignore')
    temp_lib = lib_path.parent / f"_temp_norm_{lib_path.name}"
    temp_lib.write_text(normalize_pspice_netlist(raw))

    try:
        circuit = Circuit("Gate_Charge")
        circuit.include(str(temp_lib.resolve()))
        circuit.X("M1", subcircuit_name, "d", "g", circuit.gnd)
        circuit.V("bus", "d", circuit.gnd, float(v_bus)@u_V)
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
        i_d = -np.array(analysis.branches['vbus'])
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
            time, v_gs, i_d, q_g = time[:keep], v_gs[:keep], i_d[:keep], q_g[:keep]
            q_g[-1] = i_gate * t_cross
            v_gs[-1] = v_gs_max

        return q_g, v_gs, i_d
    finally:
        if temp_lib.exists():
            temp_lib.unlink()
