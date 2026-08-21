"""Capacitance testbench: Ciss/Coss/Crss vs Vds via small-signal AC analysis.

Method (per Vds bias point, f = 1 MHz):
  Sim A: 1 A AC current injected into the gate (drain AC-grounded by its ideal
         DC bias source). |Vg| gives Ciss = Cgs + Cgd = 1/(w*|Vg|).
  Sim B: 1 A AC current injected into the drain (biased through a 1 H
         inductor: short at DC, open at 1 MHz; gate floats on a 100 MOhm
         bleed). |Vd| gives Coss = Cgd + Cds = 1/(w*|Vd|) and the gate/drain
         voltage ratio r = |Vg|/|Vd| = Cgd/(Cgd + Cgs) yields Crss = r*Ciss.
"""

from pathlib import Path
from typing import List

import numpy as np
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

from common.types import CurveData
from parser.preprocessor import normalize_pspice_netlist
from simulation.engine import SpiceEngine

F_MEAS = 1e6          # measurement frequency [Hz]
BIAS_INDUCTOR_H = 1.0  # DC bias feed: ~open at 1 MHz for any realistic Coss
GATE_BLEED_OHM = 100e6  # defines gate DC potential without loading 1 MHz


def _ac_probe(node: str, circuit: Circuit) -> complex:
    simulator = circuit.simulator(temperature=25.0, nominal_temperature=25.0)
    analysis = simulator.ac(
        start_frequency=F_MEAS@u_Hz,
        stop_frequency=F_MEAS@u_Hz,
        number_of_points=1,
        variation='lin',
    )
    try:
        return complex(np.asarray(analysis.nodes[node])[0])
    except (AttributeError, KeyError):
        return complex(np.asarray(analysis[node])[0])


class CapacitanceTestbench:
    """Extracts Ciss, Coss, Crss vs Vds from the vendor subcircuit."""

    def __init__(self, engine: SpiceEngine | None = None):
        self.engine = engine or SpiceEngine()

    def run(self, lib_path: Path, subcircuit_name: str,
            vds_min: float = 0.1, vds_max: float = 60.0,
            points: int = 25) -> List[CurveData]:
        raw = lib_path.read_text(encoding='utf-8', errors='ignore')
        temp_lib = lib_path.parent / f"_temp_norm_{lib_path.name}"
        temp_lib.write_text(normalize_pspice_netlist(raw))

        vds_sweep = np.logspace(np.log10(vds_min), np.log10(vds_max), points)
        ciss = np.zeros(points)
        coss = np.zeros(points)
        crss = np.zeros(points)

        try:
            for i, vds in enumerate(vds_sweep):
                # --- Sim A: gate drive -> Ciss ---
                ca = Circuit(f"Ciss_Vds_{vds:.3g}")
                ca.include(str(temp_lib.resolve()))
                ca.X("M1", subcircuit_name, "d", "g", ca.gnd)
                ca.V("ds", "d", ca.gnd, float(vds)@u_V)   # DC bias + AC ground
                ca.R("bleed", "g", ca.gnd, GATE_BLEED_OHM@u_Ohm)
                ca.I("drive", "g", ca.gnd, "DC 0 AC 1")   # 1 A small-signal probe
                vg = abs(_ac_probe("g", ca))
                ciss[i] = 1.0 / (2 * np.pi * F_MEAS * max(vg, 1e-15))

                # --- Sim B: drain drive -> Coss, Crss ---
                cb = Circuit(f"Coss_Vds_{vds:.3g}")
                cb.include(str(temp_lib.resolve()))
                cb.X("M1", subcircuit_name, "d", "g", cb.gnd)
                cb.V("bus", "vbus", cb.gnd, float(vds)@u_V)
                cb.L("bias", "vbus", "d", BIAS_INDUCTOR_H@u_H)  # DC feed, AC open
                cb.R("bleed", "g", cb.gnd, GATE_BLEED_OHM@u_Ohm)
                cb.I("drive", "d", cb.gnd, "DC 0 AC 1")   # 1 A small-signal probe
                vd = abs(_ac_probe("d", cb))
                vg_b = abs(_ac_probe("g", cb))
                coss[i] = 1.0 / (2 * np.pi * F_MEAS * max(vd, 1e-15))
                crss[i] = (vg_b / max(vd, 1e-15)) * ciss[i]
        finally:
            if temp_lib.exists():
                temp_lib.unlink()

        return [
            CurveData(
                name="Ciss", x_name="Drain-Source Voltage (Vds)", x_unit="V",
                y_name="Capacitance", y_unit="F",
                x_data=vds_sweep.tolist(), y_data=ciss.tolist(),
            ),
            CurveData(
                name="Coss", x_name="Drain-Source Voltage (Vds)", x_unit="V",
                y_name="Capacitance", y_unit="F",
                x_data=vds_sweep.tolist(), y_data=coss.tolist(),
            ),
            CurveData(
                name="Crss", x_name="Drain-Source Voltage (Vds)", x_unit="V",
                y_name="Capacitance", y_unit="F",
                x_data=vds_sweep.tolist(), y_data=crss.tolist(),
            ),
        ]
