"""Static DC sweep testbench for extracting standard datasheet curves."""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from piecespice.common.types import CurveData
from piecespice.simulation.engine import SpiceEngine
from piecespice.parser.preprocessor import normalize_pspice_netlist


class StaticSweepTestbench:
    """Extracts output (Id-Vds), transfer (Id-Vgs), and on-resistance curves."""

    def __init__(self, engine: Optional[SpiceEngine] = None):
        self.engine = engine or SpiceEngine()

    def run_output_characteristics(
        self,
        lib_path: Path,
        subcircuit_name: str,
        vgs_list: List[float] = [3.0, 4.5, 6.0, 10.0],
        vds_max: float = 10.0,
        vds_points: int = 101,
    ) -> List[CurveData]:
        """Extract output characteristics family: Id vs Vds for each Vgs."""
        curves: List[CurveData] = []
        raw_text = lib_path.read_text(encoding='utf-8', errors='ignore')
        normalized = normalize_pspice_netlist(raw_text)

        # Temporary normalized file if needed or direct include
        temp_lib = lib_path.parent / f"_temp_norm_{lib_path.name}"
        temp_lib.write_text(normalized)

        try:
            for vgs in vgs_list:
                circuit = Circuit(f"Output_Char_Vgs_{vgs}")
                circuit.include(str(temp_lib.resolve()))
                # Standard pin order: Drain=1, Gate=2, Source=3
                circuit.X("M1", subcircuit_name, "d", "g", circuit.gnd)
                circuit.V("ds", "d", circuit.gnd, 0@u_V)
                circuit.V("gs", "g", circuit.gnd, vgs@u_V)

                step = vds_max / max(1, vds_points - 1)
                analysis = self.engine.run_dc_sweep(circuit, "Vds", 0, vds_max, step)

                vds_arr = np.array(analysis.sweep)
                id_arr = -np.array(analysis.Vds)  # Drain current into device

                curves.append(
                    CurveData(
                        name="Output Characteristic",
                        x_name="Drain-Source Voltage (Vds)",
                        x_unit="V",
                        y_name="Drain Current (Id)",
                        y_unit="A",
                        x_data=vds_arr.tolist(),
                        y_data=id_arr.tolist(),
                        family_param="Vgs",
                        family_value=float(vgs),
                    )
                )
        finally:
            if temp_lib.exists():
                temp_lib.unlink()

        return curves

    def run_transfer_characteristics(
        self,
        lib_path: Path,
        subcircuit_name: str,
        vds_fixed: float = 10.0,
        vgs_max: float = 15.0,
        vgs_points: int = 151,
    ) -> CurveData:
        """Extract transfer characteristic: Id vs Vgs at fixed Vds."""
        raw_text = lib_path.read_text(encoding='utf-8', errors='ignore')
        normalized = normalize_pspice_netlist(raw_text)
        temp_lib = lib_path.parent / f"_temp_norm_{lib_path.name}"
        temp_lib.write_text(normalized)

        try:
            circuit = Circuit("Transfer_Characteristics")
            circuit.include(str(temp_lib.resolve()))
            circuit.X("M1", subcircuit_name, "d", "g", circuit.gnd)
            circuit.V("ds", "d", circuit.gnd, vds_fixed@u_V)
            circuit.V("gs", "g", circuit.gnd, 0@u_V)

            step = vgs_max / max(1, vgs_points - 1)
            analysis = self.engine.run_dc_sweep(circuit, "Vgs", 0, vgs_max, step)

            vgs_arr = np.array(analysis.sweep)
            id_arr = -np.array(analysis.Vds)

            return CurveData(
                name=f"Transfer Characteristic (Vds={vds_fixed}V)",
                x_name="Gate-Source Voltage (Vgs)",
                x_unit="V",
                y_name="Drain Current (Id)",
                y_unit="A",
                x_data=vgs_arr.tolist(),
                y_data=id_arr.tolist(),
                family_param="Vds",
                family_value=float(vds_fixed),
            )
        finally:
            if temp_lib.exists():
                temp_lib.unlink()
