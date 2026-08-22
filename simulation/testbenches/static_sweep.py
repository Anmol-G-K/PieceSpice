"""Static DC sweep testbench for extracting standard datasheet curves."""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from common.types import CurveData
from simulation.engine import SpiceEngine
from parser.preprocessor import normalize_pspice_netlist


def _normalized_lib_file(lib_path: Path) -> Path:
    """Write a dialect-normalized copy of the library next to the source."""
    normalized = normalize_pspice_netlist(
        lib_path.read_text(encoding='utf-8', errors='ignore')
    )
    temp_lib = lib_path.parent / f"_temp_norm_{lib_path.name}"
    temp_lib.write_text(normalized)
    return temp_lib


class StaticSweepTestbench:
    """Extracts output (Id-Vds), transfer (Id-Vgs), on-resistance and body-diode curves."""

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
        temp_lib = _normalized_lib_file(lib_path)

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
        temp_lib = _normalized_lib_file(lib_path)

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

    def run_rds_on(
        self,
        lib_path: Path,
        subcircuit_name: str,
        vgs: float = 10.0,
        vds_max: float = 2.5,
        vds_points: int = 51,
        min_current: float = 1.0,
    ) -> CurveData:
        """Extract Rds(on) = Vds/Id vs Id in the ohmic region at fixed Vgs.

        Args:
            min_current: Only report points where the device conducts more than
                this current [A] — below it the Vds/Id ratio is dominated by
                leakage/numerical noise, not channel resistance.
        """
        temp_lib = _normalized_lib_file(lib_path)

        try:
            circuit = Circuit("Rds_on")
            circuit.include(str(temp_lib.resolve()))
            circuit.X("M1", subcircuit_name, "d", "g", circuit.gnd)
            circuit.V("ds", "d", circuit.gnd, 0@u_V)
            circuit.V("gs", "g", circuit.gnd, vgs@u_V)

            step = vds_max / max(1, vds_points - 1)
            analysis = self.engine.run_dc_sweep(circuit, "Vds", 0, vds_max, step)

            vds_arr = np.array(analysis.sweep)
            id_arr = np.abs(np.array(analysis.Vds))

            # Only meaningful where the device actually conducts
            mask = id_arr > min_current
            rds_mohm = vds_arr[mask] / np.maximum(id_arr[mask], 1e-12) * 1e3

            return CurveData(
                name=f"On-State Resistance (Vgs={vgs}V)",
                x_name="Drain Current (Id)",
                x_unit="A",
                y_name="On-Resistance (Rds(on))",
                y_unit="mOhm",
                x_data=id_arr[mask].tolist(),
                y_data=rds_mohm.tolist(),
                family_param="Vgs",
                family_value=float(vgs),
            )
        finally:
            if temp_lib.exists():
                temp_lib.unlink()

    def run_threshold_voltage(
        self,
        lib_path: Path,
        subcircuit_name: str,
        i_target: float = 250e-6,
        v_max: float = 6.0,
        points: int = 3001,
    ) -> Tuple[CurveData, float]:
        """Extract Vgs(th) per the datasheet/JEDEC constant-current method.

        Gate is shorted to drain so Vds = Vgs; the sweep is interpolated at
        i_target (250 uA is the IR/Infineon datasheet convention). Note: this
        returns whatever the vendor model encodes - a LEVEL=1 macromodel has
        a hard threshold (no subthreshold conduction), so its Vth can differ
        from the real device's datasheet-typical value.
        """
        temp_lib = _normalized_lib_file(lib_path)

        try:
            circuit = Circuit("Threshold_Voltage")
            circuit.include(str(temp_lib.resolve()))
            # Drain and gate tied together -> self-bias at Vds = Vgs
            circuit.X("M1", subcircuit_name, "gd", "gd", circuit.gnd)
            circuit.V("sweep", "gd", circuit.gnd, 0@u_V)

            step = v_max / max(1, points - 1)
            analysis = self.engine.run_dc_sweep(circuit, "Vsweep", 0, v_max, step)

            v_arr = np.array(analysis.sweep)
            id_arr = -np.array(analysis.Vsweep)
            v_th = float(np.interp(i_target, id_arr, v_arr))

            curve = CurveData(
                name=(
                    f"Threshold Sweep (Vds=Vgs) "
                    f"- Vth={v_th:.3f}V @ Id={i_target * 1e6:.0f}uA"
                ),
                x_name="Gate-Source Voltage (Vgs)",
                x_unit="V",
                y_name="Drain Current (Id)",
                y_unit="A",
                x_data=v_arr.tolist(),
                y_data=id_arr.tolist(),
            )
            return curve, v_th
        finally:
            if temp_lib.exists():
                temp_lib.unlink()

    def run_body_diode(
        self,
        lib_path: Path,
        subcircuit_name: str,
        vsd_max: float = 1.0,
        points: int = 51,
    ) -> CurveData:
        """Extract body-diode forward characteristic Isd vs Vsd (gate off).

        Voltage-driven with a conservative cap (~1 V): beyond that the diode
        exponential saturates Newton iterations and the DC sweep diverges.
        """
        temp_lib = _normalized_lib_file(lib_path)

        try:
            circuit = Circuit("Body_Diode")
            circuit.include(str(temp_lib.resolve()))
            # Gate tied to source (channel off); + terminal at source so the
            # internal body diode (anode=source/body, cathode=drain) conducts.
            circuit.X("M1", subcircuit_name, "drn", "src", "src")
            circuit.V("sd", "src", "drn", 0@u_V)
            # Absolute potential anchor: without it the terminal island only
            # sees leakage paths and the OP diverges (10 MOhm shunt ~ 1e-7 A).
            circuit.R("bleed", "src", circuit.gnd, 10e6@u_Ohm)

            step = vsd_max / max(1, points - 1)
            analysis = self.engine.run_dc_sweep(circuit, "Vsd", 0, vsd_max, step)

            vsd_arr = np.array(analysis.sweep)
            isd_arr = -np.array(analysis.Vsd)  # conduction current src->drain

            return CurveData(
                name="Body Diode Forward Characteristic",
                x_name="Source-Drain Voltage (Vsd)",
                x_unit="V",
                y_name="Source Current (Isd)",
                y_unit="A",
                x_data=vsd_arr.tolist(),
                y_data=isd_arr.tolist(),
            )
        finally:
            if temp_lib.exists():
                temp_lib.unlink()
