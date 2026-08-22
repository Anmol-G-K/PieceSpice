"""Static DC sweep testbench for extracting standard datasheet curves."""

import os
import tempfile
from pathlib import Path
from typing import List, Optional
import numpy as np
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from common.types import CurveData
from simulation.engine import SpiceEngine
from parser.preprocessor import normalize_pspice_netlist


def normalized_lib_copy(lib_path: Path) -> Path:
    """Write a dialect-normalized copy of the library to a unique temp file.

    A process-unique name (PID + counter) prevents two concurrent runs on the
    same vendor model from clobbering each other's file while ngspice has it
    open. The caller owns cleanup via :func:`discard_normalized_lib`.
    """
    normalized = normalize_pspice_netlist(
        lib_path.read_text(encoding='utf-8', errors='ignore')
    )
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"_norm_{os.getpid()}_", suffix=f"_{lib_path.name}",
        dir=str(lib_path.parent),
    )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(normalized)
    return Path(tmp_name)


def discard_normalized_lib(temp_lib: Path) -> None:
    """Delete a temp lib if it still exists (idempotent)."""
    try:
        temp_lib.unlink()
    except FileNotFoundError:
        pass


# Backwards-compatible alias (older call sites imported the private name).
_normalized_lib_file = normalized_lib_copy


class StaticSweepTestbench:
    """Extracts output (Id-Vds), transfer (Id-Vgs), on-resistance and body-diode curves."""

    def __init__(self, engine: Optional[SpiceEngine] = None):
        self.engine = engine or SpiceEngine()

    def run_output_characteristics(
        self,
        lib_path: Path,
        subcircuit_name: str,
        vgs_list: Optional[List[float]] = None,
        vds_max: float = 10.0,
        vds_points: int = 101,
    ) -> List[CurveData]:
        """Extract output characteristics family: Id vs Vds for each Vgs."""
        if vgs_list is None:
            vgs_list = [3.0, 4.5, 6.0, 10.0]
        curves: List[CurveData] = []
        temp_lib = normalized_lib_copy(lib_path)

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
                self._check_sign(vds_arr, id_arr, f"output char @Vgs={vgs}")

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
            discard_normalized_lib(temp_lib)

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
        temp_lib = normalized_lib_copy(lib_path)

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
            self._check_sign(vgs_arr, id_arr, "transfer char")

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
            discard_normalized_lib(temp_lib)

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
        temp_lib = normalized_lib_copy(lib_path)

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
            discard_normalized_lib(temp_lib)

    def run_threshold_voltage(
        self,
        lib_path: Path,
        subcircuit_name: str,
        i_target: float = 250e-6,
        v_max: float = 6.0,
        points: int = 3001,
    ) -> tuple[CurveData, float]:
        """Extract Vgs(th) per the datasheet/JEDEC constant-current method.

        Gate is shorted to drain so Vds = Vgs; the sweep is interpolated at
        i_target (250 uA is the IR/Infineon datasheet convention).

        Raises:
            ValueError: If the swept current never reaches ``i_target`` — a
                silent clamp to Vth≈0 would otherwise poison downstream fits.
        Note:
            This returns whatever the vendor model encodes - a LEVEL=1
            macromodel has a hard threshold (no subthreshold conduction), so
            its Vth can differ from the real device's datasheet-typical value
            even when extraction succeeds.
        """
        temp_lib = normalized_lib_copy(lib_path)

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
            self._check_sign(v_arr, id_arr, "threshold sweep")

            if float(id_arr.max()) < i_target:
                raise ValueError(
                    f"Threshold extraction failed: swept current peaks at "
                    f"{id_arr.max():.3g} A over Vgs in [0, {v_max}], never "
                    f"reaching the target {i_target:.3g} A. The vendor model "
                    f"does not conduct enough at this sweep ceiling; raise "
                    f"v_max or verify the subcircuit."
                )

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
                family_param="Id_target",
                family_value=float(i_target),
            )
            return curve, v_th
        finally:
            discard_normalized_lib(temp_lib)

    def run_body_diode(
        self,
        lib_path: Path,
        subcircuit_name: str,
        vsd_max: float = 1.0,
        points: int = 51,
        sweep_step: float | None = None,
    ) -> CurveData:
        """Extract body-diode forward characteristic Isd vs Vsd (gate off).

        Voltage-driven with a conservative cap (~1 V): beyond that the diode
        exponential saturates Newton iterations and the DC sweep diverges.

        The 10 MOhm absolute-potential anchor injects a small parasitic current
        (Vsd / 10 MOhm, ~0.1 uA at 1 V) into every Isd sample. It is subtracted
        analytically before returning, so exported curves are clean; the raw
        convergence aid stays in the netlist.

        Args:
            sweep_step: Optional explicit DC-sweep step size. The ngspice DC
                sweep is step-sensitive near the diode's high-conduction knee
                (e.g. the TI CSD16412Q5A passes 1.0 V/39 pts but fails
                1.0 V/41 pts); if the default step diverges, retry with a
                coarser one.
        """
        temp_lib = normalized_lib_copy(lib_path)

        try:
            circuit = Circuit("Body_Diode")
            circuit.include(str(temp_lib.resolve()))
            # Gate tied to source (channel off); + terminal at source so the
            # internal body diode (anode=source/body, cathode=drain) conducts.
            circuit.X("M1", subcircuit_name, "drn", "src", "src")
            circuit.V("sd", "src", "drn", 0@u_V)
            # Absolute potential anchor: without it the terminal island only
            # sees leakage paths and the OP diverges (10 MOhm shunt ~ 1e-7 A).
            bleed_ohm = 10e6
            circuit.R("bleed", "src", circuit.gnd, bleed_ohm@u_Ohm)

            step = sweep_step if sweep_step else vsd_max / max(1, points - 1)
            try:
                analysis = self.engine.run_dc_sweep(circuit, "Vsd", 0, vsd_max, step)
            except RuntimeError as exc:
                # Step-size-sensitive convergence near the diode knee: retry on
                # a coarse 25-point grid before giving up.
                coarse_step = vsd_max / 24
                print(f"[body_diode] default step diverged ({exc}); "
                      f"retrying with coarse step {coarse_step:.4g} V")
                analysis = self.engine.run_dc_sweep(circuit, "Vsd", 0, vsd_max,
                                                    coarse_step)

            vsd_arr = np.array(analysis.sweep)
            isd_arr = -np.array(analysis.Vsd)  # conduction current src->drain

            # Remove the known parasitic bleed-anchor current so the reported
            # characteristic is the diode alone.
            isd_arr = isd_arr - vsd_arr / bleed_ohm

            return CurveData(
                name="Body Diode Forward Characteristic",
                x_name="Source-Drain Voltage (Vsd)",
                x_unit="V",
                y_name="Source Current (Isd)",
                y_unit="A",
                x_data=vsd_arr.tolist(),
                y_data=isd_arr.tolist(),
                family_param="Vgs",
                family_value=0.0,   # gate off
            )
        finally:
            discard_normalized_lib(temp_lib)

    @staticmethod
    def _check_sign(x_arr: np.ndarray, y_arr: np.ndarray, label: str) -> None:
        """Warn when extracted currents do not follow the expected convention.

        The testbenches negate ngspice source currents assuming positive
        device conduction. A vendor model with different internal source
        orientation would flip every curve's sign silently; this makes that
        visible instead.
        """
        if len(y_arr) == 0 or np.all(np.isnan(y_arr)):
            return
        # Expected: conduction grows monotonically-ish with drive; flag only
        # when the trace trends clearly negative.
        span = float(np.nanmax(y_arr) - np.nanmin(y_arr))
        if span > 0 and np.nanmean(y_arr[-max(1, len(y_arr)//5):]) < 0 and np.nanmin(y_arr) < 0:
            print(
                f"[static_sweep] WARNING: {label} currents are predominantly "
                f"negative - the sign convention (-V_source) may not match "
                f"this vendor model's internal topology. Verify curve polarity."
            )
