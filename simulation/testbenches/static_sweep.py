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
from simulation.testbenches.xinst import attach_x_instance
from common.roles import x_connections as _shared_x_connections
from parser.preprocessor import normalize_pspice_netlist


# Normalized copies are runtime artifacts, not vendor inputs: they go under
# reports/ (already git-ignored territory for generated output) so
# SpiceModels/ stays a pristine read-only input tree (Major fix M5).
NORMALIZED_DIR_NAME = "normalized"

# Module-level per-process cache: one normalized copy per source lib per run
# instead of one mkstemp per bias point (Major fix M6). Keyed by resolved
# source path; cleared via discard_normalized_lib(path) or process exit.
_COPY_CACHE: dict = {}


def _normalized_out_dir(lib_path: Path) -> Path:
    """reports/<vendor-or-model-stem>/normalized/ next to nothing vendor-side."""
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "reports" / NORMALIZED_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def normalized_lib_copy(lib_path: Path, *, fresh: bool = False) -> Path:
    """Return a dialect-normalized copy of the library for simulation.

    The original vendor file is never touched. Copies live under
    ``reports/normalized/`` (runtime artifacts, not source inputs).

    Args:
        lib_path: Vendor model file.
        fresh: Force a brand-new copy even when cached. Use this only when
            ngspice's include-path caching interferes with successive loads
            of the same path (observed on some builds after ~40 loads).

    Returns:
        Path to the normalized copy. Cleanup is optional; the cache holds one
        copy per source per process, and :func:`discard_normalized_lib`
        removes it explicitly when the caller wants deterministic cleanup.
    """
    key = str(Path(lib_path).resolve())
    if not fresh and key in _COPY_CACHE:
        cached = _COPY_CACHE[key]
        if cached.exists():
            return cached

    normalized = normalize_pspice_netlist(
        lib_path.read_text(encoding='utf-8', errors='ignore')
    )
    out_dir = _normalized_out_dir(lib_path)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{lib_path.stem}_norm{os.getpid()}_",
        suffix=lib_path.suffix or ".net",
        dir=str(out_dir),
    )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(normalized)

    path = Path(tmp_name)
    if not fresh:
        _COPY_CACHE[key] = path
    return path


def discard_normalized_lib(temp_lib: Path) -> None:
    """Remove a normalized copy (idempotent) and drop its cache entry."""
    if os.environ.get("PIECESPICE_KEEP_NORM") == "1":
        print(f"[normalizer] keeping normalized copy: {temp_lib}")
        return
    try:
        temp_lib.unlink()
    except FileNotFoundError:
        pass
    _COPY_CACHE.pop(str(temp_lib.resolve()), None)


def discard_normalized_lib(temp_lib: Path) -> None:
    """Delete a normalized copy if it still exists (idempotent).

    Note: transient copies are removed after each run; set
    ``PIECESPICE_KEEP_NORM=1`` to keep them for inspection.
    """
    if os.environ.get("PIECESPICE_KEEP_NORM") == "1":
        print(f"[normalizer] keeping normalized copy: {temp_lib}")
        return
    try:
        temp_lib.unlink()
    except FileNotFoundError:
        pass


# Backwards-compatible alias (older call sites imported the private name).
_normalized_lib_file = normalized_lib_copy


class StaticSweepTestbench:
    """Extracts output (Id-Vds), transfer (Id-Vgs), on-resistance and body-diode curves."""

    @staticmethod
    def _x_connections(pin_roles: List[str]) -> List[str]:
        """Map resolved pin roles to external node names, positionally.

        Delegates to the shared registry in common.roles (single source of
        truth for role->node mapping across all testbenches).
        """
        return _shared_x_connections(pin_roles)

    def __init__(self, engine: Optional[SpiceEngine] = None):
        self.engine = engine or SpiceEngine()

    def run_output_characteristics(
        self,
        lib_path: Path,
        subcircuit_name: str,
        vgs_list: Optional[List[float]] = None,
        vds_max: float = 10.0,
        vds_points: int = 101,
        pin_roles: Optional[List[str]] = None,
    ) -> List[CurveData]:
        """Extract output characteristics family: Id vs Vds for each Vgs."""
        if vgs_list is None:
            vgs_list = [3.0, 4.5, 6.0, 10.0]
        roles = pin_roles or ["drain", "gate", "source"]
        curves: List[CurveData] = []
        temp_lib = normalized_lib_copy(lib_path)

        try:
            for vgs in vgs_list:
                circuit = Circuit(f"Output_Char_Vgs_{vgs}")
                circuit.include(str(temp_lib.resolve()))
                attach_x_instance(circuit, "M1", subcircuit_name, roles, temperature=self.engine.temperature)
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
        pin_roles: Optional[List[str]] = None,
    ) -> CurveData:
        """Extract transfer characteristic: Id vs Vgs at fixed Vds."""
        roles = pin_roles or ["drain", "gate", "source"]
        temp_lib = normalized_lib_copy(lib_path)

        try:
            circuit = Circuit("Transfer_Characteristics")
            circuit.include(str(temp_lib.resolve()))
            attach_x_instance(circuit, "M1", subcircuit_name, roles, temperature=self.engine.temperature)
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
        pin_roles: Optional[List[str]] = None,
    ) -> CurveData:
        """Extract Rds(on) = Vds/Id vs Id in the ohmic region at fixed Vgs.

        Args:
            min_current: Only report points where the device conducts more than
                this current [A] — below it the Vds/Id ratio is dominated by
                leakage/numerical noise, not channel resistance.
        """
        temp_lib = normalized_lib_copy(lib_path)
        roles = pin_roles or ["drain", "gate", "source"]

        try:
            circuit = Circuit("Rds_on")
            circuit.include(str(temp_lib.resolve()))
            attach_x_instance(circuit, "M1", subcircuit_name, roles, temperature=self.engine.temperature)
            circuit.V("ds", "d", circuit.gnd, 0@u_V)
            circuit.V("gs", "g", circuit.gnd, vgs@u_V)

            step = vds_max / max(1, vds_points - 1)
            analysis = self.engine.run_dc_sweep(circuit, "Vds", 0, vds_max, step)

            vds_arr = np.array(analysis.sweep)
            id_arr = -np.array(analysis.Vds)  # drain current into device
            self._check_sign(vds_arr, id_arr, f"rds_on @Vgs={vgs}")

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
        pin_roles: Optional[List[str]] = None,
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
        # Drain-gate-tied topology: build positional connections from roles,
        # tying the drain and gate role positions to the same node 'gd'.
        roles = pin_roles or ["drain", "gate", "source"]
        core_roles = [r for r in roles if r.lower() != "tj"]
        has_tj = len(core_roles) != len(roles)
        xargs = ["gd" if r != "source" else "0" for r in core_roles]

        try:
            circuit = Circuit("Threshold_Voltage")
            circuit.include(str(temp_lib.resolve()))
            # Drain and gate tied together -> self-bias at Vds = Vgs
            if has_tj:
                circuit.V("tjsrc_M1", "tj_M1", circuit.gnd,
                          float(self.engine.temperature)@u_V)
                xargs = xargs + ["tj_M1"]
            circuit.X("M1", subcircuit_name, *xargs)
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
        pin_roles: Optional[List[str]] = None,
    ) -> CurveData:
        """Extract body-diode / reverse-conduction characteristic Isd vs Vsd.

        For Si MOSFETs this is the body-diode forward curve; for GaN FETs
        (e.g. EPC2216, no body diode) the same topology measures reverse
        channel conduction. Gate is tied to source (channel off for Si; for
        GaN the device conducts in the third quadrant through the channel).
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
            pin_roles: Resolved pin roles (see resolve_pin_connections).
        """
        temp_lib = normalized_lib_copy(lib_path)
        # Reverse-conduction topology: gate tied to source, drain node driven.
        # Connections are POSITIONAL (roles order = subckt port order): the
        # gate and source role positions both tie to node 'src'.
        roles = pin_roles or ["drain", "gate", "source"]
        core_roles = [r for r in roles if r.lower() != "tj"]
        has_tj = len(core_roles) != len(roles)
        xargs = ["drn" if r == "drain" else "src" for r in core_roles]
        src_node = "src"

        try:
            circuit = Circuit("Body_Diode")
            circuit.include(str(temp_lib.resolve()))
            if has_tj:
                circuit.V("tjsrc_M1", "tj_M1", circuit.gnd,
                          float(self.engine.temperature)@u_V)
                xargs = xargs + ["tj_M1"]
            circuit.X("M1", subcircuit_name, *xargs)
            circuit.V("sd", src_node, "drn", 0@u_V)
            # Absolute potential anchor: without it the terminal island only
            # sees leakage paths and the OP diverges (10 MOhm shunt ~ 1e-7 A).
            bleed_ohm = 10e6
            circuit.R("bleed", src_node, circuit.gnd, bleed_ohm@u_Ohm)

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
                metadata={"condition": "gate_off (vgs=0)"},
            )
        finally:
            discard_normalized_lib(temp_lib)

    @staticmethod
    def _check_sign(x_arr: np.ndarray, y_arr: np.ndarray, label: str,
                    expect_positive: bool = True) -> None:
        """Assert extracted currents follow the expected polarity.

        The testbenches negate ngspice source currents so positive values mean
        device conduction. A vendor model with different internal orientation
        flips every sample silently - that corrupts downstream fits, so this
        check FAILS instead of warning.

        Raises:
            AssertionError: when conduction-current samples are predominantly
                of the wrong sign over the upper half of the sweep.
        """
        y = np.asarray(y_arr, dtype=float)
        if len(y) == 0 or np.all(np.isnan(y)):
            return
        upper = y[len(y)//2:]
        frac_wrong = float(np.mean(upper < 0)) if expect_positive else \
            float(np.mean(upper > 0))
        assert frac_wrong < 0.5, (
            f"{label}: {(frac_wrong*100):.0f}% of upper-sweep current samples "
            f"have the wrong polarity (expected "
            f"{'positive' if expect_positive else 'negative'}). The sign "
            f"convention does not match this vendor model's topology - do not "
            f"use these curves."
        )
        # Legacy soft warning for mildly negative tails (kept for diagnostics)
        if expect_positive and np.nanmin(y) < 0:
            print(f"[static_sweep] NOTE: {label} contains negative tail "
                  f"samples (expected for leakage-limited sweeps).")
