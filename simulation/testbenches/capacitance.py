"""Capacitance testbench: Ciss/Coss/Crss vs Vds via small-signal AC analysis.

Method (per Vds bias point, f = 1 MHz):
  Sim A: 1 A AC current injected into the gate (drain AC-grounded by its ideal
         DC bias source). |Vg| gives Ciss = Cgs + Cgd = 1/(w*|Vg|).
  Sim B: 1 A AC current injected into the drain (biased through a 1 H
         inductor: short at DC, open at 1 MHz; gate floats on a 100 MOhm
         bleed). |Vd| gives Coss = Cgd + Cds = 1/(w*|Vd|) and the gate/drain
         voltage ratio r = |Vg|/|Vd| = Cgd/(Cgd + Cgs) yields Crss = r*Ciss.
"""

import numpy as np
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

from common.types import CurveData
from parser.preprocessor import normalize_pspice_netlist
from simulation.engine import SpiceEngine
from simulation.testbenches.static_sweep import (
    normalized_lib_copy, discard_normalized_lib,
)

F_MEAS_DEFAULT = 1e6    # measurement frequency [Hz]
BIAS_INDUCTOR_H = 1.0   # DC bias feed: ~open at 1 MHz for any realistic Coss
GATE_BLEED_OHM = 100e6  # defines gate DC potential without loading 1 MHz


class CapacitanceTestbench:
    """Extracts Ciss, Coss, Crss vs Vds from the vendor subcircuit."""

    def __init__(self, engine: SpiceEngine | None = None,
                 f_meas: float | None = None,
                 ac_retries: int = 3):
        """Args:
            engine: Shared simulator wrapper; its temperature setting is used.
            f_meas: Measurement frequency [Hz]. Defaults to the 1 MHz industry
                convention. Vendor subcircuits emulate capacitances only in a
                fitted frequency band — if the requested f_meas deviates far
                from that band the extracted values lose meaning, so a warning
                is emitted and 1 MHz used when the value looks unsafe.
        """
        self.engine = engine or SpiceEngine()
        if f_meas is None or f_meas <= 0:
            if f_meas is not None:  # explicitly bad value in config
                print(f"[capacitance] WARNING: invalid f_meas={f_meas!r}; "
                      f"falling back to {F_MEAS_DEFAULT:.1e} Hz.")
            f_meas = F_MEAS_DEFAULT
        elif not (1e5 <= f_meas <= 1e8):
            print(f"[capacitance] WARNING: f_meas={f_meas:.1e} Hz is outside "
                  f"the band vendor macromodels are typically calibrated for "
                  f"(100 kHz - 100 MHz); results may not match datasheet "
                  f"capacitances. Continuing anyway.")
        self.f_meas = float(f_meas)
        self._ac_retries = ac_retries

    def _ac_probe(self, node: str, circuit_builder) -> complex:
        """Run an AC probe with retry.

        Under heavy sequential use the ngspice shared library occasionally
        fails a run spuriously (state pollution between successive loads);
        rebuilding the circuit and retrying succeeds. Vendor-model-adjacent
        transient failures should warn-and-retry, never abort a sweep.
        """
        last_exc: Exception | None = None
        for attempt in range(self._ac_retries):
            circuit = circuit_builder(attempt)
            try:
                analysis = self.engine.run_ac_sweep(
                    circuit,
                    start_frequency=self.f_meas@u_Hz,
                    stop_frequency=self.f_meas@u_Hz,
                    number_of_points=1,
                    variation='lin',
                )
                break
            except Exception as exc:
                last_exc = exc
        else:
            raise RuntimeError(
                f"AC probe failed after {self._ac_retries} attempts "
                f"(last circuit {circuit.title!r}): {last_exc}"
            )
        try:
            return complex(np.asarray(analysis.nodes[node])[0])
        except (AttributeError, KeyError):
            return complex(np.asarray(analysis[node])[0])

    def run(self, lib_path, subcircuit_name: str = "",
            vds_min: float = 0.1, vds_max: float = 60.0,
            points: int = 25,
            pin_roles: list[str] | None = None) -> list[CurveData]:
        from pathlib import Path as _Path
        lib_path = _Path(lib_path)
        # Connections must be built in POSITIONAL role order (the order the
        # vendor subckt declares its ports), mapping each role to the external
        # node. For EPC roles [gate,drain,source] -> ['g','d','0'].
        roles = pin_roles or ["drain", "gate", "source"]
        node_by_role = {"drain": "d", "gate": "g", "source": "0"}
        xconn = [node_by_role.get(r, r) for r in roles]

        vds_sweep = np.logspace(np.log10(vds_min), np.log10(vds_max), points)
        ciss = np.zeros(points)
        coss = np.zeros(points)
        crss = np.zeros(points)

        try:
            for i, vds in enumerate(vds_sweep):
                # A FRESH normalized copy per bias point: after many successive
                # ngspice loads of the same include path, the shared library
                # intermittently fails runs; a new file path avoids it.
                temp_lib = normalized_lib_copy(lib_path)

                def mk_a(attempt):
                    ca = Circuit(f"Ciss_Vds_{vds:.3g}_{attempt}")
                    ca.include(str(temp_lib.resolve()))
                    ca.X("M1", subcircuit_name, *xconn)
                    ca.V("ds", xconn[1], ca.gnd, float(vds)@u_V)  # DC bias + AC ground
                    ca.R("bleed", xconn[0], ca.gnd, GATE_BLEED_OHM@u_Ohm)
                    ca.I("drive", xconn[0], ca.gnd, "DC 0 AC 1")  # 1 A probe
                    return ca
                vg = abs(self._ac_probe(xconn[0], mk_a))
                ciss[i] = 1.0 / (2 * np.pi * self.f_meas * max(vg, 1e-15))

                # --- Sim B: drain drive -> Coss, Crss ---
                def mk_b(attempt):
                    cb = Circuit(f"Coss_Vds_{vds:.3g}_{attempt}")
                    cb.include(str(temp_lib.resolve()))
                    cb.X("M1", subcircuit_name, *xconn)
                    cb.V("bus", "vbus", cb.gnd, float(vds)@u_V)
                    cb.L("bias", "vbus", xconn[1],
                         BIAS_INDUCTOR_H@u_H)  # DC feed, AC open
                    cb.R("bleed", xconn[0], cb.gnd, GATE_BLEED_OHM@u_Ohm)
                    cb.I("drive", xconn[1], cb.gnd, "DC 0 AC 1")  # 1 A probe
                    return cb
                vd = abs(self._ac_probe(xconn[1], mk_b))
                vg_b = abs(self._ac_probe(xconn[0], mk_b))
                coss[i] = 1.0 / (2 * np.pi * self.f_meas * max(vd, 1e-15))
                crss[i] = (vg_b / max(vd, 1e-15)) * ciss[i]
                discard_normalized_lib(temp_lib)

            # Sanity guard: avalanche/breakdown makes the AC probe collapse and
            # capacitances explode past physical bounds near the sweep ceiling.
            for name, arr in (("Ciss", ciss), ("Coss", coss), ("Crss", crss)):
                if arr[-1] > 100 * max(arr[0], 1e-15):
                    print(f"[capacitance] WARNING: {name} grows >100x across "
                          f"the sweep (last point {arr[-1]:.3e} F) - probable "
                          f"breakdown region included in Vds_cap_max; top-end "
                          f"points are likely unphysical.")
        except Exception:
            discard_normalized_lib(temp_lib)
            raise

        family_note = f"measured @ {self.f_meas:.1e} Hz"
        return [
            CurveData(
                name="Ciss", x_name="Drain-Source Voltage (Vds)", x_unit="V",
                y_name="Capacitance", y_unit="F",
                x_data=vds_sweep.tolist(), y_data=ciss.tolist(),
                metadata={"note": family_note},
            ),
            CurveData(
                name="Coss", x_name="Drain-Source Voltage (Vds)", x_unit="V",
                y_name="Capacitance", y_unit="F",
                x_data=vds_sweep.tolist(), y_data=coss.tolist(),
                metadata={"note": family_note},
            ),
            CurveData(
                name="Crss", x_name="Drain-Source Voltage (Vds)", x_unit="V",
                y_name="Capacitance", y_unit="F",
                x_data=vds_sweep.tolist(), y_data=crss.tolist(),
                metadata={"note": family_note},
            ),
        ]
