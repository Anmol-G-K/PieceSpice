"""Probe: AC works at gate drive; the failing case was drain-drive sim B
(Ciss_Vds_1.51 = sim A actually). Try sim B topology (inductor feed):"""
import sys
sys.path.insert(0, '.')
from pathlib import Path
from PySpice.Spice.Netlist import Circuit
from simulation.testbenches.static_sweep import normalized_lib_copy

lib = normalized_lib_copy(Path('SpiceModels/EPC2216_V0_PSPICE.net'))
try:
    # Sim B: drain driven through 1H inductor
    cb = Circuit("epc_simB")
    cb.include(str(lib.resolve()))
    cb.X("M1", "EPC2216", "g", "d", "0")
    cb.V("bus", "vbus", cb.gnd, 1.51)
    cb.L("bias", "vbus", "d", 1.0)
    cb.R("bleed", "g", cb.gnd, 100e6)
    cb.I("drive", "d", cb.gnd, "DC 0 AC 1")
    try:
        op = cb.simulator(temperature=25, nominal_temperature=25).ac(
            start_frequency=1e6, stop_frequency=1e6, number_of_points=1,
            variation='lin')
        print("PROBE SimB AC: OK")
    except Exception as e:
        print("PROBE SimB AC: FAIL:", str(e)[:80])

    # Sim A at the same bias but with drain DC-biased directly (no inductor)
    ca = Circuit("epc_simA")
    ca.include(str(lib.resolve()))
    ca.X("M1", "EPC2216", "g", "d", "0")
    ca.V("ds", "d", ca.gnd, 1.51)
    ca.R("bleed", "g", ca.gnd, 100e6)
    ca.I("drive", "g", ca.gnd, "DC 0 AC 1")
    try:
        op = ca.simulator(temperature=25, nominal_temperature=25).ac(
            start_frequency=1e6, stop_frequency=1e6, number_of_points=1,
            variation='lin')
        print("PROBE SimA AC: OK")
    except Exception as e:
        print("PROBE SimA AC: FAIL:", str(e)[:80])
finally:
    lib.unlink()
