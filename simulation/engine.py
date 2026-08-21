"""NgSpice execution engine using PySpice with error handling."""

from pathlib import Path
from typing import Optional
from PySpice.Spice.Netlist import Circuit


class SpiceEngine:
    """Robust circuit simulator execution wrapper."""

    def __init__(self, temperature: float = 25.0):
        self.temperature = temperature

    def create_circuit(self, title: str) -> Circuit:
        return Circuit(title)

    def run_dc_sweep(
        self,
        circuit: Circuit,
        sweep_source: str,
        start: float,
        stop: float,
        step: float
    ):
        """Execute DC operating point sweep."""
        simulator = circuit.simulator(
            temperature=self.temperature,
            nominal_temperature=self.temperature
        )
        return simulator.dc(**{sweep_source: slice(start, stop, step)})

    def run_transient(
        self,
        circuit: Circuit,
        step_time: float,
        end_time: float,
        start_time: float = 0.0
    ):
        """Execute time-domain transient analysis."""
        simulator = circuit.simulator(
            temperature=self.temperature,
            nominal_temperature=self.temperature
        )
        return simulator.transient(step_time=step_time, end_time=end_time, start_time=start_time)
