"""NgSpice execution engine using PySpice with error isolation.

Every analysis call wraps PySpice/ngspice exceptions with the circuit title and
the solver options in effect, so convergence failures are diagnosable instead
of surfacing as bare low-level errors. A hard ngspice shared-library crash
still terminates the host process (PySpice limitation) — that cannot be caught
at this layer; long unattended runs should use process-level isolation.
"""

import logging

from typing import Optional
from PySpice.Spice.Netlist import Circuit

from common.diagnostics import (ConsoleQuiet,
                                get_diagnostics)

_LOGGER = logging.getLogger(__name__)


class SpiceEngine:
    """Robust circuit simulator execution wrapper."""

    def __init__(
        self,
        temperature: float = 25.0,
        reltol: float = 1e-3,
        abstol: float = 1e-12,
        vntol: float = 1e-6,
        gmin: Optional[float] = None,
        itl1: Optional[int] = None,
        itl4: Optional[int] = None,
    ):
        """Solver options (None leaves ngspice defaults untouched).

        Args:
            temperature: Analysis and nominal temperature [degC]. NOTE: vendor
                macromodels are not always temperature-parameterized — if a
                model ignores .TEMP the option is harmless but has no effect.
            reltol: Relative error tolerance.
            abstol: Absolute current tolerance [A].
            vntol: Absolute voltage tolerance [V].
            gmin: Minimum conductance [S] (convergence aid).
            itl1: DC operating-point iteration limit.
            itl4: Transient time-step iteration limit.
        """
        self.temperature = temperature
        self.reltol = reltol
        self.abstol = abstol
        self.vntol = vntol
        self.gmin = gmin
        self.itl1 = itl1
        self.itl4 = itl4

    def _options(self) -> dict:
        opts: dict = {
            "RELTOL": self.reltol,
            "ABSTOL": self.abstol,
            "VNTOL": self.vntol,
        }
        if self.gmin is not None:
            opts["GMIN"] = self.gmin
        if self.itl1 is not None:
            opts["ITL1"] = self.itl1
        if self.itl4 is not None:
            opts["ITL4"] = self.itl4
        return opts

    def _make_simulator(self, circuit: Circuit):
        simulator = circuit.simulator(
            temperature=self.temperature,
            nominal_temperature=self.temperature,
        )
        # .OPTIONS surface; ignore failures from backends that reject keys.
        try:
            for key, value in self._options().items():
                simulator.options(**{key: value})
        except Exception as exc:  # pragma: no cover - backend-dependent
            _LOGGER.warning(f"WARNING: solver options rejected ({exc}); "
                  f"continuing with ngspice defaults.")
        return simulator

    def _collect_diagnostics(self) -> None:
        """Drain ngspice console output into the diagnostics collector."""
        try:
            from PySpice.Spice.NgSpice.Shared import NgSpiceShared
            ngs = NgSpiceShared._instances.get(0)
            if ngs is not None:
                get_diagnostics().drain_shared_buffer(ngs)
        except Exception:  # pragma: no cover - diagnostics must never break runs
            pass

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
        """Execute a DC operating point sweep.

        Runs the native ngspice ``.dc`` first; on failure retries with
        coarser steps, then a slightly shrunk stop value (model clamp
        boundaries at the sweep edge are the usual culprit), and finally
        raises RuntimeError. Each fallback is logged.
        """
        try:
            with ConsoleQuiet():
                simulator = self._make_simulator(circuit)
                self._last_simulator = simulator
                result = simulator.dc(
                    **{sweep_source: slice(start, stop, step)})
            self._collect_diagnostics()
            return result
        except Exception as dc_exc:
            print(f"[engine] .dc failed for {circuit.title!r}; "
                  f"falling back to per-point OP sweep.")
            return self._op_points_sweep(
                circuit, sweep_source, start, stop, step)

    def _op_points_sweep(
        self,
        circuit: Circuit,
        sweep_source: str,
        start: float,
        stop: float,
        step: float
    ):
        """Fallback ladder for native .dc: coarser steps, then edge shrink.

        The last sweep point frequently hits model clamp boundaries (E100
        limit, exp underflow) that the interior tolerates.
        """
        # Fallback strategy: coarser steps, then edge shrink.
        for factor in (4, 8):
            coarse = step * factor
            if coarse >= (stop - start):
                continue
            try:
                sim2 = self._make_simulator(circuit)
                result = sim2.dc(**{sweep_source: slice(start, stop, coarse)})
                _LOGGER.warning(f"fallback succeeded with step x{factor}.")
                return result
            except Exception:
                continue

        for shrink in (0.995, 0.99):
            new_stop = stop * shrink
            if new_stop <= start:
                continue
            try:
                sim3 = self._make_simulator(circuit)
                result = sim3.dc(**{sweep_source: slice(start, new_stop, step)})
                print(f"[engine] fallback succeeded with stop shrunk to "
                      f"{new_stop:.4g} ({shrink:.1%}).")
                return result
            except Exception:
                continue

        raise RuntimeError(
            f"DC sweep of {sweep_source!r} [{start}, {stop}] failed for "
            f"circuit {circuit.title!r} including coarse-step and edge-shrink "
            f"fallbacks."
        )

    def run_transient(
        self,
        circuit: Circuit,
        step_time: float,
        end_time: float,
        start_time: float = 0.0
    ):
        """Execute time-domain transient analysis."""
        try:
            with ConsoleQuiet():
                simulator = self._make_simulator(circuit)
                self._last_simulator = simulator
                result = simulator.transient(
                    step_time=step_time, end_time=end_time,
                    start_time=start_time)
            self._collect_diagnostics()
            return result
        except Exception as exc:
            raise RuntimeError(
                f"Transient [{start_time}, {end_time}], step={step_time} failed "
                f"for circuit {circuit.title!r} (options={self._options()}): {exc}"
            ) from exc

    def run_ac_sweep(self, circuit: Circuit, start_frequency, stop_frequency,
                     number_of_points: int, variation: str = 'lin'):
        """Execute small-signal AC analysis."""
        try:
            with ConsoleQuiet():
                simulator = self._make_simulator(circuit)
                self._last_simulator = simulator
                result = simulator.ac(
                    start_frequency=start_frequency,
                    stop_frequency=stop_frequency,
                    number_of_points=number_of_points,
                    variation=variation)
            self._collect_diagnostics()
            return result
        except Exception as exc:
            raise RuntimeError(
                f"AC sweep [{start_frequency}, {stop_frequency}] failed for "
                f"circuit {circuit.title!r}: {exc}"
            ) from exc
