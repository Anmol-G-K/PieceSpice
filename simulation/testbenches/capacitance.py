"""Capacitance testbench (Curve 4): Ciss/Coss/Crss vs Vds — LTspice backend.

Small-signal method per Vds bias point at f_meas (default 1 MHz):
  Sim A: 1 A AC current into the gate (drain AC-grounded by its DC bias
         source). Ciss = Cgs + Cgd = 1/(w*|Vg|).
  Sim B: 1 A AC current into the drain through a 1 H bias inductor (DC
         short, AC open); gate on a 100 MOhm bleed.
         Coss = Cgd + Cds = 1/(w*|Vd|), Crss = (|Vg|/|Vd|) * Ciss.

Out-of-process LTspice is stateless per run: the retry machinery from the
ngspice shared-library era has no purpose and is gone.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from common.types import CurveData
from simulation.engine import SpiceEngine, ACDirective

_LOGGER = logging.getLogger(__name__)

F_MEAS_DEFAULT = 1e6    # measurement frequency [Hz]
BIAS_INDUCTOR_H = 1.0   # DC bias feed: ~open at 1 MHz
GATE_BLEED_OHM = 100e6  # gate DC potential without loading 1 MHz


class CapacitanceTestbench:
    """Extracts Ciss, Coss, Crss vs Vds from the vendor subcircuit."""

    def __init__(self, engine: SpiceEngine | None = None,
                 f_meas: float | None = None):
        self.engine = engine or SpiceEngine()
        if f_meas is None or f_meas <= 0:
            if f_meas is not None:
                _LOGGER.warning("invalid f_meas=%r; falling back to %.1e Hz",
                                f_meas, F_MEAS_DEFAULT)
            f_meas = F_MEAS_DEFAULT
        elif not (1e5 <= f_meas <= 1e8):
            _LOGGER.warning(
                "f_meas=%.1e Hz outside the band vendor macromodels are "
                "typically calibrated for (100 kHz-100 MHz); continuing.",
                f_meas)
        self.f_meas = float(f_meas)

    # -- netlist assembly ------------------------------------------------------

    def _ac_netlist(self, lib_path, subcircuit_name, pin_roles, vds: float,
                    variant: str) -> str:
        """variant 'a': gate drive (Ciss). variant 'b': drain drive
        (Coss + Crss ratio)."""
        roles = [r for r in pin_roles if r.lower() != "tj"]
        # yaml pin_order may omit tj even when the subckt declares it -
        # check the header directly.
        if not any(r.lower() == "tj" for r in (pin_roles or [])):
            try:
                from parser.netlist import NetlistParser
                subs = NetlistParser.parse_subcircuits(Path(lib_path))
                info = next((s for s in subs
                             if s.name.lower() == subcircuit_name.lower()),
                            None)
                has_tj = info is not None and any(
                    pn.lower() == "tj" for pn in info.nodes)
            except Exception:
                has_tj = False
        else:
            has_tj = True
        # Role-name based mapping (NOT positional): gate-first vendors
        # like EPC would otherwise swap drain/gate nodes silently.
        canonical = {"drain": "drain", "gate": "gate", "source": "0"}
        conn = {r: canonical.get(r.lower(), r) for r in roles}
        xargs = [conn.get(r, r) for r in roles]
        if has_tj:
            xargs.append("tj_dut")

        lines = [f".include {Path(lib_path).resolve()}"]
        temp = float(getattr(self.engine, "temperature", 25.0))
        if has_tj:
            lines.append(f"Vtj tj_dut 0 {temp:g}")

        if variant == "a":
            # Drain AC-grounded by its ideal DC bias source.
            lines.append(f"Vds {conn['drain']} 0 {vds:g}")
            lines.append(f"Rbleed {conn['gate']} 0 {GATE_BLEED_OHM:g}")
            lines.append(f"Iprobe {conn['gate']} 0 DC 0 AC 1")
        else:
            # Drain biased ONLY through the inductor (DC short, AC open):
            # a direct Vds here would short the AC probe to ground.
            lines.append(f"Lbias vbus {conn['drain']} {BIAS_INDUCTOR_H:g}")
            lines.append(f"Vbus vbus 0 {vds:g}")
            lines.append(f"Rbleed {conn['gate']} 0 {GATE_BLEED_OHM:g}")
            lines.append(f"Iprobe {conn['drain']} 0 DC 0 AC 1")

        lines.append(f"Xdut {' '.join(xargs)} {subcircuit_name}")
        title = ("ciss" if variant == "a" else "coss") + f"_vds_{vds:.3g}"
        return f"* {title}\n" + "\n".join(lines) + "\n.end\n"

    # -- execution --------------------------------------------------------------

    def _run_ac(self, netlist_text: str):
        directive = ACDirective(start_freq=self.f_meas,
                                stop_freq=self.f_meas, n_points=1,
                                variation="lin")
        return self.engine.run_ac_sweep(netlist_text, directive)

    @staticmethod
    def _complex_voltage(result, node: str) -> complex:
        """Rebuild a complex node voltage from raw traces.

        LTspice ASCII .raw stores magnitude/phase pairs for AC runs as
        separate trace columns ('V(n):m', 'V(n):p' in degrees) when the
        trace is complex; some builds emit a single 'V(n)' real column for
        magnitude-only. Handle both.
        """
        traces = result.traces
        mag_name = f"V({node}):m"
        ph_name = f"V({node}):p"
        plain = f"V({node})"

        if mag_name in traces and ph_name in traces:
            mag = float(np.asarray(traces[mag_name]).ravel()[0])
            phase_deg = float(np.asarray(traces[ph_name]).ravel()[0])
            return complex(mag * np.cos(np.deg2rad(phase_deg)),
                           mag * np.sin(np.deg2rad(phase_deg)))
        if plain in traces:
            wave = np.asarray(traces[plain])
            if np.iscomplexobj(wave):
                return complex(wave.ravel()[0])
            return complex(float(wave.ravel()[0]), 0.0)
        raise KeyError(
            f"node voltage trace for {node!r} not found; "
            f"available: {sorted(traces)}")

    # -- public sweep ---------------------------------------------------------------

    def run(self, lib_path, subcircuit_name: str = "",
            vds_min: float = 0.1, vds_max: float = 60.0,
            points: int = 25, pin_roles=None):
        lib_path = Path(lib_path)
        vds_sweep = np.logspace(np.log10(vds_min), np.log10(vds_max), points)
        ciss = np.full(points, np.nan)
        coss = np.full(points, np.nan)
        crss = np.full(points, np.nan)

        omega = 2 * np.pi * self.f_meas
        for i, vds in enumerate(vds_sweep):
            try:
                text_a = self._ac_netlist(lib_path, subcircuit_name,
                                          pin_roles, vds, "a")
                res_a = self._run_ac(text_a)
                vg = abs(self._complex_voltage(res_a, sub_gate(pin_roles)))
                ciss[i] = 1.0 / (omega * max(vg, 1e-15))

                text_b = self._ac_netlist(lib_path, subcircuit_name,
                                          pin_roles, vds, "b")
                res_b = self._run_ac(text_b)
                vd = abs(self._complex_voltage(res_b, "drain"))
                vg_b = abs(self._complex_voltage(res_b, sub_gate(pin_roles)))
                coss[i] = 1.0 / (omega * max(vd, 1e-15))
                crss[i] = (vg_b / max(vd, 1e-15)) * ciss[i]
            except Exception as exc:
                _LOGGER.warning("%s @ Vds=%.2fV probe failed (%s); point "
                                "masked as NaN.",
                                "Ciss/Coss/Crss", vds, exc)

            # physical sanity mask
            for name, arr in (("Ciss", ciss), ("Coss", coss),
                              ("Crss", crss)):
                v = arr[i]
                if not np.isfinite(v) or v <= 0 or v > 1e-3:
                    _LOGGER.warning("%s @ Vds=%.2fV = %.3e F non-physical "
                                    "(limit 1 mF); masked.", name, vds, v)
                    arr[i] = np.nan

        def _trim(arr):
            valid = np.where(np.isfinite(arr) & (arr > 0))[0]
            return arr[:valid[-1] + 1] if len(valid) else arr

        ciss_t, coss_t, crss_t = map(_trim, (ciss, coss, crss))
        n_valid = min(len(ciss_t), len(coss_t), len(crss_t))
        x_valid = vds_sweep[:n_valid]
        ciss_t, coss_t, crss_t = (ciss_t[:n_valid], coss_t[:n_valid],
                                  crss_t[:n_valid])
        if n_valid < points:
            _LOGGER.info("returning first %d of %d points (valid envelope)",
                         n_valid, points)

        note = f"measured @ {self.f_meas:.1e} Hz"
        out = []
        for name, cap in (("Ciss", ciss_t), ("Coss", coss_t),
                          ("Crss", crss_t)):
            out.append(CurveData(
                name=name,
                x_name="Drain-Source Voltage (Vds)", x_unit="V",
                y_name="Capacitance", y_unit="F",
                x_data=x_valid.tolist(), y_data=cap.tolist(),
                metadata={"note": note}))
        return out


def sub_gate(pin_roles) -> str:
    """External node name bound to the gate role.

    Resolved BY ROLE NAME (not position) so gate-first vendor headers
    like EPC2216 bind correctly. Raises when no gate role exists.
    """
    canonical = {"drain": "drain", "gate": "gate", "source": "0"}
    for r in pin_roles or []:
        rl = r.lower()
        if rl == "tj":
            continue
        if rl == "gate":
            return canonical[rl]
    raise ValueError(
        f"no 'gate' role in pin_roles={pin_roles!r}; cannot place the "
        f"gate-side AC probe.")
