"""DC sweep testbenches (Curves 1, 2, 3, 6, 7) — LTspice backend.

Each bench assembles a complete netlist via simulation.netlist_text,
executes it through the spicelib engine, and extracts Id from the .raw
current trace of the drain-side source.

Sign convention: the drain supply element is declared with its positive
terminal at the device drain, so its raw current I(Vds) is positive when
the device conducts. _check_sign guards against silent polarity flips.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from common.types import CurveData
from simulation.engine import SpiceEngine, DCDirective

_LOGGER = logging.getLogger(__name__)


class StaticSweepTestbench:
    """Extracts output, transfer, Rds(on), body-diode and Vth curves."""

    def __init__(self, engine: SpiceEngine | None = None):
        self.engine = engine or SpiceEngine()

    # -- shared helpers -------------------------------------------------------

    @staticmethod
    def _core_roles(pin_roles):
        return [r for r in pin_roles if r.lower() != "tj"]

    def _has_tj(self, lib_path, subcircuit_name, pin_roles) -> bool:
        """True when the SUBCKT declares a thermal port.

        Checked from the library header itself (authoritative); some yamls
        omit 'tj' from pin_order even though the port exists.
        """
        if any(r.lower() == "tj" for r in pin_roles):
            return True
        try:
            from parser.netlist import NetlistParser
            subs = NetlistParser.parse_subcircuits(Path(lib_path))
            info = next((s for s in subs
                         if s.name.lower() == subcircuit_name.lower()), None)
            if info is None:
                return False
            return any(p.lower() == "tj" for p in info.nodes)
        except Exception:
            return False

    def _header_lines(self, lib_path, subcircuit_name, pin_roles, xargs):
        lines = [f".include {Path(lib_path).resolve()}"]
        if self._has_tj(lib_path, subcircuit_name, pin_roles):
            temp = float(getattr(self.engine, "temperature", 25.0))
            lines.append(f"Vtj tj_dut 0 {temp:g}")
        return lines, xargs

    def _finish(self, text, directive, current_trace, label,
                negate: bool = True):
        """Run a DC sweep and extract (x, |y|) with polarity validation.

        LTspice reports a voltage SOURCE's current as positive when it
        flows out of its + terminal; conduction current enters the DUT,
        so bench traces come out negative and are negated here (matching
        the historical '-analysis.Vds' convention). The sign check runs
        AFTER negation so a flipped probe still fails loudly.
        """
        result = self.engine.run_dc_sweep(text, directive)
        x = np.asarray(result.get(result.sweep_var), dtype=float)
        y = np.asarray(result.get(current_trace), dtype=float)
        if negate:
            y = -y
        self._check_sign(x, y, label)
        return x, np.abs(y)

    def _xargs_for(self, lib_path, subcircuit_name, roles):
        """Positional connections including the tj node when present.

        Roles are matched BY NAME (drain/gate/source), so gate-first
        vendor headers bind correctly. A 'tj' role appends its dedicated
        thermal node after the core three.
        """
        conn = {"drain": "drain", "gate": "gate", "source": "0"}
        xargs = [conn.get(r, r) for r in self._core_roles(roles)]
        if self._has_tj(lib_path, subcircuit_name, roles):
            xargs.append("tj_dut")
        return xargs

    # -- Curve 1: output characteristics --------------------------------------

    def run_output_characteristics(self, lib_path, subcircuit_name,
                                   vgs_list=None, vds_max=10.0,
                                   vds_points=101, pin_roles=None):
        if vgs_list is None:
            vgs_list = [3.0, 4.5, 6.0, 10.0]
        roles = pin_roles or ["drain", "gate", "source"]
        step = vds_max / max(1, vds_points - 1)
        curves = []
        for vgs in vgs_list:
            lines, xargs = self._header_lines(
                lib_path, subcircuit_name, roles,
                self._xargs_for(lib_path, subcircuit_name, roles))
            # The swept source ties DIRECTLY to drain: sweeping it raises
            # the true device Vds from 0 to vds_max (positive terminal at
            # drain, negative at ground).
            lines += [
                f"Vgs gate 0 {vgs:g}",
                "Vds drain 0 0",
                f"Xdut {' '.join(xargs)} {subcircuit_name}",
            ]
            text = "* output characteristics\n" + "\n".join(lines) \
                + "\n.end\n"
            directive = DCDirective("Vds", 0.0, float(vds_max), step)
            x, y = self._finish(text, directive, "I(Vds)",
                                f"output char @Vgs={vgs}")
            curves.append(CurveData(
                name="Output Characteristic",
                x_name="Drain-Source Voltage (Vds)", x_unit="V",
                y_name="Drain Current (Id)", y_unit="A",
                x_data=x.tolist(), y_data=y.tolist(),
                family_param="Vgs", family_value=float(vgs)))
        return curves

    # -- Curve 2: transfer characteristic --------------------------------------

    def run_transfer_characteristics(self, lib_path, subcircuit_name,
                                     vds_fixed=10.0, vgs_max=15.0,
                                     vgs_points=151, pin_roles=None):
        roles = pin_roles or ["drain", "gate", "source"]
        lines, xargs = self._header_lines(
            lib_path, subcircuit_name, roles,
            self._xargs_for(lib_path, subcircuit_name, roles))
        lines += [
            f"Vds drain 0 {float(vds_fixed):g}",
            "Vgs gate 0 0",
            f"Xdut {' '.join(xargs)} {subcircuit_name}",
        ]
        text = "* transfer characteristics\n" + "\n".join(lines) + "\n.end\n"
        directive = DCDirective("Vgs", 0.0, float(vgs_max),
                                vgs_max / max(1, vgs_points - 1))
        x, y = self._finish(text, directive, "I(Vds)", "transfer char")
        return CurveData(
            name=f"Transfer Characteristic (Vds={vds_fixed}V)",
            x_name="Gate-Source Voltage (Vgs)", x_unit="V",
            y_name="Drain Current (Id)", y_unit="A",
            x_data=x.tolist(), y_data=y.tolist(),
            family_param="Vds", family_value=float(vds_fixed))

    # -- Curve 3: Rds(on) --------------------------------------------------------

    def run_rds_on(self, lib_path, subcircuit_name, vgs=10.0, vds_max=2.5,
                   vds_points=51, min_current=1.0, pin_roles=None):
        roles = pin_roles or ["drain", "gate", "source"]
        lines, xargs = self._header_lines(
            lib_path, subcircuit_name, roles,
            self._xargs_for(lib_path, subcircuit_name, roles))
        lines += [
            "Vds drain 0 0",
            f"Vgs gate 0 {vgs:g}",
            f"Xdut {' '.join(xargs)} {subcircuit_name}",
        ]
        text = "* rds_on\n" + "\n".join(lines) + "\n.end\n"
        directive = DCDirective("Vds", 0.0, float(vds_max),
                                vds_max / max(1, vds_points - 1))
        result = self.engine.run_dc_sweep(text, directive)

        # Device Vds IS the sweep variable now (source tied drain->gnd).
        v_dev = np.asarray(result.get(result.sweep_var), dtype=float)
        i_arr = np.abs(np.asarray(result.get("I(Vds)"), dtype=float))

        mask = i_arr > float(min_current)
        rds_mohm = v_dev[mask] / np.maximum(i_arr[mask], 1e-12) * 1e3
        return CurveData(
            name=f"On-State Resistance (Vgs={vgs}V)",
            x_name="Drain Current (Id)", x_unit="A",
            y_name="On-Resistance (Rds(on))", y_unit="mOhm",
            x_data=i_arr[mask].tolist(), y_data=rds_mohm.tolist(),
            family_param="Vgs", family_value=float(vgs))

    # -- Curve 7: threshold voltage ----------------------------------------------

    def run_threshold_voltage(self, lib_path, subcircuit_name,
                              i_target=250e-6, v_max=6.0, points=3001,
                              pin_roles=None):
        """JEDEC constant-current method: drain tied to gate."""
        roles = self._core_roles(pin_roles)
        has_tj = self._has_tj(lib_path, subcircuit_name, roles)
        xargs = ["gd" if r != "source" else "0" for r in roles]
        lines = [f".include {Path(lib_path).resolve()}"]
        temp = float(getattr(self.engine, "temperature", 25.0))
        if has_tj:
            lines.append(f"Vtj tj_dut 0 {temp:g}")
            xargs.append("tj_dut")
        lines.append("Vsweep gd 0 0")
        lines.append(f"Xdut {' '.join(xargs)} {subcircuit_name}")
        text = "* threshold voltage\n" + "\n".join(lines) + "\n.end\n"

        directive = DCDirective("Vsweep", 0.0, float(v_max),
                                v_max / max(1, points - 1))
        result = self.engine.run_dc_sweep(text, directive)
        v_arr = np.asarray(result.get(result.sweep_var), dtype=float)
        i_arr = np.abs(np.asarray(result.get("I(Vsweep)"), dtype=float))

        if float(i_arr.max()) < float(i_target):
            raise ValueError(
                f"Threshold extraction failed: current peaks at "
                f"{i_arr.max():.3g} A over Vgs in [0, {v_max}], never "
                f"reaching target {i_target:.3g} A.")

        v_th = float(np.interp(i_target, i_arr, v_arr))
        curve = CurveData(
            name=(f"Threshold Sweep (Vds=Vgs) - Vth={v_th:.3f}V "
                  f"@ Id={i_target * 1e6:.0f}uA"),
            x_name="Gate-Source Voltage (Vgs)", x_unit="V",
            y_name="Drain Current (Id)", y_unit="A",
            x_data=v_arr.tolist(), y_data=i_arr.tolist(),
            family_param="Id_target", family_value=float(i_target))
        return curve, v_th

    # -- Curve 6: body diode --------------------------------------------------------

    def run_body_diode(self, lib_path, subcircuit_name, vsd_max=1.0,
                       points=51, sweep_step=None, pin_roles=None):
        """Isd vs Vsd, gate tied to source; 10 MOhm anchor subtracted."""
        roles = self._core_roles(pin_roles)
        has_tj = self._has_tj(lib_path, subcircuit_name, roles)
        conn_map = {"drain": "drn"}
        others = ["src"] * (len(roles) - 1)
        xargs = [conn_map.get(r, "src") for r in roles]
        lines = [f".include {Path(lib_path).resolve()}"]
        temp = float(getattr(self.engine, "temperature", 25.0))
        bleed_ohm = 10e6
        if has_tj:
            lines.append(f"Vtj tj_dut 0 {temp:g}")
            xargs.append("tj_dut")
        lines += [
            f"Rbleed src 0 {bleed_ohm:g}",
            "Vsd src drn 0",
            f"Xdut {' '.join(xargs)} {subcircuit_name}",
        ]

        def make_text():
            return "* body diode\n" + "\n".join(lines) + "\n.end\n"

        step = sweep_step if sweep_step else vsd_max / max(1, points - 1)
        directive = DCDirective("Vsd", 0.0, float(vsd_max), step)
        try:
            x, y_raw = self._finish(make_text(), directive, "I(Vsd)",
                                    "body diode")
        except (RuntimeError, AssertionError):
            coarse = vsd_max / 24
            _LOGGER.warning("body_diode default step diverged; retrying "
                            "coarse %.4g V", coarse)
            directive.step = coarse
            x, y_raw = self._finish(make_text(), directive,
                                    "I(Vsd)", "body diode")

        y = y_raw - x / bleed_ohm
        return CurveData(
            name="Body Diode Forward Characteristic",
            x_name="Source-Drain Voltage (Vsd)", x_unit="V",
            y_name="Source Current (Isd)", y_unit="A",
            x_data=x.tolist(), y_data=y.tolist(),
            metadata={"condition": "gate_off (vgs=0)"})

    # -- polarity guard -------------------------------------------------------------

    @staticmethod
    def _check_sign(x_arr, y_arr, label, expect_positive=True) -> None:
        y = np.asarray(y_arr, dtype=float)
        if len(y) == 0 or np.all(np.isnan(y)):
            return
        upper = y[len(y)//2:]
        frac_wrong = float(np.mean(upper < 0)) if expect_positive else \
            float(np.mean(upper > 0))
        assert frac_wrong < 0.5, (
            f"{label}: {(frac_wrong*100):.0f}% of upper-sweep samples have "
            f"the wrong polarity (expected "
            f"{'positive' if expect_positive else 'negative'}) - do not use "
            f"these curves.")
