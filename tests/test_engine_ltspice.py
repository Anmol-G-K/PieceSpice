"""Live LTspice engine tests (T2.1). Fast: each <2 s."""

import os
import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

from simulation.engine import (
    SpiceEngine, TranDirective, DCDirective, OPDirective,
    LTSpiceError, render_directive,
)

RC_NETLIST = """* rc smoke
V1 in 0 PULSE(0 5 0 1u 1u 1m 2m)
R1 in out 10k
C1 out 0 100n
.ic V(out)=0
.end
"""


def test_exe_resolution_fails_fast(monkeypatch):
    monkeypatch.delenv("LTSPICE_EXE", raising=False)
    with pytest.raises(FileNotFoundError):
        SpiceEngine(ltspice_exe="Z:/definitely/not/here/LTspice.exe")


def test_render_directives():
    assert render_directive(
        TranDirective(step_time=1e-9, end_time=3e-3)) == ".tran 0 0.003 0"
    card = render_directive(DCDirective("Vds", 0.0, 10.0, 0.1))
    assert card.startswith(".dc Vds") and "10" in card
    assert render_directive(OPDirective()) == ".op"


@pytest.mark.skipif(os.environ.get("PIECESPICE_LIVE", "1") != "1",
                    reason="live LTspice disabled via PIECESPICE_LIVE=0")
class TestLiveLTspice:
    """Real LTspice executions."""

    def setup_method(self):
        self.engine = SpiceEngine()

    def test_transient_runs_and_parses(self):
        result = self.engine.run_transient(
            RC_NETLIST, TranDirective(step_time=1e-6, end_time=3e-3))
        t = result.get("time")
        vout = result.get("V(out)")
        assert len(t) > 100
        # 2ms-period 50% duty pulse train: output charges toward 5V during
        # each high phase. Peak must exceed 60% of supply.
        assert float(np.max(vout)) > 3.0

    def test_missing_raw_raises_with_log_tail(self):
        bad = RC_NETLIST.replace(
            ".end", "Xbogus a b c nosuchsub\n.end")
        with pytest.raises(LTSpiceError) as ei:
            self.engine.run_op(bad)
        assert ei.value.log_tail != ""

    def test_timeout_raises_ltspice_error(self):
        # Deterministic timeout: force an extremely fine max timestep over
        # a long transient so LTspice cannot finish within the budget.
        slow = RC_NETLIST.replace(".end", "")
        directive = TranDirective(step_time=1e-9, end_time=10.0,
                                  max_step=1e-9)   # 1e10 steps required
        with pytest.raises(LTSpiceError) as ei:
            self.engine.run_transient(slow, directive, timeout_s=3)
        assert "timed out" in str(ei.value).lower()

    def test_dc_sweep_directive(self):
        net = "* dc sweep\nVsrc src 0 0\nR1 src 0 1k\n.end\n"
        result = self.engine.run_dc_sweep(
            net, DCDirective(source_name="Vsrc", start=0.0, stop=1.0,
                             step=0.1))
        assert len(result.traces) >= 2

    def test_temperature_card_emitted(self):
        engine = SpiceEngine(temperature=85.0)
        result = engine.run_op(RC_NETLIST)
        assert ".temp 85" in result.netlist_path.read_text()
