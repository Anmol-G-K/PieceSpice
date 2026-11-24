"""Live LTspice engine tests (T2.1). Fast: each <2 s."""

import os
import shutil
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


# --- offline netlist-preparation tests (no LTspice executable needed) -------

def _offline_engine(temperature=25.0):
    """Engine instance that never resolves or executes a real LTspice binary."""
    eng = object.__new__(SpiceEngine)
    eng.ltspice_exe = "LTspice.exe"
    eng.timeout_s = 1.0
    eng.temperature = float(temperature)
    eng.keep_files = True
    return eng


def _active_temp_cards(netlist_text):
    """Return active `.temp` lines in prepared netlist text."""
    cards = []
    for ln in netlist_text.splitlines():
        s = ln.strip()
        if not s or s.startswith("*") or s.startswith(";"):
            continue
        low = s.lower()
        if low.startswith(".temp"):
            rest = low[len(".temp"):]
            if rest == "" or not (rest[0].isalnum() or rest[0] == "_"):
                cards.append(s)
    return cards


def _prepare(text, temperature=25.0):
    """Prepare a netlist through the engine and return the written text."""
    workdir, path = _offline_engine(temperature)._prepare_workdir(text, "unit")
    try:
        return path.read_text(encoding="ascii")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


_NO_TEMP = "* dpt deck\nV1 in 0 5\nR1 in 0 1k\n.tran 0 1m\n.end\n"


@pytest.mark.parametrize("card", [
    ".temp 25",
    ".TEMP 25",
    "   .temp 85",
    "\t.Temp 25",
    ".temp 25 ; explicit deck temperature",
])
def test_explicit_temp_card_is_not_duplicated(card):
    net = _NO_TEMP.replace("R1 in 0 1k", f"R1 in 0 1k\n{card}")
    assert _active_temp_cards(_prepare(net, temperature=25.0)) == [card.strip()]


def test_temp_card_injected_when_absent():
    assert _active_temp_cards(_prepare(_NO_TEMP, temperature=25.0)) == \
        [".temp 25"]


def test_temp_card_omitted_at_ltspice_default():
    assert _active_temp_cards(_prepare(_NO_TEMP, temperature=27.0)) == []


@pytest.mark.parametrize("comment", [
    "*.temp 25",
    "; .temp 25",
    "* note: this deck used .temp 25 before",
    "; trailing .temp 85 note",
])
def test_commented_temp_is_not_a_directive(comment):
    net = _NO_TEMP.replace("V1 in 0 5", f"{comment}\nV1 in 0 5")
    assert len(_active_temp_cards(_prepare(net, temperature=25.0))) == 1


def test_longer_dot_card_is_not_a_temp_directive():
    net = _NO_TEMP.replace("V1 in 0 5", ".temperature 25\nV1 in 0 5")
    assert len(_active_temp_cards(_prepare(net, temperature=25.0))) == 1


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
