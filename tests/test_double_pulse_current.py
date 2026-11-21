"""DPT bench current identity: device current is I(Lload) - I(Dclamp)."""
from __future__ import annotations

import numpy as np

from simulation.engine import SimResult
from simulation.testbenches import double_pulse as dp


class _FakeEngine:
    def __init__(self, traces, **_kwargs):
        self._traces = traces

    def run_transient(self, _text, _directive):
        return SimResult(traces=self._traces, sweep_var="time",
                         log_text="Total iterations: 10\n")


def _run(monkeypatch, traces):
    monkeypatch.setattr(dp, "SpiceEngine", lambda **_kw: _FakeEngine(traces))
    return dp.run_double_pulse(
        "unused.lib", "TOY", ["drain", "gate", "source"],
        vbus=10.0, vgs_on=5.0, i_load_target=2.0, l_load=1e-6,
        t1_on=1e-6, t1_width=1e-6, t2_on=5e-6, t2_width=1e-6,
        edge_time=1e-9, t_end=10e-6, max_step=1e-9, temperature=25.0)


def test_dpt_uses_device_current_when_clamp_trace_exists(monkeypatch):
    t = np.linspace(0.0, 1e-5, 11)
    run = _run(monkeypatch, {
        "time": t,
        "V(drain)": np.full_like(t, 10.0),
        "V(gate)": np.zeros_like(t),
        "I(Lload)": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                              7.0, 8.0, 9.0, 10.0, 11.0]),
        "I(Dclamp)": np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                               0.5, 0.5, 0.5, 0.5, 0.5]),
    })
    assert run.current_definition == "I(Lload)-I(Dclamp)"
    assert np.allclose(run.i_d, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
                                 6.5, 7.5, 8.5, 9.5, 10.5])


def test_dpt_fallback_is_labelled_and_never_silent(monkeypatch):
    t = np.linspace(0.0, 1e-5, 3)
    run = _run(monkeypatch, {
        "time": t,
        "V(drain)": np.full_like(t, 10.0),
        "V(gate)": np.zeros_like(t),
        "I(Lload)": np.array([-1.0, 2.0, 3.0]),
    })
    assert "fallback" in run.current_definition
    assert np.allclose(run.i_d, [1.0, 2.0, 3.0])
