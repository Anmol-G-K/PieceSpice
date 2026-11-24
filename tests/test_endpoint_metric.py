"""Endpoint validation uses the configured Curve-1 Vds ceiling."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from config.yaml_io import load_input
from pipeline import fitting


class _FakeEngine:
    def __init__(self, *args, **kwargs):
        self.netlists = []

    def run_op(self, text):
        self.netlists.append(text)
        return SimpleNamespace(traces={"I(Vds)": np.array([0.0])})


def test_endpoint_comparison_uses_configured_vds_max(monkeypatch):
    vendor = "cpm2-1200"
    cond = load_input(vendor)["conditions"]
    vds_end = float(cond["Vds_sweep_max"])
    assert vds_end != 10.0  # the regression this test protects against
    engines: list[_FakeEngine] = []

    def factory(*args, **kwargs):
        engine = _FakeEngine()
        engines.append(engine)
        return engine

    monkeypatch.setattr(fitting, "SpiceEngine", factory)
    rows, _worst = fitting.compare_tier2_vs_golden(vendor)
    assert rows and engines
    assert all(f"Vds drain 0 DC {vds_end:g}" in text
               for engine in engines for text in engine.netlists)
