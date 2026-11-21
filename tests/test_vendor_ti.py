"""Unit tests for multi-vendor model support: TI CSD16412Q5A (PSpice LEVEL=7
dialect) alongside the Infineon/IR MODPEX model (SPICE3 LEVEL=1).

The LTspice backend consumes vendor netlist text directly: .PARAM hierarchies
and {} expressions are resolved natively, and pins are bound by the header
tokens declared in config/inputs/csd16412q5a.yaml.
"""

import os
import re
from pathlib import Path

import numpy as np
import pytest

from config.yaml_io import load_input
from parser.netlist import NetlistParser, resolve_pin_connections
from simulation.engine import SpiceEngine
from simulation.netlist_text import NetlistBuilder
from simulation.testbenches.static_sweep import StaticSweepTestbench

TI_LIB = Path("SpiceModels/CSD16412Q5A.lib")
TI_SUBCKT = "CSD16412Q5A"
TI_VENDOR = "csd16412q5a"

LIVE = pytest.mark.skipif(
    os.environ.get("PIECESPICE_LIVE", "1") != "1",
    reason="live LTspice disabled via PIECESPICE_LIVE=0",
)


def _ti_pin_roles():
    """Roles in the vendor header's positional order (semantic map, not guesswork)."""
    cfg = load_input(TI_VENDOR)
    return resolve_pin_connections(
        TI_LIB, cfg["subcircuit"], cfg["pins"]["pin_order_semantic"])


# ---------- parsing ----------

def test_ti_lib_subckt_discovered():
    subs = NetlistParser.parse_subcircuits(TI_LIB)
    assert any(s.name == TI_SUBCKT for s in subs)


def test_ti_lib_pins_are_three():
    """Header '.SUBCKT CSD16412Q5A 1 2 3' -> exactly 3 pins, no stray tokens."""
    subs = NetlistParser.parse_subcircuits(TI_LIB)
    s = next(x for x in subs if x.name == TI_SUBCKT)
    assert s.nodes == ["1", "2", "3"]
    assert _ti_pin_roles() == ["drain", "gate", "source"]


def test_ti_lib_params_are_native_ltspice():
    """pWidth and perim stay symbolic in the vendor file; LTspice resolves them.

    No preprocessor normalization is involved: the raw library is included as
    text in the operating-point deck below and LTspice evaluates the hierarchy.
    """
    raw = TI_LIB.read_text(encoding="utf-8", errors="ignore")
    assert re.search(r"(?im)^\s*\.PARAM\s+pWidth\s+0\.6178256\b", raw)
    assert re.search(r"(?im)^\s*\.PARAM\s+perim\s+\{2\*pWidth\}", raw)
    assert "W={pWidth}" in raw and "PS={perim}" in raw


# ---------- live LTspice ----------

@LIVE
def test_ti_model_operates_under_ltspice():
    """Raw TI subckt at Vds=2.5 V, Vgs=10 V reaches Id > 50 A in a .op deck."""
    nb = NetlistBuilder("ti_op")
    nb.include(TI_LIB)
    nb.v_dc("ds", "drain", "0", 2.5)
    nb.v_dc("gs", "gate", "0", 10.0)
    nb.x_instance("m1", TI_SUBCKT, _ti_pin_roles())

    result = SpiceEngine().run_op(nb.render())
    i_d = abs(float(np.asarray(result.get("I(Vds)"))[0]))
    assert i_d > 50.0  # device conducts hard at full overdrive


@LIVE
def test_ti_rds_on_plausible():
    """Vendor model gives ~10.4 mOhm at Vgs=10 V, ID~50 A (9-11 mOhm part)."""
    curve = StaticSweepTestbench().run_rds_on(
        TI_LIB, TI_SUBCKT, vgs=10.0, vds_max=2.5, vds_points=51,
        min_current=20.0, pin_roles=_ti_pin_roles())
    rx, ry = curve.to_numpy()
    rds_50a = float(np.interp(50.0, rx, ry))
    # The PSpice vendor model is a 9-11 mOhm part at Vgs=10 V, ID~52 A
    # (vendor_model_defects.md device reference; recorded curve3 gives
    # 10.44 mOhm). The old <10 mOhm bar came from the retired ngspice model.
    assert 8.0 < rds_50a < 12.0, f"Rds(on) {rds_50a:.2f} mOhm implausible"
