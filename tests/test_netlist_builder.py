"""Tests for Tier-2 subcircuit synthesis (M3.5) and fitting pipeline (M3.6)."""

import numpy as np
import pytest

from common.types import PWLModelSpec
from models.netlist_builder import build_tier2_subcircuit


@pytest.fixture
def sample_spec():
    return PWLModelSpec(
        device_name="TestDev",
        breakpoints_vds=[0., 1., 2., 5., 10.],
        breakpoints_id=[[0., 2., 4., 10., 20.], [0., 3., 6., 15., 30.]],
        vgs_values=[5.0, 10.0],
        q_oss_vs_v=[0., 25., 50., 100.],
        q_oss_charge_c=[0., 5e-8, 7e-8, 9e-8],
        rds_on=0.005, v_th=2.5,
        body_diode_vsd=[0.4, 0.7, 0.9],
        body_diode_isd=[0.01, 1.0, 10.0])


def test_spec_validates_structure(sample_spec):
    with pytest.raises(ValueError):
        PWLModelSpec(
            device_name="bad", breakpoints_vds=[0., 5., 1.],   # not ascending
            breakpoints_id=[[0., 1., 2.]], vgs_values=[5.],
            rds_on=0.005, v_th=2.)


def test_subckt_structure(sample_spec):
    text = build_tier2_subcircuit(sample_spec)
    assert ".SUBCKT TestDev_PIECESPICE_T2 d g s" in text
    assert f".ENDS TestDev_PIECESPICE_T2" in text
    assert "G_CH" in text and "VALUE =" in text
    assert "tanh(" in text                      # C1 knee present
    assert "C_QOSS" in text and "Q = {" in text  # charge-conserving cap
    assert "D_BODY" in text                     # body diode present


def test_no_absolute_slope_accumulation_in_channel(sample_spec):
    """Regression (audit bug class): transfer ramps must use delta slopes.

    Endpoints are (0,0), (Vth,0), (5,20), (10,30). The emitter scales
    transfer amplitudes by 1/tanh(Vmax/s) to preserve the DC endpoint after
    the fitted knee; with the cap removed that factor can exceed 1.2 on a
    synthetic curve. Assert scale-invariant properties instead: the ratio
    between the two drives and a final slope far below the runaway 8+2=10.
    """
    line = next(l for l in build_tier2_subcircuit(sample_spec).splitlines()
                if l.startswith("G_CH"))
    expr = line.split("VALUE = {", 1)[1].rsplit("}", 1)[0]
    pwl = expr[1:].split("*tanh")[0].replace("V(gi,ss)", "v")
    id_of_vgs = eval("lambda v: " + pwl, {"abs": abs})
    assert id_of_vgs(0.0) == pytest.approx(0.0)
    assert id_of_vgs(2.5) == pytest.approx(0.0)   # Vth anchor
    assert id_of_vgs(10.0) / id_of_vgs(5.0) == pytest.approx(1.5, rel=0.02)
    assert 0.0 < (id_of_vgs(10.0) - id_of_vgs(5.0)) / 5.0 < 3.0


def test_ascii_only_netlist(sample_spec):
    """Non-ASCII chars break ngspice parsing on cp1252 consoles."""
    text = build_tier2_subcircuit(sample_spec)
    text.encode("ascii")   # raises if any non-ascii slipped in


def test_tier2_simulates_and_matches(tmp_path):
    """End-to-end: emit lib, instantiate in ngspice, check Id vs spec."""
    from PySpice.Spice.Netlist import Circuit
    from PySpice.Unit import u_V

    spec = PWLModelSpec(
        device_name="AccDev",
        breakpoints_vds=[0., 1., 2., 5., 10.],
        breakpoints_id=[[0., 2., 4., 10., 20.], [0., 3., 6., 15., 30.]],
        vgs_values=[5.0, 10.0],
        q_oss_vs_v=[0., 25., 50., 100.],
        q_oss_charge_c=[0., 5e-8, 7e-8, 9e-8],
        rds_on=0.005, v_th=2.5,
        body_diode_vsd=[0.4, 0.7, 0.9],
        body_diode_isd=[0.01, 1.0, 10.0])
    text = build_tier2_subcircuit(spec)
    lib = tmp_path / "acc.lib"
    lib.write_text(text, encoding="ascii")
    name = "AccDev_PIECESPICE_T2"

    expected = {5.0: 20.0, 10.0: 30.0}
    for vgs, want in expected.items():
        c = Circuit(f"acc{vgs}")
        c.include(str(lib))
        c.X("M1", name, "d", "g", c.gnd)
        c.V("ds", "d", c.gnd, 10)
        c.V("gs", "g", c.gnd, vgs)
        op = c.simulator(temperature=25,
                         nominal_temperature=25).operating_point()
        i_d = abs(float(op.branches["vds"][0]))
        assert i_d == pytest.approx(want, rel=0.02), \
            f"vgs={vgs}: Id={i_d} vs {want}"


def test_fit_pwl_model_generates_lib(tmp_path):
    """M3.6: pipeline stage consumes outputs YAML and emits a valid .lib.

    Disabled: it needs a live LTspice install plus ~25 AC probe runs per
    vendor. Tracked in docs/migration_todo.md T6.2a; re-enable together with
    the other four skipped tests.
    """
    import pytest
    pytest.skip("needs live LTspice + re-extraction of curve 4 "
                "(docs/migration_todo.md T6.2a)")
    from pipeline.fitting import fit_pwl_model
    lib = fit_pwl_model("csd16412q5a", out_dir=tmp_path)
    assert lib.exists()
    text = lib.read_text(encoding="ascii")
    assert ".SUBCKT" in text and ".ENDS" in text
    assert "G_CH" in text and "C_QOSS" in text
