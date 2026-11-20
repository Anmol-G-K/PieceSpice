"""Tests for Tier-2 subcircuit synthesis (M3.5) and fitting pipeline (M3.6)."""

import math

import numpy as np
import pytest

from approximation.smoother import eps_from_grid, emit_qoss_expr
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


# ---------------------------------------------------------------------------
# T1 C1 emitter A/B (unified spec 4.1; C1 study P1/P3) - offline only.
# ---------------------------------------------------------------------------

def _line_with(text: str, prefix: str) -> str:
    return next(line for line in text.splitlines()
                if line.startswith(prefix))


def _qoss_expression(text: str) -> str:
    line = _line_with(text, "C_QOSS")
    return line.split("Q = {", 1)[1].rsplit("}", 1)[0]


def _eval_spice(expr: str, vds: float) -> float:
    """Numerically evaluate one emitted Q= expression at a drain voltage."""
    substituted = expr.replace("V(di,ss)", f"({vds!r})")
    return float(eval(substituted, {"sqrt": math.sqrt, "abs": abs}))


def _emitter_nodes(text: str) -> set[str]:
    """Internal/external nodes named on R/C/G/D element lines only."""
    nodes: set[str] = set()
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0][:1] in {"R", "C", "G", "D"} and len(parts) >= 3:
            nodes.update(parts[1:3])
    return nodes


def test_default_c1_eps_is_unchanged_legacy_c0(sample_spec):
    """No Tier-2 default change: None and 0.0 emit identical C0 text."""
    base = build_tier2_subcircuit(sample_spec)
    assert base == build_tier2_subcircuit(sample_spec, c1_eps=None)
    assert base == build_tier2_subcircuit(sample_spec, c1_eps=0.0)
    assert "abs(" in _line_with(base, "G_CH")
    assert "abs(" in _line_with(base, "C_QOSS")
    assert "sqrt(" not in base


def test_c1_eps_changes_only_the_two_ramp_lines(sample_spec):
    eps = eps_from_grid(sample_spec.q_oss_vs_v)
    assert eps == pytest.approx(12.5)   # median spacing 25, capped at 12.5
    base = build_tier2_subcircuit(sample_spec)
    text = build_tier2_subcircuit(sample_spec, c1_eps=eps)
    assert text.isascii()
    base_lines, new_lines = base.splitlines(), text.splitlines()
    assert len(base_lines) == len(new_lines)
    changed = {new_lines[i].split()[0]
               for i, (a, b) in enumerate(zip(base_lines, new_lines)) if a != b}
    assert changed == {"G_CH", "C_QOSS"}
    for prefix in ("RG ", "RD ", "RS ", "D_BODY ", ".MODEL", ".SUBCKT",
                   ".ENDS"):
        assert _line_with(base, prefix) == _line_with(text, prefix)
    # Both ramp sites come from the shared T0 helper, not local copies.
    q = emit_qoss_expr(sample_spec.q_oss_vs_v,
                       sample_spec.q_oss_charge_c, eps)
    assert _line_with(text, "C_QOSS") == f"C_QOSS di ss Q = {{{q}}}"
    assert "sqrt(" in _line_with(text, "G_CH")
    assert "sqrt(" in _line_with(text, "C_QOSS")


def test_emitted_qoss_is_anchored_at_zero_with_c1_eps(sample_spec):
    eps = eps_from_grid(sample_spec.q_oss_vs_v)
    text = build_tier2_subcircuit(sample_spec, c1_eps=eps)
    q0 = _eval_spice(_qoss_expression(text), 0.0)
    qmax = max(sample_spec.q_oss_charge_c)
    # Only %.6e coefficient rounding remains; no eps/2 charge is injected.
    assert abs(q0) < 1e-6 * qmax


def test_emitted_qoss_delta_slopes_rebuild_table_at_eps_zero(sample_spec):
    text = build_tier2_subcircuit(sample_spec)
    expr = _qoss_expression(text)
    for v, q in zip(sample_spec.q_oss_vs_v, sample_spec.q_oss_charge_c):
        assert _eval_spice(expr, v) == pytest.approx(q, rel=1e-5)
    grid = np.linspace(min(sample_spec.q_oss_vs_v),
                       max(sample_spec.q_oss_vs_v), 25)
    values = [_eval_spice(expr, float(v)) for v in grid]
    qmax = max(sample_spec.q_oss_charge_c)
    assert np.all(np.diff(values) >= -1e-6 * qmax)


def test_c1_eps_adds_no_nodes_and_keeps_semantic_pins(sample_spec):
    base_nodes = _emitter_nodes(build_tier2_subcircuit(sample_spec))
    assert base_nodes == {"d", "g", "s", "gi", "di", "ss"}
    for eps in (0.0, 0.5, 12.5):
        text = build_tier2_subcircuit(sample_spec, c1_eps=eps)
        assert _emitter_nodes(text) == base_nodes
        assert ".SUBCKT TestDev_PIECESPICE_T2 d g s" in text
        assert ".ENDS TestDev_PIECESPICE_T2" in text
        assert text.isascii()


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_c1_eps_rejects_invalid_widths(sample_spec, bad):
    with pytest.raises(ValueError):
        build_tier2_subcircuit(sample_spec, c1_eps=bad)
