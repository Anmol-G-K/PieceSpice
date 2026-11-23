"""Unit tests for the paper-defined parts of the Tier-3 TPWL terminal model."""
import hashlib
import re

import numpy as np
import pytest

from config.yaml_io import load_input
from models import tpwl_netlist_builder as emitter
from pipeline.tier3 import (
    _emission_record, _emission_settings, _log_iterations, _waveform_errors,
)
from approximation import tpwl_deim as tpwl
from approximation.charge_surface import ChargeSurface
from approximation.tpwl_deim import (
    WEIGHT_EXPONENT,
    TPWLTerminalModel,
    deim_indices,
    paper_weights,
)
from approximation.smoother import hyp_expr
from simulation.testbenches.dpt_trajectory import Trajectory


def test_deim_indices_are_distinct_and_interpolatory():
    rng = np.random.default_rng(0)
    basis, _ = np.linalg.qr(rng.normal(size=(8, 3)))
    idx = deim_indices(basis)
    assert idx.shape == (3,)
    assert len(set(idx.tolist())) == 3
    # DEIM interpolatory projector reproduces selected rows exactly
    P = np.eye(8)[:, idx]
    proj = basis @ np.linalg.solve(P.T @ basis, P.T)
    v = rng.normal(size=8)
    assert np.allclose((proj @ v)[idx], v[idx])


def test_paper_weights_follow_rewienski_white_formula():
    assert WEIGHT_EXPONENT == 25.0
    anchors = np.array([[0.0, 0.0], [1.0, 0.0]])
    mid = paper_weights(anchors, np.array([[0.5], [0.0]]))
    assert np.allclose(mid[:, 0], [0.5, 0.5])
    near = paper_weights(anchors, np.array([[0.1], [0.0]]))
    assert near[0, 0] > near[1, 0]
    assert np.isclose(near[:, 0].sum(), 1.0)
    # far from every anchor the weights stay normalised and finite
    far = paper_weights(anchors, np.array([[50.0], [0.0]]))
    assert np.all(np.isfinite(far[:, 0]))
    assert np.isclose(far[:, 0].sum(), 1.0)


def test_single_cell_model_predicts_its_affine_channel():
    model = TPWLTerminalModel(
        vendor="t", subcircuit="T", pin_roles=["drain", "gate", "source"],
        anchors=np.array([[0.0, 0.0]]),
        v_ref=np.zeros(2), v_span=np.ones(2),
        channel=np.array([[1.0, 2.0, 3.0]]),
        charge_d=np.zeros((1, 2)),
        charge_g=np.zeros((1, 3)), edge_window=3)
    t = np.linspace(0.0, 1e-6, 11)
    u = np.vstack([np.linspace(0.0, 1.0, 11), np.zeros(11)])
    id_pred, qg_pred = model.predict(t, u)
    assert np.allclose(id_pred, 1.0 + 2.0 * u[0])
    assert np.allclose(qg_pred, 0.0)


def _toy_model() -> TPWLTerminalModel:
    """Offline model with a real vendor's config-backed curve files."""
    return TPWLTerminalModel(
        vendor="csd16412q5a", subcircuit="TOY",
        pin_roles=["drain", "gate", "source"],
        anchors=np.array([[0.0, 0.0], [0.5, 0.5]]),
        v_ref=np.array([4.0, 8.0]), v_span=np.array([10.0, 10.0]),
        channel=np.array([[10.0, 1.0, 0.5], [20.0, 2.0, 1.0]]),
        charge_d=np.array([[1e-9, 1e-10], [2e-9, 2e-10]]),
        charge_g=np.zeros((2, 3)), edge_window=3,
        coss_v=np.array([0.1, 5.0, 20.0]),
        coss_c=np.array([1e-9, 5e-10, 2e-10]),
        v_block=1.88, v_width=0.05)


def test_grid_axes_are_deterministic_and_cover_dpt():
    model = _toy_model()
    g1, d1 = emitter.grid_axes(model)
    g2, d2 = emitter.grid_axes(model)
    assert np.array_equal(g1, g2)
    assert np.array_equal(d1, d2)
    assert g1[0] <= 0.0 and g1[-1] >= 10.0
    assert d1[0] <= 0.0 and d1[-1] >= 12.5
    assert np.all(np.diff(g1) > 0) and np.all(np.diff(d1) > 0)


def test_emission_is_ascii_and_uses_flat_tables_without_weights():
    model = _toy_model()
    text = emitter.emit_subcircuit(model, name="TOY_T3")
    assert text.isascii()
    assert text.startswith(".subckt TOY_T3 d g s\n")
    assert text.rstrip().endswith(".ends")

    gch = next(line for line in text.splitlines()
               if line.startswith("G_CH "))
    assert gch.startswith("G_CH d s VALUE={")
    value = gch[len("G_CH d s VALUE={"):]
    assert value.endswith("}")
    value = value[:-1]

    # Flat channel: nested V(d,s) tables (transfer x channel) times the
    # analytic Vds factor, with no B-source intermediates or weight nodes.
    assert "table(V(d,s)," in text
    assert "V(d,s)/(V(d,s)+2.000000000e-01)" in text
    assert "QOSS d s Q={table(limit(V(d,s)" in text
    assert "\nB" not in text and "C_QD" not in text
    assert "weight" not in text.lower()
    assert "exp(" not in text and "sqrt(" in text

    # Softplus is the shared C1 hyperbola-smoothed hinge
    # 0.5*(x + sqrt(x*x + eps^2)), not the old literal softplus form.
    env = hyp_expr("@", emitter.emission_policy(model).soft_eps).split("@")
    assert value.startswith(env[0])       # 0.5*((...
    assert value.endswith(env[-1])        # ...)+eps^2))


def test_emitted_channel_clamps_negative_drain_voltage():
    model = _toy_model()
    u = np.vstack([np.array([10.0, 10.0, 10.0]),
                   np.array([-0.1, 0.0, 10.0])])
    current = emitter.emitted_channel(model, u)
    # TI's measured soft clamp approaches zero below Vds = 0 but is C1.
    assert current[0] < 1e-2
    assert current[1] < 1e-2
    assert current[2] > 0.0


def test_emission_policies_are_vendor_specific_and_ascii():
    toy = _toy_model()
    ir = _toy_model()
    ir.vendor = "irfs7730pbf"
    epc = _toy_model()
    epc.vendor = "epc2216"
    text = emitter.emit_subcircuit(ir, name="IR_T3")
    assert text.isascii()
    assert "QOSS d s Q={table(" not in text
    assert "exp(-" in text and "sqrt(" in text
    assert "max(0," in text  # IRFS keeps the hard clamp
    assert emitter.emission_policy(toy).soft_clamp
    assert emitter.emission_policy(ir).high_vds_saturation
    assert emitter.emission_policy(epc).vgs_points == 19


def test_high_vds_saturation_extends_grid_and_floors_channel():
    ir = _toy_model()
    ir.vendor = "irfs7730pbf"
    g, d = emitter.grid_axes(ir)
    cfg = load_input(ir.vendor)["dpt"]
    assert d[-1] >= 1.10 * float(cfg["vbus"])
    table = emitter.grid_channel(ir, g, d)
    tail = table[:, np.searchsorted(d, 10.0):]
    assert np.all(np.isfinite(tail)) and np.all(tail >= 0.0)
    assert np.allclose(tail, tail[:, :1])


def test_emitted_static_rmse_matches_channel_prediction():
    from approximation.static_curves import static_samples
    model = _toy_model()
    u, ref, _ = static_samples(model.vendor)
    pred = emitter.emitted_channel(model, u)
    expected = float(np.sqrt(np.mean((pred - ref) ** 2)) /
                     (np.ptp(ref) or 1.0))
    assert np.isclose(emitter.emitted_static_rmse(model), expected)


def test_log_iterations_parses_completed_and_failed_logs():
    assert _log_iterations("Total iterations: 394001\n") == 394001
    assert _log_iterations("Simulation Failed: Iteration limit reached") is None


def test_saturation_scale_override_is_vendor_specific():
    assert emitter.saturation_scale(_toy_model()) == 0.2
    epc = _toy_model()
    epc.vendor = "epc2216"
    assert emitter.saturation_scale(epc) == 0.5


def test_emission_fails_closed_for_unregistered_vendor():
    model = _toy_model()
    model.vendor = "not_registered_vendor"
    with pytest.raises(ValueError, match="no measured TPWL emission policy"):
        emitter.emission_policy(model)
    with pytest.raises(ValueError):
        emitter.saturation_scale(model)
    with pytest.raises(ValueError):
        emitter.emit_subcircuit(model)


def test_waveform_errors_uses_common_golden_grid():
    from types import SimpleNamespace

    tg = np.linspace(0.0, 10.0, 101)
    traj = SimpleNamespace(
        time=tg,
        input=np.vstack([np.zeros_like(tg), np.full_like(tg, 5.0)]),
        output=np.vstack([np.full_like(tg, 2.0), np.zeros_like(tg)]))
    tr = np.linspace(0.0, 10.0, 1001)
    run = SimpleNamespace(time=tr, v_ds=np.full_like(tr, 5.0),
                          i_d=np.full_like(tr, 2.0))
    err = _waveform_errors(run, traj)
    assert err["compare_points"] == 101      # golden grid, not 1001 ROM samples
    assert err["vds_rel_rmse_pct"] == pytest.approx(0.0)
    assert err["id_rel_rmse_pct"] == pytest.approx(0.0)
    assert err["current_definition"] == "I(Lload)-I(Dclamp)"

    short = SimpleNamespace(time=tr[tr <= 5.0],
                            v_ds=np.full_like(tr[tr <= 5.0], 5.0),
                            i_d=np.full_like(tr[tr <= 5.0], 2.0))
    partial = _waveform_errors(short, traj)
    assert partial["compare_points"] == 51


# ---- V2 physical elements, emitted selection, multi-trajectory, surface ----

def _v2_model(qg_q=None) -> TPWLTerminalModel:
    model = _toy_model()
    model.qg_v = np.array([0.0, 1.0, 2.5, 5.0])
    model.qg_q = (np.array([0.0, 2e-10, 8e-10, 1.3e-9])
                  if qg_q is None else np.asarray(qg_q, dtype=float))
    return model


def _qg_pairs(text: str) -> tuple[np.ndarray, np.ndarray]:
    line = next(ln for ln in text.splitlines()
                if ln.startswith("CTOY_V2_QG "))
    nums = [float(x) for x in
            re.findall(r"[-+]?\d\.\d+e[-+]\d+", line)]
    return np.asarray(nums[2::2]), np.asarray(nums[3::2])


def test_default_v1_emission_is_the_closed_four_line_form():
    text = emitter.emit_subcircuit(_toy_model(), name="TOY_T3")
    lines = text.splitlines()
    assert lines[0] == ".subckt TOY_T3 d g s"
    assert lines[1].startswith("CTOY_T3_QOSS d s Q={table(limit(V(d,s),")
    assert lines[2].startswith("G_CH d s VALUE={")
    assert lines[3] == ".ends"
    assert len(lines) == 4
    assert "V(gi" not in text and "V(di" not in text and "V(ss" not in text
    assert "RG " not in text and "RD " not in text and "RS " not in text
    assert "D_BODY" not in text and "_QG " not in text
    assert text.isascii()


def _channel_rows(text: str) -> list[str]:
    """Unique nested V(d,s) channel table rows.

    The C1 soft clamp substitutes the whole channel expression three times
    inside the smoothed hinge, so the same row text appears three times.
    Deduplicate while preserving order; the distinct row count is the Vgs
    grid size.
    """
    gch = next(ln for ln in text.splitlines() if ln.startswith("G_CH "))
    return list(dict.fromkeys(
        re.findall(r"table\(V\(d,s\),([^()]*)\)", gch)))


def _row_points(row: str) -> int:
    return len(re.findall(r"[-+]?\d\.\d+e[-+]\d+", row)) // 2


def test_default_emission_is_byte_identical_with_compact_fields_present():
    model = _toy_model()
    base = emitter.emission_policy(model)
    legacy = emitter.EmissionPolicy(
        vgs_points=base.vgs_points, vds_points=base.vds_points,
        smooth_qoss=base.smooth_qoss, soft_clamp=base.soft_clamp,
        soft_eps=base.soft_eps,
        high_vds_saturation=base.high_vds_saturation,
        include_gate_charge=base.include_gate_charge,
        include_body_diode=base.include_body_diode,
        include_parasitics=base.include_parasitics)
    text = emitter.emit_subcircuit(model)
    assert text == emitter.emit_subcircuit(model, policy=base)
    assert text == emitter.emit_subcircuit(model, policy=legacy)
    # Closed v1 record: byte-identity guard for the toy model's full lib.
    assert hashlib.sha256(text.encode("ascii")).hexdigest() == (
        "78f5aba97f72bc4d192d9d31e4832a986f7ee00a0aafcd3ab4c74ce2cd6ffee0")
    assert emitter.EmissionPolicy().compact is False
    assert emitter.EmissionPolicy().compact_rel_tol == 0.005
    assert emitter.emission_metrics(model, None) == {
        "table_points": 41, "vgs_points": 30,
        "max_abs_error_a": 0.0, "max_rel_error_pct": 0.0,
        "compact": False, "rel_tol": 0.005,
    }


def test_compact_policy_preserves_vendor_clamp_and_grid():
    model = _toy_model()
    base = emitter.emission_policy(model)
    pol = emitter.COMPACT_POLICY(model)
    assert pol.compact is True and pol.compact_rel_tol == base.compact_rel_tol
    assert (pol.vgs_points, pol.vds_points, pol.soft_clamp, pol.soft_eps,
            pol.smooth_qoss, pol.high_vds_saturation) == (
        base.vgs_points, base.vds_points, base.soft_clamp, base.soft_eps,
        base.smooth_qoss, base.high_vds_saturation)
    assert emitter.COMPACT_POLICY(base, rel_tol=0.002) == (
        emitter.compact_policy(model, rel_tol=0.002))


def test_compact_emission_reduces_vds_axis_with_measured_error():
    model = _toy_model()
    pol = emitter.COMPACT_POLICY(model)
    full = emitter.emit_subcircuit(model)
    text = emitter.emit_subcircuit(model, policy=pol)
    full_rows, compact_rows = _channel_rows(full), _channel_rows(text)
    assert len(full_rows) == len(compact_rows) == 30   # same Vgs/cell count
    assert {_row_points(r) for r in full_rows} == {41}
    assert {_row_points(r) for r in compact_rows} == {26}
    assert text.isascii() and len(text.splitlines()) == len(full.splitlines())
    assert text.startswith(full.splitlines()[0] + "\n")
    metrics = emitter.emission_metrics(model, pol)
    assert metrics["compact"] is True and metrics["rel_tol"] == 0.005
    assert metrics["vgs_points"] == len(compact_rows)
    assert metrics["table_points"] == 26
    assert metrics["max_abs_error_a"] > 0.0
    assert metrics["max_rel_error_pct"] <= 0.5        # 0.005 * 100


def test_compact_channel_table_reports_measured_error_and_fails_closed():
    model = _toy_model()
    g, d = emitter.grid_axes(model)
    table = emitter.grid_channel(model, g, d)
    dc, tc, abs_err, rel_err = emitter.compact_channel_table(
        model, g, d, table, 0.005)
    assert dc[0] == d[0] and dc[-1] == d[-1] and np.all(np.diff(dc) > 0)
    assert tc.shape == (len(g), len(dc)) and 2 <= len(dc) < len(d)
    assert np.allclose(tc[:, 0], table[:, 0])
    assert np.allclose(tc[:, -1], table[:, -1])
    interp = np.array([np.interp(d, dc, row) for row in tc])
    assert np.isclose(np.max(np.abs(table - interp)), abs_err)
    assert np.isclose(rel_err, abs_err / np.max(np.abs(table)))
    assert rel_err <= 0.005
    for bad_tol in (0.0, -1e-3, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            emitter.compact_channel_table(model, g, d, table, bad_tol)
    with pytest.raises(ValueError):
        emitter.compact_channel_table(model, g, d[:1], table[:, :1], 1e-3)
    broken = table.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError):
        emitter.compact_channel_table(model, g, d, broken, 1e-3)
    with pytest.raises(ValueError):
        emitter.emit_subcircuit(
            model, policy=emitter.COMPACT_POLICY(model, rel_tol=0.0))


def test_tier3_compact_config_is_opt_in_and_recorded(monkeypatch):
    model = _toy_model()
    base_cfg = load_input(model.vendor)

    def _patched_load_input(vendor):
        cfg = dict(base_cfg)
        cfg["tier3"] = {"compact_emission": True, "compact_rel_tol": 0.002}
        return cfg

    monkeypatch.setattr("pipeline.tier3.load_input", _patched_load_input)
    tier3_cfg = _patched_load_input(model.vendor)["tier3"]
    pol, source = _emission_settings(model.vendor, model, tier3_cfg)
    assert pol.compact is True and pol.compact_rel_tol == 0.002
    assert source == "config"
    text = emitter.emit_subcircuit(model, policy=pol)
    record = _emission_record(model, pol, text, source)
    assert record["compact"] is True and record["source"] == "config"
    assert record["rel_tol"] == 0.002
    assert record["table_points"] == 28 and record["vgs_points"] == 30
    assert 0.0 < record["max_rel_error_pct"] <= 0.2
    assert record["estimated_literal_count"] > 0

    default_pol, default_source = _emission_settings(
        model.vendor, model, {})
    assert default_pol.compact is False and default_source == "default"
    default_text = emitter.emit_subcircuit(model)
    assert emitter.emit_subcircuit(model, policy=default_pol) == default_text
    default = _emission_record(model, default_pol, default_text,
                               default_source)
    assert default["compact"] is False and default["source"] == "default"
    assert default["table_points"] == 41
    assert default["max_rel_error_pct"] == 0.0
    assert default["estimated_literal_count"] > \
        record["estimated_literal_count"]

    impossible = dict(base_cfg)
    impossible["tier3"] = {"compact_emission": True, "compact_rel_tol": 0.0}
    with pytest.raises(ValueError):
        _emission_settings(model.vendor, model, impossible["tier3"])


def test_v2_policy_emits_parasitics_diode_and_monotone_gate_charge(monkeypatch):
    from models.netlist_builder import build_tier2_subcircuit

    spec = _simplis_spec()
    monkeypatch.setattr("pipeline.fitting.build_pwl_spec",
                        lambda vendor: spec)
    model = _v2_model()
    text = emitter.emit_subcircuit(model, name="TOY_V2",
                                   policy=emitter.V2_POLICY)
    lines = text.splitlines()
    assert text.isascii()
    assert lines[0] == ".subckt TOY_V2 d g s"
    assert lines[-1] == ".ends"
    assert "RG g gi 1.500000000e+00" in lines
    assert "RD d di 1.000000000e-04" in lines
    assert "RS s ss 1.000000000e-04" in lines
    assert any(ln.startswith("CTOY_V2_QOSS di ss Q={table(limit(V(di,ss),")
               for ln in lines)
    assert any(ln.startswith("G_CH di ss VALUE={") for ln in lines)
    assert any(ln.startswith("CTOY_V2_QG gi ss Q={table(limit(V(gi,ss),")
               for ln in lines)
    diode = next(ln for ln in lines if ln.startswith(".MODEL D_BD_"))
    body = next(ln for ln in lines if ln.startswith("D_BODY "))
    # The diode comes from the Tier-2 builder, not a second copy of the math.
    tier2 = build_tier2_subcircuit(
        spec, subckt_name=f"T3_{model.vendor.upper()}")
    assert diode in tier2
    assert body in tier2
    assert body == f"D_BODY ss di D_BD_T3_{model.vendor.upper()}"
    # Qg is monotone and anchored at Q(0 V) = 0.
    qg_v, qg_q = _qg_pairs(text)
    assert np.all(np.diff(qg_q) >= 0.0)
    assert np.isclose(qg_q[0], 0.0)
    assert np.allclose(qg_v, model.qg_v)


def test_v2_refuses_non_monotone_or_missing_gate_charge(monkeypatch):
    monkeypatch.setattr("pipeline.fitting.build_pwl_spec",
                        lambda vendor: _simplis_spec())
    bad = _v2_model(qg_q=[0.0, 2e-9, 1e-9, 3e-9])
    with pytest.raises(ValueError, match="non-monotone"):
        emitter.emit_subcircuit(bad, name="TOY_V2",
                                policy=emitter.V2_POLICY)
    missing = _v2_model()
    missing.qg_v = missing.qg_q = None
    with pytest.raises(ValueError, match="gate-charge table is unavailable"):
        emitter.emit_subcircuit(missing, name="TOY_V2",
                                policy=emitter.V2_POLICY)


def test_v2_requires_parasitic_nodes_for_gate_charge():
    model = _v2_model()
    policy = emitter.EmissionPolicy(include_gate_charge=True)
    with pytest.raises(ValueError, match="include_parasitics"):
        emitter.emit_subcircuit(model, name="TOY_V2", policy=policy)


def test_emitter_uses_charge_surface_when_present():
    model = _toy_model()
    vds = np.linspace(0.0, 10.0, 11)
    vgs = np.array([0.0, 2.0, 4.0])
    q = (5e-10 * (1.0 - np.exp(-vds / 4.0))[None, :]
         * (1.0 + 0.2 * vgs)[:, None])
    model.qoss_surface = ChargeSurface(vds=vds, vgs=vgs, q=q)
    text = emitter.emit_subcircuit(model, name="TOY_T3")
    line = next(ln for ln in text.splitlines() if "_QOSS " in ln)
    nums = [float(x) for x in
            re.findall(r"[-+]?\d\.\d+e[-+]\d+", line)]
    table_v, table_q = np.asarray(nums[2::2]), np.asarray(nums[3::2])
    # Table endpoints are exact surface slices, so compare there (the
    # interior is a 160-point resampling of the measured surface).
    assert np.isclose(table_v[-1], 10.0)
    assert np.isclose(table_q[-1], q[0, -1])


def test_tier3_selection_callback_is_the_emitted_static_rmse():
    from pipeline.tier3 import _emitted_selection_error
    model = _toy_model()
    assert _emitted_selection_error(model) == emitter.emitted_static_rmse(model)


def _synthetic_static_family(vendor):
    levels = np.array([0.0, 2.0, 4.0])
    grid = np.array([0.0, 5.0, 10.0])
    g, d = np.meshgrid(levels, grid, indexing="ij")
    u = np.vstack([g.ravel(), d.ravel()])
    return u, 20.0 * g.ravel() + 5.0 * d.ravel(), np.repeat(levels, grid.size)


def _synthetic_trajectory(vds_scale: float = 1.0) -> Trajectory:
    """Two-pulse DPT-shaped trajectory from an exactly known terminal law."""
    n = 241
    t = np.linspace(0.0, 4.0e-6, n)
    edge = 4.0e-8

    def pulse(t_on: float, width: float) -> np.ndarray:
        return (np.clip((t - t_on) / edge, 0.0, 1.0)
                - np.clip((t - (t_on + width)) / edge, 0.0, 1.0))

    vgs = 5.0 * (pulse(0.3e-6, 0.7e-6) + pulse(2.2e-6, 0.6e-6))
    vds = vds_scale * (12.0 - 4.0 * np.clip((t - 0.3e-6) / 1.2e-6,
                                            0.0, 1.0))
    id_ = 20.0 * vgs + 5.0 * vds
    ig = 1e-9 * np.gradient(vgs, t)
    return Trajectory(
        vendor="csd16412q5a", time=t,
        state=np.zeros((1, n)), state_names=["V(xdut:1)"],
        input=np.vstack([vgs, vds]), output=np.vstack([id_, ig]),
        pin_roles=["drain", "gate", "source"], subcircuit="TOY",
        result=None, edge_time=edge)


def test_build_model_uses_emitted_selection_error_when_supplied(monkeypatch):
    monkeypatch.setattr("approximation.static_curves.static_samples",
                        _synthetic_static_family)
    traj = _synthetic_trajectory()
    base = tpwl.build_model(traj, s_grid=(1, 2))
    assert base.meta["selection_error_source"] == "static_rmse_rel"
    assert "emitted_static_rmse_rel" not in base.meta

    calls = []

    def selection(cand):
        calls.append(cand.meta["cells"])
        return {1: 9.0, 2: 0.0}[cand.meta["cells"]]

    model = tpwl.build_model(traj, s_grid=(1, 2),
                             selection_error_fn=selection)
    assert model.meta["cells"] == 2
    assert model.meta["selection_error_source"] == "emitted_static_rmse_rel"
    assert model.meta["emitted_static_rmse_rel"] == 0.0
    assert model.meta["static_rmse_rel"] is not None
    assert model.meta["selection_error_rel"] == 0.0
    assert calls == [1, 2]


def test_extra_trajectories_concatenate_dpt_samples_and_share_static_family(
        monkeypatch):
    monkeypatch.setattr("approximation.static_curves.static_samples",
                        _synthetic_static_family)
    primary = _synthetic_trajectory()
    extra = _synthetic_trajectory(vds_scale=0.5)
    split = tpwl._split_index(primary.input[0])

    single = tpwl.build_model(primary, s_grid=(2, 3))
    assert single.meta["n_trajectories"] == 1
    assert single.meta["trajectory_sources"] == ["csd16412q5a:primary"]
    assert single.meta["dpt_samples"] == split
    assert single.meta["static_samples"] == 9

    both = tpwl.build_model(primary, s_grid=(2, 3),
                            extra_trajectories=[extra])
    assert both.meta["n_trajectories"] == 2
    assert both.meta["trajectory_sources"] == [
        "csd16412q5a:primary", "csd16412q5a:training:1"]
    assert both.meta["dpt_samples"] == 2 * split
    assert both.meta["static_samples"] == 9
    assert both.meta["holdout_id_rmse_rel"] == "not_measured"
    assert both.meta["cv_status"] == "not_measured"


def test_holdout_trajectory_is_blocked_and_reported(monkeypatch):
    monkeypatch.setattr("approximation.static_curves.static_samples",
                        _synthetic_static_family)
    primary = _synthetic_trajectory()
    extra = _synthetic_trajectory(vds_scale=0.5)
    split = tpwl._split_index(primary.input[0])

    model = tpwl.build_model(primary, s_grid=(2, 3),
                             extra_trajectories=[extra],
                             holdout_trajectory=True)
    assert model.meta["cv_status"] == "holdout"
    assert model.meta["n_trajectories"] == 2
    assert "holdout" in model.meta["trajectory_sources"][-1]
    # The held-out capture is not fitted: only the primary DPT samples fit.
    assert model.meta["dpt_samples"] == split
    assert isinstance(model.meta["holdout_id_rmse_rel"], float)
    assert isinstance(model.meta["holdout_qg_rmse_rel"], float)
    assert model.meta["holdout_id_rmse_rel"] >= 0.0
    assert model.meta["holdout_qg_rmse_rel"] >= 0.0

    blocked = tpwl.build_model(primary, s_grid=(2, 3),
                               holdout_trajectory=True)
    assert blocked.meta["holdout_id_rmse_rel"] == "not_measured"
    assert blocked.meta["holdout_qg_rmse_rel"] == "not_measured"
    assert blocked.meta["cv_status"] == "not_measured"
    assert "no extra_trajectories" in blocked.meta["cv_reason"]


def _charge_surface(*, with_temperature: bool = False) -> ChargeSurface:
    vds = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    vgs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    q = (1e-9 * (1.0 - np.exp(-vds / 3.0))[None, :]
         * (1.0 + 0.1 * vgs)[:, None])
    if not with_temperature:
        return ChargeSurface(vds=vds, vgs=vgs, q=q)
    return ChargeSurface(vds=vds, vgs=vgs, q=np.stack([q, 1.1 * q]),
                         temperature_c=np.array([25.0, 125.0]))


def test_charge_surface_bilinear_interpolation_and_capacitance():
    surface = _charge_surface()
    # Nodes reproduce the table.
    assert np.isclose(surface.charge(4.0, 2.0), surface.q[2, 2])
    # Cell centre is the mean of its four corners.
    expected = np.mean([surface.q[1, 1], surface.q[1, 2],
                        surface.q[2, 1], surface.q[2, 2]])
    assert np.isclose(surface.charge(3.0, 1.5), expected)
    # Monotone charge and positive capacitance over the measured box.
    vds = np.linspace(0.0, 10.0, 101)
    q = surface.charge(vds, 2.0)
    assert np.all(np.diff(q) >= 0.0)
    c = surface.capacitance(vds[:-1], 2.0)
    assert np.all(np.asarray(c) > 0.0)
    v, qv = surface.qoss_table()
    assert np.array_equal(v, surface.vds)
    assert np.all(np.diff(qv) >= 0.0)


def test_charge_surface_fails_closed_on_bad_axes_and_queries():
    surface = _charge_surface()
    with pytest.raises(ValueError, match="outside the measured vds range"):
        surface.charge(-1.0, 2.0)
    with pytest.raises(ValueError, match="outside the measured vgs range"):
        surface.charge(1.0, 6.0)
    with pytest.raises(ValueError, match="non-monotone in Vds"):
        ChargeSurface(vds=surface.vds, vgs=surface.vgs,
                      q=np.array(surface.q, copy=True)
                      @ np.eye(surface.vds.size)[:, ::-1])
    with pytest.raises(ValueError, match="strictly increasing"):
        ChargeSurface(vds=np.array([0.0, 0.0, 1.0]), vgs=surface.vgs,
                      q=surface.q[:, :3])
    with pytest.raises(ValueError, match="temperature is required"):
        _charge_surface(with_temperature=True).charge(1.0, 1.0)
    with pytest.raises(ValueError, match="temperature range"):
        _charge_surface(with_temperature=True).charge(1.0, 1.0,
                                                      temperature=200.0)


def test_charge_surface_temperature_slice_is_interpolated():
    surface = _charge_surface(with_temperature=True)
    cold = surface.charge(3.0, 1.5, temperature=25.0)
    hot = surface.charge(3.0, 1.5, temperature=125.0)
    mid = surface.charge(3.0, 1.5, temperature=75.0)
    assert np.isclose(mid, 0.5 * (cold + hot))
    assert np.all(surface.capacitance(np.linspace(0.0, 10.0, 7), 2.0,
                                      temperature=75.0) > 0.0)


# ---- SIMPLIS Level-2 charge-table variant (offline, no LTspice) ------------
from common.types import PWLModelSpec
from models.simplis_netlist_builder import (
    build_simplis_subcircuit,
    cgs_linearity_decision,
    charge_ramp_terms,
)


def _simplis_spec(*, linear_cgs: bool = False) -> PWLModelSpec:
    """Small deterministic spec carrying the three (V,Q) charge tables."""
    v = [0.1, 1.0, 10.0, 25.0]
    if linear_cgs:
        cgs_q = [0.0, 2e-9, 20e-9, 50e-9]  # exact chord
    else:
        cgs_q = [0.0, 3e-9, 12e-9, 26e-9]
    return PWLModelSpec(
        device_name="TOY",
        tier=3,
        breakpoints_vds=[0.0, 1.0, 10.0],
        breakpoints_id=[[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]],
        vgs_values=[4.0, 10.0],
        q_iss_vs_v=v, q_iss_charge_c=cgs_q,
        q_oss_vs_v=v, q_oss_charge_c=[0.0, 4e-9, 20e-9, 50e-9],
        q_rss_vs_v=v, q_rss_charge_c=[0.0, 1e-9, 5e-9, 12e-9],
        rds_on=0.01, v_th=1.0,
        body_diode_vsd=[0.5, 0.8],
        body_diode_isd=[1e-6, 1e-3],
        metadata={"r_gate": 1.5, "r_source": 1e-4, "r_drain": 1e-4},
    )


def test_simplis_charge_ramp_terms_are_monotonic():
    """Delta slopes integrate back to a non-decreasing Q at every point."""
    spec = _simplis_spec()
    for name, pairs in (
        ("CGS", list(zip(spec.q_iss_vs_v, spec.q_iss_charge_c))),
        ("CDS", list(zip(spec.q_oss_vs_v, spec.q_oss_charge_c))),
        ("CDG", list(zip(spec.q_rss_vs_v, spec.q_rss_charge_c))),
    ):
        terms = list(charge_ramp_terms(pairs))
        assert len(terms) == len(pairs) - 1, name
        slopes = [d for _, d in terms]
        assert all(d >= 0 or i > 0 for i, d in enumerate(slopes)), name
        rebuilt = [pairs[0][1]]
        for (v_prev, delta), (v_i, q_i) in zip(terms, pairs[1:]):
            prev_q = rebuilt[-1]
            prev_slope = 0.0 if len(rebuilt) == 1 else (
                (pairs[len(rebuilt) - 1][1] - pairs[len(rebuilt) - 2][1])
                / (pairs[len(rebuilt) - 1][0] - pairs[len(rebuilt) - 2][0]))
            rebuilt.append(prev_q + (prev_slope + delta) * (v_i - v_prev))
            assert np.isclose(rebuilt[-1], q_i), name
        assert np.all(np.diff(rebuilt) >= 0), name


def test_simplis_emission_is_ascii_and_ports_match_tier2():
    text = build_simplis_subcircuit(_simplis_spec(), cgs_axis="vgs")
    assert text.isascii()
    lines = text.splitlines()
    assert ".SUBCKT TOY_PIECESPICE_T3_simplis d g s" in lines
    assert ".ENDS TOY_PIECESPICE_T3_simplis" in lines
    for expected in ("RG g gi ", "RD d di ", "RS s ss ",
                     "G_CH di ss VALUE", "D_BODY ss di ",
                     "CGS gi ss ", "CDS di ss Q=", "CDG di gi Q="):
        assert any(line.startswith(expected) for line in lines), expected
    for removed in ("C_QOSS ", "C_ISS ", "C_RSS "):
        assert not any(line.startswith(removed) for line in lines), removed
    charges = [line.split()[0] for line in lines
               if line.split() and line.split()[0] in ("CGS", "CDS", "CDG")]
    assert charges == ["CGS", "CDS", "CDG"]


def test_simplis_cgs_linearity_decision_is_deterministic():
    v = [0.1, 1.0, 10.0, 25.0]
    linear = cgs_linearity_decision(v, [0.0, 2e-9, 20e-9, 50e-9])
    nonlinear = cgs_linearity_decision(v, [0.0, 3e-9, 12e-9, 26e-9])
    assert linear.is_linear and not nonlinear.is_linear
    assert linear == cgs_linearity_decision(v, [0.0, 2e-9, 20e-9, 50e-9])
    assert nonlinear.chord_deviation > linear.chord_deviation
    assert linear.as_dict()["mode"] == "linear"


def test_simplis_linear_cgs_emits_plain_capacitor():
    text = build_simplis_subcircuit(_simplis_spec(linear_cgs=True))
    cgs_lines = [x for x in text.splitlines() if x.startswith("CGS ")]
    assert len(cgs_lines) == 1 and "Q=" not in cgs_lines[0]
    text_pwl = build_simplis_subcircuit(_simplis_spec(), cgs_axis="vgs")
    cgs_pwl = [x for x in text_pwl.splitlines() if x.startswith("CGS ")]
    assert len(cgs_pwl) == 1 and "Q={" in cgs_pwl[0]


def test_simplis_emits_rgs_and_truthful_extrapolation_comment():
    text = build_simplis_subcircuit(_simplis_spec(), cgs_axis="vgs")
    lines = text.splitlines()
    assert len([x for x in lines if x.startswith("RGS gi ss 1e7")]) == 1
    assert "lookups end-clamp" not in " ".join(lines)
    assert any("extrapolate linearly with the last segment" in x
               for x in lines)
    tier2 = build_simplis_subcircuit(_simplis_spec(), charge_mode="tier2")
    t2_lines = tier2.splitlines()
    assert any(x.startswith("C_QOSS ") for x in t2_lines)
    assert any(x.startswith("C_ISS ") for x in t2_lines)
    assert any(x.startswith("C_RSS ") for x in t2_lines)
    assert not any(x.split() and x.split()[0] in ("CGS", "CDS", "CDG")
                   for x in t2_lines)
    assert len([x for x in t2_lines if x.startswith("RGS gi ss 1e7")]) == 1
    try:
        build_simplis_subcircuit(_simplis_spec(), charge_mode="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown charge_mode must fail")


from pipeline.tier3_simplis import (  # noqa: E402
    _CHANNEL_RULE,
    _CV_DATA_LIMITATION,
    _FULL_CV_KEYS,
    _cap_curve_to_vbus,
    _cv_sweep_from_extracted,
    _decompose_cv,
    _fidelity_metrics,
    _fit_anchored_charge,
    _fit_charge_tables,
    _full_cv_from_curve4,
    _load_cv_sweep,
    _on_state_breakpoint,
    _select_charge_candidate,
    _table_error,
)


def test_simplis_charge_fit_anchors_at_zero_and_meets_budget():
    v = np.linspace(0.1, 25.0, 40)
    c = 1e-9 * (1 + 3 * np.exp(-((v - 5) / 0.5) ** 2)
                + 3 * np.exp(-((v - 12) / 0.4) ** 2))
    fit = _fit_anchored_charge(v, c, 0.005)
    assert fit.pairs[0] == (0.0, 0.0)
    assert np.isclose(fit.q_full[1] / fit.v_full[1], c[0])
    assert np.isclose(fit.pairs[-1][0], v[-1])
    assert fit.max_abs_error_c <= fit.error_budget_c
    assert len(fit.pairs) > 5  # the old 5-point cap would have truncated


def test_simplis_charge_fit_uses_voltage_axis_not_indices():
    """Greedy breakpoints are voltages; an index axis blows the Q budget."""
    v = np.linspace(0.1, 60.0, 40)
    c = 1e-9 * (1 + 3 * np.exp(-((v - 5.0) / 0.5) ** 2)
                + 4 * np.exp(-((v - 22.0) / 1.2) ** 2))
    fit = _fit_anchored_charge(v, c, 0.005)
    volts = np.array([p[0] for p in fit.pairs], dtype=float)

    # Every breakpoint is an actual va voltage. The 60 V endpoint is far
    # outside the 0..len(va)-1 index range, so an index can never emit it.
    assert np.all(np.isin(volts, fit.v_full))
    assert volts[0] == 0.0
    assert np.isclose(volts[-1], 60.0)
    assert volts[-1] > len(fit.v_full) - 1
    assert np.all(np.diff(volts) > 0)

    # Error is measured on the full voltage axis, inside the unchanged budget.
    assert _table_error(fit, fit.pairs) <= fit.error_budget_c
    assert fit.max_abs_error_c <= fit.error_budget_c

    # Counterfactual: the same breakpoints interpreted as np.interp indices
    # (the regression operand) cannot meet the 0.5% budget on the volts axis.
    idx = np.searchsorted(fit.v_full, volts)
    index_axis_err = float(np.max(np.abs(
        fit.q_full - np.interp(fit.v_full, idx, fit.q_full[idx]))))
    assert index_axis_err > fit.error_budget_c


def test_simplis_charge_fit_forced_breakpoint_stays_on_voltage_axis():
    """forced_v/forbid_below still select voltage breakpoints."""
    v = np.linspace(0.1, 40.0, 60)
    c = 1e-9 * (1 + 3 * np.exp(-((v - 20.0) / 0.8) ** 2)
                + 4 * np.exp(-((v - 32.0) / 1.0) ** 2))
    forced = 12.0
    fit = _fit_anchored_charge(v, c, 0.005,
                               forced_v=forced, forbid_below=forced)
    volts = np.array([p[0] for p in fit.pairs], dtype=float)

    assert np.any(np.isclose(volts, forced))
    assert np.all(volts[1:] >= forced - 1e-12)
    assert np.all(np.isin(volts, fit.v_full))
    assert _table_error(fit, fit.pairs) <= fit.error_budget_c


def test_simplis_cgs_pwl_fails_closed_on_vds_axis():
    """A Vds-swept table must not be emitted as PWL on the Vgs axis."""
    with pytest.raises(ValueError, match="axis"):
        build_simplis_subcircuit(_simplis_spec())
    with pytest.raises(ValueError, match="cgs_axis"):
        build_simplis_subcircuit(_simplis_spec(), cgs_axis="drain")
    declared = build_simplis_subcircuit(_simplis_spec(), cgs_axis="vgs")
    assert any(line.startswith("CGS gi ss Q={")
               for line in declared.splitlines())
    # LINEAR vendors are unchanged by the new axis declaration.
    linear = build_simplis_subcircuit(_simplis_spec(linear_cgs=True))
    cgs_lines = [x for x in linear.splitlines() if x.startswith("CGS ")]
    assert len(cgs_lines) == 1 and "Q=" not in cgs_lines[0]


def test_simplis_charge_fit_bound_reports_achieved_error_ratio():
    """An unmeetable budget raises with the achieved ratio, never hangs."""
    v = np.array([0.1, 1.0, 2.0, 5.0])
    c = np.array([1e-9, 2.5e-9, 1.2e-9, 0.5e-9])
    with pytest.raises(RuntimeError, match="error ratio"):
        _fit_anchored_charge(v, c, -1.0)


def test_simplis_forced_on_state_breakpoint_replaces_lower_points():
    v = np.linspace(0.1, 12.5, 30)
    c = np.full_like(v, 2e-10)
    bp = _on_state_breakpoint(
        [0.0, 10.0, 10.0, 10.0],
        [12.5, 0.5, 0.1725, 0.1], vgs_on=10.0, vbus=12.5)
    assert np.isclose(bp["measured_v"], 0.5)
    assert not bp["clamped_to_vbus"]
    fit = _fit_anchored_charge(
        v, c, 0.005, forced_v=bp["used_v"],
        forbid_below=bp["used_v"])
    volts = [p[0] for p in fit.pairs]
    assert any(np.isclose(x, bp["used_v"]) for x in volts)
    assert min(volts[1:]) >= bp["used_v"] - 1e-12
    assert fit.max_abs_error_c <= fit.error_budget_c

    clamped = _on_state_breakpoint(
        [0.0, 10.0], [38.9, 38.9], vgs_on=10.0, vbus=38.0)
    assert clamped["clamped_to_vbus"] and clamped["used_v"] == 38.0
    grid, curves = _cap_curve_to_vbus(
        np.array([0.1, 10.0, 60.0]), (np.ones(3),), 38.0)
    assert np.isclose(grid[-1], 38.0) and curves[0][-1] == 1.0


# ---- Full Curve-4 C(V) arrays -> simplis charge tables (offline) -----------
def _synthetic_full_cv(n: int = 60):
    """Non-uniform Vds grid with smooth, structured Ciss/Coss/Crss."""
    v = 0.1 + 24.9 * (np.linspace(0.0, 1.0, n) ** 2.0)
    crss = 3e-11 * (1.0 + np.exp(-v / 2.0))
    coss = 6e-10 / np.sqrt(1.0 + v / 3.0) + crss
    ciss = 9e-10 / np.sqrt(1.0 + v / 10.0) + crss
    return v, ciss, coss, crss


def _full_cv_extracted(v, ciss, coss, crss):
    return {
        "curve_vds": np.asarray(v).tolist(),
        "ciss_curve_f": np.asarray(ciss).tolist(),
        "coss_curve_f": np.asarray(coss).tolist(),
        "crss_curve_f": np.asarray(crss).tolist(),
    }


def test_simplis_full_cv_arrays_use_vds_axis_and_meet_charge_budget():
    v, ciss, coss, crss = _synthetic_full_cv()
    extracted = _full_cv_extracted(v, ciss, coss, crss)
    assert set(_FULL_CV_KEYS).issubset(extracted)
    sweep = _cv_sweep_from_extracted(extracted, "TOY")
    assert sweep is not None
    assert np.allclose(sweep.v, v) and np.allclose(sweep.coss, coss)
    assert (v[1] - v[0]) < 0.1 * (v[-1] - v[-2])  # genuinely non-uniform

    tables = _fit_charge_tables(
        sweep.v, sweep.ciss, sweep.coss, sweep.crss, vbus=25.0)
    for name, fit in tables.fits.items():
        volts = np.array([p[0] for p in fit.pairs])
        assert volts[0] == 0.0
        assert np.all(np.isin(volts[1:], fit.v_full))  # voltages, not indices
        assert np.isclose(volts[-1], v[-1])
        assert np.all(np.diff(volts) > 0)
        assert fit.max_abs_error_c <= fit.error_budget_c
        assert _table_error(fit, fit.pairs) <= fit.error_budget_c

    # Regression operand: the same breakpoints read as indices miss the 0.5%
    # budget on this non-uniform voltage axis.
    fit = tables.fits["cds"]
    volts = np.array([p[0] for p in fit.pairs])
    idx = np.searchsorted(fit.v_full, volts)
    index_err = float(np.max(np.abs(
        fit.q_full - np.interp(fit.v_full, idx, fit.q_full[idx]))))
    assert index_err > fit.error_budget_c


def test_simplis_full_cv_fails_closed_on_bad_axis_or_capacitance():
    v, ciss, coss, crss = _synthetic_full_cv(20)
    bad = _full_cv_extracted(v, ciss, coss, crss)
    bad["curve_vds"] = bad["curve_vds"][::-1]
    with pytest.raises(ValueError, match="ascending"):
        _cv_sweep_from_extracted(bad, "TOY")

    bad = _full_cv_extracted(v, ciss, coss, crss)
    bad["curve_vds"][3] = bad["curve_vds"][2]
    with pytest.raises(ValueError, match="ascending"):
        _cv_sweep_from_extracted(bad, "TOY")

    bad = _full_cv_extracted(v, ciss, coss, crss)
    bad["coss_curve_f"] = (-np.asarray(bad["coss_curve_f"])).tolist()
    with pytest.raises(ValueError, match="non-negative"):
        _cv_sweep_from_extracted(bad, "TOY")

    bad = _full_cv_extracted(v, ciss, coss, crss)
    del bad["crss_curve_f"]
    assert _cv_sweep_from_extracted(bad, "TOY") is None


def _non_charge_lines(text):
    """Channel/diode/parasitic lines must not move with charge data."""
    out, inside = [], False
    for line in text.splitlines():
        if line.startswith("* --- SIMPLIS Level-2 three-capacitor charge model"):
            inside = True
            continue
        if line.startswith("* --- body diode (Curve 6) ---"):
            inside = False
        if not inside:
            out.append(line)
    return out


def test_simplis_full_cv_tables_zero_anchored_monotone_and_form_unchanged():
    v, ciss, coss, crss = _synthetic_full_cv()
    tables = _fit_charge_tables(v, ciss, coss, crss, vbus=25.0)
    for name, pairs in tables.pairs.items():
        assert pairs[0] == (0.0, 0.0), name
        assert np.all(np.diff([p[0] for p in pairs]) > 0), name
        assert np.all(np.diff([p[1] for p in pairs]) >= 0), name
        assert list(charge_ramp_terms(pairs)), name

    rich = _simplis_spec()
    for key, axis_attr, q_attr in (
        ("cds", "q_oss_vs_v", "q_oss_charge_c"),
        ("cgs", "q_iss_vs_v", "q_iss_charge_c"),
        ("cdg", "q_rss_vs_v", "q_rss_charge_c"),
    ):
        setattr(rich, axis_attr, [p[0] for p in tables.pairs[key]])
        setattr(rich, q_attr, [p[1] for p in tables.pairs[key]])
    base_text = build_simplis_subcircuit(_simplis_spec(), cgs_axis="vgs")
    rich_text = build_simplis_subcircuit(rich, cgs_axis="vgs")
    assert rich_text.isascii()
    assert _non_charge_lines(rich_text) == _non_charge_lines(base_text)
    for expected in (".SUBCKT TOY_PIECESPICE_T3_simplis d g s",
                     "RG g gi ", "RD d di ", "RS s ss ",
                     "G_CH di ss VALUE", "D_BODY ss di ",
                     "CGS gi ss ", "CDS di ss Q=", "CDG di gi Q=",
                     "RGS gi ss 1e7"):
        assert any(line.startswith(expected)
                   for line in rich_text.splitlines()), expected


class _FakeCVPoint:
    def __init__(self, v, c):
        self.x_data = v
        self.y_data = c


class _FakeCapBench:
    def __init__(self, v, curves):
        self._v, self._curves = v, curves

    def run(self, *args, **kwargs):
        return tuple(_FakeCVPoint(self._v, c) for c in self._curves)


def test_simplis_full_cv_source_prefers_persisted_arrays(monkeypatch):
    v, ciss, coss, crss = _synthetic_full_cv(12)
    sweep = _cv_sweep_from_extracted(
        _full_cv_extracted(v, ciss, coss, crss), "TOY")
    monkeypatch.setattr(
        "pipeline.tier3_simplis._full_cv_from_curve4", lambda vendor: sweep)
    out = _load_cv_sweep("TOY", {}, {})
    assert out[4] == "curve4_full_arrays" and out[5] is None
    assert np.allclose(out[0], v) and np.allclose(out[2], coss)


def test_simplis_full_cv_fallback_discloses_scalar_only_record(monkeypatch):
    scalar_only = {"ciss_f": 4e-10, "coss_f": 5e-10,
                   "crss_f": 3e-11, "cap_ref_vds": 6.0}
    assert _cv_sweep_from_extracted(scalar_only, "TOY") is None

    v = np.linspace(0.1, 25.0, 6)
    benches = []

    def _bench_factory(**kw):
        bench = _FakeCapBench(v, (np.full_like(v, 4e-10),
                                  np.full_like(v, 5e-10),
                                  np.full_like(v, 3e-11)))
        benches.append(bench)
        return bench

    monkeypatch.setattr(
        "pipeline.tier3_simplis._full_cv_from_curve4", lambda vendor: None)
    monkeypatch.setattr(
        "pipeline.tier3_simplis.CapacitanceTestbench", _bench_factory)
    cfg = {"library": "unused.lib", "subcircuit": "TOY",
           "pins": {"pin_order": ["d", "g", "s"]}}
    cond = {"F_meas": 1e6, "Vds_cap_min": 0.1, "Vds_cap_max": 25.0,
            "Cap_points": 6}
    got_v, _, got_coss, _, source, limit = _load_cv_sweep("TOY", cfg, cond)
    assert source == "live_ac_sweep" and len(benches) == 1
    assert np.allclose(got_v, v) and np.allclose(got_coss, 5e-10)
    assert limit == _CV_DATA_LIMITATION
    assert "not persisted" in limit and "scalar" in limit
    assert "No full C(V) sweep was synthesized" in limit


def _candidate(vds, id_, tr, tf, e_on, e_off) -> dict:
    comparison = {
        key: {"tier3_rel_err_pct": err}
        for key, err in (("tr_ns", tr), ("tf_ns", tf),
                         ("e_on_uj", e_on), ("e_off_uj", e_off))}
    # The residual gate checks trace_points as a completion guard before it
    # compares fidelity; give every synthetic candidate a real point count
    # so a rejection reason names the failing metric, not the guard.
    return {"fidelity": _fidelity_metrics(
        {"vds_rel_rmse_pct": vds, "id_rel_rmse_pct": id_}, comparison),
        "dpt_run": {"completed": True, "trace_points": 100}}


def test_simplis_candidate_selection_requires_strict_all_metric_gain():
    base = _candidate(6.0, 0.05, 29.0, 47.0, 51.0, 1.3)
    better = _candidate(3.8, 0.045, 20.0, 35.2, 29.1, 1.25)
    cands = {"tables": base, "tier2": better}
    assert _select_charge_candidate(cands)[0] == "tier2"
    tied = {**better, "fidelity": {
        **better["fidelity"],
        "abs_rel_err_pct": {**better["fidelity"]["abs_rel_err_pct"],
                            "e_off_uj": 1.3}}}
    assert _select_charge_candidate(
        {"tables": base, "tier2": tied})[0] == "tables"
    failed = _candidate(None, 0.045, 35.6, 35.2, 29.1, 1.25)
    assert _select_charge_candidate(
        {"tables": base, "tier2": failed})[0] == "tables"

from pipeline.tier3_simplis import _select_channel_candidate  # noqa: E402
from models.hybrid_channel import fit_hybrid_channel  # noqa: E402
from types import SimpleNamespace  # noqa: E402


_HYBRID_NOTE_PREFIXES = (
    "* Hybrid channel:", "* fit meta:", "* fit stop criterion:",
    "* measured limitation:",
)


def test_simplis_hybrid_fit_replaces_only_gch_and_keeps_provenance():
    spec = _simplis_spec()
    fit = fit_hybrid_channel(spec.vgs_values, spec.breakpoints_vds,
                             spec.breakpoints_id, rel_tolerance=0.005)
    base = build_simplis_subcircuit(spec, charge_mode="tier2")
    hybrid = build_simplis_subcircuit(
        spec, charge_mode="tier2", channel_fit=fit)
    assert base.isascii() and hybrid.isascii()

    base_lines = base.splitlines()
    hybrid_lines = hybrid.splitlines()
    assert sum(line.startswith(_HYBRID_NOTE_PREFIXES)
               for line in hybrid_lines) == len(_HYBRID_NOTE_PREFIXES)
    body = [line for line in hybrid_lines
            if not line.startswith(_HYBRID_NOTE_PREFIXES)]
    assert len(body) == len(base_lines)
    changed = [(a, b) for a, b in zip(base_lines, body) if a != b]
    assert len(changed) == 1, changed
    old, new = changed[0]
    assert old.startswith("G_CH di ss VALUE")
    assert new == f"G_CH di ss VALUE={fit.expression}"
    for prefix in ("RG ", "RD ", "RS ", "D_BODY ",
                   "C_QOSS ", "C_ISS ", "C_RSS "):
        assert [x for x in base_lines if x.startswith(prefix)] == [
            x for x in body if x.startswith(prefix)]
    header = "\n".join(hybrid_lines[:10])
    assert "hinge_c" in header
    assert "local://hybrid_channel_feasibility.md" in header
    assert str(fit.meta["n_knees"]) in header


def test_simplis_hybrid_rejects_non_ascii_expression():
    bad = SimpleNamespace(expression="V(gi,ss)*\u00e9", meta={},
                          max_abs_fit_error_a=0.0)
    try:
        build_simplis_subcircuit(_simplis_spec(), cgs_axis="vgs",
                                 channel_fit=bad)
    except ValueError:
        pass
    else:
        raise AssertionError("non-ASCII hybrid expression must fail")


def test_simplis_channel_selection_rejects_regressing_hybrid_dpt():
    base_tables = _candidate(6.0, 0.05, 29.0, 47.0, 51.0, 1.3)
    base_tier2 = _candidate(5.0, 0.04, 25.0, 40.0, 45.0, 1.2)
    # Measured feasibility outcome: static fit improves, DPT Vds regresses.
    cands = {
        "tables": base_tables, "tier2": base_tier2,
        "hybrid_tables": _candidate(11.20, 0.17, 30.0, 48.0, 52.0, 1.35),
        "hybrid_tier2": _candidate(7.10, 0.10, 26.0, 41.0, 46.0, 1.25),
    }
    choice, reason, evidence = _select_channel_candidate(cands)
    assert choice == "tier2"
    assert "vds_rmse_pct" in reason and "hybrid_tables" in reason
    for mode in ("hybrid_tables", "hybrid_tier2"):
        row = evidence["pairs"][mode]["metrics"]["vds_rmse_pct"]
        assert row["tier2_channel"] < row["hybrid"]
        assert not row["strict_gain"]
    # Charge selection is unchanged by the extra channel candidates.
    assert _select_charge_candidate(cands)[0] == "tier2"
    # A missing hybrid candidate can never be selected.
    assert _select_channel_candidate(
        {"tables": base_tables, "tier2": base_tier2})[0] == "tier2"

    # Even a strict fidelity win is rejected if adaptive speed was not measured.
    winning = {
        **cands,
        "hybrid_tables": _candidate(3.0, 0.02, 10.0, 20.0, 20.0, 1.0),
        "hybrid_tier2": _candidate(2.5, 0.02, 9.0, 19.0, 19.0, 1.0),
    }
    choice, reason, _ = _select_channel_candidate(winning)
    assert choice == "tier2" and "adaptive" in reason

    # Positive control: fidelity gain plus measured, no-worse adaptive speed.
    speed = {
        "tables": {"completed": True, "solver_speedup": 1.0,
                   "wall_speedup": 1.0},
        "tier2": {"completed": True, "solver_speedup": 1.1,
                  "wall_speedup": 1.05},
        "hybrid_tables": {"completed": True, "solver_speedup": 1.2,
                          "wall_speedup": 1.1},
        "hybrid_tier2": {"completed": True, "solver_speedup": 1.3,
                         "wall_speedup": 1.2},
    }
    assert _select_channel_candidate(
        winning, adaptive_results=speed)[0] == "hybrid"


def test_simplis_channel_gate_is_explicitly_fidelity_first():
    """No adaptive measurement exists at selection time; none is claimed."""
    assert "FIDELITY-FIRST" in _CHANNEL_RULE
    assert "deferred" in _CHANNEL_RULE
    winning = {
        "tables": _candidate(6.0, 0.05, 29.0, 47.0, 51.0, 1.3),
        "tier2": _candidate(5.0, 0.04, 25.0, 40.0, 45.0, 1.2),
        "hybrid_tables": _candidate(3.0, 0.02, 10.0, 20.0, 20.0, 1.0),
        "hybrid_tier2": _candidate(2.5, 0.02, 9.0, 19.0, 19.0, 1.0),
    }
    choice, reason, evidence = _select_channel_candidate(winning)
    assert choice == "tier2" and "deferred" in reason
    assert all("deferred" in row["status"]
               for row in evidence["adaptive"].values())



# ---- Evidence-gated ANN residual channel on the simplis path (offline) -----
def _synthetic_residual_fit(activation: str = "isru"):
    from approximation.ann_residual import AnnResidualFit

    return AnnResidualFit(
        coefficients={"w1": [[0.2, 0.01]], "b1": [0.0],
                      "w2": [0.5], "b2": 0.0},
        v_th=2.0, activation=activation, neurons=1,
        diagnostics={
            "train_samples": 12, "curve1_samples": 8, "dpt_samples": 4,
            "residual_rel_rmse": 0.05, "residual_rms_a": 0.1,
            "target_rms_a": 2.0, "zero_vds_max_abs_a": 0.0,
            "cutoff_attenuation": 0.01,
        },
        provenance={"target": "synthetic test fit (not measured)"},
    )


def _charge_element_lines(text: str, charge_mode: str):
    prefixes = (("CGS ", "CDS ", "CDG ", "RGS ") if charge_mode == "tables"
                else ("C_QOSS ", "C_ISS ", "C_RSS ", "RGS "))
    return [line for line in text.splitlines()
            if line.startswith(prefixes)]


def _element_nodes(text: str):
    """Every node used by a circuit element line (no internal nodes added)."""
    prefixes = ("RG ", "RD ", "RS ", "RGS ", "CGS ", "CDS ", "CDG ",
                "C_QOSS ", "C_ISS ", "C_RSS ", "G_CH ", "G_ANN ",
                "D_BODY ")
    nodes = set()
    for line in text.splitlines():
        if line.startswith(prefixes):
            nodes.update(line.split()[1:3])
    return nodes


def test_simplis_residual_adds_one_gann_after_gch_and_preserves_every_line():
    fit = _synthetic_residual_fit()
    for mode in ("tables", "tier2"):
        base = build_simplis_subcircuit(
            _simplis_spec(), charge_mode=mode, cgs_axis="vgs")
        residual = build_simplis_subcircuit(
            _simplis_spec(), charge_mode=mode, cgs_axis="vgs",
            residual_fit=fit)
        assert base.isascii() and residual.isascii()
        base_lines, res_lines = base.splitlines(), residual.splitlines()
        gch = [i for i, line in enumerate(base_lines)
               if line.startswith("G_CH ")]
        res_gch = [i for i, line in enumerate(res_lines)
                   if line.startswith("G_CH ")]
        gann = [i for i, line in enumerate(res_lines)
                if line.startswith("G_ANN ")]
        assert len(gch) == 1 and len(res_gch) == 1 and len(gann) == 1
        assert gann[0] == res_gch[0] + 1
        assert res_lines[res_gch[0]] == base_lines[gch[0]]
        assert res_lines[gann[0]].startswith("G_ANN di ss VALUE = {")
        assert res_lines[gann[0]].count("V(di,ss)") >= 1
        notes = [line for line in res_lines
                 if line.startswith("* ANN residual")]
        assert len(notes) == 2
        body = [line for line in res_lines
                if not line.startswith("* ANN residual")
                and not line.startswith("G_ANN ")]
        assert body == base_lines
        assert _charge_element_lines(base, mode) == \
            _charge_element_lines(residual, mode)
        assert _element_nodes(base) == _element_nodes(residual)


def test_simplis_residual_keeps_name_override_and_rejects_bad_fit():
    fit = _synthetic_residual_fit()
    # Toy CGS is nonlinear: declare the Vgs axis so this test exercises the
    # name override / bad-fit guard, not the unrelated CGS axis-mismatch guard.
    custom = build_simplis_subcircuit(
        _simplis_spec(), residual_fit=fit, cgs_axis="vgs",
        subckt_name="CUSTOM")
    assert ".SUBCKT CUSTOM d g s" in custom.splitlines()
    assert ".ENDS CUSTOM" in custom.splitlines()
    assert "D_BD_CUSTOM" in custom
    named = build_simplis_subcircuit(
        _simplis_spec(), residual_fit=fit, cgs_axis="vgs", name="NAMED")
    assert ".SUBCKT NAMED d g s" in named.splitlines()
    with pytest.raises(TypeError):
        build_simplis_subcircuit(
            _simplis_spec(), residual_fit=object(), cgs_axis="vgs")


from pipeline.tier3_simplis import _select_residual_candidate  # noqa: E402


def test_simplis_residual_gate_requires_strict_dpt_gain_and_complete_run():
    base = _candidate(6.0, 0.05, 29.0, 47.0, 51.0, 1.3)
    worse = _candidate(6.5, 0.06, 30.0, 48.0, 52.0, 1.4)
    choice, reason, evidence = _select_residual_candidate(
        {"tables": base, "tier2+ann_residual": worse}, charge_mode="tables")
    assert choice == "tier2" and "rejected" in reason
    assert "vds_rmse_pct" in reason and evidence["strict_gain"] is False
    tied = {**worse, "fidelity": {
        **worse["fidelity"],
        "abs_rel_err_pct": {**worse["fidelity"]["abs_rel_err_pct"],
                            "e_off_uj": 1.3}}}
    assert _select_residual_candidate(
        {"tables": base, "tier2+ann_residual": tied},
        charge_mode="tables")[0] == "tier2"
    failed = {**worse, "dpt_run": {"completed": False, "trace_points": 0}}
    assert "incomplete" in _select_residual_candidate(
        {"tables": base, "tier2+ann_residual": failed},
        charge_mode="tables")[1]
    assert _select_residual_candidate(
        {"tables": base}, charge_mode="tables")[0] == "tier2"

    better = _candidate(3.0, 0.02, 10.0, 20.0, 20.0, 1.0)
    choice, reason, evidence = _select_residual_candidate(
        {"tables": base, "tier2+ann_residual": better}, charge_mode="tables")
    assert choice == "tier2+ann_residual" and "selected" in reason
    assert evidence["score"]["strict_gain"] is True
    # Charge selection semantics are untouched by the extra channel candidate.
    assert _select_charge_candidate({"tables": base, "tier2": better})[0] \
        == "tier2"


def test_simplis_speed_uses_header_resolved_cpm2_tj_pin(
        monkeypatch, tmp_path):
    """CPM2's floating Tj port must not be dropped from the golden bench."""
    import pipeline.tier3_simplis as sm

    lib = tmp_path / "cpm2.lib"
    lib.write_text("* fixture\n.SUBCKT CPM2 d g s Tj\nR1 d s 1\n.ENDS\n",
                   encoding="ascii")
    seen = []

    def fake_bench(library, subckt, pins, **kwargs):
        seen.append(list(pins))
        return {
            "wall_clock_s": 0.8, "solver_elapsed_s": 0.4,
            "iterations": 1000, "trace_points": 1000,
            "trace_names": 12, "end_time_s": 176e-6,
        }

    monkeypatch.setattr(sm, "run_inductive_benchmark", fake_bench)
    monkeypatch.setattr(sm, "_adaptive_guard_run", lambda *a, **kw: {
        "t_last_s": 176e-6, "iterations": 1000, "trace_points": 1000})

    cfg = {
        "library": str(lib), "subcircuit": "CPM2", "device": "CPM2",
        "pins": {
            "pin_order": ["drain", "gate", "source"],
            "pin_order_semantic": {
                "drain": "d", "gate": "g", "source": "s", "tj": "Tj"},
        },
    }
    dpt = {
        "vbus": 20.0, "vgs_on": 18.0, "i_load_target": 20.0,
        "l_load": 5e-6, "t1_on": 1e-6, "t1_width": 5e-6,
        "t2_on": 10e-6, "t2_width": 2e-6, "edge_time": 20e-9,
    }
    row = sm._run_adaptive_benchmark("CPM2", cfg, dpt, lib, 25.0)

    assert seen[0] == ["drain", "gate", "source", "tj"]
    assert seen[1] == ["drain", "gate", "source"]
    assert row["solver_speedup"] == pytest.approx(1.0)


@pytest.mark.parametrize("residual_better", [False, True])
def test_simplis_run_vendor_records_residual_evidence_and_gate(
        monkeypatch, tmp_path, residual_better):
    """Offline wiring check: no LTspice, no live gates, synthetic fits only."""
    import pipeline.tier3_simplis as sm

    fit = _synthetic_residual_fit()
    golden_metrics = {
        "td_on_ns": 5.0, "tr_ns": 10.0, "td_off_ns": 8.0, "tf_ns": 30.0,
        "dv_dt_max_v_per_ns": 2.0, "di_dt_max_a_per_ns": 3.0,
        "e_on_uj": 2.0, "e_off_uj": 1.0, "i_peak_a": 4.0,
    }

    def metrics(**changes):
        return {**golden_metrics, **changes}

    tables = (metrics(tr_ns=20.0, tf_ns=40.0, e_on_uj=5.0, e_off_uj=2.0),
              {"vds_rel_rmse_pct": 6.0, "id_rel_rmse_pct": 0.05})
    tier2 = (metrics(tr_ns=15.0, tf_ns=35.0, e_on_uj=4.0, e_off_uj=1.5),
             {"vds_rel_rmse_pct": 4.5, "id_rel_rmse_pct": 0.04})
    residual = (
        metrics(tr_ns=12.0, tf_ns=32.0, e_on_uj=3.0, e_off_uj=1.2),
        {"vds_rel_rmse_pct": 3.0, "id_rel_rmse_pct": 0.02}) if residual_better \
        else (
        metrics(tr_ns=16.0, tf_ns=36.0, e_on_uj=4.2, e_off_uj=1.6),
        {"vds_rel_rmse_pct": 5.0, "id_rel_rmse_pct": 0.045})
    time = np.linspace(0.0, 1e-4, 100)

    def fake_dpt(lib_path, vendor):
        chosen = residual if "ann_residual" in lib_path.name else (
            tier2 if "tier2_charge" in lib_path.name else tables)
        run = SimpleNamespace(
            time=time, wall_clock_s=0.1, metrics=chosen[0],
            waveform=chosen[1])
        return run, "Total iterations: 100\n", "tier3"

    monkeypatch.setattr(sm, "ROOT", tmp_path)
    monkeypatch.setattr(sm, "capture_trajectory", lambda v: SimpleNamespace(
        time=time, input=(np.ones_like(time), np.linspace(0, 20, 100)),
        output=(np.ones_like(time),),
        pin_roles=["drain", "gate", "source"]))
    monkeypatch.setattr(sm, "load_input", lambda v: {
        "conditions": {"temperature": 300.0}, "device": "TOY",
        "library": str(tmp_path / "golden.lib"), "subcircuit": "GOLDEN"})
    monkeypatch.setattr(sm, "_dpt_conditions", lambda v: {
        "vgs_on": 10.0, "vbus": 20.0, "i_load_target": 1.0,
        "l_load": 1e-6, "t1_on": 1e-6, "t2_on": 5e-6,
        "t1_width": 1e-6, "t2_width": 2e-6, "t_end": 1e-4,
        "edge_time": 1e-9, "max_step": 2e-9})
    monkeypatch.setattr(sm, "_build_spec", lambda v, **kw: (
        _simplis_spec(linear_cgs=True), {"dummy": 1}, None))
    monkeypatch.setattr(sm, "hybrid_ann_fit_for_vendor", lambda v: {
        "residual_fit": fit, "charge_fit": object()})  # must stay unconsumed
    monkeypatch.setattr(
        sm, "extract_dpt_metrics",
        lambda run: getattr(run, "metrics", golden_metrics))
    monkeypatch.setattr(sm, "_run_emitted_dpt", fake_dpt)
    monkeypatch.setattr(sm, "_waveform_errors", lambda run, traj: run.waveform)
    monkeypatch.setattr(sm, "_emitted_static_rmse", lambda v, lib: {
        "mean_rel_rmse_pct": 1.0, "per_curve_rel_rmse_pct": [],
        "n_points_per_curve": 0})
    monkeypatch.setattr(sm, "_recorded_tier2_static", lambda v: 1.5)
    monkeypatch.setattr(sm, "run_inductive_benchmark", lambda *a, **kw: {
        "wall_clock_s": 0.1, "solver_elapsed_s": 0.05,
        "iterations": 100, "trace_points": 100})
    monkeypatch.setattr(sm, "_run_adaptive_benchmark", lambda *a, **kw: {
        "completed": True})

    record = sm.run_vendor("TOY")
    prov = record["ann_residual_candidate"]
    assert prov["status"] == "fitted"
    assert prov["sample_counts"] == {"train": 12, "curve1": 8, "dpt": 4}
    assert prov["activation"] == "isru" and prov["neurons"] == 1
    assert set(prov["dpt_fit"]) == {"tier2", "tier2+ann_residual"}
    assert set(prov["dpt_held_out"]) == {"tier2", "tier2+ann_residual"}
    assert record["charge_candidates"]["selection"] == "tier2"
    expected = "tier2+ann_residual" if residual_better else "tier2"
    assert prov["selection"] == expected
    assert record["algorithm"]["residual_channel_candidate_selected"] == \
        expected
    assert record["dpt_waveform"] == (
        residual[1] if residual_better else tier2[1])

    models = tmp_path / "reports" / "models"
    base_lib = models / "TOY_PIECESPICE_T3_simplis_tier2_charge.lib"
    residual_lib = models / (
        "TOY_PIECESPICE_T3_simplis_hybrid_ann_residual_tier2_charge.lib")
    base_text = base_lib.read_text(encoding="ascii")
    res_text = residual_lib.read_text(encoding="ascii")
    assert len([x for x in res_text.splitlines()
                if x.startswith("G_ANN ")]) == 1
    assert _charge_element_lines(base_text, "tier2") == \
        _charge_element_lines(res_text, "tier2")
    if not residual_better:
        assert "rejected" in prov["selection_reason"]
        assert prov["evidence"]["strict_gain"] is False
    else:
        assert prov["evidence"]["strict_gain"] is True

# ---- T1: Tier-3 sites call the T0 hyp/sqrt helpers (unified spec 4.1) -----

def test_tier3_emitter_calls_shared_hyp_expr(monkeypatch):
    """R11 drift guard: soft clamp and Qoss guard own no fourth hyp copy."""
    model = _toy_model()
    monkeypatch.setattr(emitter, "hyp_expr",
                        lambda x, eps=0.0: f"SHARED_HYP({x})")
    text = emitter.emit_subcircuit(model, name="TOY_T3")
    assert "SHARED_HYP(" in text
    assert text.isascii()

    ir = _toy_model()
    ir.vendor = "irfs7730pbf"       # smooth_qoss policy hits the Qoss site
    ir_text = emitter.emit_subcircuit(ir, name="IR_T3")
    assert "SHARED_HYP(V(d,s))" in ir_text


def test_tier3_smooth_qoss_guard_uses_shared_hyp_text():
    ir = _toy_model()
    ir.vendor = "irfs7730pbf"
    text = emitter.emit_subcircuit(ir, name="IR_T3")
    assert hyp_expr("V(d,s)", emitter._QOSS_V0_EPS) in text
    assert text.isascii()


def test_tier3_numeric_mirror_uses_shared_hyp_value(monkeypatch):
    """The numpy mirror is the T0 helper, not a local clamp duplicate."""
    model = _toy_model()
    u = np.vstack([np.array([10.0, 10.0]), np.array([0.0, 10.0])])
    baseline = emitter.emitted_channel(model, u)
    calls = []

    def spy(x, eps=0.0):
        calls.append(eps)
        return np.zeros_like(np.asarray(x, dtype=float))

    monkeypatch.setattr(emitter, "hyp_value", spy)
    replaced = emitter.emitted_channel(model, u)
    assert calls == [emitter.emission_policy(model).soft_eps]
    assert not np.allclose(replaced, baseline)


def test_tier3_emission_adds_no_internal_nodes():
    """Flat channel/Qoss emission keeps only the three semantic ports."""
    text = emitter.emit_subcircuit(_toy_model(), name="TOY_T3")
    lines = text.splitlines()
    assert lines[0] == ".subckt TOY_T3 d g s"
    assert lines[-1] == ".ends"
    elements = lines[1:-1]
    assert len(elements) == 2
    assert elements[0].startswith("C") and " Q={" in elements[0]
    assert elements[1].startswith("G_CH d s VALUE={")
    nodes: set[str] = set()
    for line in elements:
        nodes.update(line.split()[:3][1:3])
    assert nodes == {"d", "s"}
    assert text.isascii()


# ---- B: ANN guard-width provenance (static grid, DPT-safe) ----------------
import sys  # noqa: E402
from common.types import CurveData  # noqa: E402
from approximation import ann_channel as ann_channel_mod  # noqa: E402
from approximation.ann_channel import DptPoints  # noqa: E402
from approximation.ann_charge import AnnChargeFit  # noqa: E402
from approximation.smoother import eps_from_grid  # noqa: E402
from models.ann_netlist_builder import (  # noqa: E402
    build_ann_subcircuit, emit_ann_channel,
)
from pipeline import ann_fit as ann_fit_mod  # noqa: E402


class _FakeLeastSquaresResult:
    """Deterministic optimizer stand-in: guard width is the unit under test."""

    def __init__(self, x):
        self.x = np.asarray(x, float)
        self.cost = 0.0
        self.nfev = 1
        self.njev = 1
        self.status = 1
        self.message = "deterministic test result"


def _stub_optimizer(monkeypatch):
    monkeypatch.setattr(
        ann_channel_mod, "least_squares",
        lambda residual, p0, **kw: _FakeLeastSquaresResult(p0))


def _ann_static_families(n: int = 21):
    vds = np.linspace(0.0, 10.0, n)
    out = []
    for vgs in (2.0, 4.0, 6.0):
        id_ = (vgs - 1.0) * (1.0 - np.exp(-vds))
        out.append(CurveData(
            name=f"curve1_vgs_{vgs:g}", x_name="Vds", x_unit="V",
            y_name="Id", y_unit="A", x_data=list(vds), y_data=list(id_),
            family_param="Vgs", family_value=vgs,
            metadata={"source": f"synthetic:curve_id_vgs_{vgs:g}"}))
    return out


def test_ann_guard_width_uses_static_grid_not_dense_dpt(monkeypatch):
    """Dense DPT plateau samples must not collapse the low-VDS hinge."""
    _stub_optimizer(monkeypatch)
    fams = _ann_static_families()
    dense = 10.0 + np.linspace(0.0, 1e-4, 30)
    dpt = DptPoints(np.full_like(dense, 8.0), dense, np.full_like(dense, 20.0),
                    {"source": "dpt_trajectory", "status": "ok",
                     "sample_indices": list(range(100, 130))})
    fit = ann_channel_mod.fit_channel_iv(fams + [dpt], 1.0, 2.0, neurons=2)

    expected = eps_from_grid(np.unique(fams[0].to_numpy()[0]).tolist())
    collapsed = eps_from_grid(np.unique(np.concatenate(
        [c.to_numpy()[0] for c in fams] + [dense])).tolist())
    assert expected > 0.1                      # static grid scale
    assert collapsed < 1e-3                    # the defect being guarded
    assert fit.coefficients["guard_width"] == pytest.approx(expected)
    assert fit.diagnostics["guard_width_v"] == pytest.approx(expected)
    gw_prov = fit.provenance["guard_width"]
    assert gw_prov["dpt_points_excluded"] == dense.size
    assert "Curve-1 static Vds grid" in gw_prov["source"]
    dpt_prov = fit.provenance["curves"]["dpt_high_vi"][0]
    assert dpt_prov["sample_indices"] == list(range(100, 130))


def test_ann_emitter_uses_fit_guard_width(monkeypatch):
    """The emitted tanh blend and header carry the fit's guard width."""
    _stub_optimizer(monkeypatch)
    fit = ann_channel_mod.fit_channel_iv(
        _ann_static_families(), 1.0, 2.0, neurons=2, guard_width=0.075,
        guard_width_provenance={"source": "explicit static grid",
                                "n_grid_points": 101})
    assert fit.coefficients["guard_width"] == pytest.approx(0.075)
    assert fit.provenance["guard_width"]["source"] == "explicit static grid"
    assert "/(7.500000e-02)" in emit_ann_channel(fit)

    tables = {n: {"v": [0.0, 1.0], "q": [0.0, 1e-9]}
              for n in ("oss", "iss", "rss")}
    charge = AnnChargeFit(tables, None, "isru", 2, {}, {})
    text = build_ann_subcircuit(_simplis_spec(), fit, charge)
    assert "* Guard width: 7.500000e-02 V" in text
    assert text.isascii()


def test_ann_dpt_provenance_records_retained_sample_indices(monkeypatch):
    """sample_indices are the real captured-trajectory indices, not []."""
    module = type(sys)("simulation.testbenches.dpt_trajectory")
    n = 12
    vgs = np.linspace(0.0, 10.0, n)
    vds = np.linspace(5.0, 12.0, n)
    ids = np.ones(n)
    def _capture(vendor, timeout_s, require_complete=False):
        return SimpleNamespace(
            input=np.vstack([vgs, vds]),
            output=np.vstack([ids, np.zeros(n)]),
            pin_roles=["d", "g", "s"])

    module.capture_trajectory = _capture
    monkeypatch.setitem(sys.modules,
                        "simulation.testbenches.dpt_trajectory", module)
    dpt, prov = ann_fit_mod._dpt_points("TOY", 10.0)
    assert dpt is not None and prov["status"] == "ok"
    assert prov["sample_indices"] == [8, 9, 10, 11]
    assert dpt.provenance["sample_indices"] == prov["sample_indices"]


def test_ann_fit_record_uses_actual_q0_used(monkeypatch):
    """q0_offset is the measured log-Q0, not the unset caller override."""
    fams = _ann_static_families()
    captured: dict = {}

    def _fit(inputs, v_th, v_sat, **kwargs):
        captured["width"] = kwargs.get("guard_width")
        captured["prov"] = kwargs.get("guard_width_provenance")
        return SimpleNamespace(
            activation="isru", neurons=2, loss_weights={"id": 1.0},
            diagnostics={"derivative_terms": ["gm"]},
            train_range={"vds": [0.0, 10.0]}, out_of_range_check="pass",
            provenance={}, v_th=1.0)

    charge_prov = {"q0_used": {"oss": 1.5e-8, "iss": 2.5e-10, "rss": 1.0e-8},
                   "derivative_terms": ["dq"]}
    charge = SimpleNamespace(q0=None, provenance=charge_prov)
    monkeypatch.setattr(ann_fit_mod, "load_input",
                        lambda v: {"device": "TOY", "conditions": {}})
    monkeypatch.setattr(ann_fit_mod, "_channel_families",
                        lambda v, c: (fams, 10.0))
    def _toy_extract(v, bench):
        if bench == "curve7_threshold_voltage":
            return {
                "vth_v": 1.0,
                "curve_vgs": [0.0, 1.0, 2.0, 3.0],
                "curve_id": [0.0, 1.0e-3, 0.1, 1.0],
                "Id_th_target": 1.0e-3,
            }
        return {"vth_v": 1.0}

    monkeypatch.setattr(ann_fit_mod, "_extract", _toy_extract)
    monkeypatch.setattr(ann_fit_mod, "_charge_curves", lambda v, c: (None, None, {}))
    monkeypatch.setattr(ann_fit_mod, "_dpt_points",
                        lambda v, m: (None, {"status": "disabled",
                                             "sample_indices": []}))
    monkeypatch.setattr(ann_fit_mod, "fit_channel_iv", _fit)
    monkeypatch.setattr(ann_fit_mod, "fit_charge_qv", lambda c, p: charge)

    out = ann_fit_mod.ann_fit_for_vendor("TOY", include_dpt_points=False)
    record = out["record"]
    assert record["q0_offset"] == pytest.approx(1.5e-8)
    assert record["q0_offset_by_table"] == charge_prov["q0_used"]
    assert "q_oss entry" in record["q0_offset_source"]
    assert captured["width"] == pytest.approx(0.25)
    assert out["provenance"]["guard_width"] == captured["prov"]
    assert out["provenance"]["dpt_high_vi"]["sample_indices"] == []
