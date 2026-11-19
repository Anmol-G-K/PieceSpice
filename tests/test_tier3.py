"""Unit tests for the paper-defined parts of the Tier-3 TPWL terminal model."""
import numpy as np

from models import tpwl_netlist_builder as emitter
from pipeline.tier3 import _log_iterations
from approximation.tpwl_deim import (
    WEIGHT_EXPONENT,
    TPWLTerminalModel,
    deim_indices,
    paper_weights,
)


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
    text = emitter.emit_subcircuit(_toy_model(), name="TOY_T3")
    assert text.isascii()
    assert text.startswith(".subckt TOY_T3 d g s\n")
    assert text.rstrip().endswith(".ends")
    assert "G_CH d s VALUE={max(0," in text
    assert "table(V(d,s)," in text
    assert "V(d,s)/(V(d,s)+2.000000000e-01)" in text
    assert "QOSS d s Q={table(limit(V(d,s)" in text
    # No B-source intermediates, no per-cell weight nodes, no C_QD.
    assert "\nB" not in text and "C_QD" not in text
    assert "exp(" not in text and "sqrt(" not in text


def test_emitted_channel_clamps_negative_drain_voltage():
    model = _toy_model()
    u = np.vstack([np.array([10.0, 10.0, 10.0]),
                   np.array([-0.1, 0.0, 10.0])])
    current = emitter.emitted_channel(model, u)
    assert current[0] == 0.0
    assert current[1] == 0.0
    assert current[2] > 0.0


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
