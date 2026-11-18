"""Unit tests for the paper-defined parts of the Tier-3 TPWL terminal model."""
import numpy as np

from tier3.tpwl_deim import (
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
