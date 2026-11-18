"""Unit tests for the paper-defined parts of Tier-3 TPWL/POD-DEIM."""
import numpy as np

from tier3.tpwl_deim import TPWLModel, WEIGHT_EXPONENT, deim_indices

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


def test_tpwl_weights_follow_paper_formula():
    model = TPWLModel(
        vendor="test", basis=np.zeros((1, 1)), x_mean=np.zeros(1),
        z_points=np.array([[0.0], [1.0]]), Ared=np.zeros((2, 1, 1)),
        Bred=np.zeros((2, 1, 1)), gamma=np.zeros((2, 1)),
        C=np.zeros((1, 1)), D=np.zeros((1, 1)), e=np.zeros(1),
        deim_indices=np.array([0]), deim_coeff=np.ones((1, 1)),
        z0=np.zeros(1), time=np.array([0.0]), state_names=["n"],
        pin_roles=["drain", "gate", "source"], subcircuit="X", r=1,
        n_points=2, eps=1.0, validation_rmse=0.0,
    )
    assert WEIGHT_EXPONENT == 25.0
    mid = model.weights(np.array([0.5]))
    assert np.allclose(mid, [0.5, 0.5])
    near = model.weights(np.array([0.1]))
    assert near[0] > near[1]
    assert np.isclose(near.sum(), 1.0)
