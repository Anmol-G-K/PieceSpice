"""Unit tests for approximation and charge modeling algorithms."""

import numpy as np
from approximation.pwlf_fit import fit_piecewise_linear
from approximation.rdp_compress import compress_rdp
from approximation.charge_model import compute_charge_qoss


def test_pwlf_fit():
    x = np.linspace(0, 10, 50)
    y = np.where(x < 5, 2 * x, 10 + 0.5 * (x - 5))
    result = fit_piecewise_linear(x, y, n_segments=2)
    assert result.n_segments == 2
    assert len(result.breakpoints_x) == 3
    assert result.rmse < 0.1


def test_rdp_compress():
    x = np.linspace(0, 10, 100)
    y = 3 * x + 2  # perfectly linear
    x_comp, y_comp = compress_rdp(x, y, epsilon=0.01)
    assert len(x_comp) <= 3


def test_charge_integration():
    v = np.linspace(0, 10, 100)
    c = np.full_like(v, 1e-9)  # 1 nF constant
    v_out, q_out, e_oss = compute_charge_qoss(v, c)
    # Q = C * V = 1e-9 * 10 = 10 nC
    assert np.isclose(q_out[-1], 10e-9, rtol=1e-2)
    # E = 0.5 * C * V^2 = 0.5 * 1e-9 * 100 = 50 nJ
    assert np.isclose(e_oss, 50e-9, rtol=1e-2)
