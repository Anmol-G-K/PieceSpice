"""Unit tests for metrics and error calculation."""

import numpy as np
from metrics.static import compute_curve_error
from metrics.loss import compute_switching_energy


def test_curve_error():
    y1 = np.array([1.0, 2.0, 3.0, 4.0])
    y2 = np.array([1.0, 2.1, 2.9, 4.0])
    err = compute_curve_error(y1, y2)
    assert err.max_abs_error <= 0.11
    assert err.rmse < 0.1
    assert err.r_squared > 0.99


def test_switching_energy():
    t = np.linspace(0, 10e-6, 1000)
    v = np.full_like(t, 10.0)
    i = np.full_like(t, 2.0)
    # P = 20 W for 10 us -> E = 200 uJ
    energy_uj = compute_switching_energy(t, v, i, 0, 10e-6)
    assert np.isclose(energy_uj, 200.0, rtol=1e-2)
