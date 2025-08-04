import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from KPOKPCH.forecast import (
    seasonal_difference,
    integrate_seasonal_diff,
    forecast_dmfm,
    out_of_sample_rmse,
)


def generate_data(T=12, p1=2, p2=2):
    rng = np.random.default_rng(1)
    R = rng.normal(size=(p1, 1))
    C = rng.normal(size=(p2, 1))
    A = [0.7 * np.eye(1)]
    B = [0.5 * np.eye(1)]
    F = np.zeros((T, 1, 1))
    for t in range(1, T):
        F[t] = A[0] @ F[t - 1] @ B[0].T + rng.normal(size=(1, 1))
    Y = np.zeros((T, p1, p2))
    for t in range(T):
        Y[t] = R @ F[t] @ C.T + 0.1 * rng.normal(size=(p1, p2))
    mask = np.ones_like(Y, dtype=bool)
    return Y, mask


def test_seasonal_roundtrip():
    Y, _ = generate_data(T=8)
    diff = seasonal_difference(Y, 2)
    recon = integrate_seasonal_diff(Y[:2], diff, 2)
    np.testing.assert_allclose(recon, Y[2:], atol=1e-8)


def test_forecast_shape():
    Y, mask = generate_data(T=10)
    fcst = forecast_dmfm(Y, 2, mask=mask)
    assert fcst.shape == (2, Y.shape[1], Y.shape[2])


def test_out_of_sample_rmse_small():
    Y, mask = generate_data(T=12)
    rmse = out_of_sample_rmse(Y, 2, mask=mask)
    assert rmse < 0.5
