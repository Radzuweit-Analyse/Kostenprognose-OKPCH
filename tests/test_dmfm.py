import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from DMFM import DMFMModel, EMEstimatorDMFM


def generate_data(T=8, p1=4, p2=3, k1=2, k2=2):
    rng = np.random.default_rng(0)
    R = rng.normal(size=(p1, k1))
    C = rng.normal(size=(p2, k2))
    A = [0.5 * np.eye(k1)]
    B = [0.3 * np.eye(k2)]
    F = np.zeros((T, k1, k2))
    for t in range(1, T):
        F[t] = A[0] @ F[t - 1] @ B[0].T + rng.normal(size=(k1, k2))
    Y = np.zeros((T, p1, p2))
    for t in range(T):
        Y[t] = R @ F[t] @ C.T + 0.1 * rng.normal(size=(p1, p2))
    mask = np.ones_like(Y, dtype=bool)
    return Y, mask, F


def test_initialization_shapes():
    Y, mask, _ = generate_data()
    model = DMFMModel(p1=Y.shape[1], p2=Y.shape[2], k1=2, k2=2, P=1)
    model.initialize(Y, mask)
    assert model.R.shape == (Y.shape[1], 2)
    assert model.C.shape == (Y.shape[2], 2)
    assert model.F.shape == (Y.shape[0], 2, 2)


def test_em_loglik_increases():
    Y, mask, _ = generate_data(T=6)
    model = DMFMModel(p1=Y.shape[1], p2=Y.shape[2], k1=2, k2=2, P=1)
    model.initialize(Y, mask)
    est = EMEstimatorDMFM(model)
    est.fit(Y, mask, max_iter=5)
    ll = est.get_loglik_trace()
    assert all(x2 >= x1 - 1e-6 for x1, x2 in zip(ll, ll[1:]))


def test_em_convergence():
    Y, mask, _ = generate_data(T=6)
    model = DMFMModel(p1=Y.shape[1], p2=Y.shape[2], k1=2, k2=2, P=1)
    model.initialize(Y, mask)
    est = EMEstimatorDMFM(model)
    est.fit(Y, mask, max_iter=20, tol=1e-3)
    assert est.diff_trace[-1] < 1e-3


def test_smoothing_recovers_factors():
    Y, mask, F_true = generate_data(T=10)
    model = DMFMModel(p1=Y.shape[1], p2=Y.shape[2], k1=2, k2=2, P=1)
    model.initialize(Y, mask)
    est = EMEstimatorDMFM(model)
    est.fit(Y, mask, max_iter=20, tol=1e-4)
    F_hat = est.get_factors()
    corr = np.corrcoef(F_hat.ravel(), F_true.ravel())[0, 1]
    assert corr > 0.0
