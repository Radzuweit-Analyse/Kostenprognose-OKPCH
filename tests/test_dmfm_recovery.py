import numpy as np
import pytest

import KPOKPCH


def test_recover_mar_coefficients():
    rng = np.random.default_rng(0)
    T = 30
    A_true = np.array([[0.6]])
    B_true = np.array([[0.4]])
    F = np.zeros((T, 1, 1))
    for t in range(1, T):
        F[t] = A_true @ F[t - 1] @ B_true + rng.normal(scale=0.01)
    Y = F.copy()
    params = KPOKPCH.fit_dmfm_em(Y, 1, 1, 1, max_iter=15)
    est = params["A"][0][0, 0] * params["B"][0][0, 0]
    truth = (A_true * B_true)[0, 0]
    assert np.isclose(est, truth, atol=0.1)
