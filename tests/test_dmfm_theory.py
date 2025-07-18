import numpy as np
import pytest

import KPOKPCH


def generate_data(T=4, p1=3, p2=2):
    rng = np.random.default_rng(0)
    return rng.normal(size=(T, p1, p2))


def test_kalman_output_cov_structure():
    Y = generate_data(T=5)
    params = KPOKPCH.initialize_dmfm(Y, 2, 2, 1)
    res = KPOKPCH.kalman_smoother_dmfm(
        Y,
        params["R"],
        params["C"],
        params["A"],
        params["B"],
        params["H"],
        params["K"],
    )
    for V in res["V_smooth"]:
        assert np.allclose(V, V.T)
        eigvals = np.linalg.eigvalsh(V)
        assert np.all(eigvals >= -1e-8)
