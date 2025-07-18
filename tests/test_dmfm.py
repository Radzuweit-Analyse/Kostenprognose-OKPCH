import numpy as np
import pytest

import KPOKPCH


def generate_data(T=4, p1=3, p2=2):
    rng = np.random.default_rng(0)
    return rng.normal(size=(T, p1, p2))


def test_initialize_dmfm_shapes():
    Y = generate_data()
    k1, k2, P = 2, 1, 1
    params = KPOKPCH.initialize_dmfm(Y, k1, k2, P)
    assert params["R"].shape == (Y.shape[1], k1)
    assert params["C"].shape == (Y.shape[2], k2)
    assert len(params["A"]) == P
    assert params["A"][0].shape == (k1, k1)
    assert len(params["B"]) == P
    assert params["B"][0].shape == (k2, k2)
    assert params["H"].shape == (Y.shape[1], Y.shape[1])
    assert params["K"].shape == (Y.shape[2], Y.shape[2])
    assert params["P"].shape == (k1, k1)
    assert params["Q"].shape == (k2, k2)
    assert params["F"].shape == (Y.shape[0], k1, k2)
    assert not np.isnan(params["F"]).any()


def test_construct_state_matrices_basic():
    A = [np.array([[0.2]]), np.array([[0.3]])]
    B = [np.array([[0.4]]), np.array([[0.5]])]
    Tmat = KPOKPCH._construct_state_matrices(A, B)
    expected = np.array([[0.08, 0.15], [1.0, 0.0]])
    assert np.allclose(Tmat, expected)


def test_kalman_smoother_shapes():
    Y = generate_data(T=6)
    params = KPOKPCH.initialize_dmfm(Y, 2, 2, 1)
    result = KPOKPCH.kalman_smoother_dmfm(
        Y,
        params["R"],
        params["C"],
        params["A"],
        params["B"],
        params["H"],
        params["K"],
    )
    assert result["F_smooth"].shape == (Y.shape[0], 2, 2)
    assert result["F_pred"].shape == (Y.shape[0], 2, 2)
    assert result["F_filt"].shape == (Y.shape[0], 2, 2)
    assert np.isfinite(result["loglik"])


def test_em_step_dmfm_update():
    Y = generate_data(T=5)
    params = KPOKPCH.initialize_dmfm(Y, 2, 2, 1)
    new_params, diff, ll = KPOKPCH.em_step_dmfm(Y, params)
    assert isinstance(new_params, dict)
    assert diff >= 0
    assert np.isfinite(ll)
    assert new_params["R"].shape == params["R"].shape
    assert new_params["C"].shape == params["C"].shape


def test_fit_dmfm_em_runs():
    Y = generate_data(T=5)
    params = KPOKPCH.fit_dmfm_em(Y, 1, 1, 1, max_iter=3)
    assert "loglik" in params
    assert len(params["loglik"]) >= 1
    assert params["R"].shape == (Y.shape[1], 1)
    assert params["C"].shape == (Y.shape[2], 1)


def test_initialize_dmfm_with_mask():
    Y = generate_data(T=4, p1=2, p2=2)
    mask = np.ones_like(Y, dtype=bool)
    mask[0, 0, 0] = False
    params = KPOKPCH.initialize_dmfm(Y, 1, 1, 1, mask=mask)
    assert params["F"].shape == (Y.shape[0], 1, 1)
    assert not np.isnan(params["F"]).any()


def test_fit_dmfm_em_convergence():
    rng = np.random.default_rng(1)
    T = 8
    A_true = [np.array([[0.5]])]
    B_true = [np.array([[0.9]])]
    F_true = np.zeros((T, 1, 1))
    for t in range(1, T):
        F_true[t] = A_true[0] @ F_true[t - 1] @ B_true[0].T + rng.normal(scale=0.1)
    R = np.array([[1.0]])
    C = np.array([[1.0]])
    Y = np.einsum("ij,tjk,kl->til", R, F_true, C.T)
    params = KPOKPCH.fit_dmfm_em(Y, 1, 1, 1, max_iter=10)
    loglik = params["loglik"]
    assert all(np.diff(loglik) >= -1e-6)
    assert np.linalg.norm(params["F"] - F_true) < 0.5


def test_construct_state_matrices_higher_order():
    A = [np.array([[0.2]]), np.array([[0.3]]), np.array([[0.1]])]
    B = [np.array([[0.4]]), np.array([[0.5]]), np.array([[0.2]])]
    Tmat = KPOKPCH._construct_state_matrices(A, B)
    expected = np.array(
        [
            [0.08, 0.15, 0.02],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    assert np.allclose(Tmat, expected)


def test_covariance_sym_psd():
    Y = generate_data(T=6)
    params = KPOKPCH.fit_dmfm_em(Y, 1, 1, 1, max_iter=2)
    H = params["H"]
    Kmat = params["K"]
    assert np.allclose(H, H.T)
    assert np.all(np.linalg.eigvalsh(H) >= -1e-8)
    assert np.allclose(Kmat, Kmat.T)
    assert np.all(np.linalg.eigvalsh(Kmat) >= -1e-8)


def test_initialize_dmfm_invalid_inputs():
    Y = generate_data()
    mask = np.ones((2, 2, 2), dtype=bool)
    with pytest.raises(Exception):
        KPOKPCH.initialize_dmfm(Y, 2, 1, 1, mask=mask)
    with pytest.raises(Exception):
        KPOKPCH.initialize_dmfm(Y.reshape(-1), 2, 1, 1)


def test_construct_state_matrices_invalid():
    A = [np.eye(2)]
    B = []
    with pytest.raises(Exception):
        KPOKPCH._construct_state_matrices(A, B)
