import numpy as np
from dmfm import (
    initialize_dmfm,
    _construct_state_matrices,
    kalman_smoother_dmfm,
    em_step_dmfm,
    fit_dmfm_em,
)


def generate_data(T=4, p1=3, p2=2):
    rng = np.random.default_rng(0)
    return rng.normal(size=(T, p1, p2))


def test_initialize_dmfm_shapes():
    Y = generate_data()
    k1, k2, P = 2, 1, 1
    params = initialize_dmfm(Y, k1, k2, P)
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
    Tmat = _construct_state_matrices(A, B)
    expected = np.array([[0.08, 0.15], [1.0, 0.0]])
    assert np.allclose(Tmat, expected)


def test_kalman_smoother_shapes():
    Y = generate_data(T=6)
    params = initialize_dmfm(Y, 2, 2, 1)
    result = kalman_smoother_dmfm(
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
    params = initialize_dmfm(Y, 2, 2, 1)
    new_params, diff, ll = em_step_dmfm(Y, params)
    assert isinstance(new_params, dict)
    assert diff >= 0
    assert np.isfinite(ll)
    assert new_params["R"].shape == params["R"].shape
    assert new_params["C"].shape == params["C"].shape


def test_fit_dmfm_em_runs():
    Y = generate_data(T=5)
    params = fit_dmfm_em(Y, 1, 1, 1, max_iter=3)
    assert "loglik" in params
    assert len(params["loglik"]) >= 1
    assert params["R"].shape == (Y.shape[1], 1)
    assert params["C"].shape == (Y.shape[2], 1)
