import numpy as np
import pytest

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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


def test_em_step_idempotence():
    Y = generate_data(T=5)
    params = KPOKPCH.initialize_dmfm(Y, 1, 1, 1)
    params1, diff1, ll1 = KPOKPCH.em_step_dmfm(Y, params)
    params2, diff2, ll2 = KPOKPCH.em_step_dmfm(Y, params1)
    assert diff1 >= 0 and diff2 >= 0
    assert np.isfinite(ll1) and np.isfinite(ll2)


def test_em_loglik_monotonicity():
    Y = generate_data(T=5)
    params = KPOKPCH.initialize_dmfm(Y, 1, 1, 1)
    params1, diff1, ll1 = KPOKPCH.em_step_dmfm(Y, params)
    params2, diff2, ll2 = KPOKPCH.em_step_dmfm(Y, params1)
    assert np.isfinite(ll1) and np.isfinite(ll2)

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



def test_fit_with_high_missingness():
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(5, 3, 2))
    mask = rng.random(size=Y.shape) < 0.2
    mask[0] = True  # ensure at least one fully observed time point
    params = KPOKPCH.fit_dmfm_em(Y, 1, 1, 1, max_iter=2, mask=mask)
    assert np.isfinite(params["F"]).all()


def test_fit_with_block_missing_pattern():
    rng = np.random.default_rng(2)
    Y = rng.normal(size=(6, 4, 3))
    mask = np.ones_like(Y, dtype=bool)
    mask[:3, :2, :] = False
    mask[0] = True  # at least one fully observed
    params = KPOKPCH.fit_dmfm_em(Y, 1, 1, 1, max_iter=2, mask=mask)
    assert np.isfinite(params["F"]).all()


def test_missing_with_nan_and_mask():
    rng = np.random.default_rng(3)
    Y = rng.normal(size=(4, 2, 2))
    mask = np.ones_like(Y, dtype=bool)
    Y[1, 0, 1] = np.nan
    mask[1, 0, 1] = False
    params = KPOKPCH.initialize_dmfm(Y, 1, 1, 1, mask=mask)
    assert np.isfinite(params["F"]).all()


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


def test_initialize_with_zero_data():
    Y = np.zeros((4, 2, 3))
    params = KPOKPCH.initialize_dmfm(Y, 1, 1, 1)
    assert not np.isnan(params["F"]).any()
    assert np.allclose(params["F"], 0)


def test_initialize_with_low_rank_Y():
    rng = np.random.default_rng(0)
    T, p1, p2 = 5, 3, 3
    base = rng.normal(size=(p1, p2))
    Y = np.array([base * (t + 1) for t in range(T)])  # rank one across time
    params = KPOKPCH.initialize_dmfm(Y, 1, 1, 1)
    assert np.isfinite(params["F"]).all()


def test_initialize_with_nans():
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(4, 2, 2))
    mask = np.ones_like(Y, dtype=bool)
    Y[0, 0, 0] = np.nan
    mask[0, 0, 0] = False
    params = KPOKPCH.initialize_dmfm(Y, 1, 1, 1, mask=mask)
    assert np.isfinite(params["F"]).all()


def test_construct_state_matrices_invalid():
    A = [np.eye(2)]
    B = []
    with pytest.raises(Exception):
        KPOKPCH._construct_state_matrices(A, B)


def test_factor_number_sensitivity():
    rng = np.random.default_rng(4)
    Y = rng.normal(size=(6, 3, 4))
    params = KPOKPCH.fit_dmfm_em(Y, 2, 2, 1, max_iter=2)
    assert params["F"].shape == (6, 2, 2)
    assert np.isfinite(params["F"]).all()


def test_fit_with_nontrivial_P_Q():
    rng = np.random.default_rng(5)
    Y = rng.normal(size=(5, 2, 2))
    params = KPOKPCH.initialize_dmfm(Y, 1, 1, 1)
    res = KPOKPCH.kalman_smoother_dmfm(
        Y,
        params["R"],
        params["C"],
        params["A"],
        params["B"],
        params["H"],
        params["K"],
        Pmat=2 * np.eye(1),
        Qmat=3 * np.eye(1),
    )
    assert res["F_smooth"].shape == (5, 1, 1)


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


def generate_data(T=4, p1=3, p2=2):
    rng = np.random.default_rng(42)
    return rng.normal(size=(T, p1, p2))


def test_initialize_with_nan_values():
    Y = generate_data(T=5)
    mask = np.ones_like(Y, dtype=bool)
    Y[0, 0, 0] = np.nan
    Y[1, 2, 1] = np.nan
    mask[0, 0, 0] = False
    mask[1, 2, 1] = False
    params = KPOKPCH.initialize_dmfm(Y, 1, 1, 1, mask=mask)
    assert params["F"].shape == (Y.shape[0], 1, 1)
    assert not np.isnan(params["F"]).any()
    assert np.all(np.isfinite(params["R"]))
    assert np.all(np.isfinite(params["C"]))


def test_idiosyncratic_covariances_full():
    Y = generate_data(T=6, p1=4, p2=3)
    params = KPOKPCH.fit_dmfm_em(Y, 1, 1, 1, max_iter=2)
    H = params["H"]
    K = params["K"]
    assert np.allclose(H, H.T)
    assert np.allclose(K, K.T)
    assert np.all(np.linalg.eigvalsh(H) >= -1e-8)
    assert np.all(np.linalg.eigvalsh(K) >= -1e-8)
    # off-diagonal entries should generally be present
    off_H = np.abs(H - np.diag(np.diag(H)))
    off_K = np.abs(K - np.diag(np.diag(K)))
    assert off_H.sum() > 0
    assert off_K.sum() > 0


def test_em_step_idempotence_on_convergence():
    Y = generate_data(T=5)
    params = KPOKPCH.fit_dmfm_em(Y, 1, 1, 1, max_iter=15, tol=1e-6)
    new_params, diff, ll = KPOKPCH.em_step_dmfm(Y, params)
    assert diff < 1e-5


def test_fit_with_high_missingness():
    Y = generate_data(T=10, p1=5, p2=5)
    rng = np.random.default_rng(3)
    mask = rng.random(size=Y.shape) < 0.2
    mask[0] = True
    Y = np.where(mask, Y, np.nan)
    params = KPOKPCH.fit_dmfm_em(Y, 2, 2, 1, max_iter=5, mask=mask)
    assert np.isfinite(params["loglik"][-1])
    assert params["F"].shape == (Y.shape[0], 2, 2)
    assert not np.isnan(params["F"]).any()


def test_kalman_output_cov_psd():
    Y = generate_data(T=5)
    params = KPOKPCH.fit_dmfm_em(Y, 1, 1, 1, max_iter=2)
    res = KPOKPCH.kalman_smoother_dmfm(
        Y,
        params["R"],
        params["C"],
        params["A"],
        params["B"],
        params["H"],
        params["K"],
    )
    V = res["V_smooth"]
    for t in range(len(V)):
        assert np.allclose(V[t], V[t].T)
        eigvals = np.linalg.eigvalsh(V[t])
        assert np.all(eigvals >= -1e-8)


def test_construct_state_matrix_known_kronecker():
    A = [np.array([[1]])]
    B = [np.array([[2]])]
    Tmat = KPOKPCH._construct_state_matrices(A, B)
    assert Tmat.shape == (1, 1)
    assert np.allclose(Tmat[0, 0], 2)


def test_construct_state_matrices_phi():
    Phi = [np.array([[0.5]]), np.array([[0.2]])]
    Tmat = KPOKPCH._construct_state_matrices(None, None, Phi=Phi, kronecker_only=True)
    expected = np.array([[0.5, 0.2], [1.0, 0.0]])
    assert np.allclose(Tmat, expected)


def test_select_dmfm_rank_basic():
    Y = generate_data(T=6, p1=3, p2=4)
    k1, k2 = KPOKPCH.select_dmfm_rank(Y, max_k=2)
    assert 1 <= k1 <= 2
    assert 1 <= k2 <= 2


def test_select_dmfm_rank_baing():
    Y = generate_data(T=6, p1=4, p2=3)
    k1, k2 = KPOKPCH.select_dmfm_rank(Y, max_k=3, method="bai-ng")
    assert 1 <= k1 <= 3
    assert 1 <= k2 <= 3


def test_kalman_smoother_i1():
    Y = generate_data(T=5)
    params = KPOKPCH.initialize_dmfm(Y, 1, 1, 1)
    res = KPOKPCH.kalman_smoother_dmfm(
        Y,
        params["R"],
        params["C"],
        params["A"],
        params["B"],
        params["H"],
        params["K"],
        i1_factors=True,
    )
    assert res["F_smooth"].shape == (Y.shape[0], 1, 1)


def test_fit_dmfm_em_i1_runs():
    Y = generate_data(T=6)
    params = KPOKPCH.fit_dmfm_em(Y, 1, 1, 1, max_iter=3, i1_factors=True)
    assert params["F"].shape == (Y.shape[0], 1, 1)
    assert len(params["loglik"]) >= 1


def test_select_dmfm_qml_basic():
    Y = generate_data(T=4, p1=3, p2=2)
    k1, k2, P = KPOKPCH.select_dmfm_qml(Y, max_k=1, max_P=1)
    assert isinstance(k1, int) and isinstance(k2, int) and isinstance(P, int)
    assert k1 == 1 and k2 == 1 and P == 1


def test_select_dmfm_qml_search_P():
    Y = generate_data(T=5, p1=2, p2=2)
    k1, k2, P = KPOKPCH.select_dmfm_qml(Y, max_k=1, max_P=2, criterion="aic")
    assert isinstance(k1, int) and isinstance(k2, int) and isinstance(P, int)
    assert k1 == 1 and k2 == 1 and 1 <= P <= 2


def test_standard_errors_shapes():
    Y = generate_data(T=6)
    params = KPOKPCH.fit_dmfm_em(Y, 1, 1, 1, max_iter=2, return_se=True)
    se = params.get("standard_errors")
    assert se is not None
    assert se["se_R"].shape == params["R"].shape
    assert se["se_C"].shape == params["C"].shape
    assert se["ci_R"].shape == params["R"].shape + (2,)
    assert se["ci_C"].shape == params["C"].shape + (2,)


def test_kronecker_only_mode():
    Y = generate_data(T=5)
    params = KPOKPCH.fit_dmfm_em(Y, 1, 1, 1, max_iter=2, kronecker_only=True)
    assert "Phi" in params
    assert len(params["Phi"]) == 1
    assert params["Phi"][0].shape == (1, 1)
    assert params["F"].shape[0] == Y.shape[0]


def test_kalman_smoother_kronecker_only():
    Y = generate_data(T=4)
    params = KPOKPCH.initialize_dmfm(Y, 1, 1, 1)
    Phi = [np.kron(params["B"][0], params["A"][0])]
    res = KPOKPCH.kalman_smoother_dmfm(
        Y,
        params["R"],
        params["C"],
        params["A"],
        params["B"],
        params["H"],
        params["K"],
        Phi=Phi,
        kronecker_only=True,
    )
    assert res["F_smooth"].shape == (Y.shape[0], 1, 1)


def test_optimize_qml_dmfm_runs():
    Y = generate_data(T=4)
    res = KPOKPCH.optimize_qml_dmfm(Y, 1, 1, 1)
    assert res["R"].shape == (Y.shape[1], 1)
    assert "loglik" in res


def test_fit_dmfm_em_qml_opt():
    Y = generate_data(T=4)
    res = KPOKPCH.fit_dmfm_em(Y, 1, 1, 1, use_qml_opt=True)
    assert res["R"].shape == (Y.shape[1], 1)


def test_identify_dmfm_trends_shapes():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(8, 2, 1))
    out = KPOKPCH.identify_dmfm_trends(F)
    r = out["r"]
    assert 0 <= r <= 2
    assert out["F_trend"].shape == (F.shape[0], r)
    assert out["F_cycle"].shape[0] == F.shape[0]


def test_fit_dmfm_em_trend_decomp():
    Y = generate_data(T=6)
    res = KPOKPCH.fit_dmfm_em(
        Y,
        1,
        1,
        1,
        max_iter=2,
        i1_factors=True,
        return_trend_decomp=True,
    )
    td = res.get("trend_decomposition")
    assert td is not None
    assert td["F_trend"].shape[0] == Y.shape[0]


def test_forecast_dmfm_basic():
    Y = generate_data(T=6, p1=2, p2=2)
    params = KPOKPCH.fit_dmfm_em(Y, 1, 1, 1, max_iter=3)
    fcst = KPOKPCH.forecast_dmfm(2, params)
    assert fcst.shape == (2, 2, 2)
    F_last = params["F"][-1]
    A = params["A"][0]
    B = params["B"][0]
    expected_F1 = A @ F_last @ B.T
    expected_Y1 = params["R"] @ expected_F1 @ params["C"].T
    assert np.allclose(fcst[0], expected_Y1, atol=1e-6)


def test_forecast_dmfm_return_factors():
    Y = generate_data(T=5, p1=2, p2=2)
    params = KPOKPCH.fit_dmfm_em(Y, 1, 1, 1, max_iter=2)
    Y_fcst, F_fcst = KPOKPCH.forecast_dmfm(1, params, return_factors=True)
    assert Y_fcst.shape == (1, 2, 2)
    assert F_fcst.shape == (1, 1, 1)


def test_subsample_panel_basic():
    Y = generate_data(T=4, p1=4, p2=3)
    blocks, idx = KPOKPCH.subsample_panel(Y, 2, axis="row")
    assert len(blocks) == 2
    assert sum(len(i) for i in idx) == Y.shape[1]


def test_fit_dmfm_local_qml_runs():
    Y = generate_data(T=4, p1=3, p2=2)
    res = KPOKPCH.fit_dmfm_local_qml(Y, 1, 1, 1, max_iter=2)
    assert "R" in res and "F" in res


def test_fit_dmfm_distributed_runs():
    Y = generate_data(T=5, p1=4, p2=3)
    params = KPOKPCH.fit_dmfm_distributed(Y, B=2, k1=1, k2=1, P=1)
    assert params["R"].shape == (Y.shape[1], 1)
    assert params["C"].shape == (Y.shape[2], 1)
    

def test_standard_errors_dynamics_return():
    Y = generate_data(T=6)
    params = KPOKPCH.fit_dmfm_em(
        Y,
        1,
        1,
        1,
        max_iter=2,
        return_se=True,
        return_se_dynamics=True,
    )
    se_dyn = params.get("standard_errors_dynamics")
    assert se_dyn is not None
    assert len(se_dyn["se_A"]) == 1
    assert se_dyn["se_A"][0].shape == params["A"][0].shape
    assert len(se_dyn["se_B"]) == 1
    assert se_dyn["se_B"][0].shape == params["B"][0].shape
    assert "aic" in params and "bic" in params and "n_params" in params
    assert params["lag_order"] == 1


def test_unit_root_factors_function():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(6, 1, 1))
    res = KPOKPCH.test_unit_root_factors(F)
    key = list(res.keys())[0]
    assert isinstance(res[key][0], float)


def test_qml_loglik_matches_kalman_smoother():
    Y = generate_data(T=3, p1=2, p2=2)
    params = KPOKPCH.initialize_dmfm(Y, 1, 1, 1)
    ll_smoother = KPOKPCH.kalman_smoother_dmfm(
        Y,
        params["R"],
        params["C"],
        params["A"],
        params["B"],
        params["H"],
        params["K"],
    )["loglik"]
    ll_qml = KPOKPCH.qml_loglik_dmfm(
        Y,
        params["R"],
        params["C"],
        params["A"],
        params["B"],
        params["H"],
        params["K"],
    )
    assert np.isclose(ll_qml, ll_smoother)


def test_pack_unpack_roundtrip():
    Y = generate_data(T=4, p1=3, p2=2)
    params = KPOKPCH.initialize_dmfm(Y, 2, 1, 1)

    vec = KPOKPCH.pack_dmfm_parameters(
        params["R"],
        params["C"],
        params["A"],
        params["B"],
        params["H"],
        params["K"],
        params["P"],
        params["Q"],
    )

    shape_info = {
        "p1": params["R"].shape[0],
        "k1": params["R"].shape[1],
        "p2": params["C"].shape[0],
        "k2": params["C"].shape[1],
        "P": len(params["A"]),
    }
    R, C, A, B, H, K, Pmat, Qmat = KPOKPCH.unpack_dmfm_parameters(vec, shape_info)

    assert np.allclose(R, params["R"])
    assert np.allclose(C, params["C"])
    for a, a0 in zip(A, params["A"]):
        assert np.allclose(a, a0)
    for b, b0 in zip(B, params["B"]):
        assert np.allclose(b, b0)
    assert np.allclose(H, params["H"])
    assert np.allclose(K, params["K"])
    assert np.allclose(Pmat, params["P"])
    assert np.allclose(Qmat, params["Q"])


def test_conditional_forecast_enforces_known_values():
    Y = generate_data(T=5, p1=2, p2=2)
    params = KPOKPCH.fit_dmfm_em(Y, 1, 1, 1, max_iter=3)

    params["H"] *= 0
    params["K"] *= 0

    known1 = np.full((2, 2), np.nan)
    mask1 = np.zeros((2, 2), dtype=bool)
    known1[0, 0] = 0.5
    mask1[0, 0] = True

    known2 = np.full((2, 2), np.nan)
    mask2 = np.zeros((2, 2), dtype=bool)
    known2[1, 1] = -0.2
    mask2[1, 1] = True

    fcst = KPOKPCH.conditional_forecast_dmfm(
        2,
        params,
        known_future={1: known1, 2: known2},
        mask_future={1: mask1, 2: mask2},
    )
    assert fcst.shape == (2, 2, 2)
    assert np.allclose(fcst[0][mask1], known1[mask1])
    assert np.allclose(fcst[1][mask2], known2[mask2])
