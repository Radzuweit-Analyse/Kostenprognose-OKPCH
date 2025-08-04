import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import KPOKPCH


def generate_data(T=5, p1=3, p2=2):
    rng = np.random.default_rng(0)
    return rng.normal(size=(T, p1, p2))


def test_direct_initialization_optional_params():
    Y = generate_data()
    base = KPOKPCH.DMFM.from_data(Y, 1, 1, 1)
    model = KPOKPCH.DMFM(base.R, base.C, base.F)
    assert model.dynamics.A == []
    assert model.idiosyncratic.H is None
    assert model.F.shape == base.F.shape


def test_from_data_and_kalman_smoother():
    Y = generate_data(T=6)
    model = KPOKPCH.DMFM.from_data(Y, 2, 2, 1)
    result = model.kalman_smoother(Y)
    assert result["F_smooth"].shape == (Y.shape[0], 2, 2)
    assert np.isfinite(result["loglik"])


def test_em_fit_and_forecast():
    Y = generate_data(T=6, p1=2, p2=2)
    model = KPOKPCH.DMFM.from_data(Y, 1, 1, 1).fit_em(Y, max_iter=3)
    fcst = model.forecast(2)
    assert fcst.shape == (2, 2, 2)


def test_conditional_forecast_enforces_known_values():
    Y = generate_data(T=5, p1=2, p2=2)
    model = KPOKPCH.DMFM.from_data(Y, 1, 1, 1).fit_em(Y, max_iter=3)
    known = {1: np.zeros((2, 2))}
    mask = {1: np.zeros((2, 2), dtype=bool)}
    mask[1][0, 0] = True
    fcst = model.conditional_forecast(2, known_future=known, mask_future=mask)
    assert np.allclose(fcst[0, 0, 0], 0)


def test_compute_standard_errors():
    Y = generate_data(T=6)
    model = KPOKPCH.DMFM.from_data(Y, 1, 1, 1).fit_em(Y, max_iter=2)
    se = model.compute_standard_errors(Y)
    assert se["se_R"].shape == model.R.shape
    dyn = model.compute_standard_errors_dynamics()
    assert "se_A" in dyn and "se_B" in dyn


def test_rank_selection():
    Y = generate_data(T=6, p1=3, p2=4)
    k1, k2 = KPOKPCH.DMFM.select_rank(Y, max_k=2)
    assert 1 <= k1 <= 2 and 1 <= k2 <= 2


def test_qml_selection():
    Y = generate_data(T=4, p1=3, p2=2)
    k1, k2, P = KPOKPCH.DMFM.select_qml(Y, max_k=1, max_P=1)
    assert (k1, k2, P) == (1, 1, 1)


def test_fit_distributed_returns_model():
    Y = generate_data(T=5, p1=4, p2=3)
    model = KPOKPCH.DMFM.fit_distributed(Y, B=2, k1=1, k2=1, P=1)
    assert isinstance(model, KPOKPCH.DMFM)
    assert model.R.shape == (Y.shape[1], 1)


def test_seasonal_difference():
    Y = generate_data(T=6, p1=2, p2=2)
    diff = KPOKPCH.seasonal_difference(Y, period=2)
    assert np.allclose(diff, Y[2:] - Y[:-2])
