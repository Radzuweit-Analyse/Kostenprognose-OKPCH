import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from KPOKPCH.forecast import (
    seasonal_difference,
    integrate_seasonal_diff,
    forecast_dmfm,
    out_of_sample_rmse,
    load_cost_matrix,
    generate_future_periods,
    compute_q4_growth,
    canton_forecast,
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


def test_load_cost_matrix(tmp_path):
    csv_path = tmp_path / "costs.csv"
    csv_path.write_text("Period,ZH,BE\n2020Q1,1.0,2.0\n2020Q2,3.0,\n")
    periods, cantons, data = load_cost_matrix(str(csv_path))
    assert periods == ["2020Q1", "2020Q2"]
    assert cantons == ["ZH", "BE"]
    np.testing.assert_allclose(data, [[1.0, 2.0], [3.0, np.nan]])


def test_generate_future_periods_rollover():
    periods = generate_future_periods("2020Q4", 3)
    assert periods == ["2021Q1", "2021Q2", "2021Q3"]


def test_compute_q4_growth():
    periods = ["2020Q3", "2020Q4"]
    data = np.array([[1.0, 2.0], [2.0, 4.0]])
    fcst = np.array([[3.0, 6.0], [4.5, 9.0]])
    future_periods = ["2021Q4", "2022Q4"]
    stats = compute_q4_growth(periods, data, fcst, future_periods)
    np.testing.assert_allclose(stats["growth_y1"], [50.0, 50.0])
    np.testing.assert_allclose(stats["growth_y2"], [50.0, 50.0])


def test_canton_forecast_separate_and_joint_consistent():
    Y = np.ones((8, 3))
    mask = np.ones_like(Y, dtype=bool)
    fcst_joint, total_joint = canton_forecast(
        Y, 2, mask=mask[..., None], separate_cantons=False
    )
    fcst_sep, total_sep = canton_forecast(
        Y, 2, mask=mask[..., None], separate_cantons=True
    )
    assert fcst_joint.shape == (2, 3)
    np.testing.assert_allclose(total_sep, fcst_sep.sum(axis=1))
    np.testing.assert_allclose(total_joint, fcst_joint.sum(axis=1))
