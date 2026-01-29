import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from KPOKPCH import DMFMModel, EMEstimatorDMFM


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


# ---------------------------------------------------------------------------
# Tests for I(1) factors (Barigozzi & Trapin 2025 Section 6)
# ---------------------------------------------------------------------------


def generate_i1_data(T=20, p1=4, p2=3, k1=2, k2=2, seed=42):
    """Generate data with I(1) (random walk) factors."""
    rng = np.random.default_rng(seed)
    R = rng.normal(size=(p1, k1))
    C = rng.normal(size=(p2, k2))

    # I(1) factors: F_t = F_{t-1} + U_t (random walk, A=B=I)
    F = np.zeros((T, k1, k2))
    for t in range(1, T):
        F[t] = F[t - 1] + 0.5 * rng.normal(size=(k1, k2))

    # Observed data
    Y = np.zeros((T, p1, p2))
    for t in range(T):
        Y[t] = R @ F[t] @ C.T + 0.1 * rng.normal(size=(p1, p2))

    return Y, F, R, C


def test_i1_dynamics_not_updated():
    """Test that A, B remain at identity when i1_factors=True."""
    from KPOKPCH.DMFM import fit_dmfm

    Y, F_true, _, _ = generate_i1_data(T=15)

    model, result = fit_dmfm(Y, k1=2, k2=2, P=1, i1_factors=True, max_iter=10)

    # Check A and B are still identity
    assert np.allclose(model.A[0], np.eye(2), atol=1e-10)
    assert np.allclose(model.B[0], np.eye(2), atol=1e-10)


def test_i1_drift_is_zero():
    """Test that drift is zero when i1_factors=True."""
    from KPOKPCH.DMFM import fit_dmfm

    Y, F_true, _, _ = generate_i1_data(T=15)

    model, result = fit_dmfm(Y, k1=2, k2=2, P=1, i1_factors=True, max_iter=10)

    # Check drift is zero
    assert np.allclose(model.dynamics.C_drift, 0.0, atol=1e-10)


def test_i1_flag_preserved():
    """Test that i1_factors flag is preserved through EM iterations."""
    from KPOKPCH.DMFM import fit_dmfm

    Y, _, _, _ = generate_i1_data(T=10)

    model, result = fit_dmfm(Y, k1=2, k2=2, P=1, i1_factors=True, max_iter=5)

    # Check flag is still set
    assert model.dynamics.i1_factors is True
    assert model.dynamics.nonstationary is True  # alias


def test_i1_vs_stationary_different():
    """Test that I(1) and stationary modes produce different results."""
    from KPOKPCH.DMFM import fit_dmfm

    Y, _, _, _ = generate_i1_data(T=15)

    model_i1, _ = fit_dmfm(Y, k1=2, k2=2, P=1, i1_factors=True, max_iter=10)
    model_stat, _ = fit_dmfm(Y, k1=2, k2=2, P=1, i1_factors=False, max_iter=10)

    # A, B should be different
    assert not np.allclose(model_i1.A[0], model_stat.A[0])


def test_forecast_with_i1_factors():
    """Test that forecasting works with i1_factors=True."""
    from KPOKPCH.forecast import forecast_dmfm, ForecastConfig

    Y, _, _, _ = generate_i1_data(T=15)

    config = ForecastConfig(k1=2, k2=2, P=1, i1_factors=True, max_iter=10)
    result = forecast_dmfm(Y, steps=4, config=config)

    # Check forecast shape
    assert result.forecast.shape == (4, Y.shape[1], Y.shape[2])
    # Check model has i1_factors set
    assert result.model.dynamics.i1_factors is True


def test_seasonal_diff_still_works():
    """Test that seasonal differencing still works (backward compatibility)."""
    from KPOKPCH.forecast import forecast_dmfm, ForecastConfig

    rng = np.random.default_rng(123)
    # Create seasonal data (period=4)
    T, p1, p2 = 20, 3, 2
    Y = rng.normal(size=(T, p1, p2))
    # Add seasonal pattern
    for t in range(T):
        Y[t] += np.sin(2 * np.pi * t / 4)

    config = ForecastConfig(
        k1=1, k2=1, P=1, seasonal_period=4, max_iter=10, i1_factors=False
    )
    result = forecast_dmfm(Y, steps=4, config=config)

    assert result.seasonal_adjusted is True
    assert result.forecast.shape == (4, p1, p2)
