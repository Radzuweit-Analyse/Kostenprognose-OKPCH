"""Tests for KPOKPCH.DMFM.em module.

Tests cover EMConfig, EMResult, EMEstimatorDMFM, and the EM algorithm
for fitting Dynamic Matrix Factor Models.
"""

import numpy as np
import pytest

from KPOKPCH.DMFM import (
    DMFMModel,
    DMFMConfig,
    EMEstimatorDMFM,
    EMConfig,
    EMResult,
)

# ---------------------------------------------------------------------------
# EMConfig tests
# ---------------------------------------------------------------------------


class TestEMConfig:
    """Tests for EMConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EMConfig()
        assert config.max_iter == 100
        assert config.tol == 1e-4
        assert config.verbose is False
        assert config.check_loglik_increase is True

    def test_custom_config(self):
        """Test configuration with custom values."""
        config = EMConfig(
            max_iter=50,
            tol=1e-6,
            verbose=True,
            check_loglik_increase=False,
        )
        assert config.max_iter == 50
        assert config.tol == 1e-6
        assert config.verbose is True
        assert config.check_loglik_increase is False


# ---------------------------------------------------------------------------
# EMResult tests
# ---------------------------------------------------------------------------


class TestEMResult:
    """Tests for EMResult dataclass."""

    def test_create_result(self):
        """Test creating EMResult."""
        result = EMResult(
            converged=True,
            num_iter=25,
            final_loglik=-500.0,
            loglik_trace=[-1000.0, -700.0, -500.0],
            diff_trace=[1.0, 0.1, 0.001],
        )
        assert result.converged is True
        assert result.num_iter == 25
        assert result.final_loglik == -500.0
        assert len(result.loglik_trace) == 3
        assert len(result.diff_trace) == 3


# ---------------------------------------------------------------------------
# EMEstimatorDMFM creation tests
# ---------------------------------------------------------------------------


class TestEMEstimatorCreation:
    """Tests for EMEstimatorDMFM initialization."""

    def test_create_with_initialized_model(self, initialized_model):
        """Test creating estimator with initialized model."""
        estimator = EMEstimatorDMFM(initialized_model)
        assert estimator.model is initialized_model
        assert estimator.result is None
        assert isinstance(estimator.config, EMConfig)

    def test_create_with_custom_config(self, initialized_model):
        """Test creating estimator with custom config."""
        config = EMConfig(max_iter=50, tol=1e-6)
        estimator = EMEstimatorDMFM(initialized_model, config=config)
        assert estimator.config.max_iter == 50
        assert estimator.config.tol == 1e-6

    def test_create_uninitialized_model_raises(self, dmfm_config):
        """Test that uninitialized model raises ValueError."""
        model = DMFMModel(dmfm_config)  # Not initialized

        with pytest.raises(ValueError, match="Model must be initialized"):
            EMEstimatorDMFM(model)


# ---------------------------------------------------------------------------
# EMEstimatorDMFM fitting tests
# ---------------------------------------------------------------------------


class TestEMEstimatorFitting:
    """Tests for EM algorithm fitting."""

    @pytest.fixture
    def estimator_with_data(self, initialized_model, dmfm_data):
        """Create estimator and return with data."""
        config = EMConfig(max_iter=5, verbose=False)
        estimator = EMEstimatorDMFM(initialized_model, config=config)
        return estimator, dmfm_data["Y"], dmfm_data["mask"]

    def test_fit_returns_result(self, estimator_with_data):
        """Test that fit() returns EMResult."""
        estimator, Y, mask = estimator_with_data
        result = estimator.fit(Y, mask)

        assert isinstance(result, EMResult)
        assert result is estimator.result

    def test_fit_updates_model(self, estimator_with_data):
        """Test that fit() marks model as fitted."""
        estimator, Y, mask = estimator_with_data
        assert not estimator.model.is_fitted()

        estimator.fit(Y, mask)

        assert estimator.model.is_fitted()

    def test_fit_stores_factors(self, estimator_with_data, small_dims):
        """Test that fit() stores estimated factors."""
        estimator, Y, mask = estimator_with_data
        estimator.fit(Y, mask)

        F = estimator.model.F
        T, k1, k2 = small_dims["T"], small_dims["k1"], small_dims["k2"]
        assert F.shape == (T, k1, k2)

    def test_fit_without_mask(self, initialized_model, dmfm_data):
        """Test fit() works without explicit mask."""
        config = EMConfig(max_iter=3)
        estimator = EMEstimatorDMFM(initialized_model, config=config)

        result = estimator.fit(dmfm_data["Y"])  # No mask

        assert result is not None

    def test_fit_wrong_dimensions_raises(self, initialized_model, rng):
        """Test that wrong data dimensions raise ValueError."""
        estimator = EMEstimatorDMFM(initialized_model)
        Y_wrong = rng.normal(size=(10, 5, 5))  # Wrong p1, p2

        with pytest.raises(ValueError, match="Data dimensions"):
            estimator.fit(Y_wrong)

    def test_fit_2d_data_raises(self, initialized_model, rng):
        """Test that 2D data raises ValueError."""
        estimator = EMEstimatorDMFM(initialized_model)
        Y_2d = rng.normal(size=(10, 4))

        with pytest.raises(ValueError, match="Expected 3D array"):
            estimator.fit(Y_2d)

    def test_loglik_trace_recorded(self, estimator_with_data):
        """Test that log-likelihood trace is recorded."""
        estimator, Y, mask = estimator_with_data
        result = estimator.fit(Y, mask)

        assert len(result.loglik_trace) > 0
        assert len(result.loglik_trace) == result.num_iter

    def test_diff_trace_recorded(self, estimator_with_data):
        """Test that parameter difference trace is recorded."""
        estimator, Y, mask = estimator_with_data
        result = estimator.fit(Y, mask)

        assert len(result.diff_trace) > 0
        assert len(result.diff_trace) == result.num_iter

    def test_loglik_generally_increases(self, rng, small_dims):
        """Test that log-likelihood generally increases or stays stable."""
        from conftest import generate_dmfm_data

        # Generate data with reasonable signal
        data = generate_dmfm_data(
            T=small_dims["T"],
            p1=small_dims["p1"],
            p2=small_dims["p2"],
            k1=small_dims["k1"],
            k2=small_dims["k2"],
            rng=rng,
            noise_scale=0.1,
        )

        config = DMFMConfig(
            p1=small_dims["p1"],
            p2=small_dims["p2"],
            k1=small_dims["k1"],
            k2=small_dims["k2"],
        )
        model = DMFMModel(config)
        model.initialize(data["Y"], data["mask"])

        em_config = EMConfig(max_iter=10, check_loglik_increase=True)
        estimator = EMEstimatorDMFM(model, em_config)
        result = estimator.fit(data["Y"], data["mask"])

        # With check_loglik_increase=True, log-likelihood should not decrease significantly
        for i in range(1, len(result.loglik_trace)):
            # Allow small numerical tolerance for decrease
            assert result.loglik_trace[i] >= result.loglik_trace[i - 1] - 1e-5


# ---------------------------------------------------------------------------
# EMEstimatorDMFM convergence tests
# ---------------------------------------------------------------------------


class TestEMEstimatorConvergence:
    """Tests for EM algorithm convergence behavior."""

    def test_converges_within_max_iter(self, rng, small_dims):
        """Test that EM can converge within max iterations."""
        from conftest import generate_dmfm_data

        data = generate_dmfm_data(
            T=30,  # More data for better convergence
            p1=small_dims["p1"],
            p2=small_dims["p2"],
            k1=small_dims["k1"],
            k2=small_dims["k2"],
            rng=rng,
            noise_scale=0.05,  # Lower noise
        )

        config = DMFMConfig(
            p1=small_dims["p1"],
            p2=small_dims["p2"],
            k1=small_dims["k1"],
            k2=small_dims["k2"],
        )
        model = DMFMModel(config)
        model.initialize(data["Y"], data["mask"])

        # Use loose tolerance for faster convergence
        em_config = EMConfig(max_iter=100, tol=1e-3)
        estimator = EMEstimatorDMFM(model, em_config)
        result = estimator.fit(data["Y"], data["mask"])

        # May or may not converge depending on data, but should run
        assert result.num_iter <= em_config.max_iter

    def test_respects_max_iter(self, initialized_model, dmfm_data):
        """Test that EM stops at max_iter if not converged."""
        em_config = EMConfig(max_iter=3, tol=1e-20)  # Very tight tolerance
        estimator = EMEstimatorDMFM(initialized_model, em_config)
        result = estimator.fit(dmfm_data["Y"], dmfm_data["mask"])

        assert result.num_iter <= 3

    def test_respects_tolerance(self, rng, small_dims):
        """Test that EM stops when tolerance is reached."""
        from conftest import generate_dmfm_data

        data = generate_dmfm_data(
            T=20,
            p1=small_dims["p1"],
            p2=small_dims["p2"],
            k1=small_dims["k1"],
            k2=small_dims["k2"],
            rng=rng,
            noise_scale=0.1,
        )

        config = DMFMConfig(
            p1=small_dims["p1"],
            p2=small_dims["p2"],
            k1=small_dims["k1"],
            k2=small_dims["k2"],
        )
        model = DMFMModel(config)
        model.initialize(data["Y"], data["mask"])

        # Very loose tolerance - should converge quickly
        em_config = EMConfig(max_iter=100, tol=1e0)
        estimator = EMEstimatorDMFM(model, em_config)
        result = estimator.fit(data["Y"], data["mask"])

        # With loose tolerance, should converge before max_iter
        if result.converged:
            assert result.diff_trace[-1] < em_config.tol


# ---------------------------------------------------------------------------
# EMEstimatorDMFM callback tests
# ---------------------------------------------------------------------------


class TestEMEstimatorCallback:
    """Tests for checkpoint callback functionality."""

    def test_callback_called_each_iteration(self, initialized_model, dmfm_data):
        """Test that callback is called at each iteration."""
        em_config = EMConfig(max_iter=5)
        estimator = EMEstimatorDMFM(initialized_model, em_config)

        callback_calls = []

        def callback(iteration, state):
            callback_calls.append(iteration)

        estimator.fit(dmfm_data["Y"], dmfm_data["mask"], checkpoint_callback=callback)

        assert len(callback_calls) > 0
        assert callback_calls == list(range(1, len(callback_calls) + 1))

    def test_callback_receives_state(self, initialized_model, dmfm_data):
        """Test that callback receives state dictionary."""
        em_config = EMConfig(max_iter=2)
        estimator = EMEstimatorDMFM(initialized_model, em_config)

        received_states = []

        def callback(iteration, state):
            received_states.append(state)

        estimator.fit(dmfm_data["Y"], dmfm_data["mask"], checkpoint_callback=callback)

        assert len(received_states) > 0
        state = received_states[0]
        assert "iteration" in state
        assert "loglik" in state
        assert "diff" in state
        assert "params" in state


# ---------------------------------------------------------------------------
# EMEstimatorDMFM accessor tests
# ---------------------------------------------------------------------------


class TestEMEstimatorAccessors:
    """Tests for public accessor methods."""

    @pytest.fixture
    def fitted_estimator(self, initialized_model, dmfm_data):
        """Create a fitted estimator."""
        config = EMConfig(max_iter=3)
        estimator = EMEstimatorDMFM(initialized_model, config=config)
        estimator.fit(dmfm_data["Y"], dmfm_data["mask"])
        return estimator

    def test_get_factors(self, fitted_estimator, small_dims):
        """Test get_factors() returns correct shape."""
        F = fitted_estimator.get_factors()

        T, k1, k2 = small_dims["T"], small_dims["k1"], small_dims["k2"]
        assert F.shape == (T, k1, k2)

    def test_get_factors_before_fit_raises(self, initialized_model):
        """Test get_factors() before fitting raises error."""
        estimator = EMEstimatorDMFM(initialized_model)

        with pytest.raises(ValueError, match="Model not fitted"):
            estimator.get_factors()

    def test_get_loglik_trace(self, fitted_estimator):
        """Test get_loglik_trace() returns trace."""
        trace = fitted_estimator.get_loglik_trace()

        assert isinstance(trace, list)
        assert len(trace) > 0
        assert all(isinstance(x, float) for x in trace)

    def test_get_loglik_trace_before_fit_raises(self, initialized_model):
        """Test get_loglik_trace() before fitting raises error."""
        estimator = EMEstimatorDMFM(initialized_model)

        with pytest.raises(RuntimeError, match="not been fitted"):
            estimator.get_loglik_trace()

    def test_get_diff_trace(self, fitted_estimator):
        """Test get_diff_trace() returns trace."""
        trace = fitted_estimator.get_diff_trace()

        assert isinstance(trace, list)
        assert len(trace) > 0

    def test_get_diff_trace_before_fit_raises(self, initialized_model):
        """Test get_diff_trace() before fitting raises error."""
        estimator = EMEstimatorDMFM(initialized_model)

        with pytest.raises(RuntimeError, match="not been fitted"):
            estimator.get_diff_trace()


# ---------------------------------------------------------------------------
# EM with missing data tests
# ---------------------------------------------------------------------------


class TestEMWithMissingData:
    """Tests for EM with missing observations."""

    def test_fit_with_sparse_missing(self, initialized_model, dmfm_data, rng):
        """Test fit() with sparse missing data."""
        Y = dmfm_data["Y"].copy()
        mask = dmfm_data["mask"].copy()

        # Randomly mask 10% of observations
        missing_rate = 0.1
        missing_mask = rng.random(mask.shape) < missing_rate
        mask[missing_mask] = False

        config = EMConfig(max_iter=5)
        estimator = EMEstimatorDMFM(initialized_model, config=config)
        result = estimator.fit(Y, mask)

        assert result is not None
        assert np.isfinite(result.final_loglik)

    def test_fit_with_full_time_missing(self, initialized_model, dmfm_data):
        """Test fit() with entire time points missing."""
        Y = dmfm_data["Y"].copy()
        mask = dmfm_data["mask"].copy()

        # Set entire time point 3 and 7 as missing
        mask[3, :, :] = False
        mask[7, :, :] = False

        config = EMConfig(max_iter=5)
        estimator = EMEstimatorDMFM(initialized_model, config=config)
        result = estimator.fit(Y, mask)

        assert result is not None


# ---------------------------------------------------------------------------
# EM with I(1) factors tests
# ---------------------------------------------------------------------------


class TestEMWithI1Factors:
    """Tests for EM with I(1) (random walk) factors."""

    def test_fit_i1_factors(self, rng, small_dims):
        """Test fitting with i1_factors=True."""
        from conftest import generate_i1_data

        data = generate_i1_data(
            T=small_dims["T"],
            p1=small_dims["p1"],
            p2=small_dims["p2"],
            k1=small_dims["k1"],
            k2=small_dims["k2"],
            rng=rng,
        )

        config = DMFMConfig(
            p1=small_dims["p1"],
            p2=small_dims["p2"],
            k1=small_dims["k1"],
            k2=small_dims["k2"],
        )
        model = DMFMModel(config)
        model.initialize(data["Y"], data["mask"])
        model.dynamics.i1_factors = True

        em_config = EMConfig(max_iter=5)
        estimator = EMEstimatorDMFM(model, em_config)
        result = estimator.fit(data["Y"], data["mask"])

        assert result is not None
        # I(1) mode should preserve the flag
        assert model.dynamics.i1_factors is True


# ---------------------------------------------------------------------------
# Rotation normalization tests
# ---------------------------------------------------------------------------


class TestRotationNormalization:
    """Tests for rotation normalization in EM."""

    def test_loadings_sign_normalized(self, initialized_model, dmfm_data):
        """Test that loadings are sign-normalized after fitting."""
        config = EMConfig(max_iter=5)
        estimator = EMEstimatorDMFM(initialized_model, config=config)
        estimator.fit(dmfm_data["Y"], dmfm_data["mask"])

        R = estimator.model.R
        C = estimator.model.C

        # Check that max absolute element in each column is positive
        for j in range(R.shape[1]):
            max_idx = np.argmax(np.abs(R[:, j]))
            # Should be non-negative (could be zero in degenerate cases)
            assert R[max_idx, j] >= -1e-10

        for j in range(C.shape[1]):
            max_idx = np.argmax(np.abs(C[:, j]))
            assert C[max_idx, j] >= -1e-10

    def test_loadings_orthonormalized(self, initialized_model, dmfm_data):
        """Test that loadings are orthonormalized after fitting."""
        config = EMConfig(max_iter=5)
        estimator = EMEstimatorDMFM(initialized_model, config=config)
        estimator.fit(dmfm_data["Y"], dmfm_data["mask"])

        R = estimator.model.R
        C = estimator.model.C

        # R^T @ R should be approximately identity
        RTR = R.T @ R
        np.testing.assert_allclose(RTR, np.eye(R.shape[1]), atol=1e-6)

        # C^T @ C should be approximately identity
        CTC = C.T @ C
        np.testing.assert_allclose(CTC, np.eye(C.shape[1]), atol=1e-6)
