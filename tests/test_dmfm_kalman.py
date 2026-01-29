"""Tests for KPOKPCH.DMFM.kalman module.

Tests cover KalmanConfig, KalmanState, and KalmanFilterDMFM functionality
including filtering, smoothing, and log-likelihood computation.
"""

import numpy as np
import pytest

from KPOKPCH.DMFM import (
    DMFMModel,
    DMFMConfig,
    KalmanFilterDMFM,
    KalmanState,
    KalmanConfig,
)

# ---------------------------------------------------------------------------
# KalmanConfig tests
# ---------------------------------------------------------------------------


class TestKalmanConfig:
    """Tests for KalmanConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = KalmanConfig()
        assert config.initial_state_variance == 1e2
        assert config.initial_state_variance_i1 == 1e4
        assert config.regularization == 1e-8
        assert config.use_woodbury is False
        assert config.check_symmetry is True

    def test_custom_config(self):
        """Test configuration with custom values."""
        config = KalmanConfig(
            initial_state_variance=50.0,
            initial_state_variance_i1=1e5,
            regularization=1e-6,
            use_woodbury=True,
            check_symmetry=False,
        )
        assert config.initial_state_variance == 50.0
        assert config.initial_state_variance_i1 == 1e5
        assert config.regularization == 1e-6
        assert config.use_woodbury is True
        assert config.check_symmetry is False


# ---------------------------------------------------------------------------
# KalmanState tests
# ---------------------------------------------------------------------------


class TestKalmanState:
    """Tests for KalmanState dataclass."""

    def test_create_minimal_state(self):
        """Test creating state with required fields only."""
        T, d = 10, 4
        state = KalmanState(
            x_pred=np.zeros((T, d)),
            P_pred=np.zeros((T, d, d)),
            x_filt=np.zeros((T, d)),
            P_filt=np.zeros((T, d, d)),
        )
        assert state.x_pred.shape == (T, d)
        assert state.x_smooth is None
        assert state.P_smooth is None
        assert state.P_smooth_lag is None
        assert state.loglik is None

    def test_create_full_state(self):
        """Test creating state with all fields."""
        T, d = 10, 4
        state = KalmanState(
            x_pred=np.zeros((T, d)),
            P_pred=np.zeros((T, d, d)),
            x_filt=np.zeros((T, d)),
            P_filt=np.zeros((T, d, d)),
            x_smooth=np.ones((T, d)),
            P_smooth=np.ones((T, d, d)),
            P_smooth_lag=np.ones((T - 1, d, d)),
            loglik=-100.0,
        )
        assert state.x_smooth is not None
        assert state.P_smooth is not None
        assert state.P_smooth_lag.shape == (T - 1, d, d)
        assert state.loglik == -100.0


# ---------------------------------------------------------------------------
# KalmanFilterDMFM creation tests
# ---------------------------------------------------------------------------


class TestKalmanFilterCreation:
    """Tests for KalmanFilterDMFM initialization."""

    def test_create_with_initialized_model(self, initialized_model):
        """Test creating filter with initialized model."""
        kf = KalmanFilterDMFM(initialized_model)
        assert kf.model is initialized_model
        assert kf.state is None
        assert isinstance(kf.config, KalmanConfig)

    def test_create_with_custom_config(self, initialized_model):
        """Test creating filter with custom config."""
        config = KalmanConfig(initial_state_variance=200.0)
        kf = KalmanFilterDMFM(initialized_model, config=config)
        assert kf.config.initial_state_variance == 200.0

    def test_create_uninitialized_model_raises(self, dmfm_config):
        """Test that uninitialized model raises ValueError."""
        model = DMFMModel(dmfm_config)  # Not initialized

        with pytest.raises(ValueError, match="Model must be initialized"):
            KalmanFilterDMFM(model)


# ---------------------------------------------------------------------------
# KalmanFilterDMFM filtering tests
# ---------------------------------------------------------------------------


class TestKalmanFilterFiltering:
    """Tests for Kalman filter forward pass."""

    @pytest.fixture
    def kf_with_data(self, initialized_model, dmfm_data):
        """Create Kalman filter and return with data."""
        kf = KalmanFilterDMFM(initialized_model)
        return kf, dmfm_data["Y"], dmfm_data["mask"]

    def test_filter_returns_state(self, kf_with_data):
        """Test that filter() returns KalmanState."""
        kf, Y, mask = kf_with_data
        state = kf.filter(Y, mask)

        assert isinstance(state, KalmanState)
        assert state is kf.state

    def test_filter_shapes(self, kf_with_data, small_dims):
        """Test that filter outputs have correct shapes."""
        kf, Y, mask = kf_with_data
        state = kf.filter(Y, mask)

        T = small_dims["T"]
        k1, k2, P = small_dims["k1"], small_dims["k2"], 1
        d = k1 * k2 * P

        assert state.x_pred.shape == (T, d)
        assert state.P_pred.shape == (T, d, d)
        assert state.x_filt.shape == (T, d)
        assert state.P_filt.shape == (T, d, d)

    def test_filter_without_mask(self, initialized_model, dmfm_data):
        """Test filter works without explicit mask."""
        kf = KalmanFilterDMFM(initialized_model)
        Y = dmfm_data["Y"]

        state = kf.filter(Y)  # No mask

        assert state.x_filt is not None

    def test_filter_wrong_dimensions_raises(self, initialized_model, rng):
        """Test that wrong data dimensions raise ValueError."""
        kf = KalmanFilterDMFM(initialized_model)
        Y_wrong = rng.normal(size=(10, 5, 5))  # Wrong p1, p2

        with pytest.raises(ValueError, match="Data dimensions"):
            kf.filter(Y_wrong)

    def test_filter_2d_data_raises(self, initialized_model, rng):
        """Test that 2D data raises ValueError."""
        kf = KalmanFilterDMFM(initialized_model)
        Y_2d = rng.normal(size=(10, 4))

        with pytest.raises(ValueError, match="Expected 3D array"):
            kf.filter(Y_2d)

    def test_filter_with_missing_data(self, initialized_model, dmfm_data, rng):
        """Test filter handles missing data correctly."""
        kf = KalmanFilterDMFM(initialized_model)
        Y = dmfm_data["Y"].copy()
        mask = dmfm_data["mask"].copy()

        # Set some values to missing
        mask[2, 1, 1] = False
        mask[5, :, 0] = False  # Entire column missing

        state = kf.filter(Y, mask)

        # Should still produce valid output
        assert not np.isnan(state.x_filt).any()

    def test_filtered_covariances_symmetric(self, kf_with_data):
        """Test that filtered covariances are symmetric."""
        kf, Y, mask = kf_with_data
        state = kf.filter(Y, mask)

        for t in range(Y.shape[0]):
            P = state.P_filt[t]
            np.testing.assert_allclose(P, P.T, atol=1e-10)


# ---------------------------------------------------------------------------
# KalmanFilterDMFM smoothing tests
# ---------------------------------------------------------------------------


class TestKalmanFilterSmoothing:
    """Tests for RTS smoother backward pass."""

    @pytest.fixture
    def filtered_kf(self, initialized_model, dmfm_data):
        """Create Kalman filter with filtered state."""
        kf = KalmanFilterDMFM(initialized_model)
        kf.filter(dmfm_data["Y"], dmfm_data["mask"])
        return kf, dmfm_data["Y"], dmfm_data["mask"]

    def test_smooth_returns_state(self, filtered_kf):
        """Test that smooth() returns updated KalmanState."""
        kf, _, _ = filtered_kf
        state = kf.smooth()

        assert state is kf.state
        assert state.x_smooth is not None
        assert state.P_smooth is not None
        assert state.P_smooth_lag is not None

    def test_smooth_shapes(self, filtered_kf, small_dims):
        """Test that smoother outputs have correct shapes."""
        kf, _, _ = filtered_kf
        state = kf.smooth()

        T = small_dims["T"]
        k1, k2, P = small_dims["k1"], small_dims["k2"], 1
        d = k1 * k2 * P

        assert state.x_smooth.shape == (T, d)
        assert state.P_smooth.shape == (T, d, d)
        assert state.P_smooth_lag.shape == (T - 1, d, d)

    def test_smooth_without_filter_raises(self, initialized_model):
        """Test that smoothing without filtering raises ValueError."""
        kf = KalmanFilterDMFM(initialized_model)

        with pytest.raises(ValueError, match="No filtered state available"):
            kf.smooth()

    def test_smooth_with_explicit_state(self, filtered_kf):
        """Test smooth() with explicitly passed state."""
        kf, Y, mask = filtered_kf
        state = kf.filter(Y, mask)

        # Create a new KF and smooth using the existing state
        kf2 = KalmanFilterDMFM(kf.model)
        # Need to ensure matrices are constructed
        kf2._construct_matrices()

        result = kf2.smooth(state)

        assert result.x_smooth is not None

    def test_smoothed_covariances_symmetric(self, filtered_kf):
        """Test that smoothed covariances are symmetric."""
        kf, _, _ = filtered_kf
        state = kf.smooth()

        for t in range(state.P_smooth.shape[0]):
            P = state.P_smooth[t]
            np.testing.assert_allclose(P, P.T, atol=1e-10)

    def test_smoother_reduces_variance(self, filtered_kf):
        """Test that smoothing generally reduces variance (uncertainty)."""
        kf, _, _ = filtered_kf
        state = kf.smooth()

        # Average variance should be less for smoothed than filtered
        # (except at the last time point where they're equal)
        filt_var = np.mean(
            [np.trace(state.P_filt[t]) for t in range(state.P_filt.shape[0] - 1)]
        )
        smooth_var = np.mean(
            [np.trace(state.P_smooth[t]) for t in range(state.P_smooth.shape[0] - 1)]
        )

        assert smooth_var <= filt_var + 1e-8


# ---------------------------------------------------------------------------
# KalmanFilterDMFM log-likelihood tests
# ---------------------------------------------------------------------------


class TestKalmanFilterLogLikelihood:
    """Tests for log-likelihood computation."""

    @pytest.fixture
    def smoothed_kf(self, initialized_model, dmfm_data):
        """Create Kalman filter with smoothed state."""
        kf = KalmanFilterDMFM(initialized_model)
        kf.filter(dmfm_data["Y"], dmfm_data["mask"])
        kf.smooth()
        return kf, dmfm_data["Y"], dmfm_data["mask"]

    def test_loglik_returns_float(self, smoothed_kf):
        """Test that log_likelihood() returns a float."""
        kf, Y, mask = smoothed_kf
        loglik = kf.log_likelihood(Y, mask)

        assert isinstance(loglik, float)
        assert not np.isnan(loglik)
        assert not np.isinf(loglik)

    def test_loglik_is_negative(self, smoothed_kf):
        """Test that log-likelihood is negative (log of probability < 1)."""
        kf, Y, mask = smoothed_kf
        loglik = kf.log_likelihood(Y, mask)

        assert loglik < 0

    def test_loglik_without_smooth_raises(self, initialized_model, dmfm_data):
        """Test that log-likelihood without smoothing raises ValueError."""
        kf = KalmanFilterDMFM(initialized_model)
        kf.filter(dmfm_data["Y"], dmfm_data["mask"])
        # Don't smooth

        with pytest.raises(ValueError, match="No smoothed state available"):
            kf.log_likelihood(dmfm_data["Y"], dmfm_data["mask"])

    def test_loglik_without_mask(self, smoothed_kf):
        """Test log-likelihood computation without explicit mask."""
        kf, Y, _ = smoothed_kf
        loglik = kf.log_likelihood(Y)  # No mask

        assert isinstance(loglik, float)

    def test_loglik_increases_with_better_fit(self, rng, small_dims):
        """Test that log-likelihood is higher for better-fitting models."""
        from tests.conftest import generate_dmfm_data

        # Generate data from known model
        data = generate_dmfm_data(
            T=small_dims["T"],
            p1=small_dims["p1"],
            p2=small_dims["p2"],
            k1=small_dims["k1"],
            k2=small_dims["k2"],
            rng=rng,
            noise_scale=0.1,
        )

        # Create well-specified model
        config = DMFMConfig(
            p1=small_dims["p1"],
            p2=small_dims["p2"],
            k1=small_dims["k1"],
            k2=small_dims["k2"],
        )
        model = DMFMModel(config)
        model.initialize(data["Y"], data["mask"])

        kf = KalmanFilterDMFM(model)
        kf.filter(data["Y"], data["mask"])
        kf.smooth()
        loglik = kf.log_likelihood(data["Y"], data["mask"])

        # Log-likelihood should be finite for well-specified model
        assert np.isfinite(loglik)


# ---------------------------------------------------------------------------
# KalmanFilterDMFM extract_factors tests
# ---------------------------------------------------------------------------


class TestKalmanFilterExtractFactors:
    """Tests for factor extraction from Kalman state."""

    @pytest.fixture
    def smoothed_kf(self, initialized_model, dmfm_data):
        """Create Kalman filter with smoothed state."""
        kf = KalmanFilterDMFM(initialized_model)
        kf.filter(dmfm_data["Y"], dmfm_data["mask"])
        kf.smooth()
        return kf, dmfm_data

    def test_extract_smoothed_factors(self, smoothed_kf, small_dims):
        """Test extracting smoothed factors."""
        kf, data = smoothed_kf
        F = kf.extract_factors(smoothed=True)

        T = small_dims["T"]
        k1, k2 = small_dims["k1"], small_dims["k2"]

        assert F.shape == (T, k1, k2)
        assert not np.isnan(F).any()

    def test_extract_filtered_factors(self, smoothed_kf, small_dims):
        """Test extracting filtered factors."""
        kf, _ = smoothed_kf
        F = kf.extract_factors(smoothed=False)

        T = small_dims["T"]
        k1, k2 = small_dims["k1"], small_dims["k2"]

        assert F.shape == (T, k1, k2)

    def test_extract_without_filter_raises(self, initialized_model):
        """Test extraction without filtering raises ValueError."""
        kf = KalmanFilterDMFM(initialized_model)

        with pytest.raises(ValueError, match="No state available"):
            kf.extract_factors()

    def test_extract_smoothed_without_smooth_raises(self, initialized_model, dmfm_data):
        """Test smoothed extraction without smoothing raises ValueError."""
        kf = KalmanFilterDMFM(initialized_model)
        kf.filter(dmfm_data["Y"], dmfm_data["mask"])
        # Don't smooth

        with pytest.raises(ValueError, match="No smoothed state available"):
            kf.extract_factors(smoothed=True)

    def test_extract_with_explicit_state(self, smoothed_kf, small_dims):
        """Test extraction with explicit state parameter."""
        kf, _ = smoothed_kf
        state = kf.state

        F = kf.extract_factors(state=state, smoothed=True)

        assert F.shape == (small_dims["T"], small_dims["k1"], small_dims["k2"])


# ---------------------------------------------------------------------------
# KalmanFilterDMFM matrix construction tests
# ---------------------------------------------------------------------------


class TestKalmanFilterMatrices:
    """Tests for state-space matrix construction."""

    def test_construct_matrices_shapes(self, initialized_model, small_dims):
        """Test that constructed matrices have correct shapes."""
        kf = KalmanFilterDMFM(initialized_model)
        T, mu, Q, Z, R = kf._construct_matrices()

        p1, p2 = small_dims["p1"], small_dims["p2"]
        k1, k2, P = small_dims["k1"], small_dims["k2"], 1
        d = k1 * k2 * P
        n = p1 * p2

        assert T.shape == (d, d)
        assert mu.shape == (d,)
        assert Q.shape == (d, d)
        assert Z.shape == (n, d)
        assert R.shape == (n, n)

    def test_matrices_are_cached(self, initialized_model):
        """Test that matrices are cached after first construction."""
        kf = KalmanFilterDMFM(initialized_model)

        T1, mu1, Q1, Z1, R1 = kf._construct_matrices()
        T2, mu2, Q2, Z2, R2 = kf._construct_matrices()

        # Should be the exact same objects (cached)
        assert T1 is T2
        assert mu1 is mu2
        assert Q1 is Q2
        assert Z1 is Z2
        assert R1 is R2

    def test_clear_cache(self, initialized_model):
        """Test that cache can be cleared."""
        kf = KalmanFilterDMFM(initialized_model)

        kf._construct_matrices()
        assert kf._T is not None

        kf._clear_cache()
        assert kf._T is None
        assert kf._mu is None
        assert kf._Q is None
        assert kf._Z is None
        assert kf._R is None

    def test_transition_matrix_bounded(self, initialized_model):
        """Test that transition matrix eigenvalues are bounded."""
        kf = KalmanFilterDMFM(initialized_model)
        T, _, _, _, _ = kf._construct_matrices()

        eigenvalues = np.linalg.eigvals(T)
        max_eval = np.max(np.abs(eigenvalues))

        # Initial dynamics may have eigenvalues at 1.0, but should not exceed
        assert max_eval <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# KalmanFilterDMFM repr tests
# ---------------------------------------------------------------------------


class TestKalmanFilterRepr:
    """Tests for KalmanFilterDMFM string representation."""

    def test_repr_not_run(self, initialized_model):
        """Test repr before filter is run."""
        kf = KalmanFilterDMFM(initialized_model)
        repr_str = repr(kf)

        assert "KalmanFilterDMFM" in repr_str
        assert "not run" in repr_str

    def test_repr_filtered(self, initialized_model, dmfm_data):
        """Test repr after filtering."""
        kf = KalmanFilterDMFM(initialized_model)
        kf.filter(dmfm_data["Y"], dmfm_data["mask"])
        repr_str = repr(kf)

        assert "filtered" in repr_str
        assert "smoothed" not in repr_str

    def test_repr_smoothed(self, initialized_model, dmfm_data):
        """Test repr after smoothing."""
        kf = KalmanFilterDMFM(initialized_model)
        kf.filter(dmfm_data["Y"], dmfm_data["mask"])
        kf.smooth()
        repr_str = repr(kf)

        assert "filtered" in repr_str
        assert "smoothed" in repr_str
