"""Tests for KPOKPCH.forecast.forecast module.

Tests cover ForecastConfig, ForecastResult, forecasting functions,
and utility functions for period generation.
"""

import numpy as np
import pytest

from KPOKPCH.forecast import (
    forecast_dmfm,
    ForecastConfig,
    ForecastResult,
)

# ---------------------------------------------------------------------------
# ForecastConfig tests
# ---------------------------------------------------------------------------


class TestForecastConfig:
    """Tests for ForecastConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ForecastConfig()
        assert config.k1 == 1
        assert config.k2 == 1
        assert config.P == 1
        assert config.max_iter == 50
        assert config.tol == 1e-4
        assert config.diagonal_idiosyncratic is False
        assert config.init_method == "svd"
        assert config.verbose is False
        assert config.i1_factors is False

    def test_custom_config(self):
        """Test configuration with custom values."""
        config = ForecastConfig(
            k1=3,
            k2=2,
            P=2,
            max_iter=100,
            tol=1e-6,
            diagonal_idiosyncratic=True,
            init_method="pe",
            verbose=True,
            i1_factors=True,
        )
        assert config.k1 == 3
        assert config.k2 == 2
        assert config.P == 2
        assert config.max_iter == 100
        assert config.tol == 1e-6
        assert config.diagonal_idiosyncratic is True
        assert config.init_method == "pe"
        assert config.verbose is True
        assert config.i1_factors is True


# ---------------------------------------------------------------------------
# ForecastResult tests
# ---------------------------------------------------------------------------


class TestForecastResult:
    """Tests for ForecastResult dataclass."""

    def test_create_result(self, initialized_model):
        """Test creating ForecastResult."""
        config = ForecastConfig()
        forecast = np.zeros((8, 4, 3))

        result = ForecastResult(
            forecast=forecast,
            model=initialized_model,
            config=config,
        )

        assert result.forecast.shape == (8, 4, 3)
        assert result.model is initialized_model
        assert result.config is config


# ---------------------------------------------------------------------------
# forecast_dmfm tests
# ---------------------------------------------------------------------------


class TestForecastDMFM:
    """Tests for forecast_dmfm function."""

    @pytest.fixture
    def forecast_data(self, rng, small_dims):
        """Generate data for forecasting tests."""
        from conftest import generate_dmfm_data

        return generate_dmfm_data(
            T=small_dims["T"],
            p1=small_dims["p1"],
            p2=small_dims["p2"],
            k1=small_dims["k1"],
            k2=small_dims["k2"],
            rng=rng,
        )

    def test_returns_forecast_result(self, forecast_data):
        """Test that function returns ForecastResult."""
        config = ForecastConfig(k1=2, k2=2, max_iter=3)
        result = forecast_dmfm(
            forecast_data["Y"],
            steps=4,
            config=config,
            mask=forecast_data["mask"],
        )

        assert isinstance(result, ForecastResult)

    def test_forecast_shape(self, forecast_data, small_dims):
        """Test that forecast has correct shape."""
        config = ForecastConfig(k1=2, k2=2, max_iter=3)
        result = forecast_dmfm(
            forecast_data["Y"],
            steps=4,
            config=config,
            mask=forecast_data["mask"],
        )

        assert result.forecast.shape == (4, small_dims["p1"], small_dims["p2"])

    def test_model_fitted(self, forecast_data):
        """Test that model is fitted after forecasting."""
        config = ForecastConfig(k1=2, k2=2, max_iter=3)
        result = forecast_dmfm(
            forecast_data["Y"],
            steps=4,
            config=config,
            mask=forecast_data["mask"],
        )

        assert result.model.is_fitted()

    def test_config_in_result(self, forecast_data):
        """Test that config is stored in result."""
        config = ForecastConfig(k1=3, k2=2, max_iter=5)
        result = forecast_dmfm(
            forecast_data["Y"],
            steps=4,
            config=config,
            mask=forecast_data["mask"],
        )

        assert result.config.k1 == 3
        assert result.config.k2 == 2

    def test_without_config(self, forecast_data):
        """Test forecasting without explicit config."""
        result = forecast_dmfm(
            forecast_data["Y"],
            steps=4,
            k1=2,
            k2=2,
            max_iter=3,
        )

        assert result.config.k1 == 2
        assert result.config.k2 == 2

    def test_kwargs_override_config(self, forecast_data):
        """Test that kwargs override config values."""
        config = ForecastConfig(k1=1, k2=1, max_iter=3)
        result = forecast_dmfm(
            forecast_data["Y"],
            steps=4,
            config=config,
            k1=2,  # Override
        )

        assert result.config.k1 == 2
        assert result.config.k2 == 1  # Not overridden

    def test_invalid_steps_raises(self, forecast_data):
        """Test that invalid steps raises ValueError."""
        with pytest.raises(ValueError, match="steps must be positive"):
            forecast_dmfm(forecast_data["Y"], steps=0)

        with pytest.raises(ValueError, match="steps must be positive"):
            forecast_dmfm(forecast_data["Y"], steps=-1)

    def test_invalid_y_shape_raises(self, rng):
        """Test that invalid Y shape raises ValueError."""
        Y_2d = rng.normal(size=(20, 5))

        with pytest.raises(ValueError, match="must be 3D array"):
            forecast_dmfm(Y_2d, steps=4)

    def test_forecast_not_nan(self, forecast_data):
        """Test that forecast does not contain NaN."""
        config = ForecastConfig(k1=2, k2=2, max_iter=3)
        result = forecast_dmfm(
            forecast_data["Y"],
            steps=4,
            config=config,
            mask=forecast_data["mask"],
        )

        assert not np.isnan(result.forecast).any()
