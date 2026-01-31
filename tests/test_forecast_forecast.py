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
    generate_future_periods,
    compute_q4_growth,
    canton_forecast,
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
# generate_future_periods tests
# ---------------------------------------------------------------------------


class TestGenerateFuturePeriods:
    """Tests for generate_future_periods function."""

    def test_basic_generation(self):
        """Test basic period generation."""
        periods = generate_future_periods("2024Q4", 4)

        assert periods == ["2025Q1", "2025Q2", "2025Q3", "2025Q4"]

    def test_year_rollover(self):
        """Test year rollover from Q4 to Q1."""
        periods = generate_future_periods("2024Q4", 2)

        assert periods == ["2025Q1", "2025Q2"]

    def test_two_year_forecast(self):
        """Test two-year forecast (8 quarters)."""
        periods = generate_future_periods("2024Q4", 8)

        expected = [
            "2025Q1",
            "2025Q2",
            "2025Q3",
            "2025Q4",
            "2026Q1",
            "2026Q2",
            "2026Q3",
            "2026Q4",
        ]
        assert periods == expected

    def test_single_period(self):
        """Test generating single period."""
        periods = generate_future_periods("2024Q3", 1)

        assert periods == ["2024Q4"]

    def test_from_q1(self):
        """Test generation starting from Q1."""
        periods = generate_future_periods("2024Q1", 3)

        assert periods == ["2024Q2", "2024Q3", "2024Q4"]


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


# ---------------------------------------------------------------------------
# compute_q4_growth tests
# ---------------------------------------------------------------------------


class TestComputeQ4Growth:
    """Tests for compute_q4_growth function."""

    def test_basic_growth_calculation(self):
        """Test basic growth calculation."""
        periods = ["2023Q1", "2023Q2", "2023Q3", "2023Q4"]
        data = np.array([100.0, 105.0, 110.0, 120.0])  # Last is Q4
        fcst = np.array(
            [
                125.0,
                130.0,
                135.0,
                140.0,  # Year 1, Q4 at idx 3
                145.0,
                150.0,
                155.0,
                160.0,
            ]
        )  # Year 2, Q4 at idx 7
        future_periods = [
            "2024Q1",
            "2024Q2",
            "2024Q3",
            "2024Q4",
            "2025Q1",
            "2025Q2",
            "2025Q3",
            "2025Q4",
        ]

        result = compute_q4_growth(periods, data, fcst, future_periods)

        assert "growth_y1" in result
        assert "growth_y2" in result
        assert "mean_y1" in result
        assert "mean_y2" in result

    def test_no_q4_in_history_raises(self):
        """Test that no Q4 in history raises ValueError."""
        periods = ["2023Q1", "2023Q2", "2023Q3"]
        data = np.array([100.0, 105.0, 110.0])
        fcst = np.array([125.0, 130.0, 135.0, 140.0])
        future_periods = ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]

        with pytest.raises(ValueError, match="No Q4 observation"):
            compute_q4_growth(periods, data, fcst, future_periods)

    def test_insufficient_future_q4_raises(self):
        """Test that insufficient future Q4s raises ValueError."""
        periods = ["2023Q4"]
        data = np.array([100.0])
        fcst = np.array([125.0, 130.0])  # Only one Q4
        future_periods = ["2024Q1", "2024Q4"]

        with pytest.raises(ValueError, match="Need two future Q4"):
            compute_q4_growth(periods, data, fcst, future_periods)


# ---------------------------------------------------------------------------
# canton_forecast tests
# ---------------------------------------------------------------------------


class TestCantonForecast:
    """Tests for canton_forecast function."""

    @pytest.fixture
    def canton_data(self, rng):
        """Generate canton data for testing."""
        T, num_cantons = 20, 5
        Y = rng.normal(size=(T, num_cantons)) + 100  # Positive values
        return Y

    def test_joint_forecast(self, canton_data):
        """Test joint forecast across cantons."""
        config = ForecastConfig(k1=1, k2=1, max_iter=3)

        fcst, total = canton_forecast(canton_data, steps=4, config=config)

        assert fcst.shape == (4, canton_data.shape[1])
        assert total.shape == (4,)

    def test_total_is_sum(self, canton_data):
        """Test that total is sum of canton forecasts."""
        config = ForecastConfig(k1=1, k2=1, max_iter=3)

        fcst, total = canton_forecast(canton_data, steps=4, config=config)

        np.testing.assert_allclose(total, np.nansum(fcst, axis=1))

    def test_separate_cantons(self, canton_data):
        """Test separate forecasts per canton."""
        config = ForecastConfig(k1=1, k2=1, max_iter=3)

        fcst, total = canton_forecast(
            canton_data, steps=4, config=config, separate_cantons=True
        )

        assert fcst.shape == (4, canton_data.shape[1])

    def test_3d_input(self, canton_data):
        """Test with 3D input."""
        Y_3d = canton_data[:, :, None]
        config = ForecastConfig(k1=1, k2=1, max_iter=3)

        fcst, total = canton_forecast(Y_3d, steps=4, config=config)

        assert fcst.shape == (4, canton_data.shape[1])

    def test_invalid_shape_raises(self, rng):
        """Test that invalid shape raises ValueError."""
        Y_4d = rng.normal(size=(20, 5, 3, 2))

        with pytest.raises(ValueError, match="must be 2D"):
            canton_forecast(Y_4d, steps=4)
