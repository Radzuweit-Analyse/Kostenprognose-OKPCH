"""Tests for KPOKPCH.forecast.forecast module.

Tests cover ForecastConfig, ForecastResult, forecasting functions,
and utility functions for seasonal differencing and period generation.
"""

import numpy as np
import pytest

from KPOKPCH.forecast import (
    forecast_dmfm,
    ForecastConfig,
    ForecastResult,
    seasonal_difference,
    integrate_seasonal_diff,
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
        assert config.seasonal_period is None
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
            seasonal_period=4,
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
        assert config.seasonal_period == 4
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
            seasonal_adjusted=True,
        )

        assert result.forecast.shape == (8, 4, 3)
        assert result.model is initialized_model
        assert result.config is config
        assert result.seasonal_adjusted is True


# ---------------------------------------------------------------------------
# seasonal_difference tests
# ---------------------------------------------------------------------------


class TestSeasonalDifference:
    """Tests for seasonal_difference function."""

    def test_basic_differencing(self, rng):
        """Test basic seasonal differencing."""
        T, p1, p2 = 20, 4, 3
        Y = rng.normal(size=(T, p1, p2))

        Y_diff = seasonal_difference(Y, period=4)

        assert Y_diff.shape == (T - 4, p1, p2)

    def test_difference_calculation(self, rng):
        """Test that differences are calculated correctly."""
        T, p1, p2 = 20, 4, 3
        Y = rng.normal(size=(T, p1, p2))

        Y_diff = seasonal_difference(Y, period=4)

        # Verify calculation
        expected = Y[4:] - Y[:-4]
        np.testing.assert_allclose(Y_diff, expected)

    def test_period_1(self, rng):
        """Test differencing with period 1."""
        T, p1, p2 = 20, 4, 3
        Y = rng.normal(size=(T, p1, p2))

        Y_diff = seasonal_difference(Y, period=1)

        assert Y_diff.shape == (T - 1, p1, p2)
        expected = Y[1:] - Y[:-1]
        np.testing.assert_allclose(Y_diff, expected)

    def test_invalid_period_raises(self, rng):
        """Test that invalid period raises ValueError."""
        Y = rng.normal(size=(20, 4, 3))

        with pytest.raises(ValueError, match="period must be between"):
            seasonal_difference(Y, period=0)

        with pytest.raises(ValueError, match="period must be between"):
            seasonal_difference(Y, period=-1)

        with pytest.raises(ValueError, match="period must be between"):
            seasonal_difference(Y, period=20)

    def test_non_3d_raises(self, rng):
        """Test that non-3D input raises ValueError."""
        Y_2d = rng.normal(size=(20, 4))

        with pytest.raises(ValueError, match="must be 3D array"):
            seasonal_difference(Y_2d, period=4)


# ---------------------------------------------------------------------------
# integrate_seasonal_diff tests
# ---------------------------------------------------------------------------


class TestIntegrateSeasonalDiff:
    """Tests for integrate_seasonal_diff function."""

    def test_integration_shape(self, rng):
        """Test that integration returns correct shape."""
        p1, p2 = 4, 3
        period = 4
        steps = 8

        last_obs = rng.normal(size=(period, p1, p2))
        diffs = rng.normal(size=(steps, p1, p2))

        levels = integrate_seasonal_diff(last_obs, diffs, period)

        assert levels.shape == (steps, p1, p2)

    def test_integration_calculation(self):
        """Test that integration is calculated correctly."""
        p1, p2 = 2, 2
        period = 2

        # Simple test case
        last_obs = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        diffs = np.array([[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]])

        levels = integrate_seasonal_diff(last_obs, diffs, period)

        # First step: diff + last_obs[-2] = 1 + 1 = 2 (for first element)
        assert levels.shape == (2, p1, p2)
        # First level = diff[0] + last_obs[0]
        expected_first = diffs[0] + last_obs[0]
        np.testing.assert_allclose(levels[0], expected_first)

    def test_roundtrip(self, rng):
        """Test that difference then integrate recovers original (approximately)."""
        T, p1, p2 = 20, 4, 3
        period = 4

        # Generate deterministic data with trend
        Y = np.zeros((T, p1, p2))
        for t in range(T):
            Y[t] = t * np.ones((p1, p2))

        # Difference
        Y_diff = seasonal_difference(Y, period)

        # Take "forecasts" from actual differences
        steps = 4
        diff_fcst = Y_diff[-steps:]
        last_obs = Y[-steps - period : -steps]

        # Integrate back - should recover something close to original trajectory
        levels = integrate_seasonal_diff(last_obs, diff_fcst, period)

        # Should match the actual values
        np.testing.assert_allclose(levels, Y[-steps:])


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
# forecast_dmfm with seasonal differencing tests
# ---------------------------------------------------------------------------


class TestForecastDMFMSeasonal:
    """Tests for forecast_dmfm with seasonal differencing."""

    @pytest.fixture
    def seasonal_data(self, rng):
        """Generate data with seasonal pattern."""
        T, p1, p2 = 24, 4, 3  # 6 years of quarterly data
        # Generate with seasonal pattern
        Y = np.zeros((T, p1, p2))
        for t in range(T):
            seasonal = np.sin(2 * np.pi * t / 4)  # Quarterly seasonality
            trend = 0.1 * t
            Y[t] = trend + seasonal + 0.1 * rng.normal(size=(p1, p2))

        mask = np.ones_like(Y, dtype=bool)
        return Y, mask

    def test_seasonal_adjustment_applied(self, seasonal_data):
        """Test that seasonal adjustment is applied."""
        Y, mask = seasonal_data
        config = ForecastConfig(k1=1, k2=1, seasonal_period=4, max_iter=3)

        result = forecast_dmfm(Y, steps=4, config=config, mask=mask)

        assert result.seasonal_adjusted is True

    def test_seasonal_forecast_shape(self, seasonal_data):
        """Test forecast shape with seasonal differencing."""
        Y, mask = seasonal_data
        config = ForecastConfig(k1=1, k2=1, seasonal_period=4, max_iter=3)

        result = forecast_dmfm(Y, steps=4, config=config, mask=mask)

        # Shape should be (steps, p1, p2)
        assert result.forecast.shape == (4, Y.shape[1], Y.shape[2])

    def test_no_seasonal_adjustment(self, seasonal_data):
        """Test that no seasonal adjustment when period is None."""
        Y, mask = seasonal_data
        config = ForecastConfig(k1=1, k2=1, seasonal_period=None, max_iter=3)

        result = forecast_dmfm(Y, steps=4, config=config, mask=mask)

        assert result.seasonal_adjusted is False


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
