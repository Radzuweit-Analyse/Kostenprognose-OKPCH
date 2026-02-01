"""Tests for KPOKPCH.forecast.validation module.

Tests cover ValidationConfig, ValidationResult, error metrics,
and out-of-sample validation functions.
"""

import numpy as np
import pytest

from KPOKPCH.forecast import ForecastConfig
from KPOKPCH.forecast.validation import (
    ValidationConfig,
    ValidationResult,
    compute_rmse,
    compute_mae,
    compute_mape,
    compute_bias,
    compute_metrics,
    out_of_sample_validate,
    rolling_window_validate,
    average_validation_results,
)

# ---------------------------------------------------------------------------
# ValidationConfig tests
# ---------------------------------------------------------------------------


class TestValidationConfig:
    """Tests for ValidationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ValidationConfig(steps=4)
        assert config.steps == 4
        assert config.window_type == "expanding"
        assert config.window_size is None
        assert config.min_train_size == 20

    def test_custom_config(self):
        """Test configuration with custom values."""
        config = ValidationConfig(
            steps=8,
            window_type="rolling",
            window_size=40,
            min_train_size=30,
        )
        assert config.steps == 8
        assert config.window_type == "rolling"
        assert config.window_size == 40
        assert config.min_train_size == 30

    def test_invalid_window_type_raises(self):
        """Test that invalid window_type raises ValueError."""
        with pytest.raises(ValueError, match="window_type must be"):
            ValidationConfig(steps=4, window_type="invalid")

    def test_rolling_without_window_size_raises(self):
        """Test that rolling without window_size raises ValueError."""
        with pytest.raises(ValueError, match="window_size must be specified"):
            ValidationConfig(steps=4, window_type="rolling")

    def test_invalid_steps_raises(self):
        """Test that invalid steps raises ValueError."""
        with pytest.raises(ValueError, match="steps must be positive"):
            ValidationConfig(steps=0)

        with pytest.raises(ValueError, match="steps must be positive"):
            ValidationConfig(steps=-1)


# ---------------------------------------------------------------------------
# ValidationResult tests
# ---------------------------------------------------------------------------


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_create_result(self):
        """Test creating ValidationResult."""
        config = ValidationConfig(steps=4)
        forecasts = np.array([1.0, 2.0, 3.0, 4.0])
        actuals = np.array([1.1, 2.1, 3.1, 4.1])
        errors = forecasts - actuals

        result = ValidationResult(
            rmse=0.1,
            mae=0.1,
            mape=5.0,
            bias=-0.1,
            forecasts=forecasts,
            actuals=actuals,
            errors=errors,
            config=config,
        )

        assert result.rmse == 0.1
        assert result.mae == 0.1
        assert result.mape == 5.0
        assert result.bias == -0.1


# ---------------------------------------------------------------------------
# Error metrics tests
# ---------------------------------------------------------------------------


class TestComputeRMSE:
    """Tests for compute_rmse function."""

    def test_perfect_forecast(self):
        """Test RMSE is zero for perfect forecast."""
        forecasts = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.0, 2.0, 3.0])

        rmse = compute_rmse(forecasts, actuals)

        assert rmse == 0.0

    def test_known_rmse(self):
        """Test RMSE calculation with known values."""
        forecasts = np.array([1.0, 2.0, 3.0])
        actuals = np.array([2.0, 3.0, 4.0])  # All errors are 1.0

        rmse = compute_rmse(forecasts, actuals)

        np.testing.assert_allclose(rmse, 1.0)

    def test_handles_nan(self):
        """Test that NaN values are ignored."""
        forecasts = np.array([1.0, np.nan, 3.0])
        actuals = np.array([1.0, 2.0, 3.0])

        rmse = compute_rmse(forecasts, actuals)

        assert rmse == 0.0


class TestComputeMAE:
    """Tests for compute_mae function."""

    def test_perfect_forecast(self):
        """Test MAE is zero for perfect forecast."""
        forecasts = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.0, 2.0, 3.0])

        mae = compute_mae(forecasts, actuals)

        assert mae == 0.0

    def test_known_mae(self):
        """Test MAE calculation with known values."""
        forecasts = np.array([1.0, 2.0, 3.0])
        actuals = np.array([2.0, 3.0, 4.0])  # All errors are 1.0

        mae = compute_mae(forecasts, actuals)

        assert mae == 1.0

    def test_handles_negative_errors(self):
        """Test MAE correctly handles negative errors."""
        forecasts = np.array([1.0, 2.0, 3.0])
        actuals = np.array([2.0, 1.0, 4.0])  # Errors: -1, 1, -1

        mae = compute_mae(forecasts, actuals)

        assert mae == 1.0


class TestComputeMAPE:
    """Tests for compute_mape function."""

    def test_perfect_forecast(self):
        """Test MAPE is zero for perfect forecast."""
        forecasts = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.0, 2.0, 3.0])

        mape = compute_mape(forecasts, actuals)

        assert mape == 0.0

    def test_known_mape(self):
        """Test MAPE calculation with known values."""
        forecasts = np.array([1.1, 2.2, 3.3])
        actuals = np.array([1.0, 2.0, 3.0])  # 10% error each

        mape = compute_mape(forecasts, actuals)

        np.testing.assert_allclose(mape, 10.0)

    def test_excludes_zeros(self):
        """Test that zeros in actuals are excluded."""
        forecasts = np.array([1.1, 0.5, 2.2])
        actuals = np.array([1.0, 0.0, 2.0])  # Second value is zero

        mape = compute_mape(forecasts, actuals)

        # Only uses first and third values
        assert np.isfinite(mape)

    def test_all_zeros_returns_nan(self):
        """Test that all zeros returns NaN."""
        forecasts = np.array([1.0, 2.0])
        actuals = np.array([0.0, 0.0])

        mape = compute_mape(forecasts, actuals)

        assert np.isnan(mape)


class TestComputeBias:
    """Tests for compute_bias function."""

    def test_unbiased_forecast(self):
        """Test bias is zero for unbiased forecast."""
        forecasts = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.0, 2.0, 3.0])

        bias = compute_bias(forecasts, actuals)

        assert bias == 0.0

    def test_positive_bias(self):
        """Test positive bias (over-forecasting)."""
        forecasts = np.array([2.0, 3.0, 4.0])
        actuals = np.array([1.0, 2.0, 3.0])

        bias = compute_bias(forecasts, actuals)

        assert bias == 1.0

    def test_negative_bias(self):
        """Test negative bias (under-forecasting)."""
        forecasts = np.array([0.0, 1.0, 2.0])
        actuals = np.array([1.0, 2.0, 3.0])

        bias = compute_bias(forecasts, actuals)

        assert bias == -1.0


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_returns_all_metrics(self):
        """Test that all metrics are returned."""
        forecasts = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.1, 2.1, 3.1])

        metrics = compute_metrics(forecasts, actuals)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "mape" in metrics
        assert "bias" in metrics

    def test_metrics_consistent(self):
        """Test that metrics are consistent with individual functions."""
        forecasts = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.5, 2.5, 3.5])

        metrics = compute_metrics(forecasts, actuals)

        assert metrics["rmse"] == compute_rmse(forecasts, actuals)
        assert metrics["mae"] == compute_mae(forecasts, actuals)
        assert metrics["mape"] == compute_mape(forecasts, actuals)
        assert metrics["bias"] == compute_bias(forecasts, actuals)


# ---------------------------------------------------------------------------
# out_of_sample_validate tests
# ---------------------------------------------------------------------------


class TestOutOfSampleValidate:
    """Tests for out_of_sample_validate function."""

    @pytest.fixture
    def validation_data(self, rng):
        """Generate data for validation tests."""
        T, p1, p2 = 30, 4, 3
        Y = rng.normal(size=(T, p1, p2))
        return Y

    def test_returns_validation_result(self, validation_data):
        """Test that function returns ValidationResult."""
        val_config = ValidationConfig(steps=4)
        fcst_config = ForecastConfig(k1=1, k2=1, max_iter=3)

        result = out_of_sample_validate(validation_data, val_config, fcst_config)

        assert isinstance(result, ValidationResult)

    def test_forecast_shape(self, validation_data):
        """Test that forecasts have correct shape."""
        val_config = ValidationConfig(steps=4)
        fcst_config = ForecastConfig(k1=1, k2=1, max_iter=3)

        result = out_of_sample_validate(validation_data, val_config, fcst_config)

        p1, p2 = validation_data.shape[1:]
        assert result.forecasts.shape == (4, p1, p2)
        assert result.actuals.shape == (4, p1, p2)
        assert result.errors.shape == (4, p1, p2)

    def test_errors_computed_correctly(self, validation_data):
        """Test that errors = forecasts - actuals."""
        val_config = ValidationConfig(steps=4)
        fcst_config = ForecastConfig(k1=1, k2=1, max_iter=3)

        result = out_of_sample_validate(validation_data, val_config, fcst_config)

        expected_errors = result.forecasts - result.actuals
        np.testing.assert_allclose(result.errors, expected_errors)

    def test_invalid_steps_raises(self, validation_data):
        """Test that invalid steps raises ValueError."""
        val_config = ValidationConfig(steps=100)  # Too many steps
        fcst_config = ForecastConfig(k1=1, k2=1, max_iter=3)

        with pytest.raises(ValueError, match="steps must be between"):
            out_of_sample_validate(validation_data, val_config, fcst_config)

    def test_invalid_y_shape_raises(self, rng):
        """Test that invalid Y shape raises ValueError."""
        Y_2d = rng.normal(size=(30, 4))
        val_config = ValidationConfig(steps=4)

        with pytest.raises(ValueError, match="must be 3D array"):
            out_of_sample_validate(Y_2d, val_config)


# ---------------------------------------------------------------------------
# rolling_window_validate tests
# ---------------------------------------------------------------------------


class TestRollingWindowValidate:
    """Tests for rolling_window_validate function."""

    @pytest.fixture
    def rolling_data(self, rng):
        """Generate data for rolling validation tests."""
        T, p1, p2 = 50, 4, 3
        Y = rng.normal(size=(T, p1, p2))
        return Y

    def test_expanding_window(self, rolling_data):
        """Test expanding window validation."""
        val_config = ValidationConfig(
            steps=4,
            window_type="expanding",
            min_train_size=20,
        )
        fcst_config = ForecastConfig(k1=1, k2=1, max_iter=3)

        results = rolling_window_validate(rolling_data, val_config, fcst_config)

        # Should have multiple results
        assert len(results) > 0
        assert all(isinstance(r, ValidationResult) for r in results)

    def test_rolling_window(self, rolling_data):
        """Test rolling window validation."""
        val_config = ValidationConfig(
            steps=4,
            window_type="rolling",
            window_size=20,
        )
        fcst_config = ForecastConfig(k1=1, k2=1, max_iter=3)

        results = rolling_window_validate(rolling_data, val_config, fcst_config)

        assert len(results) > 0

    def test_insufficient_data_raises(self, rng):
        """Test that insufficient data raises ValueError."""
        Y = rng.normal(size=(10, 4, 3))  # Very short
        val_config = ValidationConfig(
            steps=4,
            window_type="expanding",
            min_train_size=20,
        )

        with pytest.raises(ValueError, match="Insufficient data"):
            rolling_window_validate(Y, val_config)


# ---------------------------------------------------------------------------
# average_validation_results tests
# ---------------------------------------------------------------------------


class TestAverageValidationResults:
    """Tests for average_validation_results function."""

    def test_average_single_result(self):
        """Test averaging with single result."""
        config = ValidationConfig(steps=4)
        result = ValidationResult(
            rmse=1.0,
            mae=0.8,
            mape=10.0,
            bias=0.1,
            forecasts=np.zeros((4, 4, 3)),
            actuals=np.zeros((4, 4, 3)),
            errors=np.zeros((4, 4, 3)),
            config=config,
        )

        avg = average_validation_results([result])

        assert avg["rmse"] == 1.0
        assert avg["mae"] == 0.8
        assert avg["mape"] == 10.0
        assert avg["bias"] == 0.1
        assert avg["n_windows"] == 1

    def test_average_multiple_results(self):
        """Test averaging with multiple results."""
        config = ValidationConfig(steps=4)
        results = [
            ValidationResult(
                rmse=1.0,
                mae=0.8,
                mape=10.0,
                bias=0.1,
                forecasts=np.zeros((4, 4, 3)),
                actuals=np.zeros((4, 4, 3)),
                errors=np.zeros((4, 4, 3)),
                config=config,
            ),
            ValidationResult(
                rmse=2.0,
                mae=1.6,
                mape=20.0,
                bias=0.2,
                forecasts=np.zeros((4, 4, 3)),
                actuals=np.zeros((4, 4, 3)),
                errors=np.zeros((4, 4, 3)),
                config=config,
            ),
        ]

        avg = average_validation_results(results)

        np.testing.assert_allclose(avg["rmse"], 1.5)
        np.testing.assert_allclose(avg["mae"], 1.2)
        np.testing.assert_allclose(avg["mape"], 15.0)
        np.testing.assert_allclose(avg["bias"], 0.15)
        assert avg["n_windows"] == 2

    def test_includes_standard_deviations(self):
        """Test that standard deviations are included."""
        config = ValidationConfig(steps=4)
        results = [
            ValidationResult(
                rmse=1.0,
                mae=0.8,
                mape=10.0,
                bias=0.1,
                forecasts=np.zeros((4, 4, 3)),
                actuals=np.zeros((4, 4, 3)),
                errors=np.zeros((4, 4, 3)),
                config=config,
            ),
            ValidationResult(
                rmse=3.0,
                mae=2.4,
                mape=30.0,
                bias=0.3,
                forecasts=np.zeros((4, 4, 3)),
                actuals=np.zeros((4, 4, 3)),
                errors=np.zeros((4, 4, 3)),
                config=config,
            ),
        ]

        avg = average_validation_results(results)

        assert "rmse_std" in avg
        assert "mae_std" in avg
        assert "mape_std" in avg
        assert "bias_std" in avg

    def test_empty_results_raises(self):
        """Test that empty results raises ValueError."""
        with pytest.raises(ValueError, match="No results to average"):
            average_validation_results([])
