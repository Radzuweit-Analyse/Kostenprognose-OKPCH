"""Out-of-sample validation for DMFM forecasts.

This module provides functions for evaluating forecast accuracy using
rolling window validation, expanding window validation, and various
error metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List
import numpy as np

from .forecast import forecast_dmfm, ForecastConfig


@dataclass
class ValidationConfig:
    """Configuration for forecast validation.

    Parameters
    ----------
    steps : int
        Number of forecast steps to evaluate.
    window_type : str, default "expanding"
        Type of validation window:
        - "expanding": Use all data up to validation point
        - "rolling": Use fixed-size rolling window
    window_size : int or None, default None
        Size of rolling window (only used if window_type="rolling").
    min_train_size : int, default 20
        Minimum training sample size.
    """

    steps: int
    window_type: str = "expanding"
    window_size: int | None = None
    min_train_size: int = 20

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.window_type not in ("expanding", "rolling"):
            raise ValueError(
                f"window_type must be 'expanding' or 'rolling', "
                f"got '{self.window_type}'"
            )
        if self.window_type == "rolling" and self.window_size is None:
            raise ValueError(
                "window_size must be specified for rolling window validation"
            )
        if self.steps <= 0:
            raise ValueError(f"steps must be positive, got {self.steps}")


@dataclass
class ValidationResult:
    """Results from forecast validation.

    Attributes
    ----------
    rmse : float
        Root mean squared error.
    mae : float
        Mean absolute error.
    mape : float
        Mean absolute percentage error (%).
    bias : float
        Mean error (bias).
    forecasts : np.ndarray
        Point forecasts that were validated.
    actuals : np.ndarray
        Actual values used for validation.
    errors : np.ndarray
        Forecast errors (forecasts - actuals).
    config : ValidationConfig
        Validation configuration used.
    forecast_config : ForecastConfig or None
        Forecast configuration used for this window (useful when config_selector
        is used to select parameters dynamically).
    """

    rmse: float
    mae: float
    mape: float
    bias: float
    forecasts: np.ndarray
    actuals: np.ndarray
    errors: np.ndarray
    config: ValidationConfig
    forecast_config: ForecastConfig | None = None


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------


def compute_rmse(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """Compute root mean squared error.

    Parameters
    ----------
    forecasts : np.ndarray
        Forecast values.
    actuals : np.ndarray
        Actual values.

    Returns
    -------
    float
        RMSE, ignoring NaN values.
    """
    err = forecasts - actuals
    return float(np.sqrt(np.nanmean(err**2)))


def compute_mae(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """Compute mean absolute error.

    Parameters
    ----------
    forecasts : np.ndarray
        Forecast values.
    actuals : np.ndarray
        Actual values.

    Returns
    -------
    float
        MAE, ignoring NaN values.
    """
    err = np.abs(forecasts - actuals)
    return float(np.nanmean(err))


def compute_mape(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """Compute mean absolute percentage error.

    Parameters
    ----------
    forecasts : np.ndarray
        Forecast values.
    actuals : np.ndarray
        Actual values.

    Returns
    -------
    float
        MAPE in percentage points, ignoring NaN values and zeros.

    Notes
    -----
    Values where actuals are zero or very small (< 1e-10) are excluded
    to avoid division issues.
    """
    # Avoid division by zero
    valid = (np.abs(actuals) > 1e-10) & ~np.isnan(actuals) & ~np.isnan(forecasts)
    if not valid.any():
        return np.nan

    err = np.abs((forecasts[valid] - actuals[valid]) / actuals[valid])
    return float(100.0 * np.mean(err))


def compute_bias(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """Compute mean error (bias).

    Parameters
    ----------
    forecasts : np.ndarray
        Forecast values.
    actuals : np.ndarray
        Actual values.

    Returns
    -------
    float
        Mean error, ignoring NaN values.
        Positive indicates over-forecasting, negative under-forecasting.
    """
    err = forecasts - actuals
    return float(np.nanmean(err))


def compute_metrics(forecasts: np.ndarray, actuals: np.ndarray) -> dict:
    """Compute all error metrics.

    Parameters
    ----------
    forecasts : np.ndarray
        Forecast values.
    actuals : np.ndarray
        Actual values.

    Returns
    -------
    dict
        Dictionary with keys: rmse, mae, mape, bias.
    """
    return {
        "rmse": compute_rmse(forecasts, actuals),
        "mae": compute_mae(forecasts, actuals),
        "mape": compute_mape(forecasts, actuals),
        "bias": compute_bias(forecasts, actuals),
    }


# ---------------------------------------------------------------------------
# Main validation functions
# ---------------------------------------------------------------------------


def out_of_sample_validate(
    Y: np.ndarray,
    val_config: ValidationConfig,
    forecast_config: ForecastConfig | None = None,
    mask: np.ndarray | None = None,
    config_selector: (
        Callable[[np.ndarray, np.ndarray | None], ForecastConfig] | None
    ) = None,
) -> ValidationResult:
    """Perform out-of-sample validation for DMFM forecast.

    The model is estimated on Y excluding the last 'steps' observations.
    A forecast is produced for these periods and compared with the held-out
    data.

    Parameters
    ----------
    Y : np.ndarray
        Full data of shape (T, p1, p2).
    val_config : ValidationConfig
        Validation configuration.
    forecast_config : ForecastConfig, optional
        Forecast model configuration. Used when config_selector is None.
    mask : np.ndarray, optional
        Boolean mask for missing values (True = observed).
    config_selector : callable, optional
        Function that takes (Y_train, mask_train) and returns a ForecastConfig.
        When provided, this is called to dynamically select model parameters.

    Returns
    -------
    ValidationResult
        Validation results including all error metrics.

    Raises
    ------
    ValueError
        If Y is not 3D or validation steps exceed available data.

    Examples
    --------
    >>> from forecast import ForecastConfig
    >>> from validation import ValidationConfig, out_of_sample_validate
    >>>
    >>> val_config = ValidationConfig(steps=8)
    >>> fcst_config = ForecastConfig(k1=2, k2=2, P=1)
    >>> result = out_of_sample_validate(Y, val_config, fcst_config)
    >>> print(f"RMSE: {result.rmse:.2f}, MAE: {result.mae:.2f}")
    """
    # Validate inputs
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 3:
        raise ValueError(f"Y must be 3D array, got shape {Y.shape}")

    steps = val_config.steps
    if steps <= 0 or steps >= Y.shape[0]:
        raise ValueError(f"steps must be between 1 and {Y.shape[0]-1}, got {steps}")

    # Split data
    Y_train = Y[:-steps]
    Y_test = Y[-steps:]
    mask_train = mask[:-steps] if mask is not None else None

    # Get forecast config (either fixed or dynamically selected)
    if config_selector is not None:
        used_config = config_selector(Y_train, mask_train)
    else:
        used_config = forecast_config

    # Generate forecast
    result = forecast_dmfm(
        Y_train,
        steps,
        config=used_config,
        mask=mask_train,
    )

    forecasts = result.forecast
    actuals = Y_test
    errors = forecasts - actuals

    # Compute metrics
    metrics = compute_metrics(forecasts, actuals)

    return ValidationResult(
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        mape=metrics["mape"],
        bias=metrics["bias"],
        forecasts=forecasts,
        actuals=actuals,
        errors=errors,
        config=val_config,
        forecast_config=used_config,
    )


def rolling_window_validate(
    Y: np.ndarray,
    val_config: ValidationConfig,
    forecast_config: ForecastConfig | None = None,
    mask: np.ndarray | None = None,
    config_selector: (
        Callable[[np.ndarray, np.ndarray | None], ForecastConfig] | None
    ) = None,
) -> List[ValidationResult]:
    """Perform rolling window out-of-sample validation.

    Creates multiple train/test splits using either expanding or rolling
    windows, evaluates forecasts on each, and returns individual results.

    Parameters
    ----------
    Y : np.ndarray
        Full data of shape (T, p1, p2).
    val_config : ValidationConfig
        Validation configuration specifying window type and size.
    forecast_config : ForecastConfig, optional
        Forecast model configuration. Used when config_selector is None.
    mask : np.ndarray, optional
        Boolean mask for missing values.
    config_selector : callable, optional
        Function that takes (Y_train, mask_train) and returns a ForecastConfig.
        When provided, this is called for each validation window to dynamically
        select model parameters (e.g., using BIC for rank selection).
        Signature: config_selector(Y_train: np.ndarray, mask_train: np.ndarray | None) -> ForecastConfig

    Returns
    -------
    list[ValidationResult]
        List of validation results, one per window. Each result includes
        the forecast_config used for that window.

    Raises
    ------
    ValueError
        If insufficient data for validation windows.

    Examples
    --------
    >>> # Fixed config
    >>> val_config = ValidationConfig(
    ...     steps=4,
    ...     window_type="rolling",
    ...     window_size=40,
    ...     min_train_size=20
    ... )
    >>> results = rolling_window_validate(Y, val_config)
    >>> avg_rmse = np.mean([r.rmse for r in results])
    >>> print(f"Average RMSE: {avg_rmse:.2f}")

    >>> # Dynamic config selection using BIC
    >>> def select_config(Y_train, mask_train):
    ...     from KPOKPCH.DMFM import select_rank
    ...     Y_diff = np.diff(Y_train, n=4, axis=0)
    ...     result = select_rank(Y_diff, k1_range=(1,2), k2_range=(1,6), criterion="bic")
    ...     return ForecastConfig(k1=result.best_k1, k2=result.best_k2, P=1)
    >>> results = rolling_window_validate(Y, val_config, config_selector=select_config)
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 3:
        raise ValueError(f"Y must be 3D array, got shape {Y.shape}")

    T = Y.shape[0]
    steps = val_config.steps
    min_train = val_config.min_train_size

    # Determine validation windows
    if val_config.window_type == "expanding":
        # Start with min_train, expand by one each iteration
        start_indices = range(min_train, T - steps)
    else:  # rolling
        window_size = val_config.window_size
        if window_size is None:
            raise ValueError("window_size required for rolling validation")
        # Fixed window size
        start_indices = range(window_size, T - steps)

    if not start_indices:
        raise ValueError(
            f"Insufficient data: T={T}, steps={steps}, min_train={min_train}"
        )

    results = []
    for train_end in start_indices:
        # Determine training window
        if val_config.window_type == "expanding":
            train_start = 0
        else:
            train_start = train_end - val_config.window_size

        # Split data
        Y_train = Y[train_start:train_end]
        Y_test = Y[train_end : train_end + steps]
        mask_train = mask[train_start:train_end] if mask is not None else None

        # Get forecast config (either fixed or dynamically selected)
        if config_selector is not None:
            window_config = config_selector(Y_train, mask_train)
        else:
            window_config = forecast_config

        # Generate forecast
        fcst_result = forecast_dmfm(
            Y_train,
            steps,
            config=window_config,
            mask=mask_train,
        )

        forecasts = fcst_result.forecast
        actuals = Y_test
        errors = forecasts - actuals

        # Compute metrics
        metrics = compute_metrics(forecasts, actuals)

        results.append(
            ValidationResult(
                rmse=metrics["rmse"],
                mae=metrics["mae"],
                mape=metrics["mape"],
                bias=metrics["bias"],
                forecasts=forecasts,
                actuals=actuals,
                errors=errors,
                config=val_config,
                forecast_config=window_config,
            )
        )

    return results


def average_validation_results(results: List[ValidationResult]) -> dict:
    """Compute average metrics across multiple validation results.

    Parameters
    ----------
    results : list[ValidationResult]
        List of validation results from rolling window validation.

    Returns
    -------
    dict
        Average metrics with keys: rmse, mae, mape, bias, and their
        standard deviations (rmse_std, mae_std, etc.).

    Examples
    --------
    >>> results = rolling_window_validate(Y, val_config)
    >>> avg_metrics = average_validation_results(results)
    >>> print(f"Avg RMSE: {avg_metrics['rmse']:.2f} ± {avg_metrics['rmse_std']:.2f}")
    """
    if not results:
        raise ValueError("No results to average")

    rmses = [r.rmse for r in results]
    maes = [r.mae for r in results]
    mapes = [r.mape for r in results if not np.isnan(r.mape)]
    biases = [r.bias for r in results]

    return {
        "rmse": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses, ddof=1) if len(rmses) > 1 else 0),
        "mae": float(np.mean(maes)),
        "mae_std": float(np.std(maes, ddof=1) if len(maes) > 1 else 0),
        "mape": float(np.mean(mapes)) if mapes else np.nan,
        "mape_std": float(np.std(mapes, ddof=1) if len(mapes) > 1 else 0),
        "bias": float(np.mean(biases)),
        "bias_std": float(np.std(biases, ddof=1) if len(biases) > 1 else 0),
        "n_windows": len(results),
    }
