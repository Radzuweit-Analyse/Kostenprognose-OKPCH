"""Forecasting utilities for DMFM models.

This module provides functions for forecasting with Dynamic Matrix Factor Models,
including support for seasonal differencing, canton-level forecasting, and
growth rate calculations.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from ..DMFM import (
    DMFMModel,
    fit_dmfm,
)


@dataclass
class ForecastConfig:
    """Configuration for DMFM forecasting.
    
    Parameters
    ----------
    k1, k2 : int, default 1
        Number of row and column factors.
    P : int, default 1
        MAR order.
    seasonal_period : int or None, default None
        Seasonal differencing period (e.g., 4 for quarterly data).
        If None, no differencing is applied.
    max_iter : int, default 50
        Maximum EM iterations for model fitting.
    tol : float, default 1e-4
        EM convergence tolerance.
    diagonal_idiosyncratic : bool, default False
        Whether to use diagonal idiosyncratic covariance.
    init_method : str, default "svd"
        Initialization method ("svd" or "pe").
    verbose : bool, default False
        Whether to print fitting progress.
    """
    
    k1: int = 1
    k2: int = 1
    P: int = 1
    seasonal_period: int | None = None
    max_iter: int = 50
    tol: float = 1e-4
    diagonal_idiosyncratic: bool = False
    init_method: str = "svd"
    verbose: bool = False


@dataclass
class ForecastResult:
    """Results from DMFM forecasting.
    
    Attributes
    ----------
    forecast : np.ndarray
        Point forecasts of shape (steps, p1, p2).
    model : DMFMModel
        Fitted DMFM model.
    config : ForecastConfig
        Configuration used for forecasting.
    seasonal_adjusted : bool
        Whether seasonal differencing was applied.
    """
    
    forecast: np.ndarray
    model: DMFMModel
    config: ForecastConfig
    seasonal_adjusted: bool = False


# ---------------------------------------------------------------------------
# Seasonal differencing utilities
# ---------------------------------------------------------------------------

def seasonal_difference(Y: np.ndarray, period: int) -> np.ndarray:
    """Compute seasonal differences of Y.
    
    Parameters
    ----------
    Y : np.ndarray
        Data of shape (T, p1, p2).
    period : int
        Seasonal period (e.g., 4 for quarterly data).
        
    Returns
    -------
    np.ndarray
        Seasonal differences of shape (T-period, p1, p2).
        
    Raises
    ------
    ValueError
        If Y is not 3D or period is invalid.
        
    Examples
    --------
    >>> Y_diff = seasonal_difference(Y, period=4)  # Quarterly seasonal diff
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 3:
        raise ValueError(f"Y must be 3D array, got shape {Y.shape}")
    
    T = Y.shape[0]
    if period <= 0 or period >= T:
        raise ValueError(
            f"period must be between 1 and {T-1}, got {period}"
        )
    
    return Y[period:] - Y[:-period]


def integrate_seasonal_diff(
    last_obs: np.ndarray, diffs: np.ndarray, period: int
) -> np.ndarray:
    """Integrate seasonal differences back to levels.
    
    Parameters
    ----------
    last_obs : np.ndarray
        Last 'period' observations before forecasting, shape (period, p1, p2).
    diffs : np.ndarray
        Seasonal difference forecasts, shape (steps, p1, p2).
    period : int
        Seasonal period.
        
    Returns
    -------
    np.ndarray
        Level forecasts of shape (steps, p1, p2).
        
    Examples
    --------
    >>> # Y has shape (100, 10, 5), we forecast 8 steps with period=4
    >>> last_obs = Y[-4:]  # Last 4 observations
    >>> diff_forecast = ...  # Shape (8, 10, 5)
    >>> level_forecast = integrate_seasonal_diff(last_obs, diff_forecast, 4)
    """
    history = list(np.asarray(last_obs))
    result = []
    
    for diff in np.asarray(diffs):
        baseline = history[-period]
        next_level = diff + baseline
        history.append(next_level)
        result.append(next_level)
    
    return np.stack(result, axis=0)


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def load_cost_matrix(path: str) -> (
    Tuple[List[str], List[str], np.ndarray]
    | Tuple[List[str], List[str], List[str], np.ndarray]
):
    """Load canton cost data from CSV.

    The function supports two layouts:

    * **Wide 2D format**: Column headers are ``Period`` followed by canton names.
      Returns ``(periods, cantons, data)`` where ``data`` has shape
      ``(T, num_cantons)``.
      
    * **Tensor format**: Column headers are ``<Canton>|<Group>`` for each
      cost group. Returns ``(periods, cantons, groups, data)`` where ``data``
      has shape ``(T, num_cantons, num_groups)``.

    Parameters
    ----------
    path : str
        Path to CSV file.

    Returns
    -------
    periods : list[str]
        Time period labels.
    cantons : list[str]
        Canton names.
    data : np.ndarray
        Cost data (2D or 3D depending on format).
    groups : list[str], optional
        Cost group names (only for tensor format).

    Examples
    --------
    >>> # 2D format
    >>> periods, cantons, data = load_cost_matrix("costs_2d.csv")
    >>> print(data.shape)  # (T, num_cantons)
    
    >>> # 3D tensor format
    >>> periods, cantons, groups, data = load_cost_matrix("costs_3d.csv")
    >>> print(data.shape)  # (T, num_cantons, num_groups)
    """
    periods: List[str] = []
    data_rows = []
    
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        raw_columns = header[1:]
        
        # Detect format
        has_groups = all("|" in col for col in raw_columns)
        
        if has_groups:
            # Parse canton|group headers
            canton_group_pairs: List[Tuple[str, str]] = []
            for col in raw_columns:
                canton, group = col.split("|", maxsplit=1)
                canton_group_pairs.append((canton, group))

            # Extract unique cantons and groups
            cantons: List[str] = []
            groups: List[str] = []
            for canton, group in canton_group_pairs:
                if canton not in cantons:
                    cantons.append(canton)
                if group not in groups:
                    groups.append(group)
        else:
            canton_group_pairs = [(canton, "") for canton in raw_columns]
            cantons = list(raw_columns)
            groups = [""]
        
        # Read data rows
        for row in reader:
            periods.append(row[0])
            values = []
            for x in row[1:]:
                try:
                    values.append(float(x))
                except ValueError:
                    values.append(np.nan)
            data_rows.append(values)
    
    flat = np.array(data_rows, dtype=float)
    
    if has_groups:
        # Reshape to 3D tensor
        data = np.full(
            (len(periods), len(cantons), len(groups)), np.nan, dtype=float
        )
        canton_idx = {c: i for i, c in enumerate(cantons)}
        group_idx = {g: j for j, g in enumerate(groups)}
        
        for col, (canton, group) in enumerate(canton_group_pairs):
            data[:, canton_idx[canton], group_idx[group]] = flat[:, col]
        
        return periods, cantons, groups, data

    return periods, cantons, flat


# ---------------------------------------------------------------------------
# Period utilities
# ---------------------------------------------------------------------------

def generate_future_periods(last_period: str, steps: int) -> List[str]:
    """Generate future quarterly period labels.
    
    Parameters
    ----------
    last_period : str
        Last observed period in format "YYYYQQ" (e.g., "2024Q4").
    steps : int
        Number of future periods to generate.
        
    Returns
    -------
    list[str]
        Future period labels.
        
    Examples
    --------
    >>> generate_future_periods("2024Q4", 8)
    ['2025Q1', '2025Q2', '2025Q3', '2025Q4', '2026Q1', '2026Q2', '2026Q3', '2026Q4']
    """
    year = int(last_period[:4])
    quarter = int(last_period[-1])
    periods = []
    
    for _ in range(steps):
        quarter += 1
        if quarter > 4:
            quarter = 1
            year += 1
        periods.append(f"{year}Q{quarter}")
    
    return periods


# ---------------------------------------------------------------------------
# Growth rate calculations
# ---------------------------------------------------------------------------

def compute_q4_growth(
    periods: List[str],
    data: np.ndarray,
    fcst: np.ndarray,
    future_periods: List[str],
) -> dict:
    """Compute Q4-over-Q4 growth metrics for cantonal forecasts.
    
    Parameters
    ----------
    periods : list[str]
        Historical period labels.
    data : np.ndarray
        Historical data (last Q4 used as base).
    fcst : np.ndarray
        Forecast values.
    future_periods : list[str]
        Future period labels.
        
    Returns
    -------
    dict
        Growth metrics including:
        - growth_y1: Year 1 Q4-over-Q4 growth rates (%)
        - growth_y2: Year 2 Q4-over-Q4 growth rates (%)
        - mean_y1, sd_y1, ci_y1: Statistics for year 1
        - mean_y2, sd_y2, ci_y2: Statistics for year 2
        
    Raises
    ------
    ValueError
        If no Q4 observation in historical data or insufficient future Q4s.
    """
    # Find last Q4 in historical data
    base_idx = None
    for i in range(len(periods) - 1, -1, -1):
        if periods[i].endswith("Q4"):
            base_idx = i
            break
    
    if base_idx is None:
        raise ValueError("No Q4 observation in historical data")
    
    base = data[base_idx]
    
    # Find future Q4 indices
    q4_indices = [i for i, p in enumerate(future_periods) if p.endswith("Q4")]
    if len(q4_indices) < 2:
        raise ValueError(
            f"Need two future Q4 values, found {len(q4_indices)}"
        )
    
    # Compute growth rates
    fcst_y1 = fcst[q4_indices[0]]
    fcst_y2 = fcst[q4_indices[1]]
    
    growth_y1 = 100.0 * (fcst_y1 - base) / base
    growth_y2 = 100.0 * (fcst_y2 - fcst_y1) / fcst_y1
    
    return {
        "growth_y1": growth_y1,
        "growth_y2": growth_y2,
        "mean_y1": float(np.nanmean(growth_y1)),
        "sd_y1": float(np.nanstd(growth_y1, ddof=1)),
        "ci_y1": tuple(np.nanpercentile(growth_y1, [5, 95])),
        "mean_y2": float(np.nanmean(growth_y2)),
        "sd_y2": float(np.nanstd(growth_y2, ddof=1)),
        "ci_y2": tuple(np.nanpercentile(growth_y2, [5, 95])),
    }


# ---------------------------------------------------------------------------
# Main forecasting functions
# ---------------------------------------------------------------------------

def forecast_dmfm(
    Y: np.ndarray,
    steps: int,
    config: ForecastConfig | None = None,
    mask: np.ndarray | None = None,
    **kwargs,
) -> ForecastResult:
    """Forecast with a DMFM model.
    
    This function fits a DMFM to the data and generates forecasts by
    iteratively applying the learned dynamics.
    
    Parameters
    ----------
    Y : np.ndarray
        Historical data of shape (T, p1, p2).
    steps : int
        Number of periods to forecast ahead.
    config : ForecastConfig, optional
        Forecasting configuration. If None, uses defaults.
    mask : np.ndarray, optional
        Boolean mask for missing values (True = observed).
    **kwargs
        Additional arguments override config values (e.g., k1=2, P=2).
        
    Returns
    -------
    ForecastResult
        Forecast results including point forecasts and fitted model.
        
    Raises
    ------
    ValueError
        If Y is not 3D or steps is non-positive.
        
    Examples
    --------
    >>> # Basic usage with defaults
    >>> result = forecast_dmfm(Y, steps=8)
    >>> print(result.forecast.shape)  # (8, p1, p2)
    
    >>> # Custom configuration
    >>> config = ForecastConfig(k1=2, k2=2, P=2, seasonal_period=4)
    >>> result = forecast_dmfm(Y, steps=8, config=config)
    
    >>> # Override config with kwargs
    >>> result = forecast_dmfm(Y, steps=8, config=config, k1=3)
    """
    # Validate inputs
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 3:
        raise ValueError(f"Y must be 3D array, got shape {Y.shape}")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    
    # Create/merge configuration
    if config is None:
        config = ForecastConfig(**kwargs)
    else:
        # Override config with kwargs
        config_dict = config.__dict__.copy()
        config_dict.update(kwargs)
        config = ForecastConfig(**config_dict)
    
    if mask is None:
        mask = ~np.isnan(Y)
    
    # Apply seasonal differencing if requested
    seasonal_adjusted = False
    if config.seasonal_period is not None:
        Y_fit = seasonal_difference(Y, config.seasonal_period)
        mask_fit = mask[config.seasonal_period:] & mask[:-config.seasonal_period]
        seasonal_adjusted = True
    else:
        Y_fit = Y
        mask_fit = mask
    
    # Fit model
    model, em_result = fit_dmfm(
        Y_fit,
        k1=config.k1,
        k2=config.k2,
        P=config.P,
        mask=mask_fit,
        diagonal_idiosyncratic=config.diagonal_idiosyncratic,
        init_method=config.init_method,
        max_iter=config.max_iter,
        tol=config.tol,
        verbose=config.verbose,
    )
    
    # Generate forecasts by iterating dynamics
    F_hist = [model.F[-l] for l in range(1, model.P + 1)]
    fcst = []
    
    for _ in range(steps):
        F_next = model.dynamics.evolve(F_hist)
        Y_next = model.R @ F_next @ model.C.T
        fcst.append(Y_next)
        
        # Update history
        F_hist = [F_next] + F_hist[:-1]
    
    fcst = np.stack(fcst, axis=0)
    
    # Integrate seasonal differences if applied
    if seasonal_adjusted:
        last_obs = Y[-config.seasonal_period:]
        fcst = integrate_seasonal_diff(last_obs, fcst, config.seasonal_period)
    
    return ForecastResult(
        forecast=fcst,
        model=model,
        config=config,
        seasonal_adjusted=seasonal_adjusted,
    )


def canton_forecast(
    Y: np.ndarray,
    steps: int,
    config: ForecastConfig | None = None,
    mask: np.ndarray | None = None,
    separate_cantons: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast cantonal costs with per-canton and aggregate totals.
    
    Parameters
    ----------
    Y : np.ndarray
        Historical data with shape (T, cantons) or (T, cantons, 1).
    steps : int
        Number of periods to forecast ahead.
    config : ForecastConfig, optional
        Forecasting configuration.
    mask : np.ndarray, optional
        Boolean mask for missing values.
    separate_cantons : bool, default False
        If True, fit a univariate DMFM for each canton independently and
        sum the resulting forecasts. Otherwise, estimate one panel DMFM
        across all cantons jointly.
        
    Returns
    -------
    forecasts : np.ndarray
        Per-canton forecasts of shape (steps, num_cantons).
    total : np.ndarray
        Aggregate forecast of shape (steps,).
        
    Raises
    ------
    ValueError
        If Y has invalid dimensions or mask doesn't match.
        
    Examples
    --------
    >>> # Joint forecast across all cantons
    >>> fcst_cantons, fcst_total = canton_forecast(Y, steps=8)
    
    >>> # Separate forecast per canton
    >>> fcst_cantons, fcst_total = canton_forecast(
    ...     Y, steps=8, separate_cantons=True
    ... )
    """
    # Validate and reshape
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 2:
        Y = Y[:, :, None]
    if Y.ndim != 3:
        raise ValueError(
            f"Y must be 2D (T, cantons) or 3D (T, cantons, 1), "
            f"got shape {Y.shape}"
        )
    
    if mask is not None and mask.shape[:2] != Y.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape[:2]} must match Y shape {Y.shape[:2]}"
        )
    
    if config is None:
        config = ForecastConfig()
    
    if separate_cantons:
        # Fit separate model for each canton
        forecasts = []
        for idx in range(Y.shape[1]):
            result = forecast_dmfm(
                Y[:, idx : idx + 1, :],
                steps,
                config=config,
                mask=mask[:, idx : idx + 1, :] if mask is not None else None,
            )
            forecasts.append(result.forecast[:, 0, 0])
        fcst_array = np.stack(forecasts, axis=1)
    else:
        # Fit joint panel model
        result = forecast_dmfm(Y, steps, config=config, mask=mask)
        fcst_array = result.forecast[:, :, 0]
    
    # Compute total
    total = np.nansum(fcst_array, axis=1)
    
    return fcst_array, total
