"""Forecasting utilities for DMFM models.

This module provides functions for forecasting with Dynamic Matrix Factor Models,
including support for seasonal differencing, canton-level forecasting, growth
rate calculations, and shock/intervention handling.
"""

from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

from ..DMFM import (
    DMFMModel,
    fit_dmfm,
)

if TYPE_CHECKING:
    from ..DMFM.shocks import Shock, ShockSchedule, ShockEffects


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
    i1_factors : bool, default False
        Whether factors are integrated of order 1 (I(1) / random walk).
        Per Barigozzi & Trapin (2025) Section 6, when True:
        - Dynamics A, B are fixed at identity
        - No drift is estimated
        - Data is estimated in levels (no differencing)
        Note: This is different from seasonal_period differencing.
        You can use seasonal_period for seasonal adjustment while
        i1_factors handles stochastic trends.
    shock_schedule : ShockSchedule, optional
        Schedule of known historical shocks/interventions. These are used
        during model estimation to separate shock effects from underlying
        dynamics.
    future_shocks : list[Shock], optional
        Additional shocks that occur in the forecast horizon. Their start_t
        should be relative to the forecast origin (i.e., start_t=0 means
        first forecast period). Used for scenario analysis.
    include_shock_uncertainty : bool, default False
        Whether to include shock effect estimation uncertainty in forecast
        confidence intervals.
    n_shock_simulations : int, default 100
        Number of Monte Carlo simulations for shock uncertainty quantification.
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
    i1_factors: bool = False
    shock_schedule: Optional["ShockSchedule"] = None
    future_shocks: Optional[List["Shock"]] = None
    include_shock_uncertainty: bool = False
    n_shock_simulations: int = 100


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
    shock_effects : ShockEffects, optional
        Estimated shock effect parameters (if shocks were provided).
    forecast_factors : np.ndarray, optional
        Forecasted factors of shape (steps, k1, k2).
    forecast_lower : np.ndarray, optional
        Lower confidence bound for forecasts (if uncertainty quantified).
    forecast_upper : np.ndarray, optional
        Upper confidence bound for forecasts (if uncertainty quantified).
    """

    forecast: np.ndarray
    model: DMFMModel
    config: ForecastConfig
    seasonal_adjusted: bool = False
    shock_effects: Optional["ShockEffects"] = None
    forecast_factors: Optional[np.ndarray] = None
    forecast_lower: Optional[np.ndarray] = None
    forecast_upper: Optional[np.ndarray] = None


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
        raise ValueError(f"period must be between 1 and {T-1}, got {period}")

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


def load_cost_matrix(
    path: str,
) -> (
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
        data = np.full((len(periods), len(cantons), len(groups)), np.nan, dtype=float)
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
        raise ValueError(f"Need two future Q4 values, found {len(q4_indices)}")

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
    """Forecast with a DMFM model, with optional shock/intervention support.

    This function fits a DMFM to the data and generates forecasts by
    iteratively applying the learned dynamics. When shocks are specified,
    their effects are estimated during model fitting and applied during
    forecasting.

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
        Forecast results including point forecasts, fitted model, and
        estimated shock effects (if shocks were provided).

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

    >>> # With shocks
    >>> from KPOKPCH.DMFM.shocks import Shock, ShockSchedule
    >>> schedule = ShockSchedule([
    ...     Shock("covid", start_t=32, end_t=35),
    ... ])
    >>> config = ForecastConfig(k1=2, k2=2, shock_schedule=schedule)
    >>> result = forecast_dmfm(Y, steps=8, config=config)
    >>> print(result.shock_effects.factor_effects.shape)
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
        # Override config with kwargs (excluding shock objects which can't be
        # merged via dict update)
        config_dict = {
            k: v
            for k, v in config.__dict__.items()
            if k not in ("shock_schedule", "future_shocks")
        }
        override_dict = {
            k: v
            for k, v in kwargs.items()
            if k not in ("shock_schedule", "future_shocks")
        }
        config_dict.update(override_dict)

        # Handle shock parameters separately
        shock_schedule = kwargs.get("shock_schedule", config.shock_schedule)
        future_shocks = kwargs.get("future_shocks", config.future_shocks)

        config = ForecastConfig(
            **config_dict,
            shock_schedule=shock_schedule,
            future_shocks=future_shocks,
        )

    if mask is None:
        mask = ~np.isnan(Y)

    T_original = Y.shape[0]

    # Apply seasonal differencing if requested
    seasonal_adjusted = False
    if config.seasonal_period is not None:
        Y_fit = seasonal_difference(Y, config.seasonal_period)
        mask_fit = mask[config.seasonal_period :] & mask[: -config.seasonal_period]
        seasonal_adjusted = True
        T_fit = Y_fit.shape[0]
    else:
        Y_fit = Y
        mask_fit = mask
        T_fit = T_original

    # Adjust shock schedule for seasonal differencing
    shock_schedule_fit = config.shock_schedule
    if seasonal_adjusted and config.shock_schedule is not None:
        # When data is differenced, shock timing shifts by seasonal_period
        # Create adjusted schedule with shifted start/end times
        from ..DMFM.shocks import Shock, ShockSchedule

        adjusted_shocks = []
        for shock in config.shock_schedule.shocks:
            # Shift timing back by seasonal_period
            new_start = max(0, shock.start_t - config.seasonal_period)
            new_end = None
            if shock.end_t is not None:
                new_end = max(0, shock.end_t - config.seasonal_period)
            adjusted = Shock(
                name=shock.name,
                start_t=new_start,
                end_t=new_end,
                level=shock.level,
                scope=shock.scope,
                cantons=shock.cantons,
                categories=shock.categories,
                decay_type=shock.decay_type,
                decay_rate=shock.decay_rate,
                fixed_effect=shock.fixed_effect,
            )
            adjusted_shocks.append(adjusted)
        shock_schedule_fit = ShockSchedule(adjusted_shocks)

    # Fit model with shocks if provided
    model, em_result = _fit_dmfm_with_shocks(
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
        i1_factors=config.i1_factors,
        shock_schedule=shock_schedule_fit,
    )

    shock_effects = em_result.shock_effects

    # Build future shock schedule for forecast horizon
    future_shock_contrib = None
    future_obs_shock_contrib = None

    if shock_schedule_fit is not None or config.future_shocks is not None:
        from ..DMFM.shocks import apply_factor_shocks, apply_observation_shocks

        # Extend shock schedule into forecast horizon
        if shock_schedule_fit is not None:
            X_future, extended_schedule = shock_schedule_fit.extend_to_forecast_horizon(
                T_fit, steps, config.future_shocks
            )
        else:
            from ..DMFM.shocks import ShockSchedule

            extended_schedule = ShockSchedule(config.future_shocks or [])

    # Generate forecasts by iterating dynamics with shock support
    F_hist = [model.F[-l] for l in range(1, model.P + 1)]
    fcst = []
    fcst_factors = []

    for h in range(steps):
        # Compute factor-level shock effect for this forecast step
        factor_shock_effect = None
        if shock_effects is not None and shock_effects.factor_effects is not None:
            if shock_schedule_fit is not None or config.future_shocks:
                t_abs = T_fit + h
                factor_shock_effect = np.zeros((model.k1, model.k2))
                # Check extended schedule for active shocks
                if "extended_schedule" in dir() and extended_schedule is not None:
                    for s, shock in enumerate(extended_schedule.factor_shocks):
                        intensity = shock.indicator(t_abs)
                        if intensity > 0 and s < shock_effects.n_factor_shocks:
                            factor_shock_effect += (
                                intensity * shock_effects.factor_effects[s]
                            )

        # Evolve factors
        F_next = model.dynamics.evolve(F_hist, shock_effect=factor_shock_effect)
        fcst_factors.append(F_next.copy())

        # Map to observations
        Y_next = model.R @ F_next @ model.C.T

        # Add observation-level shock effects
        if shock_effects is not None and shock_effects.observation_effects is not None:
            if "extended_schedule" in dir() and extended_schedule is not None:
                t_abs = T_fit + h
                for s, shock in enumerate(extended_schedule.observation_shocks):
                    intensity = shock.indicator(t_abs)
                    if intensity > 0 and s < shock_effects.n_observation_shocks:
                        Y_next += intensity * shock_effects.observation_effects[s]

        fcst.append(Y_next)

        # Update history
        F_hist = [F_next] + F_hist[:-1]

    fcst = np.stack(fcst, axis=0)
    fcst_factors = np.stack(fcst_factors, axis=0)

    # Integrate seasonal differences if applied
    if seasonal_adjusted:
        last_obs = Y[-config.seasonal_period :]
        fcst = integrate_seasonal_diff(last_obs, fcst, config.seasonal_period)

    return ForecastResult(
        forecast=fcst,
        model=model,
        config=config,
        seasonal_adjusted=seasonal_adjusted,
        shock_effects=shock_effects,
        forecast_factors=fcst_factors,
    )


def _fit_dmfm_with_shocks(
    Y: np.ndarray,
    k1: int,
    k2: int,
    P: int = 1,
    mask: np.ndarray | None = None,
    diagonal_idiosyncratic: bool = False,
    init_method: str = "svd",
    max_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = False,
    i1_factors: bool = False,
    shock_schedule: Optional["ShockSchedule"] = None,
):
    """Fit DMFM with optional shock schedule.

    This is an internal helper that extends fit_dmfm to support shocks.
    """
    from ..DMFM import (
        DMFMModel,
        DMFMConfig,
        EMEstimatorDMFM,
        EMConfig,
    )

    T, p1, p2 = Y.shape

    # Create model
    config = DMFMConfig(
        p1=p1,
        p2=p2,
        k1=k1,
        k2=k2,
        P=P,
        diagonal_idiosyncratic=diagonal_idiosyncratic,
    )
    model = DMFMModel(config)

    # Initialize
    model.initialize(Y, mask=mask, method=init_method)

    # Set I(1) factors flag if requested
    if i1_factors:
        model.dynamics.i1_factors = True

    # Fit with shocks
    em_config = EMConfig(max_iter=max_iter, tol=tol, verbose=verbose)
    estimator = EMEstimatorDMFM(model, em_config)
    result = estimator.fit(Y, mask=mask, shock_schedule=shock_schedule)

    return model, result


def scenario_forecast(
    Y: np.ndarray,
    steps: int,
    base_config: ForecastConfig,
    scenarios: Dict[str, List["Shock"]],
    mask: np.ndarray | None = None,
) -> Dict[str, ForecastResult]:
    """Generate forecasts under multiple shock scenarios.

    This function fits the model once (using base_config.shock_schedule for
    historical shocks) and then generates separate forecasts for each
    scenario with different future shock assumptions.

    Parameters
    ----------
    Y : np.ndarray
        Historical data of shape (T, p1, p2).
    steps : int
        Number of periods to forecast ahead.
    base_config : ForecastConfig
        Base configuration including historical shock_schedule.
    scenarios : dict[str, list[Shock]]
        Dictionary mapping scenario names to lists of future shocks.
        Each shock's start_t should be relative to forecast origin
        (start_t=0 means first forecast period).
    mask : np.ndarray, optional
        Boolean mask for missing values.

    Returns
    -------
    dict[str, ForecastResult]
        Dictionary mapping scenario names to forecast results.

    Examples
    --------
    >>> scenarios = {
    ...     "baseline": [],  # No additional shocks
    ...     "covid_wave": [
    ...         Shock("covid2", start_t=2, end_t=4, scope="global")
    ...     ],
    ...     "policy_reform": [
    ...         Shock("reform", start_t=4, end_t=None, scope="category",
    ...               categories=[2])
    ...     ],
    ... }
    >>> results = scenario_forecast(Y, steps=12, base_config, scenarios)
    >>> baseline_fcst = results["baseline"].forecast
    >>> covid_fcst = results["covid_wave"].forecast
    """
    results = {}

    for name, future_shocks in scenarios.items():
        # Create config variant with these future shocks
        scenario_config = replace(base_config, future_shocks=future_shocks)
        results[name] = forecast_dmfm(Y, steps, config=scenario_config, mask=mask)

    return results


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
            f"Y must be 2D (T, cantons) or 3D (T, cantons, 1), " f"got shape {Y.shape}"
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
