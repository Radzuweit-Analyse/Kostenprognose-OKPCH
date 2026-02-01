"""Forecasting utilities for DMFM models.

This module provides functions for forecasting with Dynamic Matrix Factor Models,
including data loading, period utilities, and shock/intervention handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
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
        When True:
        - Dynamics A, B are fixed at identity
        - Drift is still estimated (random walk WITH drift for trending data)
        - Data is estimated in levels
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
    shock_effects: Optional["ShockEffects"] = None
    forecast_factors: Optional[np.ndarray] = None
    forecast_lower: Optional[np.ndarray] = None
    forecast_upper: Optional[np.ndarray] = None


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
    >>> config = ForecastConfig(k1=2, k2=2, P=2)
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

    T_fit = Y.shape[0]
    Y_fit = Y
    mask_fit = mask
    shock_schedule_fit = config.shock_schedule

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

    # Level adjustment: ensure forecast connects smoothly to last observation
    # This corrects for any gap between model fit and actual data at forecast origin
    Y_last_actual = Y_fit[-1]  # Last actual observation
    Y_last_fitted = model.R @ model.F[-1] @ model.C.T  # Model's fitted value at T
    level_adjustment = Y_last_actual - Y_last_fitted
    # Only adjust where we have valid observations (not NaN)
    valid_mask = ~np.isnan(level_adjustment)
    level_adjustment = np.where(valid_mask, level_adjustment, 0.0)
    fcst = fcst + level_adjustment  # Add adjustment to all forecast steps

    return ForecastResult(
        forecast=fcst,
        model=model,
        config=config,
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
