from .model import DMFMModel, DMFMConfig
from .dynamics import DMFMDynamics, DynamicsConfig
from .kalman import KalmanFilterDMFM, KalmanState, KalmanConfig
from .em import EMEstimatorDMFM, EMResult, EMConfig
from .utils import (
    initialize_model,
    init_factor_loadings,
    init_idiosyncratic,
    init_dynamics,
    InitMethod,
    InitializationResult,
)
from .selection import (
    select_rank,
    compute_information_criteria,
    count_parameters,
    print_selection_summary,
    ModelSelectionResult,
    InformationCriteria,
)
from .shocks import (
    # Shocks (stochastic)
    Shock,
    ShockSchedule,
    ShockEffects,
    ShockLevel,
    ShockScope,
    DecayType,
    estimate_shock_effects,
    apply_factor_shocks,
    apply_observation_shocks,
    # Interventions (deterministic)
    Intervention,
    InterventionSchedule,
    InterventionType,
    # Factory
    ScheduleFactory,
)

__all__ = [
    # Model
    "DMFMModel",
    "DMFMConfig",
    # Dynamics
    "DMFMDynamics",
    "DynamicsConfig",
    # Kalman
    "KalmanFilterDMFM",
    "KalmanConfig",
    "KalmanState",
    # EM
    "EMEstimatorDMFM",
    "EMConfig",
    "EMResult",
    # Utilities
    "initialize_model",
    "init_factor_loadings",
    "init_idiosyncratic",
    "init_dynamics",
    "InitMethod",
    "InitializationResult",
    # Model selection
    "select_rank",
    "compute_information_criteria",
    "count_parameters",
    "print_selection_summary",
    "ModelSelectionResult",
    "InformationCriteria",
    # Shocks (stochastic)
    "Shock",
    "ShockSchedule",
    "ShockEffects",
    "ShockLevel",
    "ShockScope",
    "DecayType",
    "estimate_shock_effects",
    "apply_factor_shocks",
    "apply_observation_shocks",
    # Interventions (deterministic)
    "Intervention",
    "InterventionSchedule",
    "InterventionType",
    # Factory
    "ScheduleFactory",
    # Convenience
    "fit_dmfm",
]


# Convenience workflow function
def fit_dmfm(
    Y,
    k1,
    k2,
    P=1,
    mask=None,
    diagonal_idiosyncratic=False,
    init_method="svd",
    max_iter=100,
    tol=1e-4,
    verbose=False,
    i1_factors=False,
    shock_schedule=None,
):
    """Convenience function to fit DMFM in one call.

    This function creates a model, initializes it, and runs EM estimation
    in a single step. For more control, use the individual classes.

    Parameters
    ----------
    Y : np.ndarray
        Observed data of shape (T, p1, p2).
    k1, k2 : int
        Number of row and column factors.
    P : int, default 1
        MAR order.
    mask : np.ndarray, optional
        Boolean mask for missing values (True = observed).
    diagonal_idiosyncratic : bool, default False
        Whether to use diagonal idiosyncratic covariance.
    init_method : str, default "svd"
        Initialization method ("svd" or "pe").
    max_iter : int, default 100
        Maximum EM iterations.
    tol : float, default 1e-4
        Convergence tolerance.
    verbose : bool, default False
        Whether to print progress.
    i1_factors : bool, default False
        Whether factors are integrated of order 1 (I(1) / random walk).
        When True, implements Barigozzi & Trapin (2025) Section 6:
        - Dynamics A, B are fixed at identity (random walk)
        - No drift is estimated
        - Estimation is done in levels (no differencing needed)
        This is useful for data with stochastic trends.
    shock_schedule : ShockSchedule, optional
        Schedule of known shocks/interventions. If provided, shock effects
        will be estimated as part of the EM algorithm.

    Returns
    -------
    model : DMFMModel
        Fitted model.
    result : EMResult
        EM fitting results (includes shock_effects if shocks provided).

    Examples
    --------
    >>> from KPOKPCH.DMFM import fit_dmfm
    >>> model, result = fit_dmfm(Y, k1=3, k2=2, verbose=True)
    >>> factors = model.F
    >>> print(f"Converged: {result.converged}")

    >>> # For non-stationary data with stochastic trends
    >>> model, result = fit_dmfm(Y, k1=2, k2=2, i1_factors=True)

    >>> # With shocks
    >>> from KPOKPCH.DMFM import Shock, ShockSchedule
    >>> schedule = ShockSchedule([
    ...     Shock("covid", start_t=32, end_t=35),
    ... ])
    >>> model, result = fit_dmfm(Y, k1=2, k2=2, shock_schedule=schedule)
    >>> print(result.shock_effects.factor_effects.shape)
    """
    import numpy as np

    # Infer dimensions
    Y = np.asarray(Y)
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

    # Fit
    em_config = EMConfig(max_iter=max_iter, tol=tol, verbose=verbose)
    estimator = EMEstimatorDMFM(model, em_config)
    result = estimator.fit(Y, mask=mask, shock_schedule=shock_schedule)

    return model, result
