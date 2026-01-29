"""Model selection tools for DMFM: AIC, BIC, and automatic rank selection.

This module provides information criteria and automatic selection of the
number of factors (k1, k2) for the Dynamic Matrix Factor Model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from .model import DMFMModel, DMFMConfig
from .em import EMEstimatorDMFM, EMConfig, EMResult
from .kalman import KalmanFilterDMFM


@dataclass
class ModelSelectionResult:
    """Results from model selection.

    Attributes
    ----------
    best_k1 : int
        Selected number of row factors.
    best_k2 : int
        Selected number of column factors.
    best_model : DMFMModel
        Fitted model with selected ranks.
    best_em_result : EMResult
        EM results for best model.
    criterion : str
        Criterion used for selection ("aic" or "bic").
    best_value : float
        Value of the selection criterion for best model.
    results_grid : dict
        Dictionary mapping (k1, k2) to selection results.
    """

    best_k1: int
    best_k2: int
    best_model: DMFMModel
    best_em_result: EMResult
    criterion: str
    best_value: float
    results_grid: dict


@dataclass
class InformationCriteria:
    """Information criteria for a fitted DMFM.

    Attributes
    ----------
    loglik : float
        Log-likelihood value.
    n_params : int
        Number of estimated parameters.
    n_obs : int
        Number of observations (T * p1 * p2 minus missing).
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    aicc : float
        Corrected AIC for small samples.
    """

    loglik: float
    n_params: int
    n_obs: int
    aic: float
    bic: float
    aicc: float


def count_parameters(
    model: DMFMModel,
    include_dynamics: bool = True,
    include_drift: bool = True,
) -> int:
    """Count the number of free parameters in a DMFM.

    Parameters
    ----------
    model : DMFMModel
        Fitted DMFM model.
    include_dynamics : bool, default True
        Whether to count dynamics parameters (A, B, P, Q).
    include_drift : bool, default True
        Whether to count drift parameters.

    Returns
    -------
    int
        Number of free parameters.

    Notes
    -----
    Parameter counting for factor models accounts for identification:
    - Loadings R, C are orthonormalized, reducing free parameters
    - For orthonormal R (p1 x k1): p1*k1 - k1*(k1+1)/2 free params
    - Idiosyncratic H, K: p1 + p2 if diagonal, more if full
    - Dynamics A_l, B_l: k1^2 + k2^2 per lag
    - Innovation covariances P, Q: k1*(k1+1)/2 + k2*(k2+1)/2
    - Drift: k1 * k2
    """
    p1, p2 = model.p1, model.p2
    k1, k2 = model.k1, model.k2
    P = model.P

    n_params = 0

    # Loadings (with orthonormalization constraint)
    # R: p1*k1 parameters minus k1*(k1+1)/2 for orthonormality
    # C: p2*k2 parameters minus k2*(k2+1)/2 for orthonormality
    n_params += p1 * k1 - k1 * (k1 + 1) // 2
    n_params += p2 * k2 - k2 * (k2 + 1) // 2

    # Idiosyncratic covariances
    if model.diagonal_idiosyncratic:
        # Diagonal: p1 + p2 parameters (minus 2 for trace normalization)
        n_params += p1 + p2 - 2
    else:
        # Full symmetric: (p1*(p1+1)/2 - 1) + (p2*(p2+1)/2 - 1)
        n_params += p1 * (p1 + 1) // 2 - 1
        n_params += p2 * (p2 + 1) // 2 - 1

    if include_dynamics:
        # Check if I(1) mode (dynamics fixed)
        is_i1 = model.dynamics is not None and model.dynamics.i1_factors

        if not is_i1:
            # Dynamics coefficients A_l, B_l for each lag
            n_params += P * (k1 * k1 + k2 * k2)

        # Innovation covariances P, Q (symmetric)
        n_params += k1 * (k1 + 1) // 2
        n_params += k2 * (k2 + 1) // 2

    if include_drift:
        is_i1 = model.dynamics is not None and model.dynamics.i1_factors
        if not is_i1:
            # Drift matrix
            n_params += k1 * k2

    return max(1, n_params)


def compute_information_criteria(
    model: DMFMModel,
    Y: np.ndarray,
    mask: np.ndarray | None = None,
    loglik: float | None = None,
) -> InformationCriteria:
    """Compute AIC, BIC, and AICc for a fitted DMFM.

    Parameters
    ----------
    model : DMFMModel
        Fitted DMFM model.
    Y : np.ndarray
        Observed data of shape (T, p1, p2).
    mask : np.ndarray, optional
        Boolean mask for missing values.
    loglik : float, optional
        Pre-computed log-likelihood. If None, will be computed.

    Returns
    -------
    InformationCriteria
        Computed information criteria.

    Examples
    --------
    >>> model, em_result = fit_dmfm(Y, k1=2, k2=2)
    >>> ic = compute_information_criteria(model, Y)
    >>> print(f"AIC: {ic.aic:.2f}, BIC: {ic.bic:.2f}")
    """
    if mask is None:
        mask = ~np.isnan(Y)

    # Compute log-likelihood if not provided
    if loglik is None:
        kf = KalmanFilterDMFM(model)
        state = kf.filter(Y, mask)
        state = kf.smooth(state)
        loglik = kf.log_likelihood(Y, mask, state)

    # Count parameters
    n_params = count_parameters(model)

    # Count observations
    n_obs = int(np.sum(mask))

    # AIC = -2*loglik + 2*k
    aic = -2 * loglik + 2 * n_params

    # BIC = -2*loglik + log(n)*k
    bic = -2 * loglik + np.log(n_obs) * n_params

    # AICc = AIC + 2*k*(k+1)/(n-k-1) (corrected for small samples)
    if n_obs > n_params + 1:
        aicc = aic + 2 * n_params * (n_params + 1) / (n_obs - n_params - 1)
    else:
        aicc = np.inf

    return InformationCriteria(
        loglik=loglik,
        n_params=n_params,
        n_obs=n_obs,
        aic=aic,
        bic=bic,
        aicc=aicc,
    )


def select_rank(
    Y: np.ndarray,
    k1_range: tuple[int, int] | list[int] = (1, 5),
    k2_range: tuple[int, int] | list[int] = (1, 5),
    P: int = 1,
    criterion: str = "bic",
    mask: np.ndarray | None = None,
    diagonal_idiosyncratic: bool = True,
    init_method: str = "svd",
    max_iter: int = 50,
    tol: float = 1e-4,
    verbose: bool = False,
    callback: Callable[[int, int, InformationCriteria], None] | None = None,
) -> ModelSelectionResult:
    """Select optimal factor ranks (k1, k2) via information criterion.

    Fits DMFM models for a grid of (k1, k2) values and selects the
    combination that minimizes the specified information criterion.

    Parameters
    ----------
    Y : np.ndarray
        Observed data of shape (T, p1, p2).
    k1_range : tuple or list, default (1, 5)
        Range of k1 values to try. If tuple, interpreted as (min, max).
    k2_range : tuple or list, default (1, 5)
        Range of k2 values to try. If tuple, interpreted as (min, max).
    P : int, default 1
        MAR order.
    criterion : {"aic", "bic", "aicc"}, default "bic"
        Information criterion to minimize.
    mask : np.ndarray, optional
        Boolean mask for missing values.
    diagonal_idiosyncratic : bool, default True
        Whether to use diagonal idiosyncratic covariance.
    init_method : str, default "svd"
        Initialization method.
    max_iter : int, default 50
        Maximum EM iterations per model.
    tol : float, default 1e-4
        EM convergence tolerance.
    verbose : bool, default False
        Whether to print progress.
    callback : callable, optional
        Function called after each (k1, k2) fit with signature
        ``callback(k1, k2, ic)``.

    Returns
    -------
    ModelSelectionResult
        Selection results including best model and full results grid.

    Examples
    --------
    >>> result = select_rank(Y, k1_range=(1, 4), k2_range=(1, 4))
    >>> print(f"Selected: k1={result.best_k1}, k2={result.best_k2}")
    >>> print(f"BIC: {result.best_value:.2f}")

    >>> # Use the selected model
    >>> model = result.best_model
    >>> factors = model.F
    """
    Y = np.asarray(Y, dtype=float)
    T, p1, p2 = Y.shape

    if mask is None:
        mask = ~np.isnan(Y)

    # Convert ranges to lists
    if isinstance(k1_range, tuple):
        k1_values = list(range(k1_range[0], k1_range[1] + 1))
    else:
        k1_values = list(k1_range)

    if isinstance(k2_range, tuple):
        k2_values = list(range(k2_range[0], k2_range[1] + 1))
    else:
        k2_values = list(k2_range)

    # Validate ranges
    k1_values = [k for k in k1_values if 1 <= k <= p1]
    k2_values = [k for k in k2_values if 1 <= k <= p2]

    if not k1_values or not k2_values:
        raise ValueError("No valid (k1, k2) combinations in specified ranges")

    # Grid search
    results_grid = {}
    best_k1, best_k2 = k1_values[0], k2_values[0]
    best_value = np.inf
    best_model = None
    best_em_result = None

    total = len(k1_values) * len(k2_values)
    count = 0

    for k1 in k1_values:
        for k2 in k2_values:
            count += 1
            if verbose:
                print(f"[{count}/{total}] Fitting k1={k1}, k2={k2}...")

            try:
                # Create and fit model
                config = DMFMConfig(
                    p1=p1,
                    p2=p2,
                    k1=k1,
                    k2=k2,
                    P=P,
                    diagonal_idiosyncratic=diagonal_idiosyncratic,
                )
                model = DMFMModel(config)
                model.initialize(Y, mask=mask, method=init_method)

                em_config = EMConfig(max_iter=max_iter, tol=tol, verbose=False)
                estimator = EMEstimatorDMFM(model, em_config)
                em_result = estimator.fit(Y, mask=mask)

                # Compute information criteria
                ic = compute_information_criteria(
                    model, Y, mask, loglik=em_result.final_loglik
                )

                # Store results
                results_grid[(k1, k2)] = {
                    "ic": ic,
                    "em_result": em_result,
                    "model": model,
                }

                # Get criterion value
                if criterion == "aic":
                    crit_value = ic.aic
                elif criterion == "bic":
                    crit_value = ic.bic
                elif criterion == "aicc":
                    crit_value = ic.aicc
                else:
                    raise ValueError(f"Unknown criterion: {criterion}")

                if verbose:
                    print(
                        f"    loglik={ic.loglik:.2f}, "
                        f"AIC={ic.aic:.2f}, BIC={ic.bic:.2f}"
                    )

                # Update best
                if crit_value < best_value:
                    best_value = crit_value
                    best_k1, best_k2 = k1, k2
                    best_model = model
                    best_em_result = em_result

                # Callback
                if callback is not None:
                    callback(k1, k2, ic)

            except Exception as e:
                if verbose:
                    print(f"    Failed: {e}")
                results_grid[(k1, k2)] = {"error": str(e)}

    if best_model is None:
        raise RuntimeError("All model fits failed")

    if verbose:
        print(f"\nSelected: k1={best_k1}, k2={best_k2} ({criterion}={best_value:.2f})")

    return ModelSelectionResult(
        best_k1=best_k1,
        best_k2=best_k2,
        best_model=best_model,
        best_em_result=best_em_result,
        criterion=criterion,
        best_value=best_value,
        results_grid=results_grid,
    )


def print_selection_summary(result: ModelSelectionResult) -> None:
    """Print a summary table of model selection results.

    Parameters
    ----------
    result : ModelSelectionResult
        Results from select_rank().
    """
    print(f"\nModel Selection Summary (criterion: {result.criterion.upper()})")
    print("=" * 60)
    print(
        f"{'k1':>4} {'k2':>4} {'loglik':>12} {'AIC':>12} {'BIC':>12} {'n_params':>10}"
    )
    print("-" * 60)

    # Sort by criterion value
    sorted_keys = sorted(
        [k for k in result.results_grid.keys() if "ic" in result.results_grid[k]],
        key=lambda k: getattr(result.results_grid[k]["ic"], result.criterion),
    )

    for k1, k2 in sorted_keys:
        entry = result.results_grid[(k1, k2)]
        ic = entry["ic"]
        marker = " *" if k1 == result.best_k1 and k2 == result.best_k2 else ""
        print(
            f"{k1:>4} {k2:>4} {ic.loglik:>12.2f} {ic.aic:>12.2f} "
            f"{ic.bic:>12.2f} {ic.n_params:>10}{marker}"
        )

    print("-" * 60)
    print(f"Selected: k1={result.best_k1}, k2={result.best_k2}")
    print(f"Best {result.criterion.upper()}: {result.best_value:.2f}")
