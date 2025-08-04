from __future__ import annotations

import numpy as np

from ..DMFM.model import DMFMModel
from ..DMFM.em import EMEstimatorDMFM
from ..DMFM.dynamics import DMFMDynamics


# ---------------------------------------------------------------------------
def seasonal_difference(Y: np.ndarray, period: int) -> np.ndarray:
    """Return seasonal differences of ``Y`` with given period."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 3:
        raise ValueError("Y must be a 3D array")
    T = Y.shape[0]
    if period <= 0 or period >= T:
        raise ValueError("period must be between 1 and T-1")
    return Y[period:] - Y[:-period]


# ---------------------------------------------------------------------------
def integrate_seasonal_diff(last_obs: np.ndarray, diffs: np.ndarray, period: int) -> np.ndarray:
    """Integrate seasonal differences back to levels."""
    history = list(np.asarray(last_obs))
    result = []
    for diff in np.asarray(diffs):
        baseline = history[-period]
        next_level = diff + baseline
        history.append(next_level)
        result.append(next_level)
    return np.stack(result, axis=0)


# ---------------------------------------------------------------------------
def forecast_dmfm(
    Y: np.ndarray,
    steps: int,
    *,
    mask: np.ndarray | None = None,
    seasonal_period: int | None = None,
    k1: int = 1,
    k2: int = 1,
    P: int = 1,
) -> np.ndarray:
    """Forecast ``Y`` ``steps`` periods ahead using a DMFM."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 3:
        raise ValueError("Y must be a 3D array")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if mask is None:
        mask = ~np.isnan(Y)
    diff_applied = False
    if seasonal_period is not None:
        Y_fit = seasonal_difference(Y, seasonal_period)
        mask_fit = mask[seasonal_period:] & mask[:-seasonal_period]
        diff_applied = True
    else:
        Y_fit = Y
        mask_fit = mask
    model = DMFMModel(p1=Y_fit.shape[1], p2=Y_fit.shape[2], k1=k1, k2=k2, P=P)
    model.initialize(Y_fit, mask_fit)
    estimator = EMEstimatorDMFM(model)
    estimator.fit(Y_fit, mask_fit, max_iter=50)
    dynamics = DMFMDynamics(model.A, model.B, model.Pmat, model.Qmat)
    if model.F is not None:
        dynamics.estimate(model.F)
    F_hist = [model.F[-l] for l in range(1, model.P + 1)]
    fcst = []
    for _ in range(steps):
        F_next = dynamics.evolve(F_hist)
        fcst.append(model.R @ F_next @ model.C.T)
        F_hist = [F_next] + F_hist[:-1]
    fcst = np.stack(fcst, axis=0)
    if diff_applied:
        last_obs = Y[-seasonal_period:]
        fcst = integrate_seasonal_diff(last_obs, fcst, seasonal_period)
    return fcst
