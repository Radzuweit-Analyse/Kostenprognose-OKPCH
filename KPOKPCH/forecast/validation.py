# Out-of-sample validation
from __future__ import annotations

import numpy as np

from .forecast import forecast_dmfm


# ---------------------------------------------------------------------------
def out_of_sample_rmse(
    Y: np.ndarray,
    steps: int,
    *,
    mask: np.ndarray | None = None,
    seasonal_period: int | None = None,
    k1: int = 1,
    k2: int = 1,
    P: int = 1,
) -> float:
    """Compute out-of-sample RMSE for a DMFM forecast.

    The model is estimated on ``Y`` excluding the last ``steps`` observations.
    A forecast is produced for these periods and compared with the held-out
    data using the root mean squared error (RMSE).
    """

    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 3:
        raise ValueError("Y must be a 3D array")
    if steps <= 0 or steps >= Y.shape[0]:
        raise ValueError("steps must be between 1 and T-1")

    Y_train = Y[:-steps]
    mask_train = mask[:-steps] if mask is not None else None

    fcst = forecast_dmfm(
        Y_train,
        steps,
        mask=mask_train,
        seasonal_period=seasonal_period,
        k1=k1,
        k2=k2,
        P=P,
    )
    actual = Y[-steps:]
    err = fcst - actual
    rmse = float(np.sqrt(np.nanmean(err**2)))
    return rmse
