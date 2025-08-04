import numpy as np


def seasonal_difference(Y: np.ndarray, period: int) -> np.ndarray:
    """Return seasonal differences of ``Y`` with the given period.

    Parameters
    ----------
    Y : ndarray
        Data array ``(T, p1, p2)``.
    period : int
        Seasonal period used for differencing (e.g. ``4`` for quarterly data).

    Returns
    -------
    ndarray
        Array with shape ``(T - period, p1, p2)`` containing ``Y_t - Y_{t-period}``.
    """

    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 3:
        raise ValueError("Y must be a 3D array")
    T = Y.shape[0]
    if period <= 0 or period >= T:
        raise ValueError("period must be between 1 and T-1")
    return Y[period:] - Y[:-period]