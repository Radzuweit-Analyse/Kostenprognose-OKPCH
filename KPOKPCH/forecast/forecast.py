from __future__ import annotations

import csv
import numpy as np
from typing import List, Tuple

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
def load_cost_matrix(path: str) -> Tuple[List[str], List[str], List[str], np.ndarray]:
    """Load canton cost tensor from CSV produced by ``prepare-MOKKE-data.py``.

    The CSV is expected to have ``Periode`` as the first column and flattened
    headers of the form ``<Canton>|<Groupe_de_couts>`` for each remaining
    column. Values are arranged into a 3D array with shape
    ``(T, num_cantons, num_groups)``.

    Parameters
    ----------
    path:
        Path to the CSV file containing period identifiers in the first column
        followed by cantonal cost values.

    Returns
    -------
    periods, cantons, groups, data:
        ``periods`` contains the period labels, ``cantons`` the canton names,
        ``groups`` the cost group labels and ``data`` the numeric values as a
        ``float`` array with shape ``(T, num_cantons, num_groups)``.
    """

    periods: List[str] = []
    data_rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        raw_columns = header[1:]
        canton_group_pairs: List[Tuple[str, str]] = []
        for col in raw_columns:
            if "|" not in col:
                raise ValueError(
                    "Expected column headers in the form '<Canton>|<Group>'"
                )
            canton, group = col.split("|", maxsplit=1)
            canton_group_pairs.append((canton, group))

        cantons: List[str] = []
        groups: List[str] = []
        for canton, group in canton_group_pairs:
            if canton not in cantons:
                cantons.append(canton)
            if group not in groups:
                groups.append(group)
        
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
    data = np.full((len(periods), len(cantons), len(groups)), np.nan, dtype=float)
    canton_idx = {c: i for i, c in enumerate(cantons)}
    group_idx = {g: j for j, g in enumerate(groups)}
    for col, (canton, group) in enumerate(canton_group_pairs):
        data[:, canton_idx[canton], group_idx[group]] = flat[:, col]

    return periods, cantons, groups, data


# ---------------------------------------------------------------------------
def generate_future_periods(last_period: str, steps: int) -> List[str]:
    """Generate future quarterly period labels following ``last_period``."""

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
def compute_q4_growth(
    periods: List[str],
    data: np.ndarray,
    fcst: np.ndarray,
    future_periods: List[str],
) -> dict:
    """Compute Q4-over-Q4 growth metrics for cantonal forecasts."""

    base_idx = None
    for i in range(len(periods) - 1, -1, -1):
        if periods[i].endswith("Q4"):
            base_idx = i
            break
    if base_idx is None:
        raise ValueError("No Q4 observation in historical data")
    base = data[base_idx]
    q4_indices = [i for i, p in enumerate(future_periods) if p.endswith("Q4")]
    if len(q4_indices) < 2:
        raise ValueError("Need two future Q4 values")
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


# ---------------------------------------------------------------------------
def canton_forecast(
    Y: np.ndarray,
    steps: int,
    *,
    mask: np.ndarray | None = None,
    seasonal_period: int | None = None,
    k1: int = 1,
    k2: int = 1,
    P: int = 1,
    separate_cantons: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast cantonal costs and return per-canton and aggregate totals.

    Parameters
    ----------
    Y:
        Historical data with shape ``(T, cantons)`` or ``(T, cantons, 1)``.
    steps:
        Number of periods to forecast ahead.
    separate_cantons:
        If ``True``, fit a univariate DMFM for each canton independently and
        sum the resulting forecasts. Otherwise, estimate one panel DMFM across
        all cantons jointly. The function always returns per-canton forecasts
        and their sum across cantons.
    """

    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 2:
        Y = Y[:, :, None]
    if Y.ndim != 3:
        raise ValueError("Y must be 2D (T, cantons) or 3D (T, cantons, 1)")
    if mask is not None and mask.shape[:2] != Y.shape[:2]:
        raise ValueError("mask shape must match the first two dimensions of Y")

    if separate_cantons:
        forecasts = []
        for idx in range(Y.shape[1]):
            fcst_single = forecast_dmfm(
                Y[:, idx : idx + 1, :],
                steps,
                mask=mask[:, idx : idx + 1, :] if mask is not None else None,
                seasonal_period=seasonal_period,
                k1=k1,
                k2=k2,
                P=P,
            )
            forecasts.append(fcst_single[:, 0, 0])
        fcst_array = np.stack(forecasts, axis=1)
    else:
        fcst_panel = forecast_dmfm(
            Y,
            steps,
            mask=mask,
            seasonal_period=seasonal_period,
            k1=k1,
            k2=k2,
            P=P,
        )
        fcst_array = fcst_panel[:, :, 0]

    total = np.nansum(fcst_array, axis=1)
    return fcst_array, total
