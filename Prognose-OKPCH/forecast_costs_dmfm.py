import csv
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

import KPOKPCH


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


def integrate_seasonal_diff(
    last_obs: np.ndarray, diffs: np.ndarray, period: int
) -> np.ndarray:
    """Return level forecasts from seasonal differences."""

    history = list(np.asarray(last_obs))
    result = []
    for diff in diffs:
        baseline = history[-period]
        next_level = diff + baseline
        history.append(next_level)
        result.append(next_level)
    return np.stack(result, axis=0)


def load_cost_matrix(path: str) -> Tuple[List[str], List[str], np.ndarray]:
    """Load canton cost matrix from CSV produced by prepare-MOKKE-data.py."""
    periods: List[str] = []
    data_rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        cantons = header[1:]
        for row in reader:
            periods.append(row[0])
            values = []
            for x in row[1:]:
                try:
                    values.append(float(x))
                except ValueError:
                    values.append(np.nan)
            data_rows.append(values)
    data = np.array(data_rows, dtype=float)
    return periods, cantons, data


def generate_future_periods(last_period: str, steps: int) -> List[str]:
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


def compute_q4_growth(
    periods: List[str],
    data: np.ndarray,
    fcst: np.ndarray,
    future_periods: List[str],
) -> dict:
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


def main():
    csv_path = "C:/Dev/KOSTENPROGNOSE-OKPCH/Prognose-OKPCH/health_costs_matrix.csv"
    periods, cantons, data = load_cost_matrix(csv_path)
    scale = 1000.0
    Y = (data / scale)[:, :, None]  # (T, cantons, 1)

    period = 4  # quarterly seasonality
    Y_sd = KPOKPCH.utils.seasonal_difference(Y, period)
    mask = ~np.isnan(Y_sd)
    model = KPOKPCH.DMFM.DMFMModel(p1=Y_sd.shape[1], p2=Y_sd.shape[2], k1=1, k2=1, P=1)
    model.initialize(Y_sd, mask)
    estimator = KPOKPCH.DMFM.EMEstimatorDMFM(model)
    estimator.fit(Y_sd, mask, max_iter=50)

    dynamics = KPOKPCH.DMFM.DMFMDynamics(model.A, model.B, model.Pmat, model.Qmat)
    if model.F is not None:
        dynamics.estimate(model.F)
    steps = 8  # two years ahead
    F_hist = [model.F[-l] for l in range(1, model.P + 1)]
    fcst_diff = []
    for _ in range(steps):
        F_next = dynamics.evolve(F_hist)
        fcst_diff.append(model.R @ F_next @ model.C.T)
        F_hist = [F_next] + F_hist[:-1]
    fcst_diff = np.stack(fcst_diff, axis=0)
    fcst_levels = integrate_seasonal_diff(Y[-period:], fcst_diff, period)
    fcst = fcst_levels[:, :, 0] * scale
    future_periods = generate_future_periods(periods[-1], steps)

    # Compute yearly totals from historical and forecast data. If the last
    # historical year is incomplete, use its available quarters together with
    # the forecasts to produce a full-year total.
    yearly_totals: dict[str, np.ndarray] = {}
    quarter_counts: dict[str, int] = {}

    # Accumulate historical quarters for the final year in the data
    last_period = periods[-1]
    last_year = last_period[:4]
    last_quarter = int(last_period[-1])
    for period, row in zip(periods, data):
        year = period[:4]
        if year == last_year and int(period[-1]) <= last_quarter:
            yearly_totals.setdefault(year, np.zeros(data.shape[1]))
            quarter_counts[year] = quarter_counts.get(year, 0) + 1
            yearly_totals[year] += np.nan_to_num(row)

    # Accumulate the forecast quarters
    for i, period in enumerate(future_periods):
        year = period[:4]
        yearly_totals.setdefault(year, np.zeros(fcst.shape[1]))
        quarter_counts[year] = quarter_counts.get(year, 0) + 1
        yearly_totals[year] += np.nan_to_num(fcst[i])

    # Only display years where all four quarters are available from
    # historical + forecast data.
    start_year = int(last_year) if last_quarter < 4 else int(last_year) + 1
    for year in sorted(yearly_totals.keys()):
        if int(year) < start_year:
            continue
        if quarter_counts.get(year, 0) == 4:
            print(f"\nForecast totals for {year}:")
            for canton, value in zip(cantons, yearly_totals[year]):
                print(f"  {canton}: {value:,.0f}")

    # Plot historical data with forecasts appended
    combined_periods = periods + future_periods
    combined_data = np.vstack([data, fcst])

    if "CH" in cantons:
        idx = cantons.index("CH")
    else:
        idx = 0
    plt.figure(figsize=(10, 6))
    plt.plot(combined_periods, combined_data[:, idx], label=cantons[idx])
    plt.xticks(rotation=45)
    plt.xlabel("Period")
    plt.ylabel("Cost")
    plt.title("Historical and Forecasted Costs")
    plt.legend()
    plt.tight_layout()

    plt.savefig("forecast_plot.pdf")
    plt.close()


if __name__ == "__main__":
    main()
