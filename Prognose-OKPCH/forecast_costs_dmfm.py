import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from KPOKPCH import *


def main():
    csv_path = Path(__file__).resolve().parent / "health_costs_tensor.csv"
    loaded = load_cost_matrix(str(csv_path))
    if len(loaded) == 4:
        periods, cantons, groups, data = loaded
    else:
        periods, cantons, data = loaded
        groups = ["Total"]
        data = data[:, :, None] if data.ndim == 2 else data
    scale = 1000.0
    Y = data / scale  # (T, cantons, groups)

    period = 4  # quarterly seasonality
    Y_sd = seasonal_difference(Y, period)
    mask = ~np.isnan(Y_sd)
    model = DMFMModel(p1=Y_sd.shape[1], p2=Y_sd.shape[2], k1=1, k2=1, P=1)
    model.initialize(Y_sd, mask)
    estimator = EMEstimatorDMFM(model)
    estimator.fit(Y_sd, mask, max_iter=50)

    dynamics = DMFMDynamics(model.A, model.B, model.Pmat, model.Qmat)
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
    fcst = fcst_levels * scale
    future_periods = generate_future_periods(periods[-1], steps)

    try:
        total_idx = groups.index("Total")
    except ValueError:
        raise ValueError("'Total' cost group is required in the prepared data")
    data_total = data[:, :, total_idx]
    fcst_total = fcst[:, :, total_idx]

    # Compute yearly totals from historical and forecast data. If the last
    # historical year is incomplete, use its available quarters together with
    # the forecasts to produce a full-year total.
    yearly_totals: dict[str, np.ndarray] = {}
    quarter_counts: dict[str, int] = {}

    # Accumulate historical quarters for the final year in the data
    last_period = periods[-1]
    last_year = last_period[:4]
    last_quarter = int(last_period[-1])
    for period, row in zip(periods, data_total):
        year = period[:4]
        if year == last_year and int(period[-1]) <= last_quarter:
            yearly_totals.setdefault(year, np.zeros(data_total.shape[1]))
            quarter_counts[year] = quarter_counts.get(year, 0) + 1
            yearly_totals[year] += np.nan_to_num(row)

    # Accumulate the forecast quarters
    for i, period in enumerate(future_periods):
        year = period[:4]
        yearly_totals.setdefault(year, np.zeros(fcst_total.shape[1]))
        quarter_counts[year] = quarter_counts.get(year, 0) + 1
        yearly_totals[year] += np.nan_to_num(fcst_total[i])

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
    combined_data = np.vstack([data_total, fcst_total])

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
