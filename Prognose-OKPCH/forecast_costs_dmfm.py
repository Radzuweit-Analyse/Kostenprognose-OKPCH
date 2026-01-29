"""Forecast Swiss health costs using DMFM."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from KPOKPCH.forecast import (
    load_cost_matrix,
    generate_future_periods,
    ForecastConfig,
    forecast_dmfm,
)

VERBOSE = False


def main():
    """Run health cost forecasting."""
    # Load data
    csv_path = Path(__file__).resolve().parent / "health_costs_tensor.csv"
    loaded = load_cost_matrix(str(csv_path))

    if len(loaded) == 4:
        periods, cantons, groups, data = loaded
    else:
        periods, cantons, data = loaded
        groups = ["Total"]
        data = data[:, :, None] if data.ndim == 2 else data

    print(f"\n📊 Data loaded:")
    print(f"   Periods: {len(periods)} ({periods[0]} to {periods[-1]})")
    print(f"   Cantons: {len(cantons)}")
    print(f"   Cost groups: {len(groups)}")
    print(f"   Groups: {groups}")
    print(f"   Data shape: {data.shape}")

    # Scale to thousands
    scale = 1000.0
    Y = data / scale  # (T, cantons, groups)

    # Merge Psychothérapeutes into Autres (instead of excluding)
    if "Psychothérapeutes" in groups and "Autres" in groups:
        psycho_idx = groups.index("Psychothérapeutes")
        autres_idx = groups.index("Autres")

        # Add Psychothérapeutes to Autres (treating NaN as 0)
        Y[:, :, autres_idx] = np.nansum(
            np.stack([Y[:, :, autres_idx], Y[:, :, psycho_idx]]), axis=0
        )

        # Remove Psychothérapeutes column
        keep_indices = [i for i in range(len(groups)) if i != psycho_idx]
        Y_forecast = Y[:, :, keep_indices]
        groups_forecast = [groups[i] for i in keep_indices]

        print(f"\n🔍 Data processing:")
        print(f"   ✓ Merged 'Psychothérapeutes' into 'Autres' (94.9% missing values)")
    else:
        Y_forecast = Y
        groups_forecast = groups

    print(f"\n🔮 Forecasting configuration:")
    print(f"   Input shape: {Y_forecast.shape}")
    print(f"   Cost groups: {len(groups_forecast)}")
    print(f"   Using MAR specification to forecast all dimensions simultaneously")

    # Forecast using MAR specification (Barigozzi & Trapin 2025)
    config = ForecastConfig(
        k1=2,
        k2=2,
        P=1,
        seasonal_period=4,
        max_iter=100,
        verbose=VERBOSE,
    )

    steps = 9  # Two years ahead
    result = forecast_dmfm(Y_forecast, steps=steps, config=config)

    # Extract forecasts and scale back
    fcst = result.forecast * scale  # (steps, cantons, groups)

    # Generate future periods
    future_periods = generate_future_periods(periods[-1], steps)

    # Check stability and dynamics with NaN protection
    print(f"\n📈 Model diagnostics:")
    try:
        is_stable, max_eval = result.model.dynamics.check_stability()
        print(f"   Stable: {is_stable}, λ_max: {max_eval:.3f}")
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"   Could not check stability: {str(e)[:50]}")
        print(
            f"   A matrices contain NaN: {any(np.isnan(A).any() for A in result.model.dynamics.A)}"
        )
        print(
            f"   B matrices contain NaN: {any(np.isnan(B).any() for B in result.model.dynamics.B)}"
        )

    print(f"   Seasonal adjusted: {result.seasonal_adjusted}")
    print(f"   Forecast contains NaN: {np.isnan(fcst).any()}")
    if not np.isnan(fcst).all():
        print(f"   Forecast range: [{np.nanmin(fcst):.1f}, {np.nanmax(fcst):.1f}]")

    # Aggregate across all forecasted cost groups (MAR specification)
    # Note: Psychothérapeutes was merged into Autres earlier
    data_total = np.sum(
        data[:, :, keep_indices] if "Psychothérapeutes" in groups else data, axis=2
    )  # (T, cantons)
    fcst_total = np.sum(fcst, axis=2)  # (steps, cantons)

    # Check forecast trend for CH
    ch_idx = cantons.index("CH")
    print(f"\n📊 CH forecast diagnostics:")
    print(f"   Last 4 quarters ({periods[-4:]}): {data_total[-4:, ch_idx]}")
    print(f"   First 5 forecast qtrs ({future_periods[:5]}): {fcst_total[:5, ch_idx]}")

    first_forecast_date = fcst_total[0, ch_idx]
    one_year_earlier = data_total[-4, ch_idx]
    one_year_later = fcst_total[4, ch_idx] if len(fcst_total) > 4 else None

    growth_current_year = (
        100 * (first_forecast_date - one_year_earlier) / one_year_earlier
    )
    print(f"   One year earlier, actual: {one_year_earlier:.0f}")
    print(
        f"   First forecast date: {first_forecast_date:.0f} (YoY growth: {growth_current_year:.1f}%)"
    )
    if one_year_later:
        growth_next_year = (
            100 * (one_year_later - first_forecast_date) / first_forecast_date
        )
        print(
            f"   One year later, forecast: {one_year_later:.0f} (YoY growth: {growth_next_year:.1f}%)"
        )
    print(f"   Historical average YoY growth: 3.0%, Recent: 4.4%")

    # Compute yearly totals (using aggregated totals across all cost groups)
    yearly_totals = compute_yearly_totals(
        periods, data_total, future_periods, fcst_total
    )

    # Print forecast totals
    label = f"Total ({len(groups_forecast)} groups, Psycho→Autres)"
    print_yearly_forecasts(yearly_totals, cantons, periods[-1], label)

    # Plot results
    plot_forecasts(
        periods, data_total, future_periods, fcst_total, cantons, cost_group=label
    )

    # Compute CH yearly totals (partial years + forecast)
    ch_yearly_totals = compute_yearly_ch_totals_with_forecast(
        periods,
        data_total,
        future_periods,
        fcst_total,
        cantons,
    )

    # Save CSV
    save_yearly_ch_totals_csv(ch_yearly_totals)

    print(f"\n✅ Forecasting complete!")


def compute_yearly_totals(
    periods: list[str],
    data_total: np.ndarray,
    future_periods: list[str],
    fcst_total: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute yearly totals from historical and forecast data.

    Parameters
    ----------
    periods : list[str]
        Historical period labels.
    data_total : np.ndarray
        Historical data (T, cantons).
    future_periods : list[str]
        Future period labels.
    fcst_total : np.ndarray
        Forecast data (steps, cantons).

    Returns
    -------
    dict
        Yearly totals by year string.
    """
    yearly_totals: dict[str, np.ndarray] = {}
    quarter_counts: dict[str, int] = {}

    # Get last historical period info
    last_period = periods[-1]
    last_year = last_period[:4]
    last_quarter = int(last_period[-1])

    # Accumulate historical quarters for final year
    for period, row in zip(periods, data_total):
        year = period[:4]
        if year == last_year and int(period[-1]) <= last_quarter:
            yearly_totals.setdefault(year, np.zeros(data_total.shape[1]))
            quarter_counts[year] = quarter_counts.get(year, 0) + 1
            yearly_totals[year] += np.nan_to_num(row)

    # Accumulate forecast quarters
    for i, period in enumerate(future_periods):
        year = period[:4]
        yearly_totals.setdefault(year, np.zeros(fcst_total.shape[1]))
        quarter_counts[year] = quarter_counts.get(year, 0) + 1
        yearly_totals[year] += np.nan_to_num(fcst_total[i])

    # Filter to complete years only
    complete_totals = {}
    start_year = int(last_year) if last_quarter < 4 else int(last_year) + 1

    for year in sorted(yearly_totals.keys()):
        if int(year) < start_year:
            continue
        if quarter_counts.get(year, 0) == 4:
            complete_totals[year] = yearly_totals[year]

    return complete_totals


def print_yearly_forecasts(
    yearly_totals: dict[str, np.ndarray],
    cantons: list[str],
    last_period: str,
    cost_group: str,
) -> None:
    """Print yearly forecast totals.

    Parameters
    ----------
    yearly_totals : dict
        Yearly totals by year.
    cantons : list[str]
        Canton names.
    last_period : str
        Last historical period.
    cost_group : str
        Name of the cost group being forecast.
    """
    print(f"\n📅 Yearly forecasts ({cost_group}):")
    for year in sorted(yearly_totals.keys()):
        print(f"\n  {year}:")
        for canton, value in zip(cantons, yearly_totals[year]):
            print(f"    {canton}: {value:,.0f}")


def plot_forecasts(
    periods: list[str],
    data_total: np.ndarray,
    future_periods: list[str],
    fcst_total: np.ndarray,
    cantons: list[str],
    cost_group: str = "Total",
    output_path: str = "forecast_plot.pdf",
) -> None:
    """Plot historical data with forecasts.

    Parameters
    ----------
    periods : list[str]
        Historical period labels.
    data_total : np.ndarray
        Historical data (T, cantons).
    future_periods : list[str]
        Future period labels.
    fcst_total : np.ndarray
        Forecast data (steps, cantons).
    cantons : list[str]
        Canton names.
    cost_group : str, default "Total"
        Name of the cost group.
    output_path : str, default "forecast_plot.pdf"
        Output file path.
    """
    # Combine periods and data
    combined_periods = periods + future_periods
    combined_data = np.vstack([data_total, fcst_total])

    # Select canton to plot (prefer CH, otherwise first)
    if "CH" in cantons:
        idx = cantons.index("CH")
    else:
        idx = 0
    canton_name = cantons[idx]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Determine x-axis positions
    n_hist = len(periods)
    n_fcst = len(future_periods)
    x_hist = np.arange(n_hist)
    x_fcst = np.arange(n_hist, n_hist + n_fcst)

    # Historical data
    ax.plot(
        x_hist,
        data_total[:, idx],
        "o-",
        label=f"{canton_name} (Historical)",
        color="steelblue",
        linewidth=2,
        markersize=4,
    )

    # Forecast data
    ax.plot(
        x_fcst,
        fcst_total[:, idx],
        "s--",
        label=f"{canton_name} (Forecast)",
        color="coral",
        linewidth=2,
        markersize=5,
    )

    # Connect last historical point to first forecast point
    ax.plot(
        [x_hist[-1], x_fcst[0]],
        [data_total[-1, idx], fcst_total[0, idx]],
        linestyle="--",
        color="coral",
        linewidth=2,
        alpha=0.8,
        label="_nolegend_",
    )

    # Add vertical line at forecast start
    ax.axvline(
        x=n_hist - 0.5,
        color="gray",
        linestyle=":",
        alpha=0.5,
        linewidth=1.5,
    )

    # Add text annotation
    ax.text(
        n_hist - 0.5,
        ax.get_ylim()[1] * 0.95,
        "Forecast Start",
        rotation=90,
        verticalalignment="top",
        horizontalalignment="right",
        color="gray",
        fontsize=9,
    )

    # Formatting
    # Show every 4th period label (yearly)
    tick_positions = np.arange(0, len(combined_periods), 4)
    tick_labels = [combined_periods[i] for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.set_xlabel("Period", fontsize=11)
    ax.set_ylabel("Cost (thousands CHF)", fontsize=11)
    ax.set_title(
        f"Health Costs Forecast - {canton_name} ({cost_group})",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Plot saved to {output_path}")


def compute_yearly_ch_totals_with_forecast(
    periods: list[str],
    data_total: np.ndarray,  # (T, cantons)
    future_periods: list[str],
    fcst_total: np.ndarray,  # (steps, cantons)
    cantons: list[str],
) -> dict[str, float]:
    """
    Compute yearly totals for CH, combining observed quarters
    with forecasted quarters if the year is incomplete.
    """
    if "CH" not in cantons:
        raise ValueError("CH not found in cantons")

    ch_idx = cantons.index("CH")

    yearly_sum: dict[str, float] = {}
    yearly_quarters: dict[str, int] = {}

    # Historical data
    for period, row in zip(periods, data_total):
        year = period[:4]
        yearly_sum.setdefault(year, 0.0)
        yearly_quarters[year] = yearly_quarters.get(year, 0) + 1
        yearly_sum[year] += float(np.nan_to_num(row[ch_idx]))

    # Forecast data
    for i, period in enumerate(future_periods):
        year = period[:4]
        yearly_sum.setdefault(year, 0.0)
        yearly_quarters[year] = yearly_quarters.get(year, 0) + 1
        yearly_sum[year] += float(np.nan_to_num(fcst_total[i, ch_idx]))

    # Keep only years with at least 1 observed or forecasted quarter
    return dict(sorted(yearly_sum.items()))


def save_yearly_ch_totals_csv(
    yearly_totals: dict[str, float],
    output_path: str = "ch_total_yearly_forecast.csv",
) -> None:
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "ch_total_cost"])
        for year, value in yearly_totals.items():
            writer.writerow([year, round(value, 2)])

    print(f"\n💾 CH yearly totals saved to {output_path}")


if __name__ == "__main__":
    main()
