"""Forecast Swiss health costs using DMFM.

Includes shock/intervention handling:
- COVID-19 (2020Q2-Q3): Temporary disruption modeled as factor-level shock
- ZG hospital policy (2026Q1+): ZG pays hospital stays directly (costs = 0)
"""

import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from KPOKPCH.forecast import (
    load_cost_matrix,
    generate_future_periods,
    ForecastConfig,
    forecast_dmfm,
)
from KPOKPCH.DMFM import select_rank, print_selection_summary, ShockSchedule

from shocks_config import (
    create_covid_shock,
    apply_zg_policy_to_forecast,
    get_zg_policy_info,
)

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = "health_costs_tensor.csv"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Forecast Swiss health costs using DMFM"
    )
    parser.add_argument(
        "--select-rank",
        action="store_true",
        help="Automatically select optimal k1, k2 using BIC",
    )
    parser.add_argument(
        "--k1-range",
        type=str,
        default="1,2",
        help="Range of k1 values to search (e.g., '1,4')",
    )
    parser.add_argument(
        "--k2-range",
        type=str,
        default="4,8",
        help="Range of k2 values to search (e.g., '1,4')",
    )
    parser.add_argument(
        "--k1",
        type=int,
        default=1,
        help="Number of row factors (default: 1)",
    )
    parser.add_argument(
        "--k2",
        type=int,
        default=4,
        help="Number of column factors (default: 4)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output during fitting",
    )
    return parser.parse_args()


def main():
    """Run health cost forecasting."""
    args = parse_args()

    # Load data
    csv_path = BASE_DIR / INPUT_FILE
    periods, cantons, groups, data = load_cost_matrix(str(csv_path))

    # Remove CH if present (we compute it as aggregate)
    if "CH" in cantons:
        ch_idx = cantons.index("CH")
        cantons = [c for i, c in enumerate(cantons) if i != ch_idx]
        data = np.delete(data, ch_idx, axis=1)

    print(f"Data: {len(periods)} periods, {len(cantons)} cantons, {len(groups)} groups")

    # Scale to thousands
    scale = 1000.0
    Y = data / scale

    # Merge Psychotherapeutes into Autres (mostly missing)
    if "Psychothérapeutes" in groups and "Autres" in groups:
        psycho_idx = groups.index("Psychothérapeutes")
        autres_idx = groups.index("Autres")
        Y[:, :, autres_idx] = np.nansum(
            np.stack([Y[:, :, autres_idx], Y[:, :, psycho_idx]]), axis=0
        )
        keep_indices = [i for i in range(len(groups)) if i != psycho_idx]
        Y = Y[:, :, keep_indices]
        groups = [groups[i] for i in keep_indices]
        data = data[:, :, keep_indices]

    # Determine rank
    verbose = args.verbose

    if args.select_rank:
        k1_min, k1_max = map(int, args.k1_range.split(","))
        k2_min, k2_max = map(int, args.k2_range.split(","))

        print(f"Selecting rank (k1: [{k1_min},{k1_max}], k2: [{k2_min},{k2_max}])...")

        Y_diff = np.diff(Y, n=4, axis=0)
        mask_diff = ~np.isnan(Y_diff)

        selection_result = select_rank(
            Y_diff,
            k1_range=(k1_min, k1_max),
            k2_range=(k2_min, k2_max),
            P=1,
            criterion="bic",
            mask=mask_diff,
            diagonal_idiosyncratic=True,
            max_iter=50,
            verbose=verbose,
        )

        if verbose:
            print_selection_summary(selection_result)

        k1, k2 = selection_result.best_k1, selection_result.best_k2
        print(f"Selected: k1={k1}, k2={k2}")
    else:
        k1, k2 = args.k1, args.k2
        print(f"Using: k1={k1}, k2={k2}")

    # Create COVID shock schedule (data includes 2020Q2-Q3)
    covid_shock = create_covid_shock()
    shock_schedule = ShockSchedule([covid_shock])
    print(
        f"Including shock: {covid_shock.name} (t={covid_shock.start_t}-{covid_shock.end_t})"
    )

    # Forecast with shock handling
    config = ForecastConfig(
        k1=k1,
        k2=k2,
        P=1,
        seasonal_period=4,
        max_iter=100,
        verbose=verbose,
        shock_schedule=shock_schedule,
    )

    steps = 4  # One year ahead
    result = forecast_dmfm(Y, steps=steps, config=config)

    # Report shock effects if estimated
    if result.shock_effects is not None:
        if result.shock_effects.factor_effects is not None:
            print(
                f"Estimated COVID factor effect magnitude: "
                f"{np.linalg.norm(result.shock_effects.factor_effects[0]):.4f}"
            )

    fcst = result.forecast * scale
    future_periods = generate_future_periods(periods[-1], steps)

    # Apply ZG hospital policy: ZG séjours = 0 for 2026 periods
    zg_policy = get_zg_policy_info()
    fcst = apply_zg_policy_to_forecast(fcst, future_periods[0], cantons, groups)
    print(
        f"Applied policy: {zg_policy['name']} ({zg_policy['start_period']}-{zg_policy['end_period']})"
    )

    # Add CH as aggregate (27th canton)
    data_ch = np.nansum(data, axis=1, keepdims=True)
    fcst_ch = np.sum(fcst, axis=1, keepdims=True)
    data_with_ch = np.concatenate([data, data_ch], axis=1)
    fcst_with_ch = np.concatenate([fcst, fcst_ch], axis=1)
    cantons_with_ch = cantons + ["CH"]

    # Table 1: Canton x Group (detailed)
    save_canton_group_table(
        periods, data_with_ch, future_periods, fcst_with_ch, cantons_with_ch, groups
    )

    # Aggregate over groups for canton totals
    data_total = np.nansum(data_with_ch, axis=2)
    fcst_total = np.sum(fcst_with_ch, axis=2)

    # Table 2: Canton totals
    save_canton_total_table(
        periods, data_total, future_periods, fcst_total, cantons_with_ch
    )

    # Table 3: YoY growth rates
    save_yoy_growth_table(
        periods, data_total, future_periods, fcst_total, cantons_with_ch
    )

    # Plot CH forecast
    plot_ch_forecast(periods, data_total, future_periods, fcst_total, cantons_with_ch)

    print("Done.")


def save_canton_group_table(
    periods: list[str],
    data: np.ndarray,
    future_periods: list[str],
    fcst: np.ndarray,
    cantons: list[str],
    groups: list[str],
) -> None:
    """Save detailed canton x group table with estimates and forecasts."""
    output_path = BASE_DIR / "output_canton_group.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        header = ["period", "type", "canton", "group", "value"]
        writer.writerow(header)

        # Historical data
        for t, period in enumerate(periods):
            for i, canton in enumerate(cantons):
                for j, group in enumerate(groups):
                    val = data[t, i, j]
                    if not np.isnan(val):
                        writer.writerow(
                            [period, "estimate", canton, group, f"{val:.2f}"]
                        )

        # Forecasts
        for t, period in enumerate(future_periods):
            for i, canton in enumerate(cantons):
                for j, group in enumerate(groups):
                    val = fcst[t, i, j]
                    writer.writerow([period, "forecast", canton, group, f"{val:.2f}"])

    print(f"Saved: {output_path.name}")


def save_canton_total_table(
    periods: list[str],
    data_total: np.ndarray,
    future_periods: list[str],
    fcst_total: np.ndarray,
    cantons: list[str],
) -> None:
    """Save canton totals (aggregated over groups)."""
    output_path = BASE_DIR / "output_canton_total.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header: period, type, then cantons
        header = ["period", "type"] + cantons
        writer.writerow(header)

        # Historical
        for t, period in enumerate(periods):
            row = [period, "estimate"] + [
                f"{data_total[t, i]:.2f}" for i in range(len(cantons))
            ]
            writer.writerow(row)

        # Forecasts
        for t, period in enumerate(future_periods):
            row = [period, "forecast"] + [
                f"{fcst_total[t, i]:.2f}" for i in range(len(cantons))
            ]
            writer.writerow(row)

    print(f"Saved: {output_path.name}")


def save_yoy_growth_table(
    periods: list[str],
    data_total: np.ndarray,
    future_periods: list[str],
    fcst_total: np.ndarray,
    cantons: list[str],
) -> None:
    """Save YoY growth rates per quarter per canton."""
    output_path = BASE_DIR / "output_yoy_growth.csv"

    # Combine all periods and values
    all_periods = periods + future_periods
    all_values = np.vstack([data_total, fcst_total])
    n_hist = len(periods)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        header = ["period", "type"] + cantons
        writer.writerow(header)

        # YoY requires 4 quarters lag
        for t in range(4, len(all_periods)):
            period = all_periods[t]
            period_type = "estimate" if t < n_hist else "forecast"

            growth_rates = []
            for i in range(len(cantons)):
                prev_val = all_values[t - 4, i]
                curr_val = all_values[t, i]
                if prev_val > 0 and not np.isnan(prev_val) and not np.isnan(curr_val):
                    growth = 100 * (curr_val - prev_val) / prev_val
                    growth_rates.append(f"{growth:.2f}")
                else:
                    growth_rates.append("")

            writer.writerow([period, period_type] + growth_rates)

    print(f"Saved: {output_path.name}")


def plot_ch_forecast(
    periods: list[str],
    data_total: np.ndarray,
    future_periods: list[str],
    fcst_total: np.ndarray,
    cantons: list[str],
) -> None:
    """Plot CH forecast."""
    output_path = BASE_DIR / "forecast_ch.pdf"

    ch_idx = cantons.index("CH")

    fig, ax = plt.subplots(figsize=(12, 6))

    n_hist = len(periods)
    x_hist = np.arange(n_hist)
    x_fcst = np.arange(n_hist, n_hist + len(future_periods))

    # Historical
    ax.plot(
        x_hist,
        data_total[:, ch_idx],
        "o-",
        label="Historical",
        color="steelblue",
        linewidth=2,
        markersize=4,
    )

    # Forecast
    ax.plot(
        x_fcst,
        fcst_total[:, ch_idx],
        "s--",
        label="Forecast",
        color="coral",
        linewidth=2,
        markersize=5,
    )

    # Connect
    ax.plot(
        [x_hist[-1], x_fcst[0]],
        [data_total[-1, ch_idx], fcst_total[0, ch_idx]],
        "--",
        color="coral",
        linewidth=2,
        alpha=0.8,
    )

    # Forecast start line
    ax.axvline(x=n_hist - 0.5, color="gray", linestyle=":", alpha=0.5)

    # X-axis labels (every 4th = yearly)
    all_periods = periods + future_periods
    tick_positions = np.arange(0, len(all_periods), 4)
    tick_labels = [all_periods[i] for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.set_xlabel("Period")
    ax.set_ylabel("Cost per insured (CHF)")
    ax.set_title("Switzerland Health Costs Forecast")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path.name}")


if __name__ == "__main__":
    main()
