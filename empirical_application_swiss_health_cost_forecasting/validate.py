"""Validate Swiss health cost DMFM forecast using out-of-sample methods.

This script performs comprehensive validation of the DMFM model using
rolling window validation with automatic rank selection at each window.

Uses annualized data (rolling 4-quarter sums) for stability.
"""

import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from KPOKPCH.forecast import (
    load_cost_matrix,
    ForecastConfig,
)
from KPOKPCH.forecast.validation import (
    ValidationConfig,
    ValidationResult,
    out_of_sample_validate,
    rolling_window_validate,
    average_validation_results,
)
from KPOKPCH.DMFM import select_rank

from shocks_config import create_intervention_schedule

# Intervention schedule for deterministic policy changes (e.g., ZG 2026)
INTERVENTION_SCHEDULE = create_intervention_schedule()


def annualize(Y: np.ndarray, window: int = 4) -> np.ndarray:
    """Compute rolling sum over trailing window (annualization).

    Parameters
    ----------
    Y : np.ndarray
        Data of shape (T, ...).
    window : int
        Rolling window size (default 4 for quarterly -> annual).

    Returns
    -------
    np.ndarray
        Rolling sums of shape (T - window + 1, ...).
    """
    T = Y.shape[0]
    result = np.zeros((T - window + 1,) + Y.shape[1:])
    for t in range(T - window + 1):
        result[t] = np.nansum(Y[t : t + window], axis=0)
    return result


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = "health_costs_tensor.csv"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate Swiss health cost DMFM forecast"
    )
    parser.add_argument(
        "--k1-range",
        type=str,
        default="1,2",
        help="Range of k1 values to search (e.g., '1,3')",
    )
    parser.add_argument(
        "--k2-range",
        type=str,
        default="1,4",
        help="Range of k2 values to search (e.g., '1,8')",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Forecast horizon in quarters (default: 4)",
    )
    parser.add_argument(
        "--min-train",
        type=int,
        default=20,
        help="Minimum training size for rolling validation (default: 20)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=28,
        help="Window size for rolling validation (default: 28)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output during fitting",
    )
    return parser.parse_args()


def apply_interventions_to_results(
    results: list[ValidationResult],
    min_train_size: int,
    window_type: str = "expanding",
    window_size: int | None = None,
) -> list[ValidationResult]:
    """Apply intervention schedule to validation forecasts.

    Modifies forecasts in-place to reflect deterministic policy changes
    (e.g., ZG hospital policy setting costs to 0 in 2026).

    Parameters
    ----------
    results : list[ValidationResult]
        Validation results with forecasts to modify.
    min_train_size : int
        Minimum training size used in validation.
    window_type : str
        "expanding" or "rolling".
    window_size : int, optional
        Window size for rolling validation.

    Returns
    -------
    list[ValidationResult]
        Same results with interventions applied to forecasts.
    """
    if len(INTERVENTION_SCHEDULE) == 0:
        return results

    for i, result in enumerate(results):
        # Reconstruct train_end for this result
        if window_type == "expanding":
            train_end = min_train_size + i
        else:  # rolling
            train_end = (window_size or min_train_size) + i

        steps = result.forecasts.shape[0]

        # Apply interventions to each forecast step
        for h in range(steps):
            t = train_end + h
            result.forecasts[h] = INTERVENTION_SCHEDULE.apply(result.forecasts[h], t)

        # Recompute errors after intervention
        result.errors[:] = result.forecasts - result.actuals

    return results


def load_and_preprocess_data():
    """Load and preprocess health cost data with annualization."""
    csv_path = BASE_DIR / INPUT_FILE
    periods, cantons, groups, data = load_cost_matrix(str(csv_path))

    # Remove CH if present (we compute it as aggregate)
    if "CH" in cantons:
        ch_idx = cantons.index("CH")
        cantons = [c for i, c in enumerate(cantons) if i != ch_idx]
        data = np.delete(data, ch_idx, axis=1)

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

    # Annualize: rolling 4-quarter sums
    Y_annual = annualize(Y, window=4)
    # Period labels: use the end quarter of each window
    periods_annual = periods[3:]

    return periods_annual, cantons, groups, Y_annual, scale


def make_config_selector(
    k1_range: tuple[int, int],
    k2_range: tuple[int, int],
    verbose: bool = False,
):
    """Create a config selector function for dynamic rank selection.

    Returns a callable that selects optimal k1, k2 using BIC on first
    differences of the (already annualized) training data.

    Parameters
    ----------
    k1_range : tuple[int, int]
        Range of k1 values to search (min, max).
    k2_range : tuple[int, int]
        Range of k2 values to search (min, max).
    verbose : bool, default False
        Print selection details.

    Returns
    -------
    callable
        Function with signature (Y_train, mask_train) -> ForecastConfig
    """

    def config_selector(
        Y_train: np.ndarray, mask_train: np.ndarray | None
    ) -> ForecastConfig:
        training_end_t = Y_train.shape[0]

        # Need at least 2 observations for first differences
        if training_end_t <= 2:
            return ForecastConfig(
                k1=k1_range[0],
                k2=k2_range[0],
                P=1,
                i1_factors=True,  # Factors follow random walk (for trending data)
                max_iter=100,
                verbose=verbose,
            )

        # First differences of annualized data for rank selection
        Y_diff = np.diff(Y_train, n=1, axis=0)
        mask_diff = ~np.isnan(Y_diff)

        # Select rank using BIC
        result = select_rank(
            Y_diff,
            k1_range=k1_range,
            k2_range=k2_range,
            P=1,
            criterion="bic",
            mask=mask_diff,
            diagonal_idiosyncratic=True,
            max_iter=50,
            verbose=False,
        )

        if verbose:
            print(f"    Selected k1={result.best_k1}, k2={result.best_k2}")

        return ForecastConfig(
            k1=result.best_k1,
            k2=result.best_k2,
            P=1,
            i1_factors=True,  # Factors follow random walk (for trending data)
            max_iter=100,
            verbose=verbose,
        )

    return config_selector


def plot_rolling_window_metrics(
    results: list[ValidationResult],
    periods: list[str],
    min_train_size: int,
    output_path: Path,
) -> None:
    """Plot error metrics over rolling validation windows.

    Creates a multi-panel figure showing RMSE, MAE, MAPE, and Bias
    evolution across validation windows.
    """
    n_results = len(results)

    # Extract metrics
    rmses = [r.rmse for r in results]
    maes = [r.mae for r in results]
    mapes = [r.mape for r in results]
    biases = [r.bias for r in results]

    # X-axis: validation window end points (training end period)
    window_ends = list(range(min_train_size, min_train_size + n_results))
    # Map to period labels (every window ends at a different point)
    x_labels = [
        periods[i] if i < len(periods) else f"t+{i-len(periods)+1}" for i in window_ends
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # RMSE
    ax = axes[0, 0]
    ax.plot(window_ends, rmses, "o-", color="steelblue", linewidth=2, markersize=5)
    ax.axhline(
        np.mean(rmses),
        color="coral",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {np.mean(rmses):.3f}",
    )
    ax.fill_between(
        window_ends,
        np.mean(rmses) - np.std(rmses),
        np.mean(rmses) + np.std(rmses),
        alpha=0.2,
        color="coral",
    )
    ax.set_ylabel("RMSE")
    ax.set_title("Root Mean Squared Error")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")

    # MAE
    ax = axes[0, 1]
    ax.plot(window_ends, maes, "s-", color="seagreen", linewidth=2, markersize=5)
    ax.axhline(
        np.mean(maes),
        color="coral",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {np.mean(maes):.3f}",
    )
    ax.fill_between(
        window_ends,
        np.mean(maes) - np.std(maes),
        np.mean(maes) + np.std(maes),
        alpha=0.2,
        color="coral",
    )
    ax.set_ylabel("MAE")
    ax.set_title("Mean Absolute Error")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")

    # MAPE
    ax = axes[1, 0]
    valid_mapes = [m for m in mapes if not np.isnan(m)]
    if valid_mapes:
        ax.plot(
            window_ends[: len(valid_mapes)],
            valid_mapes,
            "^-",
            color="darkorange",
            linewidth=2,
            markersize=5,
        )
        ax.axhline(
            np.mean(valid_mapes),
            color="coral",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {np.mean(valid_mapes):.2f}%",
        )
        ax.fill_between(
            window_ends[: len(valid_mapes)],
            np.mean(valid_mapes) - np.std(valid_mapes),
            np.mean(valid_mapes) + np.std(valid_mapes),
            alpha=0.2,
            color="coral",
        )
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Mean Absolute Percentage Error")
    ax.set_xlabel("Training window end (observation index)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Bias
    ax = axes[1, 1]
    ax.plot(window_ends, biases, "d-", color="purple", linewidth=2, markersize=5)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.axhline(
        np.mean(biases),
        color="coral",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {np.mean(biases):.3f}",
    )
    ax.fill_between(
        window_ends,
        np.mean(biases) - np.std(biases),
        np.mean(biases) + np.std(biases),
        alpha=0.2,
        color="coral",
    )
    ax.set_ylabel("Bias")
    ax.set_title("Forecast Bias (+ = over-forecast)")
    ax.set_xlabel("Training window end (observation index)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.suptitle(
        "Rolling Window Validation: Error Metrics Over Time",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path.name}")


def plot_forecast_fan(
    results: list[ValidationResult],
    Y: np.ndarray,
    periods: list[str],
    min_train_size: int,
    steps: int,
    scale: float,
    output_path: Path,
) -> None:
    """Plot forecast fan chart showing all rolling window forecasts.

    Creates a visualization showing the actual series with all
    out-of-sample forecasts overlaid, creating a "fan" of predictions.
    """
    # Aggregate to Switzerland total
    Y_ch = np.nansum(Y, axis=(1, 2)) * scale
    T = len(Y_ch)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot actual series
    ax.plot(
        range(T),
        Y_ch,
        "o-",
        color="steelblue",
        linewidth=2,
        markersize=4,
        label="Actual",
        zorder=10,
    )

    # Plot each forecast
    n_results = len(results)
    colors = plt.cm.Oranges(np.linspace(0.3, 0.9, n_results))

    for i, result in enumerate(results):
        train_end = min_train_size + i
        fcst_start = train_end
        fcst_end = fcst_start + steps

        # Aggregate forecast to CH total
        fcst_ch = np.sum(result.forecasts, axis=(1, 2)) * scale
        x_fcst = range(fcst_start, fcst_end)

        alpha = 0.4 + 0.4 * (i / n_results)  # Fade older forecasts

        # Connect last actual to first forecast
        ax.plot(
            [train_end - 1, fcst_start],
            [Y_ch[train_end - 1], fcst_ch[0]],
            "-",
            color=colors[i],
            linewidth=1.5,
            alpha=alpha,
        )
        # Plot forecast
        ax.plot(x_fcst, fcst_ch, "-", color=colors[i], linewidth=1.5, alpha=alpha)

    # Add a representative legend entry for forecasts
    ax.plot(
        [],
        [],
        "-",
        color="coral",
        linewidth=2,
        label=f"Rolling forecasts (n={n_results})",
    )

    # X-axis labels (every 4th = yearly)
    tick_positions = np.arange(0, T, 4)
    tick_labels = [periods[i] for i in tick_positions if i < len(periods)]
    ax.set_xticks(tick_positions[: len(tick_labels)])
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.set_xlabel("Period")
    ax.set_ylabel("Total Cost (CHF)")
    ax.set_title("Rolling Window Forecasts: Switzerland Total Health Costs")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path.name}")


def plot_error_by_horizon(
    results: list[ValidationResult],
    steps: int,
    scale: float,
    output_path: Path,
) -> None:
    """Plot error metrics by forecast horizon.

    Shows how forecast accuracy degrades with increasing horizon.
    """
    # Collect errors by horizon step
    errors_by_step = [[] for _ in range(steps)]

    for result in results:
        # Aggregate to total (sum over cantons and groups)
        fcst_total = np.sum(result.forecasts, axis=(1, 2)) * scale
        actual_total = np.sum(result.actuals, axis=(1, 2)) * scale

        for h in range(steps):
            if h < len(fcst_total):
                err = fcst_total[h] - actual_total[h]
                errors_by_step[h].append(err)

    # Compute statistics by horizon
    horizons = list(range(1, steps + 1))
    mean_abs_errors = [np.mean(np.abs(errors_by_step[h])) for h in range(steps)]
    std_errors = [np.std(errors_by_step[h]) for h in range(steps)]
    mean_errors = [np.mean(errors_by_step[h]) for h in range(steps)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Mean Absolute Error by horizon
    ax = axes[0]
    bars = ax.bar(
        horizons, mean_abs_errors, color="steelblue", edgecolor="navy", alpha=0.7
    )
    ax.errorbar(
        horizons, mean_abs_errors, yerr=std_errors, fmt="none", color="black", capsize=5
    )
    ax.set_xlabel("Forecast Horizon (quarters)")
    ax.set_ylabel("Mean Absolute Error (CHF)")
    ax.set_title("Forecast Error by Horizon")
    ax.set_xticks(horizons)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, mean_abs_errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height()
            + std_errors[horizons.index(int(bar.get_x() + bar.get_width() / 2))],
            f"{val:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Bias by horizon
    ax = axes[1]
    colors = ["coral" if m > 0 else "seagreen" for m in mean_errors]
    bars = ax.bar(horizons, mean_errors, color=colors, edgecolor="black", alpha=0.7)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Forecast Horizon (quarters)")
    ax.set_ylabel("Mean Error / Bias (CHF)")
    ax.set_title("Forecast Bias by Horizon (+ = over-forecast)")
    ax.set_xticks(horizons)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    # Add value labels
    for bar, val in zip(bars, mean_errors):
        offset = 50 if val >= 0 else -50
        va = "bottom" if val >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + offset,
            f"{val:,.0f}",
            ha="center",
            va=va,
            fontsize=9,
        )

    plt.suptitle("Error Analysis by Forecast Horizon", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path.name}")


def plot_actual_vs_forecast(
    results: list[ValidationResult],
    scale: float,
    output_path: Path,
) -> None:
    """Plot actual vs forecast scatter with 45-degree line.

    Shows correlation between forecasts and actuals across all windows.
    """
    all_forecasts = []
    all_actuals = []

    for result in results:
        # Aggregate to total
        fcst_total = np.sum(result.forecasts, axis=(1, 2)) * scale
        actual_total = np.sum(result.actuals, axis=(1, 2)) * scale

        all_forecasts.extend(fcst_total.flatten())
        all_actuals.extend(actual_total.flatten())

    all_forecasts = np.array(all_forecasts)
    all_actuals = np.array(all_actuals)

    # Compute correlation
    corr = np.corrcoef(all_forecasts, all_actuals)[0, 1]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        all_actuals, all_forecasts, alpha=0.5, color="steelblue", edgecolor="navy", s=50
    )

    # 45-degree line
    lims = [
        min(all_actuals.min(), all_forecasts.min()),
        max(all_actuals.max(), all_forecasts.max()),
    ]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, "k--", linewidth=1.5, label="Perfect forecast")

    # Regression line
    z = np.polyfit(all_actuals, all_forecasts, 1)
    p = np.poly1d(z)
    ax.plot(
        lims,
        p(lims),
        "r-",
        linewidth=1.5,
        alpha=0.7,
        label=f"Fit: y = {z[0]:.3f}x + {z[1]:,.0f}",
    )

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual (CHF)")
    ax.set_ylabel("Forecast (CHF)")
    ax.set_title(f"Actual vs Forecast (Correlation: {corr:.3f})")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path.name}")


def plot_rank_selection(
    results: list[ValidationResult],
    min_train_size: int,
    output_path: Path,
) -> None:
    """Plot selected k1 and k2 over validation windows.

    Shows how the BIC-selected ranks vary across windows.
    """
    # Extract k1, k2 values
    k1_vals = []
    k2_vals = []
    for r in results:
        if r.forecast_config:
            k1_vals.append(r.forecast_config.k1)
            k2_vals.append(r.forecast_config.k2)
        else:
            k1_vals.append(np.nan)
            k2_vals.append(np.nan)

    if not any(~np.isnan(k1_vals)):
        return  # No config data

    n_results = len(results)
    window_ends = list(range(min_train_size, min_train_size + n_results))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # k1 over time
    ax = axes[0]
    ax.plot(window_ends, k1_vals, "o-", color="steelblue", linewidth=2, markersize=6)
    ax.set_ylabel("k1 (row factors)")
    ax.set_xlabel("Training window end (observation index)")
    ax.set_title("Selected k1 Over Validation Windows")
    ax.set_yticks(range(int(np.nanmin(k1_vals)), int(np.nanmax(k1_vals)) + 1))
    ax.grid(True, alpha=0.3, linestyle="--")

    # k2 over time
    ax = axes[1]
    ax.plot(window_ends, k2_vals, "s-", color="coral", linewidth=2, markersize=6)
    ax.set_ylabel("k2 (column factors)")
    ax.set_xlabel("Training window end (observation index)")
    ax.set_title("Selected k2 Over Validation Windows")
    ax.set_yticks(range(int(np.nanmin(k2_vals)), int(np.nanmax(k2_vals)) + 1))
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.suptitle("BIC Rank Selection Over Time", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path.name}")


def save_validation_results(
    results: list[ValidationResult],
    avg_metrics: dict,
    periods: list[str],
    min_train_size: int,
    output_path: Path,
) -> None:
    """Save detailed validation results to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header (include k1, k2 if available)
        has_config = any(r.forecast_config for r in results)
        if has_config:
            writer.writerow(
                [
                    "window",
                    "train_end_period",
                    "k1",
                    "k2",
                    "rmse",
                    "mae",
                    "mape",
                    "bias",
                ]
            )
        else:
            writer.writerow(
                ["window", "train_end_period", "rmse", "mae", "mape", "bias"]
            )

        # Individual results
        for i, result in enumerate(results):
            train_end_idx = min_train_size + i
            period = (
                periods[train_end_idx]
                if train_end_idx < len(periods)
                else f"t+{train_end_idx - len(periods) + 1}"
            )
            row = [i + 1, period]
            if has_config:
                if result.forecast_config:
                    row.extend([result.forecast_config.k1, result.forecast_config.k2])
                else:
                    row.extend(["", ""])
            row.extend(
                [
                    f"{result.rmse:.6f}",
                    f"{result.mae:.6f}",
                    f"{result.mape:.4f}" if not np.isnan(result.mape) else "",
                    f"{result.bias:.6f}",
                ]
            )
            writer.writerow(row)

        # Summary
        writer.writerow([])
        writer.writerow(["SUMMARY"])
        writer.writerow(["Metric", "Mean", "Std Dev"])
        writer.writerow(
            ["RMSE", f"{avg_metrics['rmse']:.6f}", f"{avg_metrics['rmse_std']:.6f}"]
        )
        writer.writerow(
            ["MAE", f"{avg_metrics['mae']:.6f}", f"{avg_metrics['mae_std']:.6f}"]
        )
        writer.writerow(
            ["MAPE (%)", f"{avg_metrics['mape']:.4f}", f"{avg_metrics['mape_std']:.4f}"]
        )
        writer.writerow(
            ["Bias", f"{avg_metrics['bias']:.6f}", f"{avg_metrics['bias_std']:.6f}"]
        )
        writer.writerow(["N Windows", avg_metrics["n_windows"]])

        # Rank selection summary
        if has_config:
            k1_vals = [r.forecast_config.k1 for r in results if r.forecast_config]
            k2_vals = [r.forecast_config.k2 for r in results if r.forecast_config]
            if k1_vals:
                writer.writerow([])
                writer.writerow(["RANK SELECTION"])
                writer.writerow(["k1 mode", max(set(k1_vals), key=k1_vals.count)])
                writer.writerow(["k2 mode", max(set(k2_vals), key=k2_vals.count)])

    print(f"Saved: {output_path.name}")


def main():
    """Run comprehensive validation."""
    args = parse_args()

    # Parse k1/k2 ranges
    k1_min, k1_max = map(int, args.k1_range.split(","))
    k2_min, k2_max = map(int, args.k2_range.split(","))
    k1_range = (k1_min, k1_max)
    k2_range = (k2_min, k2_max)

    # Load data (already annualized)
    print("Loading data...")
    periods, cantons, groups, Y, scale = load_and_preprocess_data()
    print(
        f"Annualized data: {len(periods)} periods, {len(cantons)} cantons, {len(groups)} groups"
    )

    # Create config selector for dynamic rank selection
    config_selector = make_config_selector(
        k1_range=k1_range,
        k2_range=k2_range,
        verbose=args.verbose,
    )
    print(f"Rank selection: k1 in [{k1_min},{k1_max}], k2 in [{k2_min},{k2_max}]")

    # =========================================================================
    # 1. Single hold-out validation
    # =========================================================================
    print("\n" + "=" * 60)
    print("1. SINGLE HOLD-OUT VALIDATION")
    print("=" * 60)

    holdout_config = ValidationConfig(steps=args.steps)
    print("Selecting rank for hold-out window...")
    holdout_result = out_of_sample_validate(
        Y, holdout_config, config_selector=config_selector
    )

    # Apply interventions to holdout forecast
    T = Y.shape[0]
    train_end = T - args.steps
    for h in range(args.steps):
        t = train_end + h
        holdout_result.forecasts[h] = INTERVENTION_SCHEDULE.apply(
            holdout_result.forecasts[h], t
        )
    holdout_result.errors[:] = holdout_result.forecasts - holdout_result.actuals

    print(f"Hold-out validation ({args.steps} steps):")
    if holdout_result.forecast_config:
        print(
            f"  Selected: k1={holdout_result.forecast_config.k1}, k2={holdout_result.forecast_config.k2}"
        )
    print(f"  RMSE: {holdout_result.rmse:.4f}")
    print(f"  MAE:  {holdout_result.mae:.4f}")
    print(f"  MAPE: {holdout_result.mape:.2f}%")
    print(f"  Bias: {holdout_result.bias:.4f}")

    # =========================================================================
    # 2. Expanding window validation
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. EXPANDING WINDOW VALIDATION")
    print("=" * 60)

    expanding_config = ValidationConfig(
        steps=args.steps,
        window_type="expanding",
        min_train_size=args.min_train,
    )

    print(
        f"Running expanding window validation (min_train={args.min_train}, steps={args.steps})..."
    )
    expanding_results = rolling_window_validate(
        Y, expanding_config, config_selector=config_selector
    )
    # Apply interventions to expanding window forecasts
    apply_interventions_to_results(
        expanding_results, args.min_train, window_type="expanding"
    )
    expanding_avg = average_validation_results(expanding_results)

    # Report selected ranks
    k1_vals = [r.forecast_config.k1 for r in expanding_results if r.forecast_config]
    k2_vals = [r.forecast_config.k2 for r in expanding_results if r.forecast_config]
    if k1_vals:
        print(
            f"  k1 selected: min={min(k1_vals)}, max={max(k1_vals)}, mode={max(set(k1_vals), key=k1_vals.count)}"
        )
        print(
            f"  k2 selected: min={min(k2_vals)}, max={max(k2_vals)}, mode={max(set(k2_vals), key=k2_vals.count)}"
        )

    print(f"Expanding window results ({expanding_avg['n_windows']} windows):")
    print(f"  RMSE: {expanding_avg['rmse']:.4f} +/- {expanding_avg['rmse_std']:.4f}")
    print(f"  MAE:  {expanding_avg['mae']:.4f} +/- {expanding_avg['mae_std']:.4f}")
    print(f"  MAPE: {expanding_avg['mape']:.2f}% +/- {expanding_avg['mape_std']:.2f}%")
    print(f"  Bias: {expanding_avg['bias']:.4f} +/- {expanding_avg['bias_std']:.4f}")

    # =========================================================================
    # 3. Rolling window validation
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. ROLLING WINDOW VALIDATION")
    print("=" * 60)

    # Adjust window size if too large for available data
    T = Y.shape[0]
    max_window_size = T - args.steps - 1  # Need at least 1 window
    window_size = min(args.window_size, max_window_size)
    if window_size < args.window_size:
        print(
            f"Note: Adjusted window_size from {args.window_size} to {window_size} (data has {T} periods)"
        )

    rolling_config = ValidationConfig(
        steps=args.steps,
        window_type="rolling",
        window_size=window_size,
        min_train_size=args.min_train,
    )

    print(
        f"Running rolling window validation (window_size={window_size}, steps={args.steps})..."
    )
    rolling_results = rolling_window_validate(
        Y, rolling_config, config_selector=config_selector
    )
    # Apply interventions to rolling window forecasts
    apply_interventions_to_results(
        rolling_results, args.min_train, window_type="rolling", window_size=window_size
    )
    rolling_avg = average_validation_results(rolling_results)

    # Report selected ranks
    k1_vals = [r.forecast_config.k1 for r in rolling_results if r.forecast_config]
    k2_vals = [r.forecast_config.k2 for r in rolling_results if r.forecast_config]
    if k1_vals:
        print(
            f"  k1 selected: min={min(k1_vals)}, max={max(k1_vals)}, mode={max(set(k1_vals), key=k1_vals.count)}"
        )
        print(
            f"  k2 selected: min={min(k2_vals)}, max={max(k2_vals)}, mode={max(set(k2_vals), key=k2_vals.count)}"
        )

    print(f"Rolling window results ({rolling_avg['n_windows']} windows):")
    print(f"  RMSE: {rolling_avg['rmse']:.4f} +/- {rolling_avg['rmse_std']:.4f}")
    print(f"  MAE:  {rolling_avg['mae']:.4f} +/- {rolling_avg['mae_std']:.4f}")
    print(f"  MAPE: {rolling_avg['mape']:.2f}% +/- {rolling_avg['mape_std']:.2f}%")
    print(f"  Bias: {rolling_avg['bias']:.4f} +/- {rolling_avg['bias_std']:.4f}")

    # =========================================================================
    # 4. Generate visualizations (using expanding window results)
    # =========================================================================
    print("\n" + "=" * 60)
    print("4. GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Use expanding window results for main visualizations
    main_results = expanding_results
    main_avg = expanding_avg
    min_train = args.min_train

    # Plot 1: Error metrics over time
    plot_rolling_window_metrics(
        main_results,
        periods,
        min_train,
        BASE_DIR / "validation_metrics.pdf",
    )

    # Plot 2: Forecast fan chart
    plot_forecast_fan(
        main_results,
        Y,
        periods,
        min_train,
        args.steps,
        scale,
        BASE_DIR / "validation_forecast_fan.pdf",
    )

    # Plot 3: Error by horizon
    plot_error_by_horizon(
        main_results,
        args.steps,
        scale,
        BASE_DIR / "validation_error_by_horizon.pdf",
    )

    # Plot 4: Actual vs forecast scatter
    plot_actual_vs_forecast(
        main_results,
        scale,
        BASE_DIR / "validation_actual_vs_forecast.pdf",
    )

    # Plot 5: Rank selection over time
    plot_rank_selection(
        main_results,
        min_train,
        BASE_DIR / "validation_rank_selection.pdf",
    )

    # Save detailed results
    save_validation_results(
        main_results,
        main_avg,
        periods,
        min_train,
        BASE_DIR / "validation_results.csv",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
