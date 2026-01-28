"""Analyze trend in health costs to inform forecasting."""

import numpy as np
import pandas as pd
from pathlib import Path
from KPOKPCH.forecast import load_cost_matrix

def main():
    """Analyze trend in health costs."""
    # Load data
    csv_path = Path(__file__).resolve().parent / "health_costs_tensor.csv"
    periods, cantons, groups, data = load_cost_matrix(str(csv_path))

    # Exclude Psychothérapeutes (94.9% missing)
    missing_by_group = [np.isnan(data[:, :, j]).sum() / data[:, :, j].size for j in range(len(groups))]
    valid_groups = [j for j, miss_rate in enumerate(missing_by_group) if miss_rate < 0.5]

    # Aggregate across valid cost groups
    data_total = np.sum(data[:, :, valid_groups], axis=2)  # (T, cantons)

    # Focus on CH (all Switzerland)
    ch_idx = cantons.index("CH")
    ch_total = data_total[:, ch_idx]

    print(f"Health Cost Trend Analysis (CH Total, excl. Psychothérapeutes)\n")
    print(f"Period              Cost (000s)   YoY Growth   4Q Growth")
    print(f"-" * 60)

    for t in range(len(periods)):
        cost = ch_total[t]

        # Year-over-year growth (same quarter, previous year)
        yoy_growth = ""
        if t >= 4:
            prev_year = ch_total[t-4]
            if not np.isnan(prev_year) and prev_year > 0:
                yoy_pct = 100 * (cost - prev_year) / prev_year
                yoy_growth = f"{yoy_pct:+6.2f}%"

        # Quarter-over-quarter growth
        qoq_growth = ""
        if t >= 1:
            prev_q = ch_total[t-1]
            if not np.isnan(prev_q) and prev_q > 0:
                qoq_pct = 100 * (cost - prev_q) / prev_q
                qoq_growth = f"{qoq_pct:+6.2f}%"

        print(f"{periods[t]:10s}      {cost:10,.0f}   {yoy_growth:>10s}   {qoq_growth:>10s}")

    # Calculate average annual growth rate
    print(f"\nGrowth Statistics:")

    # YoY growth rates (excluding first year)
    yoy_rates = []
    for t in range(4, len(periods)):
        if not np.isnan(ch_total[t]) and not np.isnan(ch_total[t-4]) and ch_total[t-4] > 0:
            yoy_pct = 100 * (ch_total[t] - ch_total[t-4]) / ch_total[t-4]
            yoy_rates.append(yoy_pct)

    if yoy_rates:
        print(f"   Average YoY growth: {np.mean(yoy_rates):.2f}%")
        print(f"   Median YoY growth:  {np.median(yoy_rates):.2f}%")
        print(f"   Std Dev YoY growth: {np.std(yoy_rates):.2f}%")
        print(f"   Recent YoY (last 4): {np.mean(yoy_rates[-4:]):.2f}%")

    # Compound annual growth rate (CAGR)
    first_year_avg = np.mean(ch_total[:4])
    last_year_avg = np.mean(ch_total[-4:])
    years = (len(periods) - 1) / 4
    cagr = 100 * ((last_year_avg / first_year_avg) ** (1/years) - 1)
    print(f"\n   CAGR ({periods[0][:4]}-{periods[-1][:4]}): {cagr:.2f}%")

    # Check if trend is linear or accelerating
    time_index = np.arange(len(periods))
    # Simple linear regression
    A = np.vstack([time_index, np.ones(len(time_index))]).T
    slope, intercept = np.linalg.lstsq(A, ch_total, rcond=None)[0]

    print(f"\n   Linear trend: {slope*4:.1f}k per year")
    print(f"   That's {100*slope*4/intercept:.2f}% per year at baseline")

if __name__ == "__main__":
    main()
