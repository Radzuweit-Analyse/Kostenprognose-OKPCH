"""Analyze missing data patterns in health costs tensor."""

import numpy as np
import pandas as pd
from pathlib import Path
from KPOKPCH.forecast import load_cost_matrix

def main():
    """Analyze missing data patterns."""
    # Load data
    csv_path = Path(__file__).resolve().parent / "health_costs_tensor.csv"
    periods, cantons, groups, data = load_cost_matrix(str(csv_path))

    print(f"Data shape: {data.shape}")
    print(f"Periods: {len(periods)} ({periods[0]} to {periods[-1]})")
    print(f"Cantons: {len(cantons)}")
    print(f"Cost groups: {len(groups)}")

    # Count missing by dimension
    missing_mask = np.isnan(data)
    total_missing = missing_mask.sum()
    total_values = data.size

    print(f"\n📊 Overall missing data:")
    print(f"   Total missing: {total_missing}/{total_values} ({100*total_missing/total_values:.2f}%)")

    # Missing by time period
    print(f"\n📅 Missing data by period:")
    for t, period in enumerate(periods):
        period_missing = missing_mask[t].sum()
        period_total = missing_mask[t].size
        if period_missing > 0:
            print(f"   {period}: {period_missing}/{period_total} ({100*period_missing/period_total:.1f}%)")

    # Missing by canton
    print(f"\n🏛️  Missing data by canton:")
    for i, canton in enumerate(cantons):
        canton_missing = missing_mask[:, i, :].sum()
        canton_total = missing_mask[:, i, :].size
        if canton_missing > 0:
            print(f"   {canton}: {canton_missing}/{canton_total} ({100*canton_missing/canton_total:.1f}%)")

    # Missing by cost group
    print(f"\n💰 Missing data by cost group:")
    for j, group in enumerate(groups):
        group_missing = missing_mask[:, :, j].sum()
        group_total = missing_mask[:, :, j].size
        pct = 100*group_missing/group_total
        print(f"   {group:45s}: {group_missing:4d}/{group_total} ({pct:5.1f}%)")

    # Find specific missing combinations
    print(f"\n🔍 Specific missing combinations:")
    missing_combos = []
    for t, period in enumerate(periods):
        for i, canton in enumerate(cantons):
            for j, group in enumerate(groups):
                if missing_mask[t, i, j]:
                    missing_combos.append((period, canton, group, data[t, i, j]))

    # Group by cost group
    from collections import defaultdict
    by_group = defaultdict(list)
    for period, canton, group, value in missing_combos:
        by_group[group].append((period, canton))

    print(f"\nTotal missing combinations: {len(missing_combos)}")
    print(f"\nMissing by cost group:")
    for group in sorted(by_group.keys(), key=lambda g: len(by_group[g]), reverse=True):
        print(f"\n  {group} ({len(by_group[group])} missing):")
        # Show first 10 for each group
        for period, canton in sorted(by_group[group])[:10]:
            print(f"    - {period}, {canton}")
        if len(by_group[group]) > 10:
            print(f"    ... and {len(by_group[group])-10} more")

    # Check if missing data is early periods (before reporting started)
    print(f"\n📆 Missing data by year:")
    years = sorted(set(p[:4] for p in periods))
    for year in years:
        year_periods = [p for p in periods if p.startswith(year)]
        year_indices = [periods.index(p) for p in year_periods]
        year_missing = sum(missing_mask[t].sum() for t in year_indices)
        year_total = sum(missing_mask[t].size for t in year_indices)
        if year_missing > 0:
            print(f"   {year}: {year_missing}/{year_total} ({100*year_missing/year_total:.1f}%)")

if __name__ == "__main__":
    main()
