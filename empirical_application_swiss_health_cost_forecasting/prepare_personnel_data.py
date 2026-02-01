"""Prepare medical personnel density data for the DMFM.

This module processes the OFS health personnel statistics (su-f-14.04.02-sm-6A.csv)
and prepares it for integration with the quarterly health cost data.

The personnel data is annual, published with ~1 year delay:
- Data for year Y is typically published in November of year Y+1
- For the model, annual values are assigned to Q4 of the corresponding year
- Q1-Q3 observations are treated as missing (handled via mask)

Data source: OFS – Statistique des services de santé
Variables used:
- tot_d: Total FTE across all staff categories (médecins, soignants, administratifs)
- Alternative: tot_a for doctors/academics only
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal

# Personnel data file
INPUT_FILE = "su-f-14.04.02-sm-6A.csv"

# Canton codes in the expected order (matches shocks_config.py)
CANTON_CODES = [
    "AG",
    "AI",
    "AR",
    "BE",
    "BL",
    "BS",
    "FR",
    "GE",
    "GL",
    "GR",
    "JU",
    "LU",
    "NE",
    "NW",
    "OW",
    "SG",
    "SH",
    "SO",
    "SZ",
    "TG",
    "TI",
    "UR",
    "VD",
    "VS",
    "ZG",
    "ZH",
]


def load_personnel_data(
    file_path: Path | str | None = None,
    variable: Literal["tot_d", "tot_a", "tot_b", "tot_c"] = "tot_d",
) -> pd.DataFrame:
    """Load and clean personnel data from OFS statistics.

    Parameters
    ----------
    file_path : Path or str, optional
        Path to the CSV file. If None, uses default location.
    variable : str, default "tot_d"
        Which variable to extract:
        - "tot_d": Total FTE (all categories)
        - "tot_a": Doctors and academics only
        - "tot_b": Nursing and animation staff
        - "tot_c": Administrative, hotel, technical staff

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: year, canton, personnel_fte
        Indexed by (year, canton).
    """
    if file_path is None:
        file_path = Path(__file__).resolve().parent / INPUT_FILE

    df = pd.read_csv(file_path)

    # Filter to individual cantons (exclude CH aggregate)
    df = df[df["canton"] != "CH"].copy()

    # Select the relevant variable
    df = df[["year", "canton", variable]].copy()
    df = df.rename(columns={variable: "personnel_fte"})

    # Convert to numeric, handling any remaining issues
    df["personnel_fte"] = pd.to_numeric(df["personnel_fte"], errors="coerce")

    # Sort for consistency
    df = df.sort_values(["year", "canton"]).reset_index(drop=True)

    return df


def expand_to_quarterly(
    annual_df: pd.DataFrame,
    start_quarter: str = "2016Q1",
    end_quarter: str | None = None,
    assign_quarter: int = 4,
) -> pd.DataFrame:
    """Expand annual personnel data to quarterly frequency.

    Annual values are assigned to a specific quarter (default Q4) with
    NaN for other quarters. This allows the model to handle mixed-frequency
    data via the observation mask.

    Parameters
    ----------
    annual_df : pd.DataFrame
        Annual data with columns: year, canton, personnel_fte
    start_quarter : str, default "2016Q1"
        First quarter in the output series (format "YYYYQQ").
    end_quarter : str, optional
        Last quarter in the output series. If None, extends to the
        latest year in the data.
    assign_quarter : int, default 4
        Which quarter (1-4) receives the annual value. Default is Q4
        since annual data represents the full year.

    Returns
    -------
    pd.DataFrame
        Quarterly DataFrame with columns: period, canton, personnel_fte
        Missing quarters have NaN values.
    """
    # Parse start/end
    start_year = int(start_quarter[:4])
    start_q = int(start_quarter[-1])

    if end_quarter is None:
        end_year = annual_df["year"].max()
        end_q = 4
    else:
        end_year = int(end_quarter[:4])
        end_q = int(end_quarter[-1])

    # Generate all quarters
    quarters = []
    y, q = start_year, start_q
    while (y, q) <= (end_year, end_q):
        quarters.append(f"{y}Q{q}")
        q += 1
        if q > 4:
            q = 1
            y += 1

    # Create full quarterly grid
    cantons = annual_df["canton"].unique()
    grid = pd.MultiIndex.from_product(
        [quarters, cantons], names=["period", "canton"]
    ).to_frame(index=False)

    # Map annual data to quarters
    annual_df = annual_df.copy()
    annual_df["period"] = annual_df["year"].apply(lambda y: f"{y}Q{assign_quarter}")

    # Merge
    result = grid.merge(
        annual_df[["period", "canton", "personnel_fte"]],
        on=["period", "canton"],
        how="left",
    )

    # Sort by period then canton
    result = result.sort_values(["period", "canton"]).reset_index(drop=True)

    return result


def create_personnel_tensor(
    quarterly_df: pd.DataFrame,
    cantons: list[str] | None = None,
) -> tuple[list[str], list[str], np.ndarray]:
    """Convert quarterly personnel data to tensor format.

    Parameters
    ----------
    quarterly_df : pd.DataFrame
        Quarterly data with columns: period, canton, personnel_fte
    cantons : list[str], optional
        Canton order. If None, uses CANTON_CODES.

    Returns
    -------
    periods : list[str]
        Time period labels.
    cantons : list[str]
        Canton codes.
    data : np.ndarray
        Personnel data of shape (T, num_cantons), with NaN for missing.
    """
    if cantons is None:
        cantons = CANTON_CODES

    periods = sorted(quarterly_df["period"].unique())

    # Create lookup
    data_dict = {
        (row["period"], row["canton"]): row["personnel_fte"]
        for _, row in quarterly_df.iterrows()
    }

    # Build tensor
    T = len(periods)
    n_cantons = len(cantons)
    data = np.full((T, n_cantons), np.nan)

    for t, period in enumerate(periods):
        for i, canton in enumerate(cantons):
            val = data_dict.get((period, canton))
            if val is not None and pd.notna(val):
                data[t, i] = val

    return periods, cantons, data


def save_personnel_tensor(
    periods: list[str],
    cantons: list[str],
    data: np.ndarray,
    output_path: Path | str,
    group_name: str = "Personnel médical (ETP)",
) -> None:
    """Save personnel tensor in the same format as health_costs_tensor.csv.

    Parameters
    ----------
    periods : list[str]
        Time period labels.
    cantons : list[str]
        Canton codes.
    data : np.ndarray
        Personnel data of shape (T, num_cantons).
    output_path : Path or str
        Output file path.
    group_name : str, default "Personnel médical (ETP)"
        Name of the personnel "cost group" for the column headers.
    """
    # Create columns in Canton|Group format
    columns = ["Periode"] + [f"{canton}|{group_name}" for canton in cantons]

    # Build rows
    rows = []
    for t, period in enumerate(periods):
        row = [period] + [
            f"{data[t, i]:.2f}" if not np.isnan(data[t, i]) else ""
            for i in range(len(cantons))
        ]
        rows.append(row)

    # Save
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_path, index=False)


def main() -> None:
    """Prepare personnel data and save as tensor."""
    base_dir = Path(__file__).resolve().parent

    print("Loading personnel data...")
    annual_df = load_personnel_data(variable="tot_d")
    print(f"  Years: {annual_df['year'].min()} to {annual_df['year'].max()}")
    print(f"  Cantons: {annual_df['canton'].nunique()}")

    print("\nExpanding to quarterly frequency...")
    quarterly_df = expand_to_quarterly(
        annual_df,
        start_quarter="2016Q1",  # Match health cost data start
        assign_quarter=4,  # Annual value → Q4
    )

    n_obs = quarterly_df["personnel_fte"].notna().sum()
    n_total = len(quarterly_df)
    pct_obs = 100 * n_obs / n_total
    print(f"  Total observations: {n_total}")
    print(f"  Observed (non-NaN): {n_obs} ({pct_obs:.1f}%)")

    print("\nConverting to tensor format...")
    periods, cantons, data = create_personnel_tensor(quarterly_df)
    print(f"  Shape: {len(periods)} periods x {len(cantons)} cantons")

    output_path = base_dir / "personnel_tensor.csv"
    save_personnel_tensor(periods, cantons, data, output_path)
    print(f"\nSaved: {output_path.name}")

    # Show sample
    print("\nSample data (first 5 periods):")
    for t in range(min(5, len(periods))):
        vals = data[t, :]
        n_valid = np.sum(~np.isnan(vals))
        if n_valid > 0:
            mean_val = np.nanmean(vals)
            print(f"  {periods[t]}: {n_valid} cantons, mean={mean_val:.1f} FTE")
        else:
            print(f"  {periods[t]}: no observations (annual data at Q4)")


if __name__ == "__main__":
    main()
