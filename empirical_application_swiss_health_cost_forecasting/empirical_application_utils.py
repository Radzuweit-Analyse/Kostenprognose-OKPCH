"""Utility functions for the Swiss health cost forecasting application.

This module provides data loading and period generation utilities used
by the empirical application scripts (main.py, validate.py).
"""

from __future__ import annotations

import csv
from typing import List, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------


def load_matrix_data(
    path: str,
) -> (
    Tuple[List[str], List[str], np.ndarray]
    | Tuple[List[str], List[str], List[str], np.ndarray]
):
    """Load panel/tensor data from CSV.

    The function supports two layouts:

    * **Wide 2D format**: Column headers are ``Period`` followed by row entity names.
      Returns ``(periods, rows, data)`` where ``data`` has shape
      ``(T, num_rows)``.

    * **Tensor format**: Column headers are ``<Row>|<Col>`` for each
      column category. Returns ``(periods, rows, cols, data)`` where ``data``
      has shape ``(T, num_rows, num_cols)``.

    Parameters
    ----------
    path : str
        Path to CSV file.

    Returns
    -------
    periods : list[str]
        Time period labels.
    rows : list[str]
        Row entity names (first dimension after time).
    data : np.ndarray
        Data array (2D or 3D depending on format).
    cols : list[str], optional
        Column category names (only for tensor format).

    Examples
    --------
    >>> # 2D format
    >>> periods, rows, data = load_matrix_data("data_2d.csv")
    >>> print(data.shape)  # (T, num_rows)

    >>> # 3D tensor format
    >>> periods, rows, cols, data = load_matrix_data("data_3d.csv")
    >>> print(data.shape)  # (T, num_rows, num_cols)
    """
    periods: List[str] = []
    data_rows = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        raw_columns = header[1:]

        # Detect format
        has_cols = all("|" in col for col in raw_columns)

        if has_cols:
            # Parse row|col headers
            row_col_pairs: List[Tuple[str, str]] = []
            for col in raw_columns:
                row_name, col_name = col.split("|", maxsplit=1)
                row_col_pairs.append((row_name, col_name))

            # Extract unique rows and cols
            rows: List[str] = []
            cols: List[str] = []
            for row_name, col_name in row_col_pairs:
                if row_name not in rows:
                    rows.append(row_name)
                if col_name not in cols:
                    cols.append(col_name)
        else:
            row_col_pairs = [(row_name, "") for row_name in raw_columns]
            rows = list(raw_columns)
            cols = [""]

        # Read data rows
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

    if has_cols:
        # Reshape to 3D tensor
        data = np.full((len(periods), len(rows), len(cols)), np.nan, dtype=float)
        row_idx = {r: i for i, r in enumerate(rows)}
        col_idx = {c: j for j, c in enumerate(cols)}

        for col, (row_name, col_name) in enumerate(row_col_pairs):
            data[:, row_idx[row_name], col_idx[col_name]] = flat[:, col]

        return periods, rows, cols, data

    return periods, rows, flat


# ---------------------------------------------------------------------------
# Period utilities
# ---------------------------------------------------------------------------


def generate_future_periods(last_period: str, steps: int) -> List[str]:
    """Generate future quarterly period labels.

    Parameters
    ----------
    last_period : str
        Last observed period in format "YYYYQQ" (e.g., "2024Q4").
    steps : int
        Number of future periods to generate.

    Returns
    -------
    list[str]
        Future period labels.

    Examples
    --------
    >>> generate_future_periods("2024Q4", 8)
    ['2025Q1', '2025Q2', '2025Q3', '2025Q4', '2026Q1', '2026Q2', '2026Q3', '2026Q4']
    """
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
