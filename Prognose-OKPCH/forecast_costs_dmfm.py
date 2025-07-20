import csv
import numpy as np
from typing import List, Tuple
import KPOKPCH


def load_cost_matrix(path: str) -> Tuple[List[str], List[str], np.ndarray]:
    """Load canton cost matrix from CSV produced by prepare-MOKKE-data.py."""
    periods: List[str] = []
    data_rows = []
    with open(path, newline='', encoding="utf-8") as f:
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


def compute_q4_growth(periods: List[str], data: np.ndarray, fcst: np.ndarray, future_periods: List[str]) -> dict:
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
    csv_path = "Prognose-OKPCH/health_costs_matrix.csv"
    periods, cantons, data = load_cost_matrix(csv_path)
    Y = data[:, :, None]  # (T, cantons, 1)
    mask = ~np.isnan(Y)
    params = KPOKPCH.fit_dmfm_em(Y, k1=1, k2=1, P=1, mask=mask, max_iter=50)
    steps = 8  # two years ahead
    fcst = KPOKPCH.forecast_dmfm(steps, params)[:, :, 0]
    future_periods = generate_future_periods(periods[-1], steps)
    stats = compute_q4_growth(periods, data, fcst, future_periods)

    print("Year 1 Q4/Q4 mean increase: {:.2f}%".format(stats["mean_y1"]))
    print("Std Dev: {:.2f}".format(stats["sd_y1"]))
    print("10% CI: [{:.2f}, {:.2f}]".format(*stats["ci_y1"]))
    print("Year 2 Q4/Q4 mean increase: {:.2f}%".format(stats["mean_y2"]))
    print("Std Dev: {:.2f}".format(stats["sd_y2"]))
    print("10% CI: [{:.2f}, {:.2f}]".format(*stats["ci_y2"]))


if __name__ == "__main__":
    main()
