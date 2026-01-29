"""Shock/intervention definitions for Swiss health cost forecasting.

This module defines known shocks and policy changes:
1. COVID-19 (2020Q2-Q3): Temporary disruption affecting all cantons/groups
2. ZG hospital policy (2026Q1+): ZG pays hospital stays directly (costs = 0 in OKP)
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from KPOKPCH.DMFM import Shock, ShockSchedule, ShockLevel, ShockScope

# =============================================================================
# Period mapping
# =============================================================================


def period_to_index(period: str, base_period: str = "2016Q1") -> int:
    """Convert period string (e.g., '2020Q2') to time index.

    Parameters
    ----------
    period : str
        Period in format 'YYYYQn'.
    base_period : str, default '2016Q1'
        First period in the dataset (t=0).

    Returns
    -------
    int
        Time index (0-based).
    """

    def parse_period(p: str) -> tuple[int, int]:
        year = int(p[:4])
        quarter = int(p[-1])
        return year, quarter

    base_year, base_q = parse_period(base_period)
    year, q = parse_period(period)

    return (year - base_year) * 4 + (q - base_q)


# =============================================================================
# COVID-19 shock definition
# =============================================================================

# COVID-19 affected Q2 and Q3 of 2020 (indices 17-18 with base 2016Q1)
COVID_START_T = period_to_index("2020Q2")  # t=17
COVID_END_T = period_to_index("2020Q3")  # t=18


def create_covid_shock() -> Shock:
    """Create COVID-19 shock for 2020Q2-Q3.

    COVID-19 caused a temporary disruption to healthcare costs:
    - Some elective procedures were postponed
    - Emergency care patterns changed
    - Overall effect was temporary but significant

    Modeled as a factor-level shock affecting all cantons and cost groups.
    """
    return Shock(
        name="covid19",
        start_t=COVID_START_T,
        end_t=COVID_END_T,
        level=ShockLevel.FACTOR,  # Affects latent factors
        scope=ShockScope.GLOBAL,  # All cantons and groups
    )


def create_covid_schedule(training_end_t: int) -> Optional[ShockSchedule]:
    """Create COVID shock schedule if training data includes the shock period.

    In validation, we only include shocks that would be known at the time:
    - If training ends before 2020Q2, COVID is not yet known
    - If training includes 2020Q2+, COVID shock should be modeled

    Parameters
    ----------
    training_end_t : int
        Last time index in training data (exclusive).

    Returns
    -------
    ShockSchedule or None
        Schedule with COVID shock if applicable, None otherwise.
    """
    # COVID starts at t=17 (2020Q2)
    # We include the shock if training data includes at least the start
    if training_end_t > COVID_START_T:
        return ShockSchedule([create_covid_shock()])
    return None


# =============================================================================
# ZG hospital policy (2026)
# =============================================================================

# ZG will pay hospital stays (séjours) directly starting 2026Q1
# This means OKP costs for ZG séjours = 0
# Known since 2025Q1

ZG_POLICY_START_PERIOD = "2026Q1"
ZG_POLICY_END_PERIOD = "2026Q4"
ZG_POLICY_KNOWN_SINCE = "2025Q1"

# Canton index for ZG (after removing CH from the alphabetical list)
# AG=0, AI=1, AR=2, BE=3, BL=4, BS=5, FR=6, GE=7, GL=8, GR=9, JU=10, LU=11,
# NE=12, NW=13, OW=14, SG=15, SH=16, SO=17, SZ=18, TG=19, TI=20, UR=21,
# VD=22, VS=23, ZG=24, ZH=25
ZG_CANTON_INDEX = 24

# Cost group index for hospital stays ("Hôpitaux (séjours)")
# After merging Psychothérapeutes into Autres, the groups are:
# 0: Autres
# 1: Etablissements médico-sociaux
# 2: Hôpitaux (ambulatoire)
# 3: Hôpitaux (séjours)  <-- this one
# 4: Laboratoire cabinet médical
# 5: Laboratoires
# 6: Médecins (ambulatoire) sans laboratoire
# 7: Médicaments (médecin)
# 8: Pharmacies
# 9: Physiothérapeutes
# 10: SPITEX (soins à domicile)
SEJOUR_GROUP_INDEX = 3


def apply_zg_policy_to_forecast(
    forecast: np.ndarray,
    forecast_start_period: str,
    cantons: list[str],
    groups: list[str],
) -> np.ndarray:
    """Apply ZG hospital policy to forecast: set ZG séjours to 0 for 2026.

    Starting 2026Q1, canton ZG pays hospital stays directly (not through OKP).
    This function sets the forecast values for ZG × séjours to 0 for affected
    periods.

    Parameters
    ----------
    forecast : np.ndarray
        Forecast array of shape (steps, n_cantons, n_groups).
    forecast_start_period : str
        Period of first forecast step (e.g., '2025Q4').
    cantons : list[str]
        List of canton codes (without 'CH').
    groups : list[str]
        List of cost group names.

    Returns
    -------
    np.ndarray
        Modified forecast with ZG séjours set to 0 for 2026 periods.
    """
    forecast = forecast.copy()
    steps = forecast.shape[0]

    # Find ZG and séjours indices
    if "ZG" not in cantons:
        return forecast  # ZG not in data, nothing to do
    zg_idx = cantons.index("ZG")

    # Find séjours group (may have slightly different name)
    sejour_idx = None
    for i, g in enumerate(groups):
        if "séjour" in g.lower() or "sejour" in g.lower():
            sejour_idx = i
            break

    if sejour_idx is None:
        return forecast  # séjours not found, nothing to do

    # Determine which forecast steps fall in 2026
    policy_start_t = period_to_index(ZG_POLICY_START_PERIOD)
    policy_end_t = period_to_index(ZG_POLICY_END_PERIOD)
    forecast_start_t = period_to_index(forecast_start_period)

    for h in range(steps):
        t = forecast_start_t + h
        if policy_start_t <= t <= policy_end_t:
            forecast[h, zg_idx, sejour_idx] = 0.0

    return forecast


def get_zg_policy_info() -> dict:
    """Get ZG policy information for documentation/output.

    Returns
    -------
    dict
        Policy details including periods and affected cells.
    """
    return {
        "name": "ZG hospital direct payment",
        "description": "Canton ZG pays hospital stays directly (not through OKP)",
        "start_period": ZG_POLICY_START_PERIOD,
        "end_period": ZG_POLICY_END_PERIOD,
        "known_since": ZG_POLICY_KNOWN_SINCE,
        "canton": "ZG",
        "cost_group": "Hôpitaux (séjours)",
        "effect": "OKP costs set to 0",
    }
