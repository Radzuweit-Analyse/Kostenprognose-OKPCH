"""Shock and intervention definitions for Swiss health cost forecasting.

This module defines known events affecting the data:

**Shocks** (stochastic, effects estimated):
    - COVID-19 (2020Q2-Q3): Temporary disruption affecting all cantons/groups
    - COVID rebound (2020Q4-2021Q1): Catch-up effect as postponed procedures performed

**Interventions** (deterministic, effects known):
    - ZG hospital policy (2026Q1-Q4): ZG pays hospital stays directly (costs = 0)
"""

from __future__ import annotations

from typing import Optional

from KPOKPCH.DMFM import (
    Shock,
    ShockSchedule,
    Intervention,
    InterventionSchedule,
    InterventionType,
    ScheduleFactory,
)

# =============================================================================
# Canton and cost group definitions
# =============================================================================

# Canton codes (alphabetical, excluding CH aggregate)
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

# Cost groups (after merging Psychothérapeutes into Autres)
COST_GROUPS = [
    "Autres",
    "Etablissements médico-sociaux",
    "Hôpitaux (ambulatoire)",
    "Hôpitaux (séjours)",
    "Laboratoire cabinet médical",
    "Laboratoires",
    "Médecins (ambulatoire) sans laboratoire",
    "Médicaments (médecin)",
    "Pharmacies",
    "Physiothérapeutes",
    "SPITEX (soins à domicile)",
]

# =============================================================================
# Factory for creating shocks and interventions
# =============================================================================

factory = ScheduleFactory(
    base_period="2016Q1",
    row_names=CANTON_CODES,
    col_names=COST_GROUPS,
)

# =============================================================================
# COVID-19 shock
# =============================================================================


def create_covid_shock() -> Shock:
    """Create COVID-19 shock for 2020Q2-Q3.

    COVID-19 caused a temporary disruption to healthcare costs:
    - Elective procedures were postponed
    - Emergency care patterns changed
    - Effect was temporary but significant

    Modeled as a factor-level shock affecting all cantons and cost groups.
    """
    return factory.shock(
        name="covid19",
        start="2020Q2",
        end="2020Q3",
    )


def create_covid_rebound_shock() -> Shock:
    """Create COVID-19 rebound shock for 2020Q4-2021Q1.

    Post-COVID catch-up effect:
    - Postponed elective procedures performed
    - Backlog clearance causing above-trend costs
    - Symmetric to COVID dip but opposite sign

    Modeled as a factor-level shock affecting all cantons and cost groups.
    """
    return factory.shock(
        name="covid19_rebound",
        start="2020Q4",
        end="2021Q1",
    )


def create_covid_schedule(training_end_period: str) -> Optional[ShockSchedule]:
    """Create COVID shock schedule if training data includes the shock period.

    Includes both the COVID dip (2020Q2-Q3) and rebound (2020Q4-2021Q1).

    Parameters
    ----------
    training_end_period : str
        Last period in training data (e.g., "2023Q4").

    Returns
    -------
    ShockSchedule or None
        Schedule with COVID shocks if applicable, None otherwise.
    """
    training_end_t = factory.period_to_index(training_end_period)
    covid_start_t = factory.period_to_index("2020Q2")
    rebound_start_t = factory.period_to_index("2020Q4")

    shocks = []

    # Include COVID dip if training extends past 2020Q2
    if training_end_t > covid_start_t:
        shocks.append(create_covid_shock())

    # Include rebound if training extends past 2020Q4
    if training_end_t > rebound_start_t:
        shocks.append(create_covid_rebound_shock())

    return ShockSchedule(shocks) if shocks else None


# =============================================================================
# ZG hospital policy intervention
# =============================================================================


def create_zg_intervention() -> Intervention:
    """Create ZG hospital policy intervention for 2026.

    Starting 2026Q1, canton ZG pays hospital stays directly (not through OKP).
    This means OKP costs for ZG × séjours = 0.

    Modeled as a deterministic override intervention.
    """
    return factory.intervention(
        name="zg_hospital_2026",
        start="2026Q1",
        end="2026Q4",
        intervention_type=InterventionType.OVERRIDE,
        value=0.0,
        rows=["ZG"],
        cols=["Hôpitaux (séjours)"],
    )


def create_intervention_schedule() -> InterventionSchedule:
    """Create schedule of all known interventions."""
    return InterventionSchedule([create_zg_intervention()])


# =============================================================================
# Combined schedules
# =============================================================================


def get_schedules(
    training_end_period: str,
) -> tuple[Optional[ShockSchedule], InterventionSchedule]:
    """Get both shock and intervention schedules for a given training period.

    Parameters
    ----------
    training_end_period : str
        Last period in training data (e.g., "2023Q4").

    Returns
    -------
    shock_schedule : ShockSchedule or None
        Schedule of shocks (for estimation).
    intervention_schedule : InterventionSchedule
        Schedule of interventions (for forecast adjustment).
    """
    shock_schedule = create_covid_schedule(training_end_period)
    intervention_schedule = create_intervention_schedule()
    return shock_schedule, intervention_schedule


def get_zg_policy_info() -> dict:
    """Get ZG policy information for documentation/output."""
    return {
        "name": "ZG hospital direct payment",
        "description": "Canton ZG pays hospital stays directly (not through OKP)",
        "start_period": "2026Q1",
        "end_period": "2026Q4",
        "canton": "ZG",
        "cost_group": "Hôpitaux (séjours)",
        "effect": "OKP costs set to 0",
    }
