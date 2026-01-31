"""Shock and intervention modeling for the DMFM.

This module provides tools for incorporating known events into the Dynamic
Matrix Factor Model. It distinguishes between two types of adjustments:

**Shocks** (stochastic):
    Unexpected events whose effects need to be estimated from data.
    Examples: COVID-19 pandemic, financial crises, natural disasters.
    Effects are additive and can decay over time.

**Interventions** (deterministic):
    Planned policy changes with known effects specified a priori.
    Examples: ZG hospital policy (costs = 0), tariff reforms, regulatory changes.
    Effects are exact: override, scale, or shift values.

Both can be targeted globally or to specific rows/columns (e.g., cantons/categories).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Literal, Optional, Tuple, Union
import numpy as np


class ShockLevel(str, Enum):
    """Where the shock enters the model."""

    FACTOR = "factor"  # Enters transition equation (affects F_t)
    OBSERVATION = "observation"  # Enters measurement equation (affects Y_t)


class ShockScope(str, Enum):
    """Scope of the shock effect."""

    GLOBAL = "global"  # Affects all cantons and categories
    CANTON = "canton"  # Affects specific cantons only
    CATEGORY = "category"  # Affects specific categories only
    CANTON_CATEGORY = "canton_category"  # Affects specific canton-category pairs


class DecayType(str, Enum):
    """How the shock intensity decays over time."""

    STEP = "step"  # Constant intensity (0 or 1)
    EXPONENTIAL = "exponential"  # Exponential decay: decay_rate^(t - start_t)
    LINEAR = "linear"  # Linear decay to zero at end_t


@dataclass
class Shock:
    """Definition of a single shock/intervention.

    A shock represents a known event that affects the data-generating process.
    Shocks can be temporary or permanent, global or targeted, and can enter
    at either the factor or observation level.

    Parameters
    ----------
    name : str
        Unique identifier for the shock.
    start_t : int
        First affected time period (0-indexed).
    end_t : int, optional
        Last affected time period. If None, shock is permanent.
    level : ShockLevel, default "factor"
        Whether shock enters at factor or observation level.
    scope : ShockScope, default "global"
        Whether shock is global or targeted to specific cantons/categories.
    cantons : list[int], optional
        Canton indices affected (required if scope involves cantons).
    categories : list[int], optional
        Category indices affected (required if scope involves categories).
    decay_type : DecayType, default "step"
        How shock intensity decays over time.
    decay_rate : float, optional
        Decay parameter. For exponential: multiplier per period (e.g., 0.8).
        For linear: ignored (uses end_t - start_t as duration).
    fixed_effect : np.ndarray, optional
        Pre-specified effect magnitude. If None, effect is estimated.
        Shape depends on level: (k1, k2) for factor, (p1, p2) for observation.

    Examples
    --------
    >>> # COVID-19: temporary global shock at factor level
    >>> covid = Shock(
    ...     name="covid_2020",
    ...     start_t=32,  # Q1 2020
    ...     end_t=35,    # Q4 2020
    ...     level=ShockLevel.FACTOR,
    ...     scope=ShockScope.GLOBAL,
    ... )

    >>> # Policy reform: permanent shock to specific categories
    >>> reform = Shock(
    ...     name="tarmed_revision",
    ...     start_t=28,
    ...     end_t=None,  # Permanent
    ...     level=ShockLevel.OBSERVATION,
    ...     scope=ShockScope.CATEGORY,
    ...     categories=[1, 2],  # Ambulatory categories
    ... )

    >>> # Regional pilot: temporary canton-specific shock with decay
    >>> pilot = Shock(
    ...     name="vaud_pilot",
    ...     start_t=36,
    ...     end_t=44,
    ...     level=ShockLevel.OBSERVATION,
    ...     scope=ShockScope.CANTON,
    ...     cantons=[22],  # Vaud
    ...     decay_type=DecayType.EXPONENTIAL,
    ...     decay_rate=0.85,
    ... )
    """

    name: str
    start_t: int
    end_t: Optional[int] = None
    level: ShockLevel = ShockLevel.FACTOR
    scope: ShockScope = ShockScope.GLOBAL
    cantons: Optional[List[int]] = None
    categories: Optional[List[int]] = None
    decay_type: DecayType = DecayType.STEP
    decay_rate: Optional[float] = None
    fixed_effect: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Validate shock configuration."""
        # Convert string literals to enums if needed
        if isinstance(self.level, str):
            self.level = ShockLevel(self.level)
        if isinstance(self.scope, str):
            self.scope = ShockScope(self.scope)
        if isinstance(self.decay_type, str):
            self.decay_type = DecayType(self.decay_type)

        # Validate targeting
        if self.scope in (ShockScope.CANTON, ShockScope.CANTON_CATEGORY):
            if self.cantons is None or len(self.cantons) == 0:
                raise ValueError(
                    f"Shock '{self.name}' with scope={self.scope} requires cantons"
                )
        if self.scope in (ShockScope.CATEGORY, ShockScope.CANTON_CATEGORY):
            if self.categories is None or len(self.categories) == 0:
                raise ValueError(
                    f"Shock '{self.name}' with scope={self.scope} requires categories"
                )

        # Validate decay
        if self.decay_type == DecayType.EXPONENTIAL:
            if self.decay_rate is None:
                raise ValueError(
                    f"Shock '{self.name}' with exponential decay requires decay_rate"
                )
            if not 0 < self.decay_rate < 1:
                raise ValueError(f"decay_rate must be in (0, 1), got {self.decay_rate}")
        if self.decay_type == DecayType.LINEAR and self.end_t is None:
            raise ValueError(
                f"Shock '{self.name}' with linear decay requires finite end_t"
            )

        # Validate time bounds
        if self.end_t is not None and self.end_t < self.start_t:
            raise ValueError(
                f"end_t ({self.end_t}) must be >= start_t ({self.start_t})"
            )

    def indicator(self, t: int) -> float:
        """Compute shock intensity at time t.

        Parameters
        ----------
        t : int
            Time period (0-indexed).

        Returns
        -------
        float
            Shock intensity in [0, 1]. Zero if outside shock window.
        """
        if t < self.start_t:
            return 0.0
        if self.end_t is not None and t > self.end_t:
            return 0.0

        periods_since_start = t - self.start_t

        if self.decay_type == DecayType.STEP:
            return 1.0
        elif self.decay_type == DecayType.EXPONENTIAL:
            return self.decay_rate**periods_since_start
        elif self.decay_type == DecayType.LINEAR:
            if self.end_t is None:
                return 1.0
            duration = self.end_t - self.start_t
            if duration == 0:
                return 1.0
            return max(0.0, 1.0 - periods_since_start / duration)

        return 1.0

    def is_active(self, t: int) -> bool:
        """Check if shock is active at time t.

        Parameters
        ----------
        t : int
            Time period.

        Returns
        -------
        bool
            True if shock has non-zero intensity at time t.
        """
        return self.indicator(t) > 0

    def build_target_mask(self, p1: int, p2: int) -> np.ndarray:
        """Build binary mask indicating affected canton-category pairs.

        Parameters
        ----------
        p1 : int
            Number of cantons.
        p2 : int
            Number of categories.

        Returns
        -------
        np.ndarray
            Binary mask of shape (p1, p2). Entry [i,j] = 1 if canton i,
            category j is affected by this shock.
        """
        mask = np.zeros((p1, p2))

        if self.scope == ShockScope.GLOBAL:
            mask[:, :] = 1.0
        elif self.scope == ShockScope.CANTON:
            for i in self.cantons:
                if 0 <= i < p1:
                    mask[i, :] = 1.0
        elif self.scope == ShockScope.CATEGORY:
            for j in self.categories:
                if 0 <= j < p2:
                    mask[:, j] = 1.0
        elif self.scope == ShockScope.CANTON_CATEGORY:
            for i in self.cantons:
                for j in self.categories:
                    if 0 <= i < p1 and 0 <= j < p2:
                        mask[i, j] = 1.0

        return mask

    def __repr__(self) -> str:
        """String representation."""
        end_str = str(self.end_t) if self.end_t is not None else "inf"
        return (
            f"Shock('{self.name}', t=[{self.start_t}, {end_str}], "
            f"level={self.level.value}, scope={self.scope.value})"
        )


@dataclass
class ShockSchedule:
    """Collection of shocks with design matrix generation.

    A ShockSchedule manages multiple shocks and provides methods to build
    the intervention design matrices needed for estimation and forecasting.

    Parameters
    ----------
    shocks : list[Shock], optional
        List of shocks. Defaults to empty list.

    Examples
    --------
    >>> schedule = ShockSchedule([
    ...     Shock("covid", start_t=32, end_t=35),
    ...     Shock("reform", start_t=40, end_t=None),
    ... ])
    >>> X = schedule.build_design_matrix(T=60)
    >>> print(X.shape)  # (60, 2)
    """

    shocks: List[Shock] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate shock names are unique."""
        names = [s.name for s in self.shocks]
        if len(names) != len(set(names)):
            raise ValueError("Shock names must be unique")

    def add(self, shock: Shock) -> None:
        """Add a shock to the schedule.

        Parameters
        ----------
        shock : Shock
            Shock to add.

        Raises
        ------
        ValueError
            If a shock with the same name already exists.
        """
        if any(s.name == shock.name for s in self.shocks):
            raise ValueError(f"Shock with name '{shock.name}' already exists")
        self.shocks.append(shock)

    def remove(self, name: str) -> None:
        """Remove a shock by name.

        Parameters
        ----------
        name : str
            Name of shock to remove.
        """
        self.shocks = [s for s in self.shocks if s.name != name]

    def get(self, name: str) -> Optional[Shock]:
        """Get a shock by name.

        Parameters
        ----------
        name : str
            Shock name.

        Returns
        -------
        Shock or None
            The shock if found, None otherwise.
        """
        for s in self.shocks:
            if s.name == name:
                return s
        return None

    @property
    def n_shocks(self) -> int:
        """Total number of shocks."""
        return len(self.shocks)

    @property
    def factor_shocks(self) -> List[Shock]:
        """Shocks that enter at the factor level."""
        return [s for s in self.shocks if s.level == ShockLevel.FACTOR]

    @property
    def observation_shocks(self) -> List[Shock]:
        """Shocks that enter at the observation level."""
        return [s for s in self.shocks if s.level == ShockLevel.OBSERVATION]

    @property
    def n_factor_shocks(self) -> int:
        """Number of factor-level shocks."""
        return len(self.factor_shocks)

    @property
    def n_observation_shocks(self) -> int:
        """Number of observation-level shocks."""
        return len(self.observation_shocks)

    def build_design_matrix(self, T: int) -> np.ndarray:
        """Build full design matrix of shock indicators.

        Parameters
        ----------
        T : int
            Number of time periods.

        Returns
        -------
        np.ndarray
            Design matrix of shape (T, n_shocks). Entry [t, s] contains
            the intensity of shock s at time t.
        """
        n = len(self.shocks)
        if n == 0:
            return np.zeros((T, 0))

        X = np.zeros((T, n))
        for s, shock in enumerate(self.shocks):
            for t in range(T):
                X[t, s] = shock.indicator(t)
        return X

    def build_factor_design_matrix(self, T: int) -> np.ndarray:
        """Build design matrix for factor-level shocks only.

        Parameters
        ----------
        T : int
            Number of time periods.

        Returns
        -------
        np.ndarray
            Design matrix of shape (T, n_factor_shocks).
        """
        factor_shocks = self.factor_shocks
        if not factor_shocks:
            return np.zeros((T, 0))

        X = np.zeros((T, len(factor_shocks)))
        for s, shock in enumerate(factor_shocks):
            for t in range(T):
                X[t, s] = shock.indicator(t)
        return X

    def build_observation_design_matrix(self, T: int) -> np.ndarray:
        """Build design matrix for observation-level shocks only.

        Parameters
        ----------
        T : int
            Number of time periods.

        Returns
        -------
        np.ndarray
            Design matrix of shape (T, n_observation_shocks).
        """
        obs_shocks = self.observation_shocks
        if not obs_shocks:
            return np.zeros((T, 0))

        X = np.zeros((T, len(obs_shocks)))
        for s, shock in enumerate(obs_shocks):
            for t in range(T):
                X[t, s] = shock.indicator(t)
        return X

    def extend_to_forecast_horizon(
        self,
        T_hist: int,
        steps: int,
        future_shocks: Optional[List[Shock]] = None,
    ) -> Tuple[np.ndarray, "ShockSchedule"]:
        """Build design matrix extending into forecast horizon.

        Ongoing shocks (those active at T_hist-1 with end_t >= T_hist or None)
        are extended into the forecast period. Additional future shocks can
        be specified for scenario analysis.

        Parameters
        ----------
        T_hist : int
            Number of historical periods.
        steps : int
            Number of forecast steps.
        future_shocks : list[Shock], optional
            Additional shocks that start in the forecast horizon.
            Their start_t should be relative to T_hist (i.e., start_t=0
            means first forecast period).

        Returns
        -------
        X_future : np.ndarray
            Design matrix for forecast horizon, shape (steps, n_extended_shocks).
        extended_schedule : ShockSchedule
            Schedule containing shocks active in forecast horizon.
        """
        # Collect shocks active in forecast horizon
        extended_shocks = []

        # Check existing shocks
        for shock in self.shocks:
            # Shock extends into forecast if:
            # - It's permanent (end_t is None) and started before T_hist
            # - Or end_t >= T_hist
            if shock.start_t < T_hist:
                if shock.end_t is None or shock.end_t >= T_hist:
                    extended_shocks.append(shock)

        # Add future shocks (adjust their start_t to absolute time)
        if future_shocks:
            for shock in future_shocks:
                # Create copy with adjusted timing
                adjusted = Shock(
                    name=shock.name,
                    start_t=T_hist + shock.start_t,
                    end_t=(T_hist + shock.end_t) if shock.end_t is not None else None,
                    level=shock.level,
                    scope=shock.scope,
                    cantons=shock.cantons,
                    categories=shock.categories,
                    decay_type=shock.decay_type,
                    decay_rate=shock.decay_rate,
                    fixed_effect=shock.fixed_effect,
                )
                extended_shocks.append(adjusted)

        extended_schedule = ShockSchedule(extended_shocks)

        # Build design matrix for forecast horizon
        n = len(extended_shocks)
        X_future = np.zeros((steps, n))
        for s, shock in enumerate(extended_shocks):
            for h in range(steps):
                t_abs = T_hist + h
                X_future[h, s] = shock.indicator(t_abs)

        return X_future, extended_schedule

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ShockSchedule({len(self.shocks)} shocks: {[s.name for s in self.shocks]})"
        )

    def __len__(self) -> int:
        """Number of shocks."""
        return len(self.shocks)

    def __iter__(self):
        """Iterate over shocks."""
        return iter(self.shocks)


@dataclass
class ShockEffects:
    """Estimated or fixed shock effect parameters.

    This class stores the effect magnitudes for each shock, either estimated
    via the EM algorithm or pre-specified.

    Attributes
    ----------
    factor_effects : np.ndarray, optional
        Effects for factor-level shocks, shape (n_factor_shocks, k1, k2).
        Entry [s, :, :] is the effect matrix Γ_s added to F_t when shock s
        is active.
    observation_effects : np.ndarray, optional
        Effects for observation-level shocks, shape (n_obs_shocks, p1, p2).
        Entry [s, :, :] is the effect matrix γ_s added to Y_t when shock s
        is active.
    factor_se : np.ndarray, optional
        Standard errors for factor effects, same shape as factor_effects.
    observation_se : np.ndarray, optional
        Standard errors for observation effects, same shape as observation_effects.
    """

    factor_effects: Optional[np.ndarray] = None
    observation_effects: Optional[np.ndarray] = None
    factor_se: Optional[np.ndarray] = None
    observation_se: Optional[np.ndarray] = None

    @property
    def n_factor_shocks(self) -> int:
        """Number of factor-level shocks."""
        if self.factor_effects is None:
            return 0
        return self.factor_effects.shape[0]

    @property
    def n_observation_shocks(self) -> int:
        """Number of observation-level shocks."""
        if self.observation_effects is None:
            return 0
        return self.observation_effects.shape[0]

    def get_factor_effect(self, idx: int) -> Optional[np.ndarray]:
        """Get effect matrix for a factor-level shock.

        Parameters
        ----------
        idx : int
            Shock index (among factor shocks).

        Returns
        -------
        np.ndarray or None
            Effect matrix of shape (k1, k2), or None if not available.
        """
        if self.factor_effects is None or idx >= self.n_factor_shocks:
            return None
        return self.factor_effects[idx]

    def get_observation_effect(self, idx: int) -> Optional[np.ndarray]:
        """Get effect matrix for an observation-level shock.

        Parameters
        ----------
        idx : int
            Shock index (among observation shocks).

        Returns
        -------
        np.ndarray or None
            Effect matrix of shape (p1, p2), or None if not available.
        """
        if self.observation_effects is None or idx >= self.n_observation_shocks:
            return None
        return self.observation_effects[idx]


# =============================================================================
# Interventions (deterministic policy changes)
# =============================================================================


class InterventionType(str, Enum):
    """Type of deterministic intervention."""

    OVERRIDE = "override"  # Set to fixed value: Y[target] = value
    SCALE = "scale"  # Multiply by factor: Y[target] *= factor
    SHIFT = "shift"  # Add fixed amount: Y[target] += delta


@dataclass
class Intervention:
    """Definition of a deterministic policy intervention.

    Unlike shocks (whose effects are estimated), interventions have known
    effects specified a priori. Use interventions for policy changes where
    the impact is deterministic.

    Parameters
    ----------
    name : str
        Unique identifier for the intervention.
    start_t : int
        First affected time period (0-indexed).
    end_t : int, optional
        Last affected time period. If None, intervention is permanent.
    intervention_type : InterventionType, default "override"
        How the intervention modifies values.
    value : float, optional
        For OVERRIDE: the value to set. For SHIFT: the amount to add.
    factor : float, optional
        For SCALE: the multiplication factor.
    rows : list[int], optional
        Row indices affected. If None, affects all rows.
    cols : list[int], optional
        Column indices affected. If None, affects all columns.

    Examples
    --------
    >>> # ZG hospital policy: set ZG × séjours to 0 for 2026
    >>> zg_policy = Intervention(
    ...     name="zg_hospital_2026",
    ...     start_t=40,  # 2026Q1
    ...     end_t=43,    # 2026Q4
    ...     intervention_type=InterventionType.OVERRIDE,
    ...     value=0.0,
    ...     rows=[24],   # ZG canton index
    ...     cols=[3],    # séjours category index
    ... )

    >>> # 10% cost reduction in ambulatory care
    >>> tariff_cut = Intervention(
    ...     name="tariff_reduction",
    ...     start_t=48,
    ...     end_t=None,  # Permanent
    ...     intervention_type=InterventionType.SCALE,
    ...     factor=0.9,
    ...     cols=[2, 6],  # Ambulatory categories
    ... )
    """

    name: str
    start_t: int
    end_t: Optional[int] = None
    intervention_type: InterventionType = InterventionType.OVERRIDE
    value: Optional[float] = None
    factor: Optional[float] = None
    rows: Optional[List[int]] = None
    cols: Optional[List[int]] = None

    def __post_init__(self) -> None:
        """Validate intervention configuration."""
        if isinstance(self.intervention_type, str):
            self.intervention_type = InterventionType(self.intervention_type)

        # Validate type-specific parameters
        if self.intervention_type == InterventionType.OVERRIDE:
            if self.value is None:
                raise ValueError(
                    f"Intervention '{self.name}' with OVERRIDE type requires 'value'"
                )
        elif self.intervention_type == InterventionType.SCALE:
            if self.factor is None:
                raise ValueError(
                    f"Intervention '{self.name}' with SCALE type requires 'factor'"
                )
        elif self.intervention_type == InterventionType.SHIFT:
            if self.value is None:
                raise ValueError(
                    f"Intervention '{self.name}' with SHIFT type requires 'value'"
                )

        # Validate time bounds
        if self.end_t is not None and self.end_t < self.start_t:
            raise ValueError(
                f"end_t ({self.end_t}) must be >= start_t ({self.start_t})"
            )

    def is_active(self, t: int) -> bool:
        """Check if intervention is active at time t."""
        if t < self.start_t:
            return False
        if self.end_t is not None and t > self.end_t:
            return False
        return True

    def build_target_mask(self, p1: int, p2: int) -> np.ndarray:
        """Build binary mask indicating affected cells.

        Parameters
        ----------
        p1 : int
            Number of rows.
        p2 : int
            Number of columns.

        Returns
        -------
        np.ndarray
            Boolean mask of shape (p1, p2).
        """
        mask = np.ones((p1, p2), dtype=bool)

        if self.rows is not None:
            row_mask = np.zeros(p1, dtype=bool)
            for i in self.rows:
                if 0 <= i < p1:
                    row_mask[i] = True
            mask &= row_mask[:, np.newaxis]

        if self.cols is not None:
            col_mask = np.zeros(p2, dtype=bool)
            for j in self.cols:
                if 0 <= j < p2:
                    col_mask[j] = True
            mask &= col_mask[np.newaxis, :]

        return mask

    def apply(self, Y: np.ndarray, t: int) -> np.ndarray:
        """Apply intervention to observation matrix at time t.

        Parameters
        ----------
        Y : np.ndarray
            Observation matrix of shape (p1, p2).
        t : int
            Time period.

        Returns
        -------
        np.ndarray
            Modified observation matrix.
        """
        if not self.is_active(t):
            return Y

        Y_mod = Y.copy()
        p1, p2 = Y.shape
        mask = self.build_target_mask(p1, p2)

        if self.intervention_type == InterventionType.OVERRIDE:
            Y_mod[mask] = self.value
        elif self.intervention_type == InterventionType.SCALE:
            Y_mod[mask] *= self.factor
        elif self.intervention_type == InterventionType.SHIFT:
            Y_mod[mask] += self.value

        return Y_mod

    def __repr__(self) -> str:
        """String representation."""
        end_str = str(self.end_t) if self.end_t is not None else "inf"
        return (
            f"Intervention('{self.name}', t=[{self.start_t}, {end_str}], "
            f"type={self.intervention_type.value})"
        )


@dataclass
class InterventionSchedule:
    """Collection of interventions.

    Parameters
    ----------
    interventions : list[Intervention], optional
        List of interventions. Defaults to empty list.
    """

    interventions: List[Intervention] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate intervention names are unique."""
        names = [i.name for i in self.interventions]
        if len(names) != len(set(names)):
            raise ValueError("Intervention names must be unique")

    def add(self, intervention: Intervention) -> None:
        """Add an intervention to the schedule."""
        if any(i.name == intervention.name for i in self.interventions):
            raise ValueError(f"Intervention '{intervention.name}' already exists")
        self.interventions.append(intervention)

    def apply(self, Y: np.ndarray, t: int) -> np.ndarray:
        """Apply all active interventions to observation matrix.

        Parameters
        ----------
        Y : np.ndarray
            Observation matrix of shape (p1, p2).
        t : int
            Time period.

        Returns
        -------
        np.ndarray
            Modified observation matrix with all interventions applied.
        """
        Y_mod = Y.copy()
        for intervention in self.interventions:
            Y_mod = intervention.apply(Y_mod, t)
        return Y_mod

    def __len__(self) -> int:
        return len(self.interventions)

    def __iter__(self):
        return iter(self.interventions)

    def __repr__(self) -> str:
        names = [i.name for i in self.interventions]
        return f"InterventionSchedule({len(self.interventions)} interventions: {names})"


# =============================================================================
# Factory for convenient creation with named dimensions
# =============================================================================


@dataclass
class ScheduleFactory:
    """Factory for creating shocks and interventions with named dimensions.

    Provides convenience methods to create shocks and interventions using
    period strings (e.g., "2020Q2") and dimension names (e.g., "ZG")
    instead of numeric indices.

    Parameters
    ----------
    base_period : str
        The period corresponding to t=0 (e.g., "2016Q1").
    row_names : list[str], optional
        Names for rows (e.g., canton codes).
    col_names : list[str], optional
        Names for columns (e.g., cost group names).

    Examples
    --------
    >>> factory = ScheduleFactory(
    ...     base_period="2016Q1",
    ...     row_names=["AG", "AI", ..., "ZH"],
    ...     col_names=["Autres", "Hôpitaux (séjours)", ...],
    ... )
    >>> covid = factory.shock("covid", start="2020Q2", end="2020Q3")
    >>> zg = factory.intervention(
    ...     "zg_policy", start="2026Q1", end="2026Q4",
    ...     rows=["ZG"], cols=["Hôpitaux (séjours)"],
    ...     intervention_type="override", value=0.0,
    ... )
    """

    base_period: str
    row_names: Optional[List[str]] = None
    col_names: Optional[List[str]] = None

    def period_to_index(self, period: str) -> int:
        """Convert period string to time index.

        Supports formats: "2020Q2" (quarterly), "2020-03" (monthly).
        """
        base_year, base_sub = self._parse_period(self.base_period)
        year, sub = self._parse_period(period)

        # Detect format from base_period
        if "Q" in self.base_period:
            return (year - base_year) * 4 + (sub - base_sub)
        else:
            return (year - base_year) * 12 + (sub - base_sub)

    def _parse_period(self, period: str) -> Tuple[int, int]:
        """Parse period string into (year, sub-period)."""
        if "Q" in period:
            # Quarterly: "2020Q2"
            year = int(period[:4])
            quarter = int(period[-1])
            return year, quarter
        elif "-" in period:
            # Monthly: "2020-03"
            parts = period.split("-")
            return int(parts[0]), int(parts[1])
        else:
            raise ValueError(f"Unknown period format: {period}")

    def _resolve_rows(self, names: Optional[List[str]]) -> Optional[List[int]]:
        """Convert row names to indices."""
        if names is None:
            return None
        if self.row_names is None:
            raise ValueError("Cannot resolve row names: row_names not configured")
        indices = []
        for name in names:
            if name not in self.row_names:
                raise ValueError(f"Unknown row name: {name}")
            indices.append(self.row_names.index(name))
        return indices

    def _resolve_cols(self, names: Optional[List[str]]) -> Optional[List[int]]:
        """Convert column names to indices."""
        if names is None:
            return None
        if self.col_names is None:
            raise ValueError("Cannot resolve col names: col_names not configured")
        indices = []
        for name in names:
            if name not in self.col_names:
                raise ValueError(f"Unknown col name: {name}")
            indices.append(self.col_names.index(name))
        return indices

    def shock(
        self,
        name: str,
        start: Union[int, str],
        end: Optional[Union[int, str]] = None,
        level: ShockLevel = ShockLevel.FACTOR,
        scope: Optional[ShockScope] = None,
        rows: Optional[List[str]] = None,
        cols: Optional[List[str]] = None,
        decay_type: DecayType = DecayType.STEP,
        decay_rate: Optional[float] = None,
        fixed_effect: Optional[np.ndarray] = None,
    ) -> Shock:
        """Create a shock with period strings and named dimensions.

        Parameters
        ----------
        name : str
            Unique identifier.
        start : int or str
            Start period (index or string like "2020Q2").
        end : int or str, optional
            End period.
        level : ShockLevel
            Factor or observation level.
        scope : ShockScope, optional
            Auto-detected from rows/cols if not provided.
        rows : list[str], optional
            Row names (e.g., ["ZG", "ZH"]).
        cols : list[str], optional
            Column names.
        decay_type : DecayType
            How shock intensity decays.
        decay_rate : float, optional
            Decay parameter for exponential decay.
        fixed_effect : np.ndarray, optional
            Pre-specified effect matrix.

        Returns
        -------
        Shock
        """
        # Convert periods to indices
        start_t = self.period_to_index(start) if isinstance(start, str) else start
        end_t = None
        if end is not None:
            end_t = self.period_to_index(end) if isinstance(end, str) else end

        # Resolve names to indices
        row_indices = self._resolve_rows(rows)
        col_indices = self._resolve_cols(cols)

        # Auto-detect scope if not provided
        if scope is None:
            if row_indices is None and col_indices is None:
                scope = ShockScope.GLOBAL
            elif row_indices is not None and col_indices is None:
                scope = ShockScope.CANTON
            elif row_indices is None and col_indices is not None:
                scope = ShockScope.CATEGORY
            else:
                scope = ShockScope.CANTON_CATEGORY

        return Shock(
            name=name,
            start_t=start_t,
            end_t=end_t,
            level=level,
            scope=scope,
            cantons=row_indices,
            categories=col_indices,
            decay_type=decay_type,
            decay_rate=decay_rate,
            fixed_effect=fixed_effect,
        )

    def intervention(
        self,
        name: str,
        start: Union[int, str],
        end: Optional[Union[int, str]] = None,
        intervention_type: InterventionType = InterventionType.OVERRIDE,
        value: Optional[float] = None,
        factor: Optional[float] = None,
        rows: Optional[List[str]] = None,
        cols: Optional[List[str]] = None,
    ) -> Intervention:
        """Create an intervention with period strings and named dimensions.

        Parameters
        ----------
        name : str
            Unique identifier.
        start : int or str
            Start period.
        end : int or str, optional
            End period.
        intervention_type : InterventionType
            OVERRIDE, SCALE, or SHIFT.
        value : float, optional
            Value for OVERRIDE or SHIFT.
        factor : float, optional
            Factor for SCALE.
        rows : list[str], optional
            Row names.
        cols : list[str], optional
            Column names.

        Returns
        -------
        Intervention
        """
        # Convert periods to indices
        start_t = self.period_to_index(start) if isinstance(start, str) else start
        end_t = None
        if end is not None:
            end_t = self.period_to_index(end) if isinstance(end, str) else end

        # Resolve names to indices
        row_indices = self._resolve_rows(rows)
        col_indices = self._resolve_cols(cols)

        return Intervention(
            name=name,
            start_t=start_t,
            end_t=end_t,
            intervention_type=intervention_type,
            value=value,
            factor=factor,
            rows=row_indices,
            cols=col_indices,
        )


# =============================================================================
# Shock effect estimation and application
# =============================================================================


def estimate_shock_effects(
    Y: np.ndarray,
    F: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    A: List[np.ndarray],
    B: List[np.ndarray],
    schedule: ShockSchedule,
    mask: Optional[np.ndarray] = None,
) -> ShockEffects:
    """Estimate shock effects from smoothed factors and observations.

    This function estimates the effect matrices for each shock by regressing
    residuals on shock indicators.

    For factor-level shocks:
        F_t - (C_drift + Σ A_l F_{t-l} B_l') = Σ_s X[t,s] * Γ_s + error

    For observation-level shocks:
        Y_t - R F_t C' = Σ_s X[t,s] * γ_s + error

    Parameters
    ----------
    Y : np.ndarray
        Observed data of shape (T, p1, p2).
    F : np.ndarray
        Smoothed factors of shape (T, k1, k2).
    R : np.ndarray
        Row loadings of shape (p1, k1).
    C : np.ndarray
        Column loadings of shape (p2, k2).
    A : list[np.ndarray]
        Row transition matrices.
    B : list[np.ndarray]
        Column transition matrices.
    schedule : ShockSchedule
        Shock schedule with shock definitions.
    mask : np.ndarray, optional
        Missing data mask (True = observed).

    Returns
    -------
    ShockEffects
        Estimated effect matrices and standard errors.
    """
    T, p1, p2 = Y.shape
    _, k1, k2 = F.shape
    P = len(A)

    effects = ShockEffects()

    # --- Estimate factor-level shock effects ---
    factor_shocks = schedule.factor_shocks
    if factor_shocks:
        n_f = len(factor_shocks)
        X_factor = schedule.build_factor_design_matrix(T)

        # Compute factor residuals (after removing AR dynamics)
        # We skip the first P periods due to lags
        F_resid = np.zeros((T - P, k1, k2))
        for t in range(P, T):
            predicted = np.zeros((k1, k2))
            for l in range(P):
                predicted += A[l] @ F[t - 1 - l] @ B[l].T
            F_resid[t - P] = F[t] - predicted

        # Estimate each shock effect separately
        Gamma = np.zeros((n_f, k1, k2))
        Gamma_se = np.zeros((n_f, k1, k2))

        for s, shock in enumerate(factor_shocks):
            if shock.fixed_effect is not None:
                Gamma[s] = shock.fixed_effect
                continue

            # Get weights for this shock (trimmed to match residuals)
            weights = X_factor[P:, s]
            active_mask = weights > 0
            n_active = active_mask.sum()

            if n_active > 0:
                # Weighted average of residuals during shock periods
                weighted_sum = np.zeros((k1, k2))
                weight_sum = 0.0
                for t_idx in range(len(weights)):
                    if weights[t_idx] > 0:
                        weighted_sum += weights[t_idx] * F_resid[t_idx]
                        weight_sum += weights[t_idx]
                if weight_sum > 0:
                    Gamma[s] = weighted_sum / weight_sum

                # Standard error estimation
                if n_active > 1:
                    # Compute variance of residuals during shock periods
                    resid_active = F_resid[active_mask]
                    var = np.var(resid_active, axis=0, ddof=1)
                    Gamma_se[s] = np.sqrt(var / n_active)

        effects.factor_effects = Gamma
        effects.factor_se = Gamma_se

    # --- Estimate observation-level shock effects ---
    obs_shocks = schedule.observation_shocks
    if obs_shocks:
        n_o = len(obs_shocks)
        X_obs = schedule.build_observation_design_matrix(T)

        # Compute observation residuals
        Y_resid = np.zeros((T, p1, p2))
        for t in range(T):
            Y_resid[t] = Y[t] - R @ F[t] @ C.T

        # Apply mask if provided
        if mask is not None:
            Y_resid = np.where(mask, Y_resid, np.nan)

        # Estimate each shock effect
        gamma = np.zeros((n_o, p1, p2))
        gamma_se = np.zeros((n_o, p1, p2))

        for s, shock in enumerate(obs_shocks):
            if shock.fixed_effect is not None:
                gamma[s] = shock.fixed_effect
                continue

            weights = X_obs[:, s]
            active_mask = weights > 0
            n_active = active_mask.sum()

            if n_active > 0:
                # Build target mask for this shock
                target_mask = shock.build_target_mask(p1, p2)

                # Weighted average of residuals, respecting target mask
                weighted_sum = np.zeros((p1, p2))
                weight_sum = 0.0
                for t in range(T):
                    if weights[t] > 0:
                        resid_t = Y_resid[t]
                        # Handle NaN from missing data
                        valid = ~np.isnan(resid_t)
                        contribution = np.where(
                            valid & (target_mask > 0),
                            weights[t] * resid_t,
                            0.0,
                        )
                        weighted_sum += contribution
                        weight_sum += weights[t]

                if weight_sum > 0:
                    gamma[s] = (weighted_sum / weight_sum) * target_mask

                # Standard error
                if n_active > 1:
                    resid_active = Y_resid[active_mask]
                    # Compute variance, handling NaN
                    var = np.nanvar(resid_active, axis=0, ddof=1)
                    gamma_se[s] = np.sqrt(var / n_active) * target_mask

        effects.observation_effects = gamma
        effects.observation_se = gamma_se

    return effects


def apply_factor_shocks(
    F_pred: np.ndarray,
    t: int,
    schedule: ShockSchedule,
    effects: ShockEffects,
) -> np.ndarray:
    """Apply factor-level shock effects to a predicted factor matrix.

    Parameters
    ----------
    F_pred : np.ndarray
        Predicted factor matrix of shape (k1, k2).
    t : int
        Time period (absolute, 0-indexed).
    schedule : ShockSchedule
        Shock schedule.
    effects : ShockEffects
        Estimated or fixed shock effects.

    Returns
    -------
    np.ndarray
        Factor matrix with shock effects added, shape (k1, k2).
    """
    if effects.factor_effects is None:
        return F_pred

    F_adjusted = F_pred.copy()
    for s, shock in enumerate(schedule.factor_shocks):
        intensity = shock.indicator(t)
        if intensity > 0 and s < effects.n_factor_shocks:
            F_adjusted += intensity * effects.factor_effects[s]

    return F_adjusted


def apply_observation_shocks(
    Y_pred: np.ndarray,
    t: int,
    schedule: ShockSchedule,
    effects: ShockEffects,
) -> np.ndarray:
    """Apply observation-level shock effects to predicted observations.

    Parameters
    ----------
    Y_pred : np.ndarray
        Predicted observations of shape (p1, p2).
    t : int
        Time period (absolute, 0-indexed).
    schedule : ShockSchedule
        Shock schedule.
    effects : ShockEffects
        Estimated or fixed shock effects.

    Returns
    -------
    np.ndarray
        Observations with shock effects added, shape (p1, p2).
    """
    if effects.observation_effects is None:
        return Y_pred

    Y_adjusted = Y_pred.copy()
    for s, shock in enumerate(schedule.observation_shocks):
        intensity = shock.indicator(t)
        if intensity > 0 and s < effects.n_observation_shocks:
            Y_adjusted += intensity * effects.observation_effects[s]

    return Y_adjusted
