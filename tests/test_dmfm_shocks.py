"""Tests for KPOKPCH.DMFM.shocks module.

Tests cover Shock, ShockSchedule, ShockEffects, and shock estimation/application
for Dynamic Matrix Factor Models with interventions.
"""

import numpy as np
import pytest

from KPOKPCH.DMFM import (
    Shock,
    ShockSchedule,
    ShockEffects,
    ShockLevel,
    ShockScope,
    DecayType,
    estimate_shock_effects,
    apply_factor_shocks,
    apply_observation_shocks,
    DMFMModel,
    DMFMConfig,
    fit_dmfm,
)


# ---------------------------------------------------------------------------
# Shock tests
# ---------------------------------------------------------------------------


class TestShock:
    """Tests for Shock dataclass."""

    def test_basic_shock_creation(self):
        """Test creating a basic shock."""
        shock = Shock(name="test", start_t=10, end_t=15)
        assert shock.name == "test"
        assert shock.start_t == 10
        assert shock.end_t == 15
        assert shock.level == ShockLevel.FACTOR
        assert shock.scope == ShockScope.GLOBAL
        assert shock.decay_type == DecayType.STEP

    def test_permanent_shock(self):
        """Test creating a permanent shock (no end_t)."""
        shock = Shock(name="permanent", start_t=10, end_t=None)
        assert shock.end_t is None
        assert shock.is_active(10)
        assert shock.is_active(100)  # Still active far in future

    def test_observation_level_shock(self):
        """Test creating an observation-level shock."""
        shock = Shock(
            name="obs_shock",
            start_t=5,
            level=ShockLevel.OBSERVATION,
            scope=ShockScope.CATEGORY,
            categories=[1, 2],
        )
        assert shock.level == ShockLevel.OBSERVATION
        assert shock.scope == ShockScope.CATEGORY
        assert shock.categories == [1, 2]

    def test_canton_specific_shock(self):
        """Test creating a canton-specific shock."""
        shock = Shock(
            name="canton_shock",
            start_t=5,
            scope=ShockScope.CANTON,
            cantons=[0, 5, 10],
        )
        assert shock.scope == ShockScope.CANTON
        assert shock.cantons == [0, 5, 10]

    def test_canton_category_shock(self):
        """Test creating a shock targeting specific canton-category pairs."""
        shock = Shock(
            name="targeted",
            start_t=5,
            scope=ShockScope.CANTON_CATEGORY,
            cantons=[1, 2],
            categories=[3, 4],
        )
        assert shock.scope == ShockScope.CANTON_CATEGORY

    def test_indicator_step_function(self):
        """Test step function indicator."""
        shock = Shock(name="step", start_t=5, end_t=10, decay_type=DecayType.STEP)

        assert shock.indicator(4) == 0.0  # Before start
        assert shock.indicator(5) == 1.0  # At start
        assert shock.indicator(7) == 1.0  # During
        assert shock.indicator(10) == 1.0  # At end
        assert shock.indicator(11) == 0.0  # After end

    def test_indicator_exponential_decay(self):
        """Test exponential decay indicator."""
        shock = Shock(
            name="exp",
            start_t=5,
            end_t=15,
            decay_type=DecayType.EXPONENTIAL,
            decay_rate=0.8,
        )

        assert shock.indicator(5) == 1.0
        assert shock.indicator(6) == pytest.approx(0.8)
        assert shock.indicator(7) == pytest.approx(0.64)
        assert shock.indicator(8) == pytest.approx(0.512)

    def test_indicator_linear_decay(self):
        """Test linear decay indicator."""
        shock = Shock(
            name="linear",
            start_t=0,
            end_t=10,
            decay_type=DecayType.LINEAR,
        )

        assert shock.indicator(0) == 1.0
        assert shock.indicator(5) == pytest.approx(0.5)
        assert shock.indicator(10) == pytest.approx(0.0)

    def test_is_active(self):
        """Test is_active method."""
        shock = Shock(name="test", start_t=5, end_t=10)

        assert not shock.is_active(4)
        assert shock.is_active(5)
        assert shock.is_active(7)
        assert shock.is_active(10)
        assert not shock.is_active(11)

    def test_build_target_mask_global(self):
        """Test target mask for global shock."""
        shock = Shock(name="global", start_t=0, scope=ShockScope.GLOBAL)
        mask = shock.build_target_mask(p1=5, p2=3)

        assert mask.shape == (5, 3)
        assert np.all(mask == 1.0)

    def test_build_target_mask_canton(self):
        """Test target mask for canton-specific shock."""
        shock = Shock(
            name="canton",
            start_t=0,
            scope=ShockScope.CANTON,
            cantons=[1, 3],
        )
        mask = shock.build_target_mask(p1=5, p2=3)

        assert mask.shape == (5, 3)
        assert np.all(mask[1, :] == 1.0)
        assert np.all(mask[3, :] == 1.0)
        assert np.all(mask[0, :] == 0.0)
        assert np.all(mask[2, :] == 0.0)

    def test_build_target_mask_category(self):
        """Test target mask for category-specific shock."""
        shock = Shock(
            name="category",
            start_t=0,
            scope=ShockScope.CATEGORY,
            categories=[0, 2],
        )
        mask = shock.build_target_mask(p1=5, p2=3)

        assert np.all(mask[:, 0] == 1.0)
        assert np.all(mask[:, 2] == 1.0)
        assert np.all(mask[:, 1] == 0.0)

    def test_validation_canton_scope_requires_cantons(self):
        """Test that canton scope requires cantons list."""
        with pytest.raises(ValueError, match="requires cantons"):
            Shock(name="bad", start_t=0, scope=ShockScope.CANTON)

    def test_validation_category_scope_requires_categories(self):
        """Test that category scope requires categories list."""
        with pytest.raises(ValueError, match="requires categories"):
            Shock(name="bad", start_t=0, scope=ShockScope.CATEGORY)

    def test_validation_exponential_requires_decay_rate(self):
        """Test that exponential decay requires decay_rate."""
        with pytest.raises(ValueError, match="requires decay_rate"):
            Shock(
                name="bad",
                start_t=0,
                decay_type=DecayType.EXPONENTIAL,
            )

    def test_validation_linear_requires_end_t(self):
        """Test that linear decay requires finite end_t."""
        with pytest.raises(ValueError, match="requires finite end_t"):
            Shock(
                name="bad",
                start_t=0,
                end_t=None,
                decay_type=DecayType.LINEAR,
            )

    def test_validation_end_before_start(self):
        """Test that end_t >= start_t."""
        with pytest.raises(ValueError, match="must be >= start_t"):
            Shock(name="bad", start_t=10, end_t=5)


# ---------------------------------------------------------------------------
# ShockSchedule tests
# ---------------------------------------------------------------------------


class TestShockSchedule:
    """Tests for ShockSchedule."""

    def test_empty_schedule(self):
        """Test creating empty schedule."""
        schedule = ShockSchedule()
        assert len(schedule) == 0
        assert schedule.n_shocks == 0

    def test_schedule_with_shocks(self):
        """Test creating schedule with shocks."""
        shocks = [
            Shock(name="s1", start_t=5, end_t=10),
            Shock(name="s2", start_t=15, end_t=20),
        ]
        schedule = ShockSchedule(shocks)
        assert len(schedule) == 2
        assert schedule.n_shocks == 2

    def test_add_shock(self):
        """Test adding shock to schedule."""
        schedule = ShockSchedule()
        schedule.add(Shock(name="new", start_t=0))
        assert len(schedule) == 1

    def test_add_duplicate_name_raises(self):
        """Test that adding duplicate name raises error."""
        schedule = ShockSchedule([Shock(name="test", start_t=0)])
        with pytest.raises(ValueError, match="already exists"):
            schedule.add(Shock(name="test", start_t=5))

    def test_remove_shock(self):
        """Test removing shock from schedule."""
        schedule = ShockSchedule([
            Shock(name="s1", start_t=0),
            Shock(name="s2", start_t=5),
        ])
        schedule.remove("s1")
        assert len(schedule) == 1
        assert schedule.get("s1") is None
        assert schedule.get("s2") is not None

    def test_get_shock(self):
        """Test getting shock by name."""
        shock = Shock(name="target", start_t=10)
        schedule = ShockSchedule([shock])
        assert schedule.get("target") is shock
        assert schedule.get("nonexistent") is None

    def test_factor_and_observation_shocks(self):
        """Test separating factor and observation shocks."""
        schedule = ShockSchedule([
            Shock(name="f1", start_t=0, level=ShockLevel.FACTOR),
            Shock(name="f2", start_t=5, level=ShockLevel.FACTOR),
            Shock(name="o1", start_t=10, level=ShockLevel.OBSERVATION,
                  scope=ShockScope.CATEGORY, categories=[0]),
        ])
        assert schedule.n_factor_shocks == 2
        assert schedule.n_observation_shocks == 1

    def test_build_design_matrix(self):
        """Test building design matrix."""
        schedule = ShockSchedule([
            Shock(name="s1", start_t=2, end_t=4),
            Shock(name="s2", start_t=6, end_t=8),
        ])
        X = schedule.build_design_matrix(T=10)

        assert X.shape == (10, 2)
        # First shock active at t=2,3,4
        assert X[1, 0] == 0.0
        assert X[2, 0] == 1.0
        assert X[4, 0] == 1.0
        assert X[5, 0] == 0.0
        # Second shock active at t=6,7,8
        assert X[5, 1] == 0.0
        assert X[6, 1] == 1.0
        assert X[8, 1] == 1.0
        assert X[9, 1] == 0.0

    def test_build_factor_design_matrix(self):
        """Test building design matrix for factor shocks only."""
        schedule = ShockSchedule([
            Shock(name="f", start_t=0, end_t=2, level=ShockLevel.FACTOR),
            Shock(name="o", start_t=0, end_t=2, level=ShockLevel.OBSERVATION,
                  scope=ShockScope.CATEGORY, categories=[0]),
        ])
        X_f = schedule.build_factor_design_matrix(T=5)
        assert X_f.shape == (5, 1)

    def test_extend_to_forecast_horizon(self):
        """Test extending schedule to forecast horizon."""
        # Permanent shock starting at t=5
        schedule = ShockSchedule([
            Shock(name="permanent", start_t=5, end_t=None),
        ])

        X_future, extended = schedule.extend_to_forecast_horizon(
            T_hist=10, steps=5
        )

        assert X_future.shape == (5, 1)
        # Shock started at t=5, so still active in forecast
        assert np.all(X_future[:, 0] == 1.0)

    def test_extend_with_future_shocks(self):
        """Test extending with additional future shocks."""
        schedule = ShockSchedule()

        future_shocks = [
            Shock(name="future", start_t=2, end_t=4),  # Relative to forecast origin
        ]

        X_future, extended = schedule.extend_to_forecast_horizon(
            T_hist=10, steps=8, future_shocks=future_shocks
        )

        # Future shock starts at relative t=2, so absolute t=12
        # In forecast horizon (h=0..7), active at h=2,3,4
        assert X_future[1, 0] == 0.0
        assert X_future[2, 0] == 1.0
        assert X_future[4, 0] == 1.0
        assert X_future[5, 0] == 0.0

    def test_unique_names_validation(self):
        """Test that duplicate names are rejected."""
        with pytest.raises(ValueError):
            ShockSchedule([
                Shock(name="dup", start_t=0),
                Shock(name="dup", start_t=5),
            ])


# ---------------------------------------------------------------------------
# ShockEffects tests
# ---------------------------------------------------------------------------


class TestShockEffects:
    """Tests for ShockEffects dataclass."""

    def test_empty_effects(self):
        """Test empty effects."""
        effects = ShockEffects()
        assert effects.n_factor_shocks == 0
        assert effects.n_observation_shocks == 0

    def test_factor_effects(self):
        """Test effects with factor shocks."""
        factor_effects = np.random.randn(3, 2, 2)  # 3 shocks, k1=k2=2
        effects = ShockEffects(factor_effects=factor_effects)

        assert effects.n_factor_shocks == 3
        assert effects.get_factor_effect(0).shape == (2, 2)
        assert effects.get_factor_effect(5) is None  # Out of range

    def test_observation_effects(self):
        """Test effects with observation shocks."""
        obs_effects = np.random.randn(2, 10, 5)  # 2 shocks, p1=10, p2=5
        effects = ShockEffects(observation_effects=obs_effects)

        assert effects.n_observation_shocks == 2
        assert effects.get_observation_effect(0).shape == (10, 5)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestShockIntegration:
    """Integration tests for shocks with DMFM."""

    @pytest.fixture
    def synthetic_data_with_shock(self):
        """Generate synthetic data with a known shock effect."""
        np.random.seed(42)
        T, p1, p2 = 50, 10, 5
        k1, k2 = 2, 2

        # Generate true parameters
        R = np.random.randn(p1, k1)
        R, _ = np.linalg.qr(R)
        C = np.random.randn(p2, k2)
        C, _ = np.linalg.qr(C)

        # Generate factors with AR dynamics
        F = np.zeros((T, k1, k2))
        F[0] = np.random.randn(k1, k2) * 0.5
        for t in range(1, T):
            F[t] = 0.8 * F[t - 1] + np.random.randn(k1, k2) * 0.2

        # Add shock effect at t=20-25
        shock_effect = np.array([[2.0, 0.5], [0.5, 1.5]])
        for t in range(20, 26):
            F[t] += shock_effect

        # Generate observations
        Y = np.zeros((T, p1, p2))
        for t in range(T):
            Y[t] = R @ F[t] @ C.T + np.random.randn(p1, p2) * 0.1

        return {
            "Y": Y,
            "true_shock_effect": shock_effect,
            "shock_start": 20,
            "shock_end": 25,
            "k1": k1,
            "k2": k2,
        }

    def test_fit_with_shock_schedule(self, synthetic_data_with_shock):
        """Test fitting model with shock schedule."""
        data = synthetic_data_with_shock

        schedule = ShockSchedule([
            Shock(
                name="test_shock",
                start_t=data["shock_start"],
                end_t=data["shock_end"],
                level=ShockLevel.FACTOR,
            )
        ])

        model, result = fit_dmfm(
            data["Y"],
            k1=data["k1"],
            k2=data["k2"],
            max_iter=20,
            shock_schedule=schedule,
        )

        assert result.converged or result.num_iter == 20
        assert result.shock_effects is not None
        assert result.shock_effects.factor_effects is not None
        assert result.shock_effects.factor_effects.shape == (1, data["k1"], data["k2"])

    def test_apply_factor_shocks(self):
        """Test applying factor shock effects."""
        k1, k2 = 2, 2
        F_pred = np.zeros((k1, k2))

        schedule = ShockSchedule([
            Shock(name="s1", start_t=0, end_t=5),
        ])

        effects = ShockEffects(
            factor_effects=np.ones((1, k1, k2)) * 2.0
        )

        # At t=3, shock is active with intensity 1.0
        F_adjusted = apply_factor_shocks(F_pred, t=3, schedule=schedule, effects=effects)
        assert np.allclose(F_adjusted, 2.0)

        # At t=10, shock is not active
        F_adjusted = apply_factor_shocks(F_pred, t=10, schedule=schedule, effects=effects)
        assert np.allclose(F_adjusted, 0.0)

    def test_apply_observation_shocks(self):
        """Test applying observation shock effects."""
        p1, p2 = 5, 3
        Y_pred = np.zeros((p1, p2))

        schedule = ShockSchedule([
            Shock(
                name="o1",
                start_t=0,
                end_t=5,
                level=ShockLevel.OBSERVATION,
                scope=ShockScope.CATEGORY,
                categories=[1],
            ),
        ])

        effect = np.zeros((p1, p2))
        effect[:, 1] = 3.0  # Only affects category 1

        effects = ShockEffects(
            observation_effects=effect[np.newaxis, ...]  # Shape (1, p1, p2)
        )

        Y_adjusted = apply_observation_shocks(Y_pred, t=3, schedule=schedule, effects=effects)
        assert np.all(Y_adjusted[:, 1] == 3.0)
        assert np.all(Y_adjusted[:, 0] == 0.0)
        assert np.all(Y_adjusted[:, 2] == 0.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_shock():
    """Create a simple shock for testing."""
    return Shock(name="simple", start_t=5, end_t=10)


@pytest.fixture
def simple_schedule(simple_shock):
    """Create a simple schedule for testing."""
    return ShockSchedule([simple_shock])
