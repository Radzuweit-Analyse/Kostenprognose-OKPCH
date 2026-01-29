"""Tests for KPOKPCH.DMFM.dynamics module.

Tests cover DynamicsConfig validation and DMFMDynamics functionality
including MAR evolution, stability enforcement, and companion form conversion.
"""

import numpy as np
import pytest

from KPOKPCH.DMFM import DMFMDynamics, DynamicsConfig

# ---------------------------------------------------------------------------
# DynamicsConfig tests
# ---------------------------------------------------------------------------


class TestDynamicsConfig:
    """Tests for DynamicsConfig dataclass validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DynamicsConfig()
        assert config.stability_threshold == 0.99
        assert config.regularization == 1e-8
        assert config.min_denominator_norm == 1e-8
        assert config.kronecker_only is False
        assert config.i1_factors is False

    def test_custom_config(self):
        """Test configuration with custom values."""
        config = DynamicsConfig(
            stability_threshold=0.95,
            regularization=1e-6,
            min_denominator_norm=1e-6,
            kronecker_only=True,
            i1_factors=True,
        )
        assert config.stability_threshold == 0.95
        assert config.regularization == 1e-6
        assert config.kronecker_only is True
        assert config.i1_factors is True

    @pytest.mark.parametrize("threshold", [0, -0.1, -1, 1.1, 2])
    def test_invalid_stability_threshold(self, threshold):
        """Test that invalid stability thresholds raise ValueError."""
        with pytest.raises(ValueError, match="stability_threshold must be in"):
            DynamicsConfig(stability_threshold=threshold)

    def test_stability_threshold_edge_cases(self):
        """Test boundary values for stability threshold."""
        # threshold = 1 should work
        config = DynamicsConfig(stability_threshold=1.0)
        assert config.stability_threshold == 1.0

        # threshold near 0 should work
        config = DynamicsConfig(stability_threshold=0.01)
        assert config.stability_threshold == 0.01

    @pytest.mark.parametrize("reg", [-1, -0.001])
    def test_negative_regularization(self, reg):
        """Test that negative regularization raises ValueError."""
        with pytest.raises(ValueError, match="regularization must be non-negative"):
            DynamicsConfig(regularization=reg)

    def test_zero_regularization(self):
        """Test that zero regularization is allowed."""
        config = DynamicsConfig(regularization=0)
        assert config.regularization == 0

    def test_nonstationary_alias(self):
        """Test that nonstationary is an alias for i1_factors."""
        config = DynamicsConfig(i1_factors=True)
        assert config.nonstationary is True

        config.nonstationary = False
        assert config.i1_factors is False
        assert config.nonstationary is False


# ---------------------------------------------------------------------------
# DMFMDynamics creation tests
# ---------------------------------------------------------------------------


class TestDMFMDynamicsCreation:
    """Tests for DMFMDynamics initialization and validation."""

    @pytest.fixture
    def simple_dynamics_params(self):
        """Simple MAR(1) dynamics parameters."""
        k1, k2 = 2, 3
        A = [0.5 * np.eye(k1)]
        B = [0.5 * np.eye(k2)]
        Pmat = np.eye(k1)
        Qmat = np.eye(k2)
        return {"A": A, "B": B, "Pmat": Pmat, "Qmat": Qmat, "k1": k1, "k2": k2}

    def test_create_mar1(self, simple_dynamics_params):
        """Test creating MAR(1) dynamics."""
        p = simple_dynamics_params
        dynamics = DMFMDynamics(p["A"], p["B"], p["Pmat"], p["Qmat"])

        assert dynamics.P == 1
        assert dynamics.k1 == p["k1"]
        assert dynamics.k2 == p["k2"]
        assert len(dynamics.A) == 1
        assert len(dynamics.B) == 1

    def test_create_mar2(self):
        """Test creating MAR(2) dynamics."""
        k1, k2 = 2, 2
        A = [0.3 * np.eye(k1), 0.2 * np.eye(k1)]
        B = [0.3 * np.eye(k2), 0.2 * np.eye(k2)]
        Pmat = np.eye(k1)
        Qmat = np.eye(k2)

        dynamics = DMFMDynamics(A, B, Pmat, Qmat)

        assert dynamics.P == 2
        assert len(dynamics.A) == 2
        assert len(dynamics.B) == 2

    def test_create_with_drift(self, simple_dynamics_params):
        """Test creating dynamics with drift matrix."""
        p = simple_dynamics_params
        drift = np.ones((p["k1"], p["k2"]))
        dynamics = DMFMDynamics(p["A"], p["B"], p["Pmat"], p["Qmat"], C=drift)

        np.testing.assert_array_equal(dynamics.C_drift, drift)

    def test_drift_defaults_to_zeros(self, simple_dynamics_params):
        """Test that drift defaults to zeros when not specified."""
        p = simple_dynamics_params
        dynamics = DMFMDynamics(p["A"], p["B"], p["Pmat"], p["Qmat"])

        expected_drift = np.zeros((p["k1"], p["k2"]))
        np.testing.assert_array_equal(dynamics.C_drift, expected_drift)

    def test_invalid_drift_shape(self, simple_dynamics_params):
        """Test that wrong drift shape raises ValueError."""
        p = simple_dynamics_params
        wrong_drift = np.ones((p["k1"] + 1, p["k2"]))

        with pytest.raises(ValueError, match="Drift C has shape"):
            DMFMDynamics(p["A"], p["B"], p["Pmat"], p["Qmat"], C=wrong_drift)

    def test_mismatched_ab_lengths(self):
        """Test that mismatched A and B lengths raise ValueError."""
        k1, k2 = 2, 2
        A = [0.5 * np.eye(k1), 0.3 * np.eye(k1)]  # 2 matrices
        B = [0.5 * np.eye(k2)]  # 1 matrix

        with pytest.raises(ValueError, match="A and B must have same length"):
            DMFMDynamics(A, B, np.eye(k1), np.eye(k2))

    def test_empty_ab_raises(self):
        """Test that empty A and B raises ValueError."""
        with pytest.raises(ValueError, match="must contain at least one matrix"):
            DMFMDynamics([], [], np.eye(2), np.eye(2))

    def test_inconsistent_a_shapes(self):
        """Test that inconsistent A matrix shapes raise ValueError."""
        A = [np.eye(2), np.eye(3)]  # Different sizes
        B = [np.eye(2), np.eye(2)]

        with pytest.raises(ValueError, match=r"A\[1\] has shape"):
            DMFMDynamics(A, B, np.eye(2), np.eye(2))

    def test_inconsistent_b_shapes(self):
        """Test that inconsistent B matrix shapes raise ValueError."""
        A = [np.eye(2), np.eye(2)]
        B = [np.eye(3), np.eye(2)]  # Different sizes

        with pytest.raises(ValueError, match=r"B\[1\] has shape"):
            DMFMDynamics(A, B, np.eye(2), np.eye(3))

    def test_wrong_pmat_shape(self):
        """Test that wrong Pmat shape raises ValueError."""
        k1, k2 = 2, 3
        A = [np.eye(k1)]
        B = [np.eye(k2)]

        with pytest.raises(ValueError, match="Pmat has shape"):
            DMFMDynamics(A, B, np.eye(k1 + 1), np.eye(k2))

    def test_wrong_qmat_shape(self):
        """Test that wrong Qmat shape raises ValueError."""
        k1, k2 = 2, 3
        A = [np.eye(k1)]
        B = [np.eye(k2)]

        with pytest.raises(ValueError, match="Qmat has shape"):
            DMFMDynamics(A, B, np.eye(k1), np.eye(k2 + 1))

    def test_custom_config(self, simple_dynamics_params):
        """Test creating dynamics with custom config."""
        p = simple_dynamics_params
        config = DynamicsConfig(stability_threshold=0.9, i1_factors=True)

        dynamics = DMFMDynamics(p["A"], p["B"], p["Pmat"], p["Qmat"], config=config)

        assert dynamics.config.stability_threshold == 0.9
        assert dynamics.i1_factors is True


# ---------------------------------------------------------------------------
# DMFMDynamics property tests
# ---------------------------------------------------------------------------


class TestDMFMDynamicsProperties:
    """Tests for DMFMDynamics property accessors."""

    @pytest.fixture
    def dynamics(self):
        """Create a simple dynamics object."""
        k1, k2 = 2, 3
        A = [0.5 * np.eye(k1)]
        B = [0.5 * np.eye(k2)]
        return DMFMDynamics(A, B, np.eye(k1), np.eye(k2))

    def test_i1_factors_property(self, dynamics):
        """Test i1_factors property getter and setter."""
        assert dynamics.i1_factors is False

        dynamics.i1_factors = True
        assert dynamics.i1_factors is True
        assert dynamics.config.i1_factors is True

    def test_nonstationary_alias(self, dynamics):
        """Test nonstationary as alias for i1_factors."""
        assert dynamics.nonstationary is False

        dynamics.nonstationary = True
        assert dynamics.nonstationary is True
        assert dynamics.i1_factors is True

    def test_kronecker_only_property(self, dynamics):
        """Test kronecker_only property getter and setter."""
        assert dynamics.kronecker_only is False

        dynamics.kronecker_only = True
        assert dynamics.kronecker_only is True
        assert dynamics.config.kronecker_only is True


# ---------------------------------------------------------------------------
# DMFMDynamics evolve tests
# ---------------------------------------------------------------------------


class TestDMFMDynamicsEvolve:
    """Tests for DMFMDynamics.evolve() method."""

    @pytest.fixture
    def mar1_dynamics(self):
        """Create MAR(1) dynamics for testing."""
        k1, k2 = 2, 3
        A = [0.5 * np.eye(k1)]
        B = [0.5 * np.eye(k2)]
        Pmat = 0.1 * np.eye(k1)
        Qmat = 0.1 * np.eye(k2)
        return DMFMDynamics(A, B, Pmat, Qmat)

    @pytest.fixture
    def mar2_dynamics(self):
        """Create MAR(2) dynamics for testing."""
        k1, k2 = 2, 2
        A = [0.4 * np.eye(k1), 0.2 * np.eye(k1)]
        B = [0.4 * np.eye(k2), 0.2 * np.eye(k2)]
        Pmat = 0.1 * np.eye(k1)
        Qmat = 0.1 * np.eye(k2)
        return DMFMDynamics(A, B, Pmat, Qmat)

    def test_evolve_mar1_deterministic(self, mar1_dynamics):
        """Test MAR(1) evolution without noise."""
        k1, k2 = mar1_dynamics.k1, mar1_dynamics.k2
        F_prev = np.ones((k1, k2))

        F_next = mar1_dynamics.evolve([F_prev], add_noise=False)

        # F_next = A @ F_prev @ B^T = 0.5 * I @ ones @ 0.5 * I = 0.25 * ones
        expected = 0.25 * np.ones((k1, k2))
        np.testing.assert_allclose(F_next, expected)

    def test_evolve_with_drift(self):
        """Test evolution includes drift term."""
        k1, k2 = 2, 2
        A = [0.5 * np.eye(k1)]
        B = [0.5 * np.eye(k2)]
        drift = np.ones((k1, k2))
        dynamics = DMFMDynamics(A, B, np.eye(k1), np.eye(k2), C=drift)

        F_prev = np.zeros((k1, k2))
        F_next = dynamics.evolve([F_prev], add_noise=False)

        # F_next = drift + A @ zeros @ B^T = drift
        np.testing.assert_allclose(F_next, drift)

    def test_evolve_mar2(self, mar2_dynamics):
        """Test MAR(2) evolution uses both lags."""
        k1, k2 = mar2_dynamics.k1, mar2_dynamics.k2
        F_lag1 = np.ones((k1, k2))
        F_lag2 = 2 * np.ones((k1, k2))

        F_next = mar2_dynamics.evolve([F_lag1, F_lag2], add_noise=False)

        # F_next = A1 @ F_lag1 @ B1^T + A2 @ F_lag2 @ B2^T
        # = 0.4*I @ ones @ 0.4*I + 0.2*I @ 2*ones @ 0.2*I
        # = 0.16*ones + 0.08*ones = 0.24*ones
        expected = 0.24 * np.ones((k1, k2))
        np.testing.assert_allclose(F_next, expected, atol=1e-10)

    def test_evolve_wrong_history_length(self, mar2_dynamics):
        """Test that wrong history length raises ValueError."""
        k1, k2 = mar2_dynamics.k1, mar2_dynamics.k2
        F_history = [np.ones((k1, k2))]  # Only 1 lag, but MAR(2) needs 2

        with pytest.raises(ValueError, match="F_history must have length P=2"):
            mar2_dynamics.evolve(F_history)

    def test_evolve_wrong_factor_shape(self, mar1_dynamics):
        """Test that wrong factor shape raises ValueError."""
        wrong_shape = np.ones((mar1_dynamics.k1 + 1, mar1_dynamics.k2))

        with pytest.raises(ValueError, match="F_history"):
            mar1_dynamics.evolve([wrong_shape])

    def test_evolve_with_noise_shape(self, mar1_dynamics, rng):
        """Test that evolve with noise returns correct shape."""
        np.random.seed(42)  # For reproducibility
        k1, k2 = mar1_dynamics.k1, mar1_dynamics.k2
        F_prev = np.zeros((k1, k2))

        F_next = mar1_dynamics.evolve([F_prev], add_noise=True)

        assert F_next.shape == (k1, k2)

    def test_evolve_with_noise_is_random(self, mar1_dynamics):
        """Test that evolve with noise gives different results each time."""
        k1, k2 = mar1_dynamics.k1, mar1_dynamics.k2
        F_prev = np.zeros((k1, k2))

        results = [mar1_dynamics.evolve([F_prev], add_noise=True) for _ in range(5)]

        # Check that not all results are identical
        all_same = all(np.allclose(results[0], r) for r in results[1:])
        assert not all_same, "Noisy evolution should produce different results"


# ---------------------------------------------------------------------------
# DMFMDynamics stability tests
# ---------------------------------------------------------------------------


class TestDMFMDynamicsStability:
    """Tests for stability checking and enforcement."""

    def test_stable_dynamics(self):
        """Test that stable dynamics are detected as stable."""
        k1, k2 = 2, 2
        A = [0.5 * np.eye(k1)]  # Spectral radius 0.5
        B = [0.5 * np.eye(k2)]
        dynamics = DMFMDynamics(A, B, np.eye(k1), np.eye(k2))

        is_stable, max_eval = dynamics.check_stability()

        assert is_stable
        assert max_eval < 1.0

    def test_unstable_dynamics(self):
        """Test that unstable dynamics are detected as unstable."""
        k1, k2 = 2, 2
        A = [1.5 * np.eye(k1)]  # Spectral radius 1.5
        B = [1.5 * np.eye(k2)]
        dynamics = DMFMDynamics(A, B, np.eye(k1), np.eye(k2))

        is_stable, max_eval = dynamics.check_stability()

        assert not is_stable
        assert max_eval > 1.0

    def test_enforce_stability(self):
        """Test that _enforce_stability limits spectral radius."""
        k1, k2 = 2, 2
        # Create dynamics with unstable A matrix
        A_unstable = np.array([[1.5, 0.2], [0.1, 1.3]])
        A = [A_unstable]
        B = [0.5 * np.eye(k2)]
        config = DynamicsConfig(stability_threshold=0.99)
        dynamics = DMFMDynamics(A, B, np.eye(k1), np.eye(k2), config=config)

        # Enforce stability
        A_stable = dynamics._enforce_stability(A_unstable)
        eigvals = np.linalg.eigvals(A_stable)
        max_eval = np.max(np.abs(eigvals))

        # Should be scaled to sqrt(0.99) ≈ 0.995
        assert max_eval <= np.sqrt(0.99) + 1e-10

    def test_already_stable_unchanged(self):
        """Test that already stable matrices are unchanged."""
        k1 = 2
        A_stable = 0.3 * np.eye(k1)
        config = DynamicsConfig(stability_threshold=0.99)
        dynamics = DMFMDynamics(
            [A_stable], [np.eye(k1)], np.eye(k1), np.eye(k1), config=config
        )

        A_result = dynamics._enforce_stability(A_stable)

        np.testing.assert_array_almost_equal(A_result, A_stable)


# ---------------------------------------------------------------------------
# DMFMDynamics companion form tests
# ---------------------------------------------------------------------------


class TestDMFMDynamicsCompanionForm:
    """Tests for VAR(1) companion form conversion."""

    def test_to_var1_mar1(self):
        """Test companion form for MAR(1)."""
        k1, k2 = 2, 2
        A = [0.5 * np.eye(k1)]
        B = [0.6 * np.eye(k2)]
        Pmat = 0.1 * np.eye(k1)
        Qmat = 0.2 * np.eye(k2)
        dynamics = DMFMDynamics(A, B, Pmat, Qmat)

        T, mu, Sigma = dynamics.to_var1()

        r = k1 * k2
        assert T.shape == (r, r)
        assert mu.shape == (r,)
        assert Sigma.shape == (r, r)

        # For MAR(1), T = B ⊗ A
        expected_T = np.kron(B[0], A[0])
        np.testing.assert_allclose(T, expected_T)

    def test_to_var1_mar2(self):
        """Test companion form for MAR(2)."""
        k1, k2 = 2, 2
        A = [0.3 * np.eye(k1), 0.2 * np.eye(k1)]
        B = [0.4 * np.eye(k2), 0.3 * np.eye(k2)]
        dynamics = DMFMDynamics(A, B, np.eye(k1), np.eye(k2))

        T, mu, Sigma = dynamics.to_var1()

        r = k1 * k2
        d = r * 2  # P=2
        assert T.shape == (d, d)

        # Check companion structure:
        # T = [[Phi1, Phi2],
        #      [I,    0  ]]
        Phi1 = np.kron(B[0], A[0])
        Phi2 = np.kron(B[1], A[1])
        np.testing.assert_allclose(T[:r, :r], Phi1)
        np.testing.assert_allclose(T[:r, r:], Phi2)
        np.testing.assert_allclose(T[r:, :r], np.eye(r))
        np.testing.assert_allclose(T[r:, r:], np.zeros((r, r)))

    def test_to_var1_with_drift(self):
        """Test companion form includes drift in mu."""
        k1, k2 = 2, 2
        drift = np.array([[1.0, 2.0], [3.0, 4.0]])
        dynamics = DMFMDynamics(
            [0.5 * np.eye(k1)], [0.5 * np.eye(k2)], np.eye(k1), np.eye(k2), C=drift
        )

        T, mu, Sigma = dynamics.to_var1()

        # Check that drift appears in first block of mu
        r = k1 * k2
        expected_mu_block = drift.ravel()
        np.testing.assert_allclose(mu[:r], expected_mu_block)

    def test_to_var1_innovation_covariance(self):
        """Test that Sigma contains Q ⊗ P in first block."""
        k1, k2 = 2, 3
        Pmat = 0.5 * np.eye(k1)
        Qmat = 0.3 * np.eye(k2)
        dynamics = DMFMDynamics([0.5 * np.eye(k1)], [0.5 * np.eye(k2)], Pmat, Qmat)

        T, mu, Sigma = dynamics.to_var1()

        r = k1 * k2
        expected_cov = np.kron(Qmat, Pmat)
        np.testing.assert_allclose(Sigma[:r, :r], expected_cov)


# ---------------------------------------------------------------------------
# DMFMDynamics estimate tests
# ---------------------------------------------------------------------------


class TestDMFMDynamicsEstimate:
    """Tests for DMFMDynamics.estimate() method."""

    @pytest.fixture
    def dynamics_for_estimation(self):
        """Create dynamics object for estimation tests."""
        k1, k2 = 2, 2
        A = [0.5 * np.eye(k1)]
        B = [0.5 * np.eye(k2)]
        return DMFMDynamics(A, B, np.eye(k1), np.eye(k2))

    def test_estimate_wrong_ndim(self, dynamics_for_estimation):
        """Test that 2D input raises ValueError."""
        F_2d = np.random.randn(10, 4)

        with pytest.raises(ValueError, match="F must be 3D array"):
            dynamics_for_estimation.estimate(F_2d)

    def test_estimate_wrong_k1(self, dynamics_for_estimation):
        """Test that wrong k1 dimension raises ValueError."""
        k1, k2 = dynamics_for_estimation.k1, dynamics_for_estimation.k2
        F_wrong = np.random.randn(20, k1 + 1, k2)

        with pytest.raises(ValueError, match="F has dimensions"):
            dynamics_for_estimation.estimate(F_wrong)

    def test_estimate_insufficient_time(self, dynamics_for_estimation):
        """Test that insufficient time steps raise ValueError."""
        k1, k2 = dynamics_for_estimation.k1, dynamics_for_estimation.k2
        P = dynamics_for_estimation.P
        F_short = np.random.randn(P, k1, k2)  # Need P+1 at minimum

        with pytest.raises(ValueError, match="Need at least"):
            dynamics_for_estimation.estimate(F_short)

    def test_estimate_updates_parameters(self, rng):
        """Test that estimate() updates A and B matrices."""
        k1, k2, T = 2, 2, 50

        # Create true dynamics
        A_true = [0.6 * np.eye(k1)]
        B_true = [0.6 * np.eye(k2)]
        dynamics_true = DMFMDynamics(A_true, B_true, 0.1 * np.eye(k1), 0.1 * np.eye(k2))

        # Generate data from true dynamics
        F = np.zeros((T, k1, k2))
        for t in range(1, T):
            F[t] = dynamics_true.evolve([F[t - 1]], add_noise=True)

        # Estimate from initialized dynamics (different from true)
        A_init = [0.3 * np.eye(k1)]
        B_init = [0.3 * np.eye(k2)]
        dynamics_est = DMFMDynamics(A_init, B_init, 0.1 * np.eye(k1), 0.1 * np.eye(k2))

        # Store initial values
        A_before = dynamics_est.A[0].copy()

        # Estimate
        dynamics_est.estimate(F)

        # Check that parameters changed
        A_after = dynamics_est.A[0]
        assert not np.allclose(A_before, A_after), "A should have been updated"


# ---------------------------------------------------------------------------
# DMFMDynamics repr tests
# ---------------------------------------------------------------------------


class TestDMFMDynamicsRepr:
    """Tests for DMFMDynamics string representation."""

    def test_repr_stable(self):
        """Test repr for stable dynamics."""
        k1, k2 = 2, 2
        dynamics = DMFMDynamics(
            [0.5 * np.eye(k1)], [0.5 * np.eye(k2)], np.eye(k1), np.eye(k2)
        )

        repr_str = repr(dynamics)

        assert "DMFMDynamics" in repr_str
        assert "P=1" in repr_str
        assert "k1=2" in repr_str
        assert "k2=2" in repr_str
        assert "stable" in repr_str

    def test_repr_unstable(self):
        """Test repr for unstable dynamics."""
        k1, k2 = 2, 2
        dynamics = DMFMDynamics(
            [1.5 * np.eye(k1)], [1.5 * np.eye(k2)], np.eye(k1), np.eye(k2)
        )

        repr_str = repr(dynamics)

        assert "unstable" in repr_str
