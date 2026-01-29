"""Tests for KPOKPCH.DMFM.utils module.

Tests cover initialization utilities including factor loading initialization,
idiosyncratic covariance estimation, and dynamics initialization.
"""

import numpy as np
import pytest

from KPOKPCH.DMFM import (
    initialize_model,
    init_factor_loadings,
    init_idiosyncratic,
    init_dynamics,
    InitMethod,
    InitializationResult,
)

# ---------------------------------------------------------------------------
# InitMethod tests
# ---------------------------------------------------------------------------


class TestInitMethod:
    """Tests for InitMethod enum."""

    def test_svd_method(self):
        """Test SVD initialization method enum."""
        assert InitMethod.SVD.value == "svd"

    def test_pe_method(self):
        """Test principal eigenvector initialization method enum."""
        assert InitMethod.PRINCIPAL_EIGENVECTOR.value == "pe"

    def test_from_string(self):
        """Test creating InitMethod from string."""
        assert InitMethod("svd") == InitMethod.SVD
        assert InitMethod("pe") == InitMethod.PRINCIPAL_EIGENVECTOR


# ---------------------------------------------------------------------------
# InitializationResult tests
# ---------------------------------------------------------------------------


class TestInitializationResult:
    """Tests for InitializationResult dataclass."""

    def test_create_result(self):
        """Test creating InitializationResult."""
        result = InitializationResult(
            R=np.eye(4, 2),
            C=np.eye(3, 2),
            F=np.zeros((10, 2, 2)),
            H=np.eye(4),
            K=np.eye(3),
            A=[np.eye(2)],
            B=[np.eye(2)],
            Pmat=np.eye(2),
            Qmat=np.eye(2),
            method="svd",
        )

        assert result.R.shape == (4, 2)
        assert result.C.shape == (3, 2)
        assert result.F.shape == (10, 2, 2)
        assert result.method == "svd"


# ---------------------------------------------------------------------------
# init_factor_loadings tests
# ---------------------------------------------------------------------------


class TestInitFactorLoadings:
    """Tests for init_factor_loadings function."""

    @pytest.fixture
    def simple_data(self, rng, small_dims):
        """Generate simple test data."""
        T, p1, p2 = small_dims["T"], small_dims["p1"], small_dims["p2"]
        Y = rng.normal(size=(T, p1, p2))
        mask = np.ones_like(Y, dtype=bool)
        return Y, mask

    def test_svd_method_returns_correct_shapes(self, simple_data, small_dims):
        """Test SVD method returns correct shapes."""
        Y, mask = simple_data
        k1, k2 = small_dims["k1"], small_dims["k2"]

        R, C, F = init_factor_loadings(Y, mask, k1, k2, method="svd")

        assert R.shape == (small_dims["p1"], k1)
        assert C.shape == (small_dims["p2"], k2)
        assert F.shape == (small_dims["T"], k1, k2)

    def test_pe_method_returns_correct_shapes(self, simple_data, small_dims):
        """Test PE method returns correct shapes."""
        Y, mask = simple_data
        k1, k2 = small_dims["k1"], small_dims["k2"]

        R, C, F = init_factor_loadings(Y, mask, k1, k2, method="pe")

        assert R.shape == (small_dims["p1"], k1)
        assert C.shape == (small_dims["p2"], k2)
        assert F.shape == (small_dims["T"], k1, k2)

    def test_invalid_method_raises(self, simple_data, small_dims):
        """Test that invalid method raises ValueError."""
        Y, mask = simple_data

        with pytest.raises(ValueError, match="Unknown method"):
            init_factor_loadings(
                Y, mask, small_dims["k1"], small_dims["k2"], method="invalid"
            )

    def test_invalid_k1_raises(self, simple_data, small_dims):
        """Test that invalid k1 raises ValueError."""
        Y, mask = simple_data

        with pytest.raises(ValueError, match="k1 and k2 must be positive"):
            init_factor_loadings(Y, mask, k1=0, k2=small_dims["k2"])

        with pytest.raises(ValueError, match="k1 and k2 must be positive"):
            init_factor_loadings(Y, mask, k1=-1, k2=small_dims["k2"])

    def test_invalid_k2_raises(self, simple_data, small_dims):
        """Test that invalid k2 raises ValueError."""
        Y, mask = simple_data

        with pytest.raises(ValueError, match="k1 and k2 must be positive"):
            init_factor_loadings(Y, mask, k1=small_dims["k1"], k2=0)

    def test_k_exceeds_dimensions_raises(self, simple_data, small_dims):
        """Test that k exceeding dimensions raises ValueError."""
        Y, mask = simple_data

        with pytest.raises(ValueError, match="cannot exceed data dimensions"):
            init_factor_loadings(Y, mask, k1=small_dims["p1"] + 1, k2=small_dims["k2"])

        with pytest.raises(ValueError, match="cannot exceed data dimensions"):
            init_factor_loadings(Y, mask, k1=small_dims["k1"], k2=small_dims["p2"] + 1)

    def test_none_mask_treated_as_all_observed(self, rng, small_dims):
        """Test that None mask is treated as all observed."""
        T, p1, p2 = small_dims["T"], small_dims["p1"], small_dims["p2"]
        Y = rng.normal(size=(T, p1, p2))

        R, C, F = init_factor_loadings(Y, None, small_dims["k1"], small_dims["k2"])

        assert R.shape[1] == small_dims["k1"]
        assert C.shape[1] == small_dims["k2"]

    def test_handles_missing_data(self, rng, small_dims):
        """Test initialization handles missing data."""
        T, p1, p2 = small_dims["T"], small_dims["p1"], small_dims["p2"]
        Y = rng.normal(size=(T, p1, p2))
        mask = np.ones_like(Y, dtype=bool)

        # Set some values as missing
        mask[0, 0, 0] = False
        mask[5, :, 1] = False

        R, C, F = init_factor_loadings(Y, mask, small_dims["k1"], small_dims["k2"])

        # Should not contain NaN
        assert not np.isnan(R).any()
        assert not np.isnan(C).any()
        assert not np.isnan(F).any()


# ---------------------------------------------------------------------------
# init_idiosyncratic tests
# ---------------------------------------------------------------------------


class TestInitIdiosyncratic:
    """Tests for init_idiosyncratic function."""

    @pytest.fixture
    def factor_data(self, rng, small_dims):
        """Generate factor and loading data for testing."""
        T, p1, p2 = small_dims["T"], small_dims["p1"], small_dims["p2"]
        k1, k2 = small_dims["k1"], small_dims["k2"]

        R = rng.normal(size=(p1, k1))
        C = rng.normal(size=(p2, k2))
        F = rng.normal(size=(T, k1, k2))

        # Generate Y = R @ F @ C^T + noise
        noise = 0.1 * rng.normal(size=(T, p1, p2))
        Y = np.zeros((T, p1, p2))
        for t in range(T):
            Y[t] = R @ F[t] @ C.T + noise[t]

        return Y, R, C, F

    def test_returns_correct_shapes(self, factor_data, small_dims):
        """Test that function returns correct shapes."""
        Y, R, C, F = factor_data

        H, K = init_idiosyncratic(Y, R, C, F)

        assert H.shape == (small_dims["p1"], small_dims["p1"])
        assert K.shape == (small_dims["p2"], small_dims["p2"])

    def test_covariances_symmetric(self, factor_data):
        """Test that returned covariances are symmetric."""
        Y, R, C, F = factor_data

        H, K = init_idiosyncratic(Y, R, C, F)

        np.testing.assert_allclose(H, H.T, atol=1e-10)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_trace_normalization(self, factor_data, small_dims):
        """Test that traces are normalized to dimensions."""
        Y, R, C, F = factor_data

        H, K = init_idiosyncratic(Y, R, C, F)

        np.testing.assert_allclose(np.trace(H), small_dims["p1"], rtol=1e-6)
        np.testing.assert_allclose(np.trace(K), small_dims["p2"], rtol=1e-6)

    def test_covariances_nonnegative_definite(self, factor_data):
        """Test that covariances are nonnegative definite."""
        Y, R, C, F = factor_data

        H, K = init_idiosyncratic(Y, R, C, F)

        # All eigenvalues should be non-negative
        eigvals_H = np.linalg.eigvalsh(H)
        eigvals_K = np.linalg.eigvalsh(K)

        assert np.all(eigvals_H >= -1e-10)
        assert np.all(eigvals_K >= -1e-10)


# ---------------------------------------------------------------------------
# init_dynamics tests
# ---------------------------------------------------------------------------


class TestInitDynamics:
    """Tests for init_dynamics function."""

    def test_returns_correct_shapes_mar1(self, small_dims):
        """Test MAR(1) dynamics initialization shapes."""
        k1, k2 = small_dims["k1"], small_dims["k2"]

        A, B, Pmat, Qmat = init_dynamics(k1, k2, P=1)

        assert len(A) == 1
        assert len(B) == 1
        assert A[0].shape == (k1, k1)
        assert B[0].shape == (k2, k2)
        assert Pmat.shape == (k1, k1)
        assert Qmat.shape == (k2, k2)

    def test_returns_correct_shapes_mar2(self, small_dims):
        """Test MAR(2) dynamics initialization shapes."""
        k1, k2 = small_dims["k1"], small_dims["k2"]

        A, B, Pmat, Qmat = init_dynamics(k1, k2, P=2)

        assert len(A) == 2
        assert len(B) == 2
        assert all(a.shape == (k1, k1) for a in A)
        assert all(b.shape == (k2, k2) for b in B)

    def test_identity_initialization(self, small_dims):
        """Test that dynamics are initialized to identity."""
        k1, k2 = small_dims["k1"], small_dims["k2"]

        A, B, Pmat, Qmat = init_dynamics(k1, k2, P=1)

        np.testing.assert_array_equal(A[0], np.eye(k1))
        np.testing.assert_array_equal(B[0], np.eye(k2))
        np.testing.assert_array_equal(Pmat, np.eye(k1))
        np.testing.assert_array_equal(Qmat, np.eye(k2))

    def test_various_dimensions(self):
        """Test initialization with various dimensions."""
        for k1 in [1, 3, 5]:
            for k2 in [1, 2, 4]:
                for P in [1, 2, 3]:
                    A, B, Pmat, Qmat = init_dynamics(k1, k2, P)

                    assert len(A) == P
                    assert len(B) == P
                    assert Pmat.shape == (k1, k1)
                    assert Qmat.shape == (k2, k2)


# ---------------------------------------------------------------------------
# initialize_model tests
# ---------------------------------------------------------------------------


class TestInitializeModel:
    """Tests for initialize_model convenience function."""

    @pytest.fixture
    def test_data(self, rng, small_dims):
        """Generate test data."""
        T, p1, p2 = small_dims["T"], small_dims["p1"], small_dims["p2"]
        Y = rng.normal(size=(T, p1, p2))
        mask = np.ones_like(Y, dtype=bool)
        return Y, mask

    def test_returns_initialization_result(self, test_data, small_dims):
        """Test that function returns InitializationResult."""
        Y, mask = test_data

        result = initialize_model(
            Y, k1=small_dims["k1"], k2=small_dims["k2"], P=1, mask=mask
        )

        assert isinstance(result, InitializationResult)

    def test_result_has_all_parameters(self, test_data, small_dims):
        """Test that result contains all parameters."""
        Y, mask = test_data

        result = initialize_model(
            Y, k1=small_dims["k1"], k2=small_dims["k2"], P=1, mask=mask
        )

        assert result.R is not None
        assert result.C is not None
        assert result.F is not None
        assert result.H is not None
        assert result.K is not None
        assert result.A is not None
        assert result.B is not None
        assert result.Pmat is not None
        assert result.Qmat is not None
        assert result.method is not None

    def test_parameter_shapes(self, test_data, small_dims):
        """Test that all parameters have correct shapes."""
        Y, mask = test_data
        T, p1, p2 = small_dims["T"], small_dims["p1"], small_dims["p2"]
        k1, k2, P = small_dims["k1"], small_dims["k2"], 1

        result = initialize_model(Y, k1=k1, k2=k2, P=P, mask=mask)

        assert result.R.shape == (p1, k1)
        assert result.C.shape == (p2, k2)
        assert result.F.shape == (T, k1, k2)
        assert result.H.shape == (p1, p1)
        assert result.K.shape == (p2, p2)
        assert len(result.A) == P
        assert len(result.B) == P
        assert result.Pmat.shape == (k1, k1)
        assert result.Qmat.shape == (k2, k2)

    def test_method_string(self, test_data, small_dims):
        """Test initialization with string method."""
        Y, mask = test_data

        result = initialize_model(
            Y, k1=small_dims["k1"], k2=small_dims["k2"], P=1, method="svd"
        )

        assert result.method == "svd"

    def test_method_enum(self, test_data, small_dims):
        """Test initialization with enum method."""
        Y, mask = test_data

        result = initialize_model(
            Y, k1=small_dims["k1"], k2=small_dims["k2"], P=1, method=InitMethod.SVD
        )

        assert result.method == "svd"

    def test_invalid_y_shape_raises(self, rng):
        """Test that non-3D Y raises ValueError."""
        Y_2d = rng.normal(size=(10, 5))

        with pytest.raises(ValueError, match="must be 3D array"):
            initialize_model(Y_2d, k1=2, k2=2, P=1)

    def test_without_mask(self, rng, small_dims):
        """Test initialization without explicit mask."""
        T, p1, p2 = small_dims["T"], small_dims["p1"], small_dims["p2"]
        Y = rng.normal(size=(T, p1, p2))

        result = initialize_model(Y, k1=small_dims["k1"], k2=small_dims["k2"], P=1)

        assert result is not None

    def test_mar2_initialization(self, test_data, small_dims):
        """Test initialization with MAR(2)."""
        Y, mask = test_data

        result = initialize_model(
            Y, k1=small_dims["k1"], k2=small_dims["k2"], P=2, mask=mask
        )

        assert len(result.A) == 2
        assert len(result.B) == 2
