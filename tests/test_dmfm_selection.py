"""Tests for KPOKPCH.DMFM.selection module.

Tests cover information criteria (AIC, BIC, AICc), parameter counting,
and automatic rank selection for DMFM models.
"""

import numpy as np
import pytest

from KPOKPCH.DMFM import (
    DMFMModel,
    DMFMConfig,
    fit_dmfm,
    select_rank,
    compute_information_criteria,
    count_parameters,
    print_selection_summary,
    ModelSelectionResult,
    InformationCriteria,
)

# ---------------------------------------------------------------------------
# InformationCriteria tests
# ---------------------------------------------------------------------------


class TestInformationCriteria:
    """Tests for InformationCriteria dataclass."""

    def test_create_criteria(self):
        """Test creating InformationCriteria."""
        ic = InformationCriteria(
            loglik=-500.0,
            n_params=20,
            n_obs=1000,
            aic=1040.0,
            bic=1138.0,
            aicc=1041.0,
        )
        assert ic.loglik == -500.0
        assert ic.n_params == 20
        assert ic.n_obs == 1000
        assert ic.aic == 1040.0
        assert ic.bic == 1138.0
        assert ic.aicc == 1041.0


# ---------------------------------------------------------------------------
# ModelSelectionResult tests
# ---------------------------------------------------------------------------


class TestModelSelectionResult:
    """Tests for ModelSelectionResult dataclass."""

    def test_create_result(self, initialized_model):
        """Test creating ModelSelectionResult."""
        from KPOKPCH.DMFM import EMResult

        em_result = EMResult(
            converged=True,
            num_iter=10,
            final_loglik=-500.0,
            loglik_trace=[-600.0, -500.0],
            diff_trace=[0.1, 0.001],
        )

        result = ModelSelectionResult(
            best_k1=2,
            best_k2=3,
            best_model=initialized_model,
            best_em_result=em_result,
            criterion="bic",
            best_value=1000.0,
            results_grid={(2, 3): {"ic": None}},
        )

        assert result.best_k1 == 2
        assert result.best_k2 == 3
        assert result.criterion == "bic"
        assert result.best_value == 1000.0


# ---------------------------------------------------------------------------
# count_parameters tests
# ---------------------------------------------------------------------------


class TestCountParameters:
    """Tests for count_parameters function."""

    def test_count_basic(self, initialized_model):
        """Test basic parameter counting."""
        n_params = count_parameters(initialized_model)
        assert isinstance(n_params, int)
        assert n_params > 0

    def test_count_increases_with_factors(self, rng, small_dims):
        """Test that more factors means more parameters."""
        from conftest import generate_dmfm_data

        data = generate_dmfm_data(
            T=small_dims["T"],
            p1=small_dims["p1"],
            p2=small_dims["p2"],
            k1=1,
            k2=1,
            rng=rng,
        )

        # Model with k1=1, k2=1
        config1 = DMFMConfig(
            p1=small_dims["p1"],
            p2=small_dims["p2"],
            k1=1,
            k2=1,
        )
        model1 = DMFMModel(config1)
        model1.initialize(data["Y"], data["mask"])

        # Model with k1=2, k2=2
        data2 = generate_dmfm_data(
            T=small_dims["T"],
            p1=small_dims["p1"],
            p2=small_dims["p2"],
            k1=2,
            k2=2,
            rng=rng,
        )
        config2 = DMFMConfig(
            p1=small_dims["p1"],
            p2=small_dims["p2"],
            k1=2,
            k2=2,
        )
        model2 = DMFMModel(config2)
        model2.initialize(data2["Y"], data2["mask"])

        n1 = count_parameters(model1)
        n2 = count_parameters(model2)

        assert n2 > n1

    def test_count_without_dynamics(self, initialized_model):
        """Test parameter counting without dynamics."""
        n_with = count_parameters(initialized_model, include_dynamics=True)
        n_without = count_parameters(initialized_model, include_dynamics=False)

        assert n_without < n_with

    def test_count_without_drift(self, initialized_model):
        """Test parameter counting without drift."""
        n_with = count_parameters(initialized_model, include_drift=True)
        n_without = count_parameters(initialized_model, include_drift=False)

        assert n_without < n_with


# ---------------------------------------------------------------------------
# compute_information_criteria tests
# ---------------------------------------------------------------------------


class TestComputeInformationCriteria:
    """Tests for compute_information_criteria function."""

    @pytest.fixture
    def fitted_model(self, rng, small_dims):
        """Create a fitted model for testing."""
        from conftest import generate_dmfm_data

        data = generate_dmfm_data(
            T=small_dims["T"],
            p1=small_dims["p1"],
            p2=small_dims["p2"],
            k1=small_dims["k1"],
            k2=small_dims["k2"],
            rng=rng,
        )

        model, _ = fit_dmfm(
            data["Y"],
            k1=small_dims["k1"],
            k2=small_dims["k2"],
            mask=data["mask"],
            max_iter=5,
        )
        return model, data["Y"], data["mask"]

    def test_returns_criteria(self, fitted_model):
        """Test that function returns InformationCriteria."""
        model, Y, mask = fitted_model
        ic = compute_information_criteria(model, Y, mask)

        assert isinstance(ic, InformationCriteria)

    def test_criteria_values_reasonable(self, fitted_model):
        """Test that criteria values are reasonable."""
        model, Y, mask = fitted_model
        ic = compute_information_criteria(model, Y, mask)

        # Log-likelihood should be negative
        assert ic.loglik < 0

        # AIC should be > -2 * loglik (has penalty)
        assert ic.aic > -2 * ic.loglik

        # BIC penalty is log(n) which is > 2 for reasonable n
        # So BIC should be > AIC
        assert ic.bic > ic.aic

        # n_obs should be positive
        assert ic.n_obs > 0

        # n_params should be positive
        assert ic.n_params > 0

    def test_with_precomputed_loglik(self, fitted_model):
        """Test computing criteria with pre-computed log-likelihood."""
        model, Y, mask = fitted_model

        # Compute with and without precomputed loglik
        ic1 = compute_information_criteria(model, Y, mask)
        ic2 = compute_information_criteria(model, Y, mask, loglik=ic1.loglik)

        # Should give same results
        np.testing.assert_allclose(ic1.aic, ic2.aic, rtol=1e-10)
        np.testing.assert_allclose(ic1.bic, ic2.bic, rtol=1e-10)

    def test_without_mask(self, fitted_model):
        """Test computing criteria without explicit mask."""
        model, Y, _ = fitted_model
        ic = compute_information_criteria(model, Y)

        assert ic is not None
        assert ic.n_obs == Y.size  # All observations


# ---------------------------------------------------------------------------
# select_rank tests
# ---------------------------------------------------------------------------


class TestSelectRank:
    """Tests for select_rank function."""

    @pytest.fixture
    def selection_data(self, rng, small_dims):
        """Generate data for rank selection tests."""
        from conftest import generate_dmfm_data

        return generate_dmfm_data(
            T=small_dims["T"],
            p1=small_dims["p1"],
            p2=small_dims["p2"],
            k1=2,
            k2=2,
            rng=rng,
        )

    def test_returns_result(self, selection_data):
        """Test that select_rank returns ModelSelectionResult."""
        result = select_rank(
            selection_data["Y"],
            k1_range=(1, 2),
            k2_range=(1, 2),
            mask=selection_data["mask"],
            max_iter=3,
        )

        assert isinstance(result, ModelSelectionResult)

    def test_result_has_best_model(self, selection_data):
        """Test that result includes best model."""
        result = select_rank(
            selection_data["Y"],
            k1_range=(1, 2),
            k2_range=(1, 2),
            mask=selection_data["mask"],
            max_iter=3,
        )

        assert result.best_model is not None
        assert result.best_model.is_fitted()
        assert 1 <= result.best_k1 <= 2
        assert 1 <= result.best_k2 <= 2

    def test_result_has_grid(self, selection_data):
        """Test that result includes full results grid."""
        result = select_rank(
            selection_data["Y"],
            k1_range=(1, 2),
            k2_range=(1, 2),
            mask=selection_data["mask"],
            max_iter=3,
        )

        # Should have 4 combinations: (1,1), (1,2), (2,1), (2,2)
        assert len(result.results_grid) == 4
        assert (1, 1) in result.results_grid
        assert (2, 2) in result.results_grid

    def test_list_ranges(self, selection_data):
        """Test that list ranges work."""
        result = select_rank(
            selection_data["Y"],
            k1_range=[1, 2],
            k2_range=[1, 2],
            mask=selection_data["mask"],
            max_iter=3,
        )

        assert result is not None

    def test_criterion_aic(self, selection_data):
        """Test selection with AIC criterion."""
        result = select_rank(
            selection_data["Y"],
            k1_range=(1, 2),
            k2_range=(1, 2),
            criterion="aic",
            mask=selection_data["mask"],
            max_iter=3,
        )

        assert result.criterion == "aic"

    def test_criterion_bic(self, selection_data):
        """Test selection with BIC criterion."""
        result = select_rank(
            selection_data["Y"],
            k1_range=(1, 2),
            k2_range=(1, 2),
            criterion="bic",
            mask=selection_data["mask"],
            max_iter=3,
        )

        assert result.criterion == "bic"

    def test_criterion_aicc(self, selection_data):
        """Test selection with AICc criterion."""
        result = select_rank(
            selection_data["Y"],
            k1_range=(1, 2),
            k2_range=(1, 2),
            criterion="aicc",
            mask=selection_data["mask"],
            max_iter=3,
        )

        assert result.criterion == "aicc"

    def test_invalid_criterion_raises(self, selection_data):
        """Test that invalid criterion raises error."""
        # Invalid criterion causes all models to fail during evaluation
        # which results in RuntimeError
        with pytest.raises((ValueError, RuntimeError)):
            select_rank(
                selection_data["Y"],
                k1_range=(1, 2),
                k2_range=(1, 2),
                criterion="invalid",
                mask=selection_data["mask"],
                max_iter=3,
            )

    def test_callback_called(self, selection_data):
        """Test that callback is called for each combination."""
        callback_calls = []

        def callback(k1, k2, ic):
            callback_calls.append((k1, k2))

        select_rank(
            selection_data["Y"],
            k1_range=(1, 2),
            k2_range=(1, 2),
            mask=selection_data["mask"],
            max_iter=3,
            callback=callback,
        )

        assert len(callback_calls) == 4
        assert (1, 1) in callback_calls
        assert (2, 2) in callback_calls

    def test_empty_range_raises(self, selection_data):
        """Test that empty valid range raises ValueError."""
        with pytest.raises(ValueError, match="No valid"):
            select_rank(
                selection_data["Y"],
                k1_range=(100, 200),  # Much larger than p1
                k2_range=(1, 2),
                mask=selection_data["mask"],
            )

    def test_filters_invalid_k_values(self, selection_data, small_dims):
        """Test that k values exceeding dimensions are filtered."""
        result = select_rank(
            selection_data["Y"],
            k1_range=(1, 10),  # Some values exceed p1
            k2_range=(1, 10),  # Some values exceed p2
            mask=selection_data["mask"],
            max_iter=3,
        )

        # Best values should be within valid range
        assert result.best_k1 <= small_dims["p1"]
        assert result.best_k2 <= small_dims["p2"]


# ---------------------------------------------------------------------------
# print_selection_summary tests
# ---------------------------------------------------------------------------


class TestPrintSelectionSummary:
    """Tests for print_selection_summary function."""

    def test_prints_without_error(self, rng, small_dims, capsys):
        """Test that print_selection_summary runs without error."""
        from conftest import generate_dmfm_data

        data = generate_dmfm_data(
            T=small_dims["T"],
            p1=small_dims["p1"],
            p2=small_dims["p2"],
            k1=2,
            k2=2,
            rng=rng,
        )

        result = select_rank(
            data["Y"],
            k1_range=(1, 2),
            k2_range=(1, 2),
            mask=data["mask"],
            max_iter=3,
        )

        # Should not raise
        print_selection_summary(result)

        # Capture output
        captured = capsys.readouterr()
        assert "Model Selection Summary" in captured.out
        assert "BIC" in captured.out  # Default criterion


# ---------------------------------------------------------------------------
# fit_dmfm convenience function tests
# ---------------------------------------------------------------------------


class TestFitDMFM:
    """Tests for fit_dmfm convenience function."""

    def test_returns_model_and_result(self, rng, small_dims):
        """Test that fit_dmfm returns model and result."""
        from conftest import generate_dmfm_data

        data = generate_dmfm_data(
            T=small_dims["T"],
            p1=small_dims["p1"],
            p2=small_dims["p2"],
            k1=small_dims["k1"],
            k2=small_dims["k2"],
            rng=rng,
        )

        model, result = fit_dmfm(
            data["Y"],
            k1=small_dims["k1"],
            k2=small_dims["k2"],
            mask=data["mask"],
            max_iter=3,
        )

        assert isinstance(model, DMFMModel)
        assert model.is_fitted()
        assert result.num_iter <= 3

    def test_respects_parameters(self, rng, small_dims):
        """Test that fit_dmfm respects all parameters."""
        from conftest import generate_dmfm_data

        data = generate_dmfm_data(
            T=small_dims["T"],
            p1=small_dims["p1"],
            p2=small_dims["p2"],
            k1=2,
            k2=2,
            rng=rng,
        )

        model, _ = fit_dmfm(
            data["Y"],
            k1=2,
            k2=2,
            P=1,
            diagonal_idiosyncratic=True,
            max_iter=5,
        )

        assert model.k1 == 2
        assert model.k2 == 2
        assert model.P == 1
        assert model.diagonal_idiosyncratic is True

    def test_i1_factors(self, rng, small_dims):
        """Test fitting with i1_factors=True."""
        from conftest import generate_i1_data

        data = generate_i1_data(
            T=small_dims["T"],
            p1=small_dims["p1"],
            p2=small_dims["p2"],
            k1=small_dims["k1"],
            k2=small_dims["k2"],
            rng=rng,
        )

        model, _ = fit_dmfm(
            data["Y"],
            k1=small_dims["k1"],
            k2=small_dims["k2"],
            mask=data["mask"],
            i1_factors=True,
            max_iter=3,
        )

        assert model.dynamics.i1_factors is True
