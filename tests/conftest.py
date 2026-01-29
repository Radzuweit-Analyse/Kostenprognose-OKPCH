"""Shared pytest fixtures for KPOKPCH tests.

This module provides common fixtures used across all test modules,
including data generators and model factories.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Data generation fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    """Provide a seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def small_dims():
    """Small dimensions for fast tests."""
    return {"T": 10, "p1": 4, "p2": 3, "k1": 2, "k2": 2}


@pytest.fixture
def medium_dims():
    """Medium dimensions for more thorough tests."""
    return {"T": 50, "p1": 10, "p2": 8, "k1": 3, "k2": 2}


def generate_dmfm_data(
    T: int,
    p1: int,
    p2: int,
    k1: int,
    k2: int,
    rng: np.random.Generator,
    noise_scale: float = 0.1,
    dynamics_scale: float = 0.5,
) -> dict:
    """Generate synthetic DMFM data.

    Parameters
    ----------
    T : int
        Number of time periods.
    p1, p2 : int
        Cross-sectional dimensions.
    k1, k2 : int
        Number of factors.
    rng : np.random.Generator
        Random number generator.
    noise_scale : float, default 0.1
        Scale of idiosyncratic noise.
    dynamics_scale : float, default 0.5
        Scale of dynamics coefficients (smaller = more stable).

    Returns
    -------
    dict
        Dictionary with keys: Y, F, R, C, A, B, mask.
    """
    # Generate loadings
    R = rng.normal(size=(p1, k1))
    C = rng.normal(size=(p2, k2))

    # Generate stable dynamics
    A = [dynamics_scale * np.eye(k1)]
    B = [dynamics_scale * np.eye(k2)]

    # Generate factors with MAR(1) dynamics
    F = np.zeros((T, k1, k2))
    for t in range(1, T):
        F[t] = A[0] @ F[t - 1] @ B[0].T + rng.normal(size=(k1, k2))

    # Generate observations
    Y = np.zeros((T, p1, p2))
    for t in range(T):
        Y[t] = R @ F[t] @ C.T + noise_scale * rng.normal(size=(p1, p2))

    # Full observation mask
    mask = np.ones_like(Y, dtype=bool)

    return {
        "Y": Y,
        "F": F,
        "R": R,
        "C": C,
        "A": A,
        "B": B,
        "mask": mask,
    }


def generate_i1_data(
    T: int,
    p1: int,
    p2: int,
    k1: int,
    k2: int,
    rng: np.random.Generator,
    noise_scale: float = 0.1,
    innovation_scale: float = 0.5,
) -> dict:
    """Generate data with I(1) (random walk) factors.

    Parameters
    ----------
    T : int
        Number of time periods.
    p1, p2 : int
        Cross-sectional dimensions.
    k1, k2 : int
        Number of factors.
    rng : np.random.Generator
        Random number generator.
    noise_scale : float, default 0.1
        Scale of idiosyncratic noise.
    innovation_scale : float, default 0.5
        Scale of factor innovations.

    Returns
    -------
    dict
        Dictionary with keys: Y, F, R, C, mask.
    """
    R = rng.normal(size=(p1, k1))
    C = rng.normal(size=(p2, k2))

    # I(1) factors: F_t = F_{t-1} + U_t
    F = np.zeros((T, k1, k2))
    for t in range(1, T):
        F[t] = F[t - 1] + innovation_scale * rng.normal(size=(k1, k2))

    # Observations
    Y = np.zeros((T, p1, p2))
    for t in range(T):
        Y[t] = R @ F[t] @ C.T + noise_scale * rng.normal(size=(p1, p2))

    mask = np.ones_like(Y, dtype=bool)

    return {"Y": Y, "F": F, "R": R, "C": C, "mask": mask}


@pytest.fixture
def dmfm_data(rng, small_dims):
    """Generate small DMFM dataset for testing."""
    return generate_dmfm_data(
        T=small_dims["T"],
        p1=small_dims["p1"],
        p2=small_dims["p2"],
        k1=small_dims["k1"],
        k2=small_dims["k2"],
        rng=rng,
    )


@pytest.fixture
def i1_data(rng, small_dims):
    """Generate small I(1) dataset for testing."""
    return generate_i1_data(
        T=small_dims["T"],
        p1=small_dims["p1"],
        p2=small_dims["p2"],
        k1=small_dims["k1"],
        k2=small_dims["k2"],
        rng=rng,
    )


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dmfm_config(small_dims):
    """Create a DMFMConfig with small dimensions."""
    from KPOKPCH.DMFM import DMFMConfig

    return DMFMConfig(
        p1=small_dims["p1"],
        p2=small_dims["p2"],
        k1=small_dims["k1"],
        k2=small_dims["k2"],
        P=1,
    )


@pytest.fixture
def initialized_model(dmfm_config, dmfm_data):
    """Create an initialized (but not fitted) DMFMModel."""
    from KPOKPCH.DMFM import DMFMModel

    model = DMFMModel(dmfm_config)
    model.initialize(dmfm_data["Y"], dmfm_data["mask"])
    return model
