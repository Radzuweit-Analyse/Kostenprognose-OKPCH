"""Utility functions for the dynamic matrix factor model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np
from numpy.linalg import svd


class InitMethod(Enum):
    """Initialization methods for factor loadings."""
    
    SVD = "svd"
    PRINCIPAL_EIGENVECTOR = "pe"


@dataclass
class InitializationResult:
    """Results from model initialization.
    
    Attributes
    ----------
    R : np.ndarray
        Row factor loadings (p1, k1).
    C : np.ndarray
        Column factor loadings (p2, k2).
    F : np.ndarray
        Initial factor estimates (T, k1, k2).
    H : np.ndarray
        Row idiosyncratic covariance (p1, p1).
    K : np.ndarray
        Column idiosyncratic covariance (p2, p2).
    A : list[np.ndarray]
        Row transition matrices (length P).
    B : list[np.ndarray]
        Column transition matrices (length P).
    Pmat : np.ndarray
        Row innovation covariance (k1, k1).
    Qmat : np.ndarray
        Column innovation covariance (k2, k2).
    method : str
        Initialization method used.
    """
    
    R: np.ndarray
    C: np.ndarray
    F: np.ndarray
    H: np.ndarray
    K: np.ndarray
    A: list[np.ndarray]
    B: list[np.ndarray]
    Pmat: np.ndarray
    Qmat: np.ndarray
    method: str


def initialize_model(
    Y: np.ndarray,
    k1: int,
    k2: int,
    P: int,
    mask: np.ndarray | None = None,
    method: str | InitMethod = InitMethod.SVD,
) -> InitializationResult:
    """Initialize all model parameters from data.
    
    This is a convenience function that performs complete initialization
    by calling the individual initialization routines in sequence.
    
    Parameters
    ----------
    Y : np.ndarray
        Observed data of shape (T, p1, p2).
    k1, k2 : int
        Number of row and column factors.
    P : int
        MAR order.
    mask : np.ndarray, optional
        Boolean mask for missing values (True = observed).
    method : str or InitMethod, default InitMethod.SVD
        Method for initializing factor loadings.
        
    Returns
    -------
    InitializationResult
        Complete set of initialized parameters.
        
    Examples
    --------
    >>> result = initialize_model(Y, k1=3, k2=2, P=1)
    >>> model = DMFMModel.from_dims(p1=10, p2=8, k1=3, k2=2)
    >>> model._R = result.R
    >>> model._C = result.C
    >>> # ... set other parameters
    """
    if isinstance(method, str):
        method = InitMethod(method)
    
    # Validate inputs
    if Y.ndim != 3:
        raise ValueError(f"Y must be 3D array, got shape {Y.shape}")
    T, p1, p2 = Y.shape
    
    if mask is None:
        mask = np.ones_like(Y, dtype=bool)
    
    # Initialize loadings and factors
    R, C, F = init_factor_loadings(Y, mask, k1, k2, method.value)
    
    # Initialize idiosyncratic covariances
    H, K = init_idiosyncratic(Y, R, C, F)
    
    # Initialize dynamics
    A, B, Pmat, Qmat = init_dynamics(k1, k2, P)
    
    return InitializationResult(
        R=R, C=C, F=F, H=H, K=K,
        A=A, B=B, Pmat=Pmat, Qmat=Qmat,
        method=method.value
    )


def init_factor_loadings(
    Y: np.ndarray,
    mask: np.ndarray | None,
    k1: int,
    k2: int,
    method: str = "svd",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize loading matrices and factors.

    Parameters
    ----------
    Y : np.ndarray
        Observed data of shape (T, p1, p2).
    mask : np.ndarray or None
        Boolean mask indicating observed entries (True = observed).
        If None, all entries are assumed observed.
    k1, k2 : int
        Number of row and column factors.
    method : {"svd", "pe"}, default "svd"
        Initialization method:
        - "svd": Singular value decomposition of mean matrix
        - "pe": Principal eigenvector approach (Barigozzi & Trapin, 2025)

    Returns
    -------
    R : np.ndarray
        Row factor loadings (p1, k1).
    C : np.ndarray
        Column factor loadings (p2, k2).
    F : np.ndarray
        Initial factor estimates (T, k1, k2).

    Raises
    ------
    ValueError
        If method is not recognized or dimensions are invalid.

    Notes
    -----
    The SVD method is faster but requires complete data or careful
    handling of missing values. The PE method is more robust to
    missingness but computationally intensive.
    """
    T, p1, p2 = Y.shape
    
    # Validate factor dimensions
    if k1 <= 0 or k2 <= 0:
        raise ValueError(f"k1 and k2 must be positive, got k1={k1}, k2={k2}")
    if k1 > p1 or k2 > p2:
        raise ValueError(
            f"Factor dimensions cannot exceed data dimensions: "
            f"k1={k1} > p1={p1} or k2={k2} > p2={p2}"
        )
    
    mask = np.ones_like(Y, dtype=bool) if mask is None else mask

    if method == "svd":
        R, C = _init_svd(Y, mask, k1, k2)
    elif method == "pe":
        R, C = _init_principal_eigenvector(Y, mask, k1, k2)
    else:
        raise ValueError(f"Unknown method '{method}', must be 'svd' or 'pe'")

    # Project data onto loadings to get initial factors
    F = _project_factors(Y, mask, R, C)
    
    return R, C, F


def _init_svd(
    Y: np.ndarray, mask: np.ndarray, k1: int, k2: int
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize via SVD of mean matrix."""
    # Compute mean matrix (handling missing values)
    Y_masked = np.where(mask, Y, np.nan)
    Y_bar = np.nanmean(Y_masked, axis=0)
    
    # Handle remaining NaNs
    Y_bar = np.nan_to_num(Y_bar, nan=0.0)
    
    # SVD decomposition
    U, s, Vt = svd(Y_bar, full_matrices=False)
    
    # Extract loadings (scaled by singular values for better initialization)
    R = U[:, :k1] * np.sqrt(s[:k1])
    C = Vt.T[:, :k2] * np.sqrt(s[:k2])
    
    return R, C


def _init_principal_eigenvector(
    Y: np.ndarray, mask: np.ndarray, k1: int, k2: int
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize via principal eigenvector method.
    
    This method follows Barigozzi and Trapin (2025) and is more
    robust to missing data.
    """
    T, p1, p2 = Y.shape
    
    # Compute sample covariances with missing data handling
    S_row_sum = np.zeros((p1, p1))
    S_col_sum = np.zeros((p2, p2))
    count_row = np.zeros((p1, p1))
    count_col = np.zeros((p2, p2))
    
    for t in range(T):
        Y_t = np.where(mask[t], Y[t], 0.0)
        M_t = mask[t].astype(float)
        
        # Accumulate products
        S_row_sum += Y_t @ Y_t.T
        S_col_sum += Y_t.T @ Y_t
        
        # Count valid pairs
        count_row += M_t @ M_t.T
        count_col += M_t.T @ M_t
    
    # Average over valid observations
    S_row = np.divide(
        S_row_sum,
        np.maximum(count_row, 1),
        where=count_row > 0,
        out=np.zeros_like(S_row_sum),
    )
    S_col = np.divide(
        S_col_sum,
        np.maximum(count_col, 1),
        where=count_col > 0,
        out=np.zeros_like(S_col_sum),
    )
    
    # Ensure symmetry
    S_row = 0.5 * (S_row + S_row.T)
    S_col = 0.5 * (S_col + S_col.T)
    
    # Extract principal eigenvectors
    evals_row, evecs_row = np.linalg.eigh(S_row)
    evals_col, evecs_col = np.linalg.eigh(S_col)
    
    # Sort by eigenvalue (descending)
    idx_row = np.argsort(evals_row)[::-1]
    idx_col = np.argsort(evals_col)[::-1]
    
    # Select top k eigenvectors
    R = evecs_row[:, idx_row[:k1]]
    C = evecs_col[:, idx_col[:k2]]
    
    return R, C


def _project_factors(
    Y: np.ndarray, mask: np.ndarray, R: np.ndarray, C: np.ndarray
) -> np.ndarray:
    """Project data onto loadings to obtain initial factor estimates."""
    T = Y.shape[0]
    k1, k2 = R.shape[1], C.shape[1]
    F = np.empty((T, k1, k2))
    
    for t in range(T):
        Y_t = np.where(mask[t], Y[t], 0.0)
        F[t] = R.T @ Y_t @ C
    
    return F


def init_idiosyncratic(
    Y: np.ndarray, R: np.ndarray, C: np.ndarray, F: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize idiosyncratic covariance matrices.

    Parameters
    ----------
    Y : np.ndarray
        Observed data (T, p1, p2).
    R : np.ndarray
        Row factor loadings (p1, k1).
    C : np.ndarray
        Column factor loadings (p2, k2).
    F : np.ndarray
        Factor estimates (T, k1, k2).

    Returns
    -------
    H : np.ndarray
        Row idiosyncratic covariance (p1, p1).
    K : np.ndarray
        Column idiosyncratic covariance (p2, p2).

    Notes
    -----
    Computes residuals Y_t - R @ F_t @ C^T and estimates covariances.
    Covariances are normalized to have trace equal to dimension.
    """
    T, p1, p2 = Y.shape
    
    # Compute residuals
    resid = Y - np.einsum("ij,tjk,kl->til", R, F, C.T)
    
    # Accumulate covariances
    H = np.zeros((p1, p1))
    K = np.zeros((p2, p2))
    
    for t in range(T):
        H += resid[t] @ resid[t].T
        K += resid[t].T @ resid[t]
    
    # Normalize
    H = 0.5 * (H + H.T) / max(1, T * p2)
    K = 0.5 * (K + K.T) / max(1, T * p1)
    
    # Scale to have trace = dimension (improves numerical stability)
    tr_H = np.trace(H)
    tr_K = np.trace(K)
    
    if tr_H > 0:
        H *= float(p1) / tr_H
    if tr_K > 0:
        K *= float(p2) / tr_K
    
    return H, K


def init_dynamics(
    k1: int, k2: int, P: int
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """Initialize MAR(P) dynamics with neutral values.

    Parameters
    ----------
    k1, k2 : int
        Factor dimensions.
    P : int
        MAR order.

    Returns
    -------
    A : list[np.ndarray]
        Row transition matrices, initialized to identity (length P).
    B : list[np.ndarray]
        Column transition matrices, initialized to identity (length P).
    Pmat : np.ndarray
        Row innovation covariance, initialized to identity (k1, k1).
    Qmat : np.ndarray
        Column innovation covariance, initialized to identity (k2, k2).

    Notes
    -----
    Identity initialization provides a neutral starting point where
    the EM algorithm can learn the dynamics from data. This corresponds
    to a weak prior that factors follow a random walk.
    """
    A = [np.eye(k1) for _ in range(P)]
    B = [np.eye(k2) for _ in range(P)]
    Pmat = np.eye(k1)
    Qmat = np.eye(k2)
    
    return A, B, Pmat, Qmat