"""Utility functions for the dynamic matrix factor model."""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd


def init_factor_loadings(
    Y: np.ndarray,
    mask: np.ndarray | None,
    k1: int,
    k2: int,
    method: str = "svd",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialise loading matrices and factors.

    Parameters
    ----------
    Y: array_like, shape (T, p1, p2)
        Observed data.
    mask: array_like or None
        Boolean mask indicating observed entries.  ``True`` means the
        entry is observed.  If ``None`` all entries are assumed
        observed.
    k1, k2: int
        Number of row and column factors.
    method: {"svd", "pe"}
        Initialisation method.  ``svd`` uses a singular value
        decomposition of the mean matrix while ``pe`` follows the
        principal eigenvector approach used by Barigozzi and Trapin
        (2025).
    """

    T, p1, p2 = Y.shape
    mask = np.ones_like(Y, dtype=bool) if mask is None else mask

    if method == "svd":
        Y_bar = np.nanmean(np.where(mask, Y, np.nan), axis=0)
        U, _, Vt = svd(np.nan_to_num(Y_bar), full_matrices=False)
        R = U[:, :k1]
        C = Vt.T[:, :k2]
    elif method == "pe":
        S_row_sum = np.zeros((p1, p1))
        S_col_sum = np.zeros((p2, p2))
        count_row = np.zeros((p1, p1))
        count_col = np.zeros((p2, p2))
        for t in range(T):
            Y_t = np.where(mask[t], Y[t], 0.0)
            M_t = mask[t]
            S_row_sum += Y_t @ Y_t.T
            S_col_sum += Y_t.T @ Y_t
            count_row += M_t @ M_t.T
            count_col += M_t.T @ M_t
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
        S_row = 0.5 * (S_row + S_row.T)
        S_col = 0.5 * (S_col + S_col.T)
        evals_row, evecs_row = np.linalg.eigh(S_row)
        evals_col, evecs_col = np.linalg.eigh(S_col)
        idx_row = np.argsort(evals_row)[::-1]
        idx_col = np.argsort(evals_col)[::-1]
        R = evecs_row[:, idx_row[:k1]]
        C = evecs_col[:, idx_col[:k2]]
    else:
        raise ValueError("method must be 'svd' or 'pe'")

    F = np.empty((T, k1, k2))
    for t in range(T):
        Y_t = np.where(mask[t], Y[t], 0.0)
        F[t] = R.T @ Y_t @ C
    return R, C, F


def init_idiosyncratic(
    Y: np.ndarray, R: np.ndarray, C: np.ndarray, F: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Initialise idiosyncratic covariance matrices ``H`` and ``K``."""

    T, p1, p2 = Y.shape
    resid = Y - np.einsum("ij,tjk,kl->til", R, F, C.T)
    H = np.zeros((p1, p1))
    K = np.zeros((p2, p2))
    for t in range(T):
        H += resid[t] @ resid[t].T
        K += resid[t].T @ resid[t]
    H = 0.5 * (H + H.T) / max(1, T * p2)
    K = 0.5 * (K + K.T) / max(1, T * p1)
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
    """Return initial MAR(P) dynamics.

    The initial transition matrices ``A_l`` and ``B_l`` are set to
    identities while the innovation covariance matrices ``Pmat`` and
    ``Qmat`` are also identity matrices.  This provides a neutral starting
    point for the EM algorithm.
    """

    A = [np.eye(k1) for _ in range(P)]
    B = [np.eye(k2) for _ in range(P)]
    Pmat = np.eye(k1)
    Qmat = np.eye(k2)
    return A, B, Pmat, Qmat
