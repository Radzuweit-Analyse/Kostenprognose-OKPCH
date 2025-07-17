"""Dynamic Matrix Factor Model (DMFM) implementation.

This module provides basic routines to estimate a Dynamic Matrix Factor
Model (DMFM) by the Expectation-Maximization algorithm using Kalman
smoothing. The implementation follows Barigozzi and Trapin (2025) in a
simplified form. Only MAR(1) dynamics and diagonal idiosyncratic
covariances are supported.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import inv, svd


def initialize_dmfm(Y: np.ndarray, k1: int, k2: int, mask: np.ndarray | None = None) -> dict:
    """Return initial parameter guesses for the DMFM.

    Parameters
    ----------
    Y : array_like
        Observation array of shape ``(T, p1, p2)``.
    k1 : int
        Number of row factors.
    k2 : int
        Number of column factors.
    mask : ndarray or None, optional
        Binary mask of same shape as ``Y`` where ``1`` indicates an observed
        entry. ``None`` treats all entries as observed.

    Returns
    -------
    dict
        Dictionary containing initial matrices ``R``, ``C``, ``A``, ``B``,
        ``H`` and ``K`` as well as an initial factor path ``F``.
    """
    Y = np.asarray(Y, dtype=float)
    T, p1, p2 = Y.shape
    if mask is None:
        mask = np.ones_like(Y, dtype=bool)

    # --- simple mean imputation for missing entries -----------------------
    Y_imp = Y.copy()
    col_mean = np.nanmean(Y_imp.reshape(T, -1), axis=0)
    inds = ~mask.reshape(T, -1)
    Y_imp.reshape(T, -1)[inds] = col_mean[inds.any(axis=0)]

    # --- average matrix and SVD ------------------------------------------
    Y_bar = np.nanmean(Y_imp, axis=0)
    U, s, Vt = svd(Y_bar, full_matrices=False)
    R = U[:, :k1]
    C = Vt.T[:, :k2]

    F = np.empty((T, k1, k2))
    for t in range(T):
        F[t] = R.T @ Y_imp[t] @ C

    A = np.eye(k1)
    B = np.eye(k2)

    resid = Y_imp - np.einsum("ij,tjk,kl->til", R, F, C.T)
    H = np.var(resid.reshape(T, p1, p2), axis=(0, 2))
    K = np.var(resid.reshape(T, p1, p2), axis=(0, 1))

    return {
        "R": R,
        "C": C,
        "A": A,
        "B": B,
        "H": np.diag(np.maximum(H, 1e-6)),
        "K": np.diag(np.maximum(K, 1e-6)),
        "F": F,
    }


def kalman_smoother_dmfm(
    Y: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    H: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray | None = None,
    P: np.ndarray | None = None,
    Q: np.ndarray | None = None,
):
    """Kalman smoother for the dynamic matrix factor model.

    Parameters
    ----------
    Y : array_like
        Data array ``(T, p1, p2)``.
    R, C : ndarray
        Loading matrices.
    A, B : ndarray
        MAR(1) coefficient matrices.
    H, K : ndarray
        Diagonal idiosyncratic covariance matrices.
    mask : ndarray or None, optional
        Binary mask of observations with shape ``Y``. ``True`` indicates an
        observed entry. ``None`` assumes full observations.
    P, Q : ndarray or None, optional
        Innovation covariance matrices for the MAR(1) process. If ``None``
        identity matrices are used.

    Returns
    -------
    dict
        Smoothed state means ``F`` and covariances ``V`` together with
        filtered and predicted values required by the EM algorithm.
    """
    Y = np.asarray(Y, dtype=float)
    T, p1, p2 = Y.shape
    k1 = R.shape[1]
    k2 = C.shape[1]
    r = k1 * k2
    if mask is None:
        mask = np.ones_like(Y, dtype=bool)

    P = np.eye(k1) if P is None else P
    Q = np.eye(k2) if Q is None else Q

    Phi = np.kron(B, A)
    Qx = np.kron(Q, P)
    R_full = np.kron(K, H)
    Z_full = np.kron(C, R)

    x_pred = np.zeros(r)
    V_pred = np.eye(r) * 1e2

    xp = np.zeros((T, r))
    Pp = np.zeros((T, r, r))
    xf = np.zeros((T, r))
    Pf = np.zeros((T, r, r))

    # --- Kalman filter ----------------------------------------------------
    for t in range(T):
        x_prior = Phi @ x_pred
        V_prior = Phi @ V_pred @ Phi.T + Qx

        y_vec = Y[t].reshape(-1)
        m_vec = mask[t].reshape(-1)
        idx = np.where(m_vec)[0]

        if idx.size > 0:
            Z = Z_full[idx, :]
            R_t = R_full[np.ix_(idx, idx)]
            y_obs = y_vec[idx]
            S = Z @ V_prior @ Z.T + R_t
            K_gain = V_prior @ Z.T @ inv(S)
            x_post = x_prior + K_gain @ (y_obs - Z @ x_prior)
            V_post = V_prior - K_gain @ Z @ V_prior
        else:
            x_post = x_prior
            V_post = V_prior

        xp[t] = x_prior
        Pp[t] = V_prior
        xf[t] = x_post
        Pf[t] = V_post

        x_pred = x_post
        V_pred = V_post

    # --- Rauch‑Tung‑Striebel smoother ------------------------------------
    xs = np.zeros_like(xf)
    Vs = np.zeros_like(Pf)
    xs[-1] = xf[-1]
    Vs[-1] = Pf[-1]

    for t in range(T - 2, -1, -1):
        Ck = Pf[t] @ Phi.T @ inv(Pp[t + 1])
        xs[t] = xf[t] + Ck @ (xs[t + 1] - xp[t + 1])
        Vs[t] = Pf[t] + Ck @ (Vs[t + 1] - Pp[t + 1]) @ Ck.T

    return {
        "F_smooth": xs.reshape(T, k1, k2),
        "V_smooth": Vs,
        "F_filt": xf.reshape(T, k1, k2),
        "V_filt": Pf,
        "F_pred": xp.reshape(T, k1, k2),
        "V_pred": Pp,
    }


def em_step_dmfm(Y: np.ndarray, params: dict, mask: np.ndarray | None = None) -> tuple[dict, float]:
    """Perform one EM iteration for the DMFM.

    Parameters
    ----------
    Y : array_like
        Observed array ``(T, p1, p2)``.
    params : dict
        Current parameter estimates as produced by ``initialize_dmfm`` or
        previous EM steps.
    mask : ndarray or None, optional
        Observation mask of the same shape as ``Y``.

    Returns
    -------
    tuple
        Updated parameter dictionary and the relative parameter change
        (used as a convergence metric).
    """
    R = params["R"]
    C = params["C"]
    A = params["A"]
    B = params["B"]
    H = params["H"]
    K = params["K"]

    smooth = kalman_smoother_dmfm(
        Y, R, C, A, B, H, K, mask,
    )
    F = smooth["F_smooth"]

    Tn, p1, p2 = Y.shape
    k1 = R.shape[1]
    k2 = C.shape[1]

    # --- Update R ---------------------------------------------------------
    R_new = np.zeros_like(R)
    for i in range(p1):
        X_stack = []
        y_stack = []
        for t in range(Tn):
            m = mask[t, i, :] if mask is not None else np.ones(p2, dtype=bool)
            if not m.any():
                continue
            X = (F[t] @ C.T)[m, :]  # (sum m) x k1
            y_stack.append(Y[t, i, m])
            X_stack.append(X)
        if X_stack:
            Xmat = np.vstack(X_stack)
            yvec = np.concatenate(y_stack)
            R_new[i] = np.linalg.lstsq(Xmat, yvec, rcond=None)[0]
        else:
            R_new[i] = R[i]

    # --- Update C ---------------------------------------------------------
    C_new = np.zeros_like(C)
    for j in range(p2):
        X_stack = []
        y_stack = []
        for t in range(Tn):
            m = mask[t, :, j] if mask is not None else np.ones(p1, dtype=bool)
            if not m.any():
                continue
            X = (R @ F[t])[m, :]  # (sum m) x k2
            y_stack.append(Y[t, m, j])
            X_stack.append(X)
        if X_stack:
            Xmat = np.vstack(X_stack)
            yvec = np.concatenate(y_stack)
            C_new[j] = np.linalg.lstsq(Xmat, yvec, rcond=None)[0]
        else:
            C_new[j] = C[j]

    # --- Update MAR matrices A and B -------------------------------------
    A_num = np.zeros((k1, k1))
    A_den = np.zeros((k1, k1))
    B_num = np.zeros((k2, k2))
    B_den = np.zeros((k2, k2))
    for t in range(1, Tn):
        Z_A = F[t - 1] @ B.T
        A_num += F[t] @ Z_A.T
        A_den += Z_A @ Z_A.T
        Z_B = F[t - 1].T @ A.T
        B_num += F[t].T @ Z_B.T
        B_den += Z_B @ Z_B.T
    A_new = A_num @ inv(A_den + 1e-8 * np.eye(k1))
    B_new = B_num @ inv(B_den + 1e-8 * np.eye(k2))

    # --- Update idiosyncratic variances (diagonal) -----------------------
    resid = np.zeros_like(Y)
    for t in range(Tn):
        resid[t] = Y[t] - R_new @ F[t] @ C_new.T
    if mask is not None:
        resid = np.where(mask, resid, np.nan)
    H_new = np.nanmean(resid ** 2, axis=(0, 2))
    K_new = np.nanmean(resid ** 2, axis=(0, 1))
    H_new = np.diag(np.maximum(H_new, 1e-6))
    K_new = np.diag(np.maximum(K_new, 1e-6))

    new_params = {
        "R": R_new,
        "C": C_new,
        "A": A_new,
        "B": B_new,
        "H": H_new,
        "K": K_new,
    }

    diff = 0.0
    for key in ["R", "C", "A", "B"]:
        diff += np.linalg.norm(params[key] - new_params[key])
    diff /= max(1.0, np.linalg.norm(params["R"]))

    new_params["F"] = F
    return new_params, diff


def fit_dmfm_em(
    Y: np.ndarray,
    k1: int,
    k2: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    mask: np.ndarray | None = None,
) -> dict:
    """Fit a dynamic matrix factor model using the EM algorithm.

    Parameters
    ----------
    Y : array_like
        Data array of shape ``(T, p1, p2)``.
    k1, k2 : int
        Number of row and column factors.
    max_iter : int, optional
        Maximum number of EM iterations.
    tol : float, optional
        Convergence threshold based on relative parameter change.
    mask : ndarray or None, optional
        Observation mask, ``True`` for observed entries.

    Returns
    -------
    dict
        Fitted parameters and smoothed factors.
    """
    params = initialize_dmfm(Y, k1, k2, mask)
    for _ in range(max_iter):
        params, diff = em_step_dmfm(Y, params, mask)
        if diff < tol:
            break
    return params
