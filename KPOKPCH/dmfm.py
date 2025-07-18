"""Dynamic Matrix Factor Model (DMFM) implementation.

This module provides routines to estimate a DMFM by the
Expectation-Maximization algorithm using Kalman smoothing.  Follows
Barigozzi and Trapin (2025), supporting MAR(P) dynamics, missing data
and full idiosyncratic covariance structure.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import inv, svd


def initialize_dmfm(
    Y: np.ndarray,
    k1: int,
    k2: int,
    P: int,
    mask: np.ndarray | None = None,
) -> dict:
    """Return initial parameter guesses for the DMFM.

    Parameters
    ----------
    Y : array_like
        Observation array of shape ``(T, p1, p2)``.
    k1, k2 : int
        Number of row and column factors.
    P : int
        Order of the matrix autoregressive dynamics.
    mask : ndarray or None, optional
        Binary mask with the same shape as ``Y`` where ``True`` indicates an
        observed entry. ``None`` treats all entries as observed.

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

    # simple mean imputation for missing entries ----------------------------
    Y_imp = Y.copy()
    col_mean = np.nanmean(Y_imp.reshape(T, -1), axis=0)
    inds = ~mask.reshape(T, -1)
    Y_imp.reshape(T, -1)[inds] = col_mean[inds.any(axis=0)]

    # average matrix and SVD -------------------------------------------------
    Y_bar = np.nanmean(Y_imp, axis=0)
    U, _, Vt = svd(Y_bar, full_matrices=False)
    R = U[:, :k1]
    C = Vt.T[:, :k2]

    F = np.empty((T, k1, k2))
    for t in range(T):
        F[t] = R.T @ Y_imp[t] @ C

    # initialize MAR coefficients as identity matrices ---------------------
    A = [np.eye(k1) for _ in range(P)]
    B = [np.eye(k2) for _ in range(P)]
    
    # residuals and covariances --------------------------------------------
    resid = Y_imp - np.einsum("ij,tjk,kl->til", R, F, C.T)
    H = np.zeros((p1, p1))
    K = np.zeros((p2, p2))
    for t in range(T):
        for j in range(p2):
            H += np.outer(resid[t, :, j], resid[t, :, j])
        for i in range(p1):
            K += np.outer(resid[t, i, :], resid[t, i, :])
    H /= max(1, T * p2)
    K /= max(1, T * p1)

    # innovation covariances ------------------------------------------------
    Pmat = np.eye(k1)
    Qmat = np.eye(k2)

    return {
        "R": R,
        "C": C,
        "A": A,
        "B": B,
        "H": H,
        "K": K,
        "P": Pmat,
        "Q": Qmat,
        "F": F,
    }


def _construct_state_matrices(A: list[np.ndarray], B: list[np.ndarray]) -> np.ndarray:
    """Return VAR(1) transition matrix for stacked MAR(P)."""
    k1 = A[0].shape[0]
    k2 = B[0].shape[0]
    P = len(A)
    r = k1 * k2
    d = r * P
    Phi = [np.kron(B[l], A[l]) for l in range(P)]
    Tmat = np.zeros((d, d))
    Tmat[:r, : r * P] = np.hstack(Phi)
    if P > 1:
        Tmat[r:, :-r] = np.eye(r * (P - 1))
    return Tmat


def kalman_smoother_dmfm(
    Y: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    A: list[np.ndarray],
    B: list[np.ndarray],
    H: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray | None = None,
    Pmat: np.ndarray | None = None,
    Qmat: np.ndarray | None = None,
) -> dict:
    """Kalman smoother for the dynamic matrix factor model.

    Parameters
    ----------
    Y : array_like
        Data array ``(T, p1, p2)``.
    R, C : ndarray
        Loading matrices.
    A, B : list of ndarray
        MAR(P) coefficient matrices.
    H, K : ndarray
        Idiosyncratic covariance matrices of shape ``(p1, p1)`` and ``(p2, p2)``.
    mask : ndarray or None, optional
        Observation mask with shape ``Y``. ``True`` indicates an observed
        entry. ``None`` assumes full observations.
    Pmat, Qmat : ndarray or None, optional
        Innovation covariance matrices for the MAR(P) process. If ``None``
        identity matrices are used.

    Returns
    -------
    dict
        Smoothed state means ``F`` of shape ``(T, k1, k2)`` and covariances
        ``V`` together with filtered and predicted values and the log-likelihood.
    """
    Y = np.asarray(Y, dtype=float)
    Tn, p1, p2 = Y.shape
    k1 = R.shape[1]
    k2 = C.shape[1]
    r = k1 * k2
    P = len(A)
    d = r * P
    
    if mask is None:
        mask = np.ones_like(Y, dtype=bool)

    Pmat = np.eye(k1) if Pmat is None else Pmat
    Qmat = np.eye(k2) if Qmat is None else Qmat

    Phi = [np.kron(B[l], A[l]) for l in range(P)]
    Qx = np.kron(Qmat, Pmat)
    Tmat = _construct_state_matrices(A, B)
    Q_full = np.zeros((d, d))
    Q_full[:r, :r] = Qx

    R_full = np.kron(K, H)
    Z0 = np.kron(C, R)
    Z_full = np.hstack([Z0] + [np.zeros((p1 * p2, r)) for _ in range(P - 1)])

    x_pred = np.zeros(d)
    V_pred = np.eye(d) * 1e2

    xp = np.zeros((Tn, d))
    Pp = np.zeros((Tn, d, d))
    xf = np.zeros((Tn, d))
    Pf = np.zeros((Tn, d, d))

    loglik = 0.0

    # Kalman filter --------------------------------------------------------
    for t in range(Tn):
        if t == 0:
            x_prior = np.zeros(d)
            V_prior = np.eye(d) * 1e2
        else:
            x_prior = Tmat @ x_pred
            V_prior = Tmat @ V_pred @ Tmat.T + Q_full

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
            sign, logdet = np.linalg.slogdet(S)
            loglik -= 0.5 * (len(idx) * np.log(2 * np.pi) + logdet + (y_obs - Z @ x_prior).T @ inv(S) @ (y_obs - Z @ x_prior))
        else:
            x_post = x_prior
            V_post = V_prior

        xp[t] = x_prior
        Pp[t] = V_prior
        xf[t] = x_post
        Pf[t] = V_post

        x_pred = x_post
        V_pred = V_post

    # Rauch‑Tung‑Striebel smoother ----------------------------------------
    xs = np.zeros_like(xf)
    Vs = np.zeros_like(Pf)
    xs[-1] = xf[-1]
    Vs[-1] = Pf[-1]
    J = np.zeros((Tn - 1, d, d))

    for t in range(Tn - 2, -1, -1):
        J[t] = Pf[t] @ Tmat.T @ inv(Pp[t + 1])
        xs[t] = xf[t] + J[t] @ (xs[t + 1] - xp[t + 1])
        Vs[t] = Pf[t] + J[t] @ (Vs[t + 1] - Pp[t + 1]) @ J[t].T

    # covariance of successive states (needed for MAR estimation) ----------
    Vss = np.zeros((Tn - 1, d, d))
    for t in range(Tn - 1):
        Vss[t] = J[t] @ Vs[t + 1]

    F_smooth = xs[:, :r].reshape(Tn, k1, k2)

    return {
        "F_smooth": F_smooth,
        "V_smooth": Vs,
        "V_ss": Vss,
        "F_filt": xf[:, :r].reshape(Tn, k1, k2),
        "F_pred": xp[:, :r].reshape(Tn, k1, k2),
        "loglik": float(loglik),
    }


def em_step_dmfm(Y: np.ndarray, params: dict, mask: np.ndarray | None = None) -> tuple[dict, float, float]:
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
        Updated parameter dictionary, the relative parameter change and the
        log-likelihood value.
    """
    R = params["R"]
    C = params["C"]
    A = params["A"]
    B = params["B"]
    H = params["H"]
    K = params["K"]
    Pmat = params.get("P", np.eye(R.shape[1]))
    Qmat = params.get("Q", np.eye(C.shape[1]))

    smooth = kalman_smoother_dmfm(Y, R, C, A, B, H, K, mask, Pmat, Qmat)
    F = smooth["F_smooth"]
    Tn, p1, p2 = Y.shape
    k1 = R.shape[1]
    k2 = C.shape[1]
    Pord = len(A)

    # Update R -------------------------------------------------------------
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

    # Update C -------------------------------------------------------------
    C_new = np.zeros_like(C)
    for j in range(p2):
        X_stack = []
        y_stack = []
        for t in range(Tn):
            m = mask[t, :, j] if mask is not None else np.ones(p1, dtype=bool)
            if not m.any():
                continue
            X = (R_new @ F[t])[m, :]  # (sum m) x k2
            y_stack.append(Y[t, m, j])
            X_stack.append(X)
        if X_stack:
            Xmat = np.vstack(X_stack)
            yvec = np.concatenate(y_stack)
            C_new[j] = np.linalg.lstsq(Xmat, yvec, rcond=None)[0]
        else:
            C_new[j] = C[j]

    # Update MAR matrices A and B -----------------------------------------
    A_new = [np.zeros_like(A[0]) for _ in range(Pord)]
    B_new = [np.zeros_like(B[0]) for _ in range(Pord)]

    for ell in range(Pord):
        A_num = np.zeros((k1, k1))
        A_den = np.zeros((k1, k1))
        B_num = np.zeros((k2, k2))
        B_den = np.zeros((k2, k2))
        for t in range(ell + 1, Tn):
            F_pred_other = np.zeros((k1, k2))
            for j in range(Pord):
                if j == ell:
                    continue
                if t - j - 1 < 0:
                    continue
                F_pred_other += A[j] @ F[t - j - 1] @ B[j].T
            Y_res = F[t] - F_pred_other
            X_A = F[t - ell - 1] @ B[ell].T
            A_num += Y_res @ X_A.T
            A_den += X_A @ X_A.T
            X_B = F[t - ell - 1].T @ A[ell].T
            B_num += Y_res.T @ X_B.T
            B_den += X_B @ X_B.T
        A_new[ell] = A_num @ inv(A_den + 1e-8 * np.eye(k1))
        B_new[ell] = B_num @ inv(B_den + 1e-8 * np.eye(k2))

    # Update innovation covariances P and Q --------------------------------
    U = np.zeros_like(F[Pord:])
    for t in range(Pord, Tn):
        F_pred = np.zeros((k1, k2))
        for j in range(Pord):
            F_pred += A_new[j] @ F[t - j - 1] @ B_new[j].T
        U[t - Pord] = F[t] - F_pred
    P_new = np.zeros((k1, k1))
    Q_new = np.zeros((k2, k2))
    for u in U:
        P_new += u @ u.T
        Q_new += u.T @ u
    denom = max(1, len(U))
    P_new /= denom * k2
    Q_new /= denom * k1

    # Update idiosyncratic covariances H and K ----------------------------
    resid = np.zeros_like(Y)
    for t in range(Tn):
        resid[t] = Y[t] - R_new @ F[t] @ C_new.T
    if mask is not None:
        resid = np.where(mask, resid, np.nan)
    H_new = np.zeros_like(H)
    K_new = np.zeros_like(K)
    count_H = 0
    count_K = 0
    for t in range(Tn):
        for j in range(p2):
            m = mask[t, :, j] if mask is not None else np.ones(p1, dtype=bool)
            rcol = resid[t, :, j]
            rcol = rcol[m]
            H_new[np.ix_(np.where(m)[0], np.where(m)[0])] += np.outer(rcol, rcol)
            count_H += 1
        for i in range(p1):
            m = mask[t, i, :] if mask is not None else np.ones(p2, dtype=bool)
            rrow = resid[t, i, :]
            rrow = rrow[m]
            K_new[np.ix_(np.where(m)[0], np.where(m)[0])] += np.outer(rrow, rrow)
            count_K += 1
    if count_H > 0:
        H_new /= count_H
    if count_K > 0:
        K_new /= count_K
    
    new_params = {
        "R": R_new,
        "C": C_new,
        "A": A_new,
        "B": B_new,
        "H": H_new,
        "K": K_new,
        "P": P_new,
        "Q": Q_new,
        "F": F,
    }

    diff = 0.0
    for key in ["R", "C"]:
        diff += np.linalg.norm(params[key] - new_params[key])
    for l in range(Pord):
        diff += np.linalg.norm(params["A"][l] - new_params["A"][l])
        diff += np.linalg.norm(params["B"][l] - new_params["B"][l])
    diff /= max(1.0, np.linalg.norm(params["R"]))

    return new_params, diff, smooth["loglik"]


def fit_dmfm_em(
    Y: np.ndarray,
    k1: int,
    k2: int,
    P: int,
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
    P : int
        Order of the MAR dynamics.
    max_iter : int, optional
        Maximum number of EM iterations.
    tol : float, optional
        Convergence threshold based on relative parameter change.
    mask : ndarray or None, optional
        Observation mask, ``True`` for observed entries.

    Returns
    -------
    dict
        Fitted parameters, smoothed factors and log-likelihood trace.
    """
    params = initialize_dmfm(Y, k1, k2, P, mask)
    loglik_trace = []
    for _ in range(max_iter):
        params, diff, ll = em_step_dmfm(Y, params, mask)
        loglik_trace.append(ll)
        if diff < tol:
            break
    params["loglik"] = loglik_trace
    return params
