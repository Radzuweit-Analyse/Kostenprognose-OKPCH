"""Dynamic Matrix Factor Model (DMFM) implementation.

This module provides routines to estimate a DMFM by the
Expectation-Maximization algorithm using Kalman smoothing.  Follows
Barigozzi and Trapin (2025), supporting MAR(P) dynamics, missing data
and full idiosyncratic covariance structure.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import inv, svd
import warnings


def initialize_dmfm(
    Y: np.ndarray,
    k1: int,
    k2: int,
    P: int,
    mask: np.ndarray | None = None,
    method: str = "pe",
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
    method : {"pe", "svd"}, optional
        Initialization method. ``"pe"`` uses the projected estimator based on
        long-run covariance matrices, while ``"svd"`` relies on singular value
        decomposition of the sample mean matrix.

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

    Y_proj = np.where(mask, Y, 0)

    if method == "svd":
        Y_bar = np.nanmean(Y_proj, axis=0)
        U, _, Vt = svd(Y_bar, full_matrices=False)
        R = U[:, :k1]
        C = Vt.T[:, :k2]
    elif method == "pe":
        S_row_sum = np.zeros((p1, p1))
        S_col_sum = np.zeros((p2, p2))
        count_row = np.zeros((p1, p1))
        count_col = np.zeros((p2, p2))

        for t in range(T):
            Y_t = Y_proj[t]
            M_t = mask[t].astype(float)
            S_row_sum += Y_t @ Y_t.T
            S_col_sum += Y_t.T @ Y_t
            count_row += M_t @ M_t.T
            count_col += M_t.T @ M_t

        S_row = np.divide(S_row_sum, np.maximum(count_row, 1), where=count_row > 0)
        S_col = np.divide(S_col_sum, np.maximum(count_col, 1), where=count_col > 0)

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
        F[t] = R.T @ np.where(mask[t], Y[t], 0) @ C

    # initialize MAR coefficients as identity matrices ---------------------
    A = [np.eye(k1) for _ in range(P)]
    B = [np.eye(k2) for _ in range(P)]
    
    # residuals and covariances --------------------------------------------
    resid = Y_proj - np.einsum("ij,tjk,kl->til", R, F, C.T)
    H = np.zeros((p1, p1))
    K = np.zeros((p2, p2))
    for t in range(T):
        for j in range(p2):
            H += np.outer(resid[t, :, j], resid[t, :, j])
        for i in range(p1):
            K += np.outer(resid[t, i, :], resid[t, i, :])
    
    H = H / max(1, T * p2)
    K = K / max(1, T * p1)
    H = 0.5 * (H + H.T)
    K = 0.5 * (K + K.T)
    tr_H = np.trace(H)
    tr_K = np.trace(K)
    if tr_H > 0:
        H *= float(p1) / tr_H
    if tr_K > 0:
        K *= float(p2) / tr_K
    
    # innovation covariances ------------------------------------------------
    Pmat = np.eye(k1)
    Qmat = np.eye(k2)

    Phi = [np.kron(B[l], A[l]) for l in range(P)]
    
    return {
        "R": R,
        "C": C,
        "A": A,
        "B": B,
        "Phi": Phi,
        "H": H,
        "K": K,
        "P": Pmat,
        "Q": Qmat,
        "F": F,
    }


def _construct_state_matrices(
    A: list[np.ndarray] | None,
    B: list[np.ndarray] | None,
    *,
    Phi: list[np.ndarray] | None = None,
    kronecker_only: bool = False,
) -> np.ndarray:
    """Return VAR(1) transition matrix for stacked MAR(P)."""
    
    if kronecker_only:
        if Phi is None:
            raise ValueError("Phi must be provided when kronecker_only=True")
        P = len(Phi)
        if P == 0:
            raise ValueError("Phi must contain at least one matrix")
        r = Phi[0].shape[0]
        d = r * P
        Tmat = np.zeros((d, d))
        Tmat[:r, : r * P] = np.hstack(Phi)
        if P > 1:
            Tmat[r:, :-r] = np.eye(r * (P - 1))
        return Tmat

    if A is None or B is None:
        raise ValueError("A and B must be provided when kronecker_only=False")
    
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
    i1_factors: bool = False,
    *,
    Phi: list[np.ndarray] | None = None,
    kronecker_only: bool = False,
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
    P = len(Phi) if kronecker_only and Phi is not None else len(A)
    d = r if i1_factors else r * P
    
    if mask is None:
        mask = np.ones_like(Y, dtype=bool)

    Pmat = np.eye(k1) if Pmat is None else Pmat
    Qmat = np.eye(k2) if Qmat is None else Qmat

    if kronecker_only:
        if Phi is None:
            raise ValueError("Phi must be provided when kronecker_only=True")
        Phi_list = Phi
    else:
        Phi_list = [np.kron(B[l], A[l]) for l in range(P)]
    Qx = np.kron(Qmat, Pmat)
    if i1_factors:
        Tmat = np.eye(r)
        Q_full = Qx
    else:
        Tmat = _construct_state_matrices(
            A if not kronecker_only else None,
            B if not kronecker_only else None,
            Phi=Phi_list if kronecker_only else None,
            kronecker_only=kronecker_only,
        )
        Q_full = np.zeros((d, d))
        Q_full[:r, :r] = Qx

    R_full = np.kron(K, H)
    Z0 = np.kron(C, R)
    if i1_factors:
        Z_full = Z0
    else:
        Z_full = np.hstack([Z0] + [np.zeros((p1 * p2, r)) for _ in range(P - 1)])

    x_pred = np.zeros(d)
    V_pred = np.eye(d) * (1e4 if i1_factors else 1e2)

    xp = np.zeros((Tn, d))
    Pp = np.zeros((Tn, d, d))
    xf = np.zeros((Tn, d))
    Pf = np.zeros((Tn, d, d))

    # Kalman filter --------------------------------------------------------
    for t in range(Tn):
        if t == 0:
            x_prior = np.zeros(d)
            V_prior = np.eye(d) * (1e4 if i1_factors else 1e2)
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
            S += 1e-8 * np.eye(S.shape[0])
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

    # Rauch‑Tung‑Striebel smoother ----------------------------------------
    xs = np.zeros_like(xf)
    Vs = np.zeros_like(Pf)
    xs[-1] = xf[-1]
    Vs[-1] = Pf[-1]
    J = np.zeros((Tn - 1, d, d))

    for t in range(Tn - 2, -1, -1):
        J[t] = Pf[t] @ Tmat.T @ inv(Pp[t + 1] + 1e-8 * np.eye(d))
        xs[t] = xf[t] + J[t] @ (xs[t + 1] - xp[t + 1])
        Vs[t] = Pf[t] + J[t] @ (Vs[t + 1] - Pp[t + 1]) @ J[t].T

    # covariance of successive states (needed for MAR estimation) ----------
    Vss = np.zeros((Tn - 1, d, d))
    for t in range(Tn - 1):
        Vss[t] = J[t] @ Vs[t + 1]

    F_smooth = xs[:, :r].reshape(Tn, k1, k2)

    # QML log-likelihood computed from smoothed states --------------------
    loglik = 0.0
    Z_base = np.kron(C, R)
    Sigma_U_base = np.kron(K, H)
    for t in range(Tn):
        f_t = xs[t, :r]
        Vt = Vs[t, :r, :r]
        y_vec = Y[t].reshape(-1)
        m_vec = mask[t].reshape(-1)
        idx = np.where(m_vec)[0]
        if idx.size == 0:
            continue
        Z_t = Z_base[idx, :]
        Sigma_U = Sigma_U_base[np.ix_(idx, idx)]
        Sigma_Y = Z_t @ Vt @ Z_t.T + Sigma_U
        Sigma_Y += 1e-8 * np.eye(idx.size)
        innov = y_vec[idx] - Z_t @ f_t
        sign, logdet = np.linalg.slogdet(Sigma_Y)
        loglik -= 0.5 * (
            logdet + innov.T @ inv(Sigma_Y) @ innov + idx.size * np.log(2 * np.pi)
        )
    
    return {
        "F_smooth": F_smooth,
        "V_smooth": Vs,
        "V_ss": Vss,
        "F_filt": xf[:, :r].reshape(Tn, k1, k2),
        "F_pred": xp[:, :r].reshape(Tn, k1, k2),
        "loglik": float(loglik),
    }


def qml_loglik_dmfm(
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
    i1_factors: bool = False,
    *,
    Phi: list[np.ndarray] | None = None,
    kronecker_only: bool = False,
) -> float:
    """Return QML log-likelihood using the Kalman filter."""

    out = kalman_smoother_dmfm(
        Y,
        R,
        C,
        A,
        B,
        H,
        K,
        mask,
        Pmat,
        Qmat,
        i1_factors=i1_factors,
        Phi=Phi,
        kronecker_only=kronecker_only,
    )
    return float(out["loglik"])


def pack_dmfm_parameters(
    R: np.ndarray,
    C: np.ndarray,
    A: list[np.ndarray],
    B: list[np.ndarray],
    H: np.ndarray,
    K: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
) -> np.ndarray:
    """Return flattened parameter vector for optimization."""

    parts = [R.ravel(), C.ravel()]
    parts.extend(a.ravel() for a in A)
    parts.extend(b.ravel() for b in B)
    parts.extend([H.ravel(), K.ravel(), P.ravel(), Q.ravel()])
    return np.concatenate(parts)


def unpack_dmfm_parameters(
    vec: np.ndarray,
    shape_info: dict,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Unpack flattened vector into DMFM parameter matrices."""

    p1 = shape_info["p1"]
    k1 = shape_info["k1"]
    p2 = shape_info["p2"]
    k2 = shape_info["k2"]
    Pord = shape_info["P"]

    idx = 0
    R = vec[idx : idx + p1 * k1].reshape(p1, k1)
    idx += p1 * k1
    C = vec[idx : idx + p2 * k2].reshape(p2, k2)
    idx += p2 * k2

    A = []
    for _ in range(Pord):
        A.append(vec[idx : idx + k1 * k1].reshape(k1, k1))
        idx += k1 * k1

    B = []
    for _ in range(Pord):
        B.append(vec[idx : idx + k2 * k2].reshape(k2, k2))
        idx += k2 * k2

    H = vec[idx : idx + p1 * p1].reshape(p1, p1)
    idx += p1 * p1
    K = vec[idx : idx + p2 * p2].reshape(p2, p2)
    idx += p2 * p2
    Pmat = vec[idx : idx + k1 * k1].reshape(k1, k1)
    idx += k1 * k1
    Qmat = vec[idx : idx + k2 * k2].reshape(k2, k2)

    def _sym_posdef(M: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        M = 0.5 * (M + M.T)
        try:
            w, V = np.linalg.eigh(M)
            w = np.clip(w, eps, None)
            return (V * w) @ V.T
        except np.linalg.LinAlgError:
            return np.eye(M.shape[0]) * eps

    H = _sym_posdef(H)
    K = _sym_posdef(K)
    Pmat = _sym_posdef(Pmat)
    Qmat = _sym_posdef(Qmat)

    return R, C, A, B, H, K, Pmat, Qmat


def qml_objective_dmfm(
    params_vec: np.ndarray,
    Y: np.ndarray,
    shape_info: dict,
    mask: np.ndarray | None = None,
    *,
    Phi: list[np.ndarray] | None = None,
    kronecker_only: bool = False,
) -> float:
    """Return negative QML log-likelihood for optimization."""

    R, C, A, B, H, K, Pmat, Qmat = unpack_dmfm_parameters(params_vec, shape_info)
    out = kalman_smoother_dmfm(
        Y,
        R,
        C,
        A,
        B,
        H,
        K,
        mask,
        Pmat,
        Qmat,
        Phi=Phi,
        kronecker_only=kronecker_only,
    )
    return -float(out["loglik"])


def optimize_qml_dmfm(
    Y: np.ndarray,
    k1: int,
    k2: int,
    P: int,
    mask: np.ndarray | None = None,
    init_params: dict | None = None,
) -> dict:
    """Optimize the QML objective for the DMFM."""

    try:
        from scipy.optimize import minimize
    except Exception:  # pragma: no cover - fallback when scipy unavailable
        minimize = None

    Y = np.asarray(Y, dtype=float)
    Tn, p1, p2 = Y.shape

    if init_params is None:
        init_params = initialize_dmfm(Y, k1, k2, P, mask)

    shape_info = {"p1": p1, "k1": k1, "p2": p2, "k2": k2, "P": P}

    x0 = pack_dmfm_parameters(
        init_params["R"],
        init_params["C"],
        init_params["A"],
        init_params["B"],
        init_params["H"],
        init_params["K"],
        init_params.get("P", np.eye(k1)),
        init_params.get("Q", np.eye(k2)),
    )

    obj = lambda v: qml_objective_dmfm(v, Y, shape_info, mask)
    if minimize is not None:
        res = minimize(obj, x0, method="L-BFGS-B")
        opt_x = res.x
    else:
        warnings.warn("scipy not installed - skipping optimization")
        res = None
        opt_x = x0

    R, C, A, B, H, K, Pmat, Qmat = unpack_dmfm_parameters(opt_x, shape_info)
    smooth = kalman_smoother_dmfm(
        Y,
        R,
        C,
        A,
        B,
        H,
        K,
        mask,
        Pmat,
        Qmat,
    )

    return {
        "R": R,
        "C": C,
        "A": A,
        "B": B,
        "H": H,
        "K": K,
        "P": Pmat,
        "Q": Qmat,
        "F": smooth["F_smooth"],
        "loglik": [smooth["loglik"]],
        "optimization_result": res,
        "frozen": True,
    }


def em_step_dmfm(
    Y: np.ndarray,
    params: dict,
    mask: np.ndarray | None = None,
    nonstationary: bool = False,
    i1_factors: bool = False,
    *,
    kronecker_only: bool = False,
) -> tuple[dict, float, float]:
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
    nonstationary : bool, optional
        If ``True`` keep MAR coefficients fixed at identity and allow larger
        state innovations.
    i1_factors : bool, optional
        If ``True`` treat the latent factors as an I(1) process.
    kronecker_only : bool, optional
        If ``True`` estimate and use MAR dynamics directly for ``vec(F_t)`` via
        stacked Kronecker matrices ``Phi``.
    
    Returns
    -------
    tuple
        Updated parameter dictionary, the relative parameter change and the
        log-likelihood value.
    """
    if params.get("frozen"):
        ll = qml_loglik_dmfm(
            Y,
            params["R"],
            params["C"],
            params["A"],
            params["B"],
            params["H"],
            params["K"],
            mask,
            params.get("P"),
            params.get("Q"),
            i1_factors=i1_factors,
            Phi=params.get("Phi"),
            kronecker_only=kronecker_only,
        )
        return params, 0.0, ll
    
    R = params["R"]
    C = params["C"]
    A = params["A"]
    B = params["B"]
    H = params["H"]
    K = params["K"]
    Pmat = params.get("P", np.eye(R.shape[1]))
    Qmat = params.get("Q", np.eye(C.shape[1]))

    scale = 1.0
    if nonstationary:
        scale = 10.0
    smooth = kalman_smoother_dmfm(
        Y,
        R,
        C,
        A,
        B,
        H,
        K,
        mask,
        Pmat * scale,
        Qmat * scale,
        i1_factors=i1_factors,
        Phi=params.get("Phi"),
        kronecker_only=kronecker_only,
    )
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
            X = (F[t] @ C.T).T[m, :]  # (sum m) x k1
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

    # Orthonormalize R and C and adjust F ---------------------------------
    R_new, R_fac = np.linalg.qr(R_new)
    C_new, C_fac = np.linalg.qr(C_new)
    for t in range(Tn):
        F[t] = R_fac @ F[t] @ C_fac.T
    
    # Update MAR matrices or Phi -----------------------------------------
    if nonstationary:
        A_new = A
        B_new = B
        Phi_new = params.get("Phi")
    else:
        if kronecker_only:
            r_vec = k1 * k2
            X_rows = []
            Y_rows = []
            for t in range(Pord, Tn):
                X_rows.append(
                    np.concatenate([F[t - l - 1].reshape(-1) for l in range(Pord)])
                )
                Y_rows.append(F[t].reshape(-1))
            if X_rows:
                Xmat = np.vstack(X_rows)
                Ymat = np.vstack(Y_rows)
                XTX = Xmat.T @ Xmat
                lam = 1e-6
                coeff = np.linalg.solve(XTX + lam * np.eye(XTX.shape[0]), Xmat.T @ Ymat)
                Phi_new = [
                    coeff[l * r_vec : (l + 1) * r_vec, :].T for l in range(Pord)
                ]
            else:
                Phi_new = params.get("Phi")
            A_new = A
            B_new = B
        else:
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
                A_est = A_num @ inv(A_den + 1e-8 * np.eye(k1))
                B_est = B_num @ inv(B_den + 1e-8 * np.eye(k2))
                if np.linalg.norm(A_den) < 1e-4:
                    A_est = A[ell]
                if np.linalg.norm(B_den) < 1e-4:
                    B_est = B[ell]
                A_est = np.clip(A_est, -0.99, 0.99)
                B_est = np.clip(B_est, -0.99, 0.99)
                A_new[ell] = A_est
                B_new[ell] = B_est
            Phi_new = [np.kron(B_new[l], A_new[l]) for l in range(Pord)]
    
    # Update innovation covariances P and Q --------------------------------
    Vs = smooth["V_smooth"]
    Vss = smooth["V_ss"]
    if i1_factors:
        Tmat_new = np.eye(k1 * k2)
        d = r = k1 * k2
    else:
        if kronecker_only:
            d = k1 * k2 * Pord
            r = k1 * k2
            Tmat_new = _construct_state_matrices(
                None,
                None,
                Phi=Phi_new,
                kronecker_only=True,
            )
        else:
            Tmat_new = _construct_state_matrices(A_new, B_new)
            d = k1 * k2 * Pord
            r = k1 * k2
    
    P_new = np.zeros((k1, k1))
    Q_new = np.zeros((k2, k2))
    count = 0
    if i1_factors:
        for t in range(1, Tn):
            diff = F[t] - F[t - 1]
            diff_vec = diff.reshape(-1)
            W = (
                Vs[t][:r, :r]
                + Vs[t - 1][:r, :r]
                - Vss[t - 1][:r, :r]
                - Vss[t - 1][:r, :r].T
                + np.outer(diff_vec, diff_vec)
            )
            for i1 in range(k1):
                idx1 = slice(i1 * k2, (i1 + 1) * k2)
                for i2 in range(k1):
                    idx2 = slice(i2 * k2, (i2 + 1) * k2)
                    P_new[i1, i2] += np.trace(W[idx1, idx2])
            for j1 in range(k2):
                idxc1 = [i * k2 + j1 for i in range(k1)]
                for j2 in range(k2):
                    idxc2 = [i * k2 + j2 for i in range(k1)]
                    Q_new[j1, j2] += np.trace(W[np.ix_(idxc1, idxc2)])
            count += 1
    else:
        for t in range(Pord, Tn):
            x_t = np.concatenate([F[t - l].reshape(-1) for l in range(Pord)])
            x_tm1 = np.concatenate([F[t - 1 - l].reshape(-1) for l in range(Pord)])
            E_tt = Vs[t] + np.outer(x_t, x_t)
            E_tm1 = Vs[t - 1] + np.outer(x_tm1, x_tm1)
            E_cross = Vss[t - 1] + np.outer(x_t, x_tm1)
            W_full = (
                E_tt
                - E_cross @ Tmat_new.T
                - Tmat_new @ E_cross.T
                + Tmat_new @ E_tm1 @ Tmat_new.T
            )
            W = W_full[:r, :r]
            for i1 in range(k1):
                idx1 = slice(i1 * k2, (i1 + 1) * k2)
                for i2 in range(k1):
                    idx2 = slice(i2 * k2, (i2 + 1) * k2)
                    P_new[i1, i2] += np.trace(W[idx1, idx2])
            for j1 in range(k2):
                idxc1 = [i * k2 + j1 for i in range(k1)]
                for j2 in range(k2):
                    idxc2 = [i * k2 + j2 for i in range(k1)]
                    Q_new[j1, j2] += np.trace(W[np.ix_(idxc1, idxc2)])
            count += 1
    denom = max(1, count)
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
    for t in range(Tn):
        res_t = resid[t]
        if mask is not None:
            res_t = np.where(mask[t], res_t, 0)
        H_new += res_t @ res_t.T
        K_new += res_t.T @ res_t

    denom_H = max(1.0, Tn * p2)
    denom_K = max(1.0, Tn * p1)
    H_new /= denom_H
    K_new /= denom_K
    H_new = 0.5 * (H_new + H_new.T)
    K_new = 0.5 * (K_new + K_new.T)
    tr_H = np.trace(H_new)
    tr_K = np.trace(K_new)
    if tr_H > 0:
        H_new *= float(p1) / tr_H
    if tr_K > 0:
        K_new *= float(p2) / tr_K
    
    new_params = {
        "R": R_new,
        "C": C_new,
        "A": A_new,
        "B": B_new,
        "Phi": Phi_new,
        "H": H_new,
        "K": K_new,
        "P": P_new,
        "Q": Q_new,
    }

    # compute smoothed factors and log-likelihood under updated parameters
    smooth_new = kalman_smoother_dmfm(
        Y,
        new_params["R"],
        new_params["C"],
        new_params["A"],
        new_params["B"],
        new_params["H"],
        new_params["K"],
        mask,
        new_params["P"],
        new_params["Q"],
        i1_factors=i1_factors,
        Phi=new_params.get("Phi"),
        kronecker_only=kronecker_only,
    )
    new_params["F"] = smooth_new["F_smooth"]
    ll = smooth_new["loglik"]
    
    diff = 0.0
    for key in ["R", "C"]:
        diff += np.linalg.norm(params[key] - new_params[key])
    for l in range(Pord):
        if kronecker_only:
            diff += np.linalg.norm(params["Phi"][l] - Phi_new[l])
        else:
            diff += np.linalg.norm(params["A"][l] - new_params["A"][l])
            diff += np.linalg.norm(params["B"][l] - new_params["B"][l])
    diff /= max(1.0, np.linalg.norm(params["R"]))

    return new_params, diff, ll


def fit_dmfm_em(
    Y: np.ndarray,
    k1: int,
    k2: int,
    P: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    mask: np.ndarray | None = None,
    nonstationary: bool = False,
    i1_factors: bool = False,
    return_se: bool = False,
    use_qml_opt: bool = False,
    *,
    kronecker_only: bool = False,
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
    nonstationary : bool, optional
        If ``True`` keep the MAR process close to a random walk and do not
        re-estimate the MAR coefficients.
    i1_factors : bool, optional
        If ``True`` model the latent factors as an I(1) process.
    return_se : bool, optional
        If ``True`` the dictionary additionally contains a ``"standard_errors"``
        field with standard errors for ``R`` and ``C``.
    kronecker_only : bool, optional
        Estimate the MAR dynamics for ``vec(F_t)`` directly using stacked
        transition matrices ``Phi``.
    
    Returns
    -------
    dict
        Fitted parameters, smoothed factors and log-likelihood trace.
    """
    if use_qml_opt:
        res = optimize_qml_dmfm(
            Y,
            k1,
            k2,
            P,
            mask=mask,
            init_params=None,
        )
        if return_se:
            res["standard_errors"] = compute_standard_errors_dmfm(
                Y, res["R"], res["C"], res["F"], mask
            )
        return res
    
    params = initialize_dmfm(Y, k1, k2, P, mask, method="pe")
    if nonstationary or kronecker_only:
        params["A"] = [np.eye(k1) for _ in range(P)]
        params["B"] = [np.eye(k2) for _ in range(P)]
    loglik_trace: list[float] = []
    diff_trace: list[float] = []
    ll_diff_trace: list[float] = []
    last_ll = -np.inf
    for it in range(max_iter):
        params, diff, ll = em_step_dmfm(
            Y,
            params,
            mask,
            nonstationary,
            i1_factors=i1_factors,
            kronecker_only=kronecker_only,
        )
        ll_change = ll - last_ll if np.isfinite(last_ll) else np.inf
        diff_trace.append(diff)
        ll_diff_trace.append(ll_change)
        if it > 0 and ll_change < -1e-6:
            warnings.warn(
                f"Log-likelihood decreased by {ll_change:.3e} at iteration {it}."
            )
            break
        loglik_trace.append(ll)
        last_ll = ll
        if diff < tol:
            break
    params["loglik"] = loglik_trace
    params["param_diff"] = diff_trace
    params["ll_diff"] = ll_diff_trace
    params["frozen"] = True
    if return_se:
        params["standard_errors"] = compute_standard_errors_dmfm(
            Y, params["R"], params["C"], params["F"], mask
        )
    return params


def compute_standard_errors_dmfm(
    Y: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    F: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict:
    """Return approximate standard errors for ``R`` and ``C``.

    Parameters
    ----------
    Y : ndarray
        Observed array ``(T, p1, p2)``.
    R, C : ndarray
        Estimated loading matrices.
    F : ndarray
        Smoothed factor path ``(T, k1, k2)``.
    mask : ndarray or None, optional
        Observation mask where ``True`` indicates observed entries.

    Returns
    -------
    dict
        Dictionary with entries ``se_R`` and ``se_C`` containing the standard
        errors. Additionally ``ci_R`` and ``ci_C`` provide 95% confidence
        intervals.
    """

    Y = np.asarray(Y, dtype=float)
    R = np.asarray(R, dtype=float)
    C = np.asarray(C, dtype=float)
    F = np.asarray(F, dtype=float)

    Tn, p1, p2 = Y.shape
    k1 = R.shape[1]
    k2 = C.shape[1]

    if mask is None:
        mask = np.ones_like(Y, dtype=bool)

    se_R = np.zeros_like(R)
    se_C = np.zeros_like(C)

    # standard errors for R -------------------------------------------------
    for i in range(p1):
        rss = 0.0
        n_obs = 0
        XtX_i = np.zeros((k1, k1))
        for t in range(Tn):
            m = mask[t, i]
            X_t = (F[t] @ C.T).T[m]
            y = Y[t, i, m]
            if y.size == 0:
                continue
            XtX_i += X_t.T @ X_t
            pred = X_t @ R[i]
            resid = y - pred
            rss += resid @ resid
            n_obs += y.size
        if n_obs > k1:
            sigma2 = rss / max(1.0, n_obs - k1)
        else:
            sigma2 = 0.0
        cov = sigma2 * inv(XtX_i + 1e-8 * np.eye(k1))
        se_R[i] = np.sqrt(np.diag(cov))

    # standard errors for C -------------------------------------------------
    for j in range(p2):
        rss = 0.0
        n_obs = 0
        ZtZ_j = np.zeros((k2, k2))
        for t in range(Tn):
            m = mask[t, :, j]
            Z_t = (R @ F[t])[m]
            y = Y[t, m, j]
            if y.size == 0:
                continue
            ZtZ_j += Z_t.T @ Z_t
            pred = Z_t @ C[j]
            resid = y - pred
            rss += resid @ resid
            n_obs += y.size
        if n_obs > k2:
            sigma2 = rss / max(1.0, n_obs - k2)
        else:
            sigma2 = 0.0
        cov = sigma2 * inv(ZtZ_j + 1e-8 * np.eye(k2))
        se_C[j] = np.sqrt(np.diag(cov))

    ci_R = np.stack([R - 1.96 * se_R, R + 1.96 * se_R], axis=-1)
    ci_C = np.stack([C - 1.96 * se_C, C + 1.96 * se_C], axis=-1)

    return {"se_R": se_R, "se_C": se_C, "ci_R": ci_R, "ci_C": ci_C}


def select_dmfm_rank(
    Y: np.ndarray,
    max_k: int = 10,
    method: str = "ratio",
) -> tuple[int, int]:
    """Return suitable ``(k1, k2)`` values for a DMFM.

    The function inspects the eigenvalues of the row and column
    covariance matrices. ``method='ratio'`` selects the factor numbers
    by the eigenvalue ratio rule while ``method='bai-ng'`` applies the
    Bai--Ng information criterion.

    Parameters
    ----------
    Y : array_like
        Data array of shape ``(T, p1, p2)``.
    max_k : int, optional
        Maximum number of row or column factors considered.
    method : {{'ratio', 'bai-ng'}}, optional
        Selection procedure to use.

    Returns
    -------
    tuple[int, int]
        Suggested number of row and column factors.
    """

    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 3:
        raise ValueError("Y must be a 3D array")
    T, p1, p2 = Y.shape

    S_row = np.zeros((p1, p1))
    S_col = np.zeros((p2, p2))
    for t in range(T):
        S_row += Y[t] @ Y[t].T
        S_col += Y[t].T @ Y[t]
    if p2 > 0:
        S_row /= T * p2
    if p1 > 0:
        S_col /= T * p1

    eval_row = np.sort(np.linalg.eigvalsh(S_row))[::-1]
    eval_col = np.sort(np.linalg.eigvalsh(S_col))[::-1]

    def _select(evals: np.ndarray, N: int, TT: int) -> int:
        evals = np.maximum(evals, 0.0)
        if method == "ratio":
            if evals.size <= 1:
                return 1
            kmax = min(max_k, evals.size - 1)
            ratios = evals[:kmax] / (evals[1 : kmax + 1] + 1e-12)
            return int(np.argmax(ratios)) + 1
        elif method in {"bai-ng", "bai", "ic"}:
            kmax = min(max_k, evals.size)
            ics = []
            for r in range(kmax + 1):
                if r >= evals.size:
                    ics.append(np.inf)
                    continue
                resid = np.mean(evals[r:])
                penalty = r * (N + TT) / (N * TT) * np.log(N * TT / (N + TT))
                ics.append(np.log(resid + 1e-12) + penalty)
            return int(np.argmin(ics))
        else:
            raise ValueError(f"Unknown method: {method}")

    k1 = _select(eval_row, p1, T * p2)
    k2 = _select(eval_col, p2, T * p1)
    k1 = max(1, min(k1, p1))
    k2 = max(1, min(k2, p2))
    return k1, k2


def select_dmfm_qml(
    Y: np.ndarray,
    max_k: int = 5,
    max_P: int = 2,
    criterion: str = "bic",
    mask: np.ndarray | None = None,
) -> tuple[int, int, int]:
    """Select (k1, k2, P) using QML-based information criteria.

    Parameters
    ----------
    Y : array_like
        Observation array ``(T, p1, p2)``.
    max_k : int, optional
        Maximum factor numbers considered for rows and columns.
    max_P : int, optional
        Maximum MAR order to consider.
    criterion : {'aic', 'bic'}, optional
        Information criterion to minimize.
    mask : ndarray or None, optional
        Binary mask indicating observed entries.

    Returns
    -------
    tuple[int, int, int]
        Selected ``(k1, k2, P)`` configuration.
    """

    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 3:
        raise ValueError("Y must be a 3D array")

    T, p1, p2 = Y.shape
    best_ic = np.inf
    best_cfg = (1, 1, 1)
    criterion = criterion.lower()

    for k1 in range(1, max_k + 1):
        for k2 in range(1, max_k + 1):
            for P in range(1, max_P + 1):
                try:
                    res = fit_dmfm_em(Y, k1, k2, P, max_iter=25, mask=mask)
                except Exception:
                    continue
                if not res.get("loglik"):
                    continue
                loglik = res["loglik"][-1]
                n_params = (
                    p1 * k1
                    + p2 * k2
                    + P * (k1**2 + k2**2)
                    + p1
                    + p2
                    + k1 * (k1 + 1) / 2
                    + k2 * (k2 + 1) / 2
                )
                n_params = int(round(n_params))
                if criterion == "aic":
                    ic = -2.0 * loglik + 2.0 * n_params
                else:
                    ic = -2.0 * loglik + np.log(T * p1 * p2) * n_params
                if ic < best_ic:
                    best_ic = ic
                    best_cfg = (k1, k2, P)

    return best_cfg
