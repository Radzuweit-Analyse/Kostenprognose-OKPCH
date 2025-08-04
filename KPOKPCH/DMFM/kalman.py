"""Kalman filtering and smoothing for the DMFM."""

from __future__ import annotations

import numpy as np
from numpy.linalg import inv

from .model import DMFMModel


class KalmanFilterDMFM:
    """Kalman filter and RTS smoother for the DMFM."""

    def __init__(self, model: DMFMModel) -> None:
        self.model = model

    # ------------------------------------------------------------------
    def _construct_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        m = self.model
        k1, k2, P = m.k1, m.k2, m.P
        r = k1 * k2
        Phi = [np.kron(m.B[l], m.A[l]) for l in range(P)]
        Tmat = np.zeros((r * P, r * P))
        Tmat[:r, : r * P] = np.hstack(Phi)
        if P > 1:
            Tmat[r:, :-r] = np.eye(r * (P - 1))
        Q_block = np.kron(m.Qmat, m.Pmat)
        Q_full = np.zeros((r * P, r * P))
        Q_full[:r, :r] = Q_block
        Z_full = np.zeros((m.p1 * m.p2, r * P))
        Z_full[:, :r] = np.kron(m.C, m.R)
        R_full = np.kron(m.K, m.H)
        if m.diagonal_idiosyncratic:
            R_full = np.kron(np.diag(np.diag(m.K)), np.diag(np.diag(m.H)))
        return Tmat, Q_full, Z_full, R_full

    # ------------------------------------------------------------------
    def filter(self, Y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Tmat, Q_full, Z_full, R_full = self._construct_matrices()
        return _kalman_filter_dmfm(
            Y, mask, Tmat, Q_full, Z_full, R_full, i1_factors=False
        )

    # ------------------------------------------------------------------
    def smooth(
        self, xp: np.ndarray, Pp: np.ndarray, xf: np.ndarray, Pf: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Tmat, _, _, _ = self._construct_matrices()
        return _kalman_smooth_dmfm(xp, Pp, xf, Pf, Tmat)

    # ------------------------------------------------------------------
    def log_likelihood(
        self, Y: np.ndarray, mask: np.ndarray, xs: np.ndarray, Vs: np.ndarray
    ) -> float:
        m = self.model
        return _kalman_loglik_dmfm(
            Y, mask, xs, Vs, m.R, m.C, m.H, m.K, m.diagonal_idiosyncratic
        )


# ----------------------------------------------------------------------
# Internal helper routines adapted from the original implementation
# ----------------------------------------------------------------------

def _kalman_filter_dmfm(
    Y: np.ndarray,
    mask: np.ndarray,
    Tmat: np.ndarray,
    Q_full: np.ndarray,
    Z_full: np.ndarray,
    R_full: np.ndarray,
    i1_factors: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Tn, d = mask.shape[0], Tmat.shape[0]
    xp = np.zeros((Tn, d))
    Pp = np.zeros((Tn, d, d))
    xf = np.zeros((Tn, d))
    Pf = np.zeros((Tn, d, d))
    x_pred = np.zeros(d)
    V_pred = np.eye(d) * (1e4 if i1_factors else 1e2)
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
            try:
                S_inv = inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)
            K_gain = V_prior @ Z.T @ S_inv
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
    return xp, Pp, xf, Pf


def _kalman_smooth_dmfm(
    xp: np.ndarray,
    Pp: np.ndarray,
    xf: np.ndarray,
    Pf: np.ndarray,
    Tmat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Tn, d = xf.shape
    xs = np.zeros_like(xf)
    Vs = np.zeros_like(Pf)
    xs[-1] = xf[-1]
    Vs[-1] = Pf[-1]
    J = np.zeros((Tn - 1, d, d))
    for t in range(Tn - 2, -1, -1):
        try:
            inv_term = inv(Pp[t + 1] + 1e-8 * np.eye(d))
        except np.linalg.LinAlgError:
            inv_term = np.linalg.pinv(Pp[t + 1] + 1e-8 * np.eye(d))
        J[t] = Pf[t] @ Tmat.T @ inv_term
        xs[t] = xf[t] + J[t] @ (xs[t + 1] - xp[t + 1])
        Vs[t] = Pf[t] + J[t] @ (Vs[t + 1] - Pp[t + 1]) @ J[t].T
    Vss = np.zeros((Tn - 1, d, d))
    for t in range(Tn - 1):
        Vss[t] = J[t] @ Vs[t + 1]
    return xs, Vs, Vss


def _kalman_loglik_dmfm(
    Y: np.ndarray,
    mask: np.ndarray,
    xs: np.ndarray,
    Vs: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    H: np.ndarray,
    K: np.ndarray,
    diagonal_idiosyncratic: bool,
) -> float:
    Tn = Y.shape[0]
    k1 = R.shape[1]
    k2 = C.shape[1]
    r = k1 * k2
    loglik = 0.0
    Z_base = np.kron(C, R)
    Sigma_U_base = np.kron(K, H)
    if diagonal_idiosyncratic:
        Sigma_U_base = np.kron(np.diag(np.diag(K)), np.diag(np.diag(H)))
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
        try:
            Sigma_inv = inv(Sigma_Y)
        except np.linalg.LinAlgError:
            Sigma_inv = np.linalg.pinv(Sigma_Y)
        loglik -= 0.5 * (
            logdet + innov.T @ Sigma_inv @ innov + idx.size * np.log(2 * np.pi)
        )
    return float(loglik)
