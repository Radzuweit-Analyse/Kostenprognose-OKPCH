"""EM algorithm for estimating the DMFM."""

from __future__ import annotations

from numpy.linalg import inv
import numpy as np

from .model import DMFMModel
from .dynamics import DMFMDynamics
from .kalman import KalmanFilterDMFM
from . import utils


class EMEstimatorDMFM:
    """Estimate DMFM parameters via the EM algorithm."""

    def __init__(self, model: DMFMModel) -> None:
        self.model = model
        self.loglik_trace: list[float] = []
        self.diff_trace: list[float] = []

    # ------------------------------------------------------------------
    def fit(
        self,
        Y: np.ndarray,
        mask: np.ndarray | None = None,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> None:
        """Run full EM iterations until convergence."""
        if mask is None:
            mask = np.ones_like(Y, dtype=bool)

        k1, k2 = self.model.k1, self.model.k2
        Pord = len(self.model.A)
        r = k1 * k2
        T = Y.shape[0]

        kf = KalmanFilterDMFM(self.model)
        last_ll = -np.inf

        for it in range(max_iter):
            # ---------- E-STEP ----------
            xp, Pp, xf, Pf = kf.filter(Y, mask)
            xs, Vs, Vss = kf.smooth(xp, Pp, xf, Pf)

            # Extract factors
            F = xs[:, :r].reshape(T, k1, k2)

            # ---------- M-STEP ----------
            # Update R
            R_new = _update_row_loadings(Y, F, self.model.C, self.model.R, mask)

            # Update C
            C_new = _update_col_loadings(Y, F, R_new, self.model.C, mask)

            # Orthonormalize
            R_new, R_fac = np.linalg.qr(R_new)
            C_new, C_fac = np.linalg.qr(C_new)
            for t in range(F.shape[0]):
                F[t] = R_fac @ F[t] @ C_fac.T

            # Update dynamics
            A_new, B_new, Phi_new = _update_dynamics(
                F,
                self.model.A,
                self.model.B,
                Pord,
                k1,
                k2,
                self.model.dynamics is not None and self.model.dynamics.nonstationary,
                self.model.dynamics is not None and self.model.dynamics.kronecker_only,
            )

            # Update innovations
            P_new, Q_new = _update_innovations(
                F, Vs, Vss, A_new, B_new, Phi_new,
                self.model.dynamics is not None and self.model.dynamics.i1_factors,
                self.model.dynamics is not None and self.model.dynamics.kronecker_only,
            )

            # Update idiosyncratic covariances
            H_new, K_new = _update_idiosyncratic(
                Y, F, R_new, C_new, mask, self.model.diagonal_idiosyncratic
            )

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

            # ---------- Convergence ----------
            ll = kf.log_likelihood(Y, mask, xs, Vs)

            diff = _compute_param_diff(
                self.model.__dict__, new_params, Pord, self.model.dynamics is not None and self.model.dynamics.kronecker_only
            )

            self.loglik_trace.append(ll)
            self.diff_trace.append(diff)

            if it > 0 and (ll - last_ll) < -1e-6:
                print(f"⚠️  Warning: Log-likelihood decreased at iteration {it}")
                break

            if diff < tol:
                break

            last_ll = ll


    # ------------------------------------------------------------------
    def get_factors(self) -> np.ndarray:
        """Return the smoothed factor sequence."""
        if self.model.F is None:
            raise RuntimeError("Estimator has not been fitted yet")
        return self.model.F

    # ------------------------------------------------------------------
    def get_loglik_trace(self) -> list[float]:
        """Return the log-likelihood values across EM iterations."""
        return self.loglik_trace


# ----------------------------------------------------------------------
# Helper routines copied from original implementation
# ----------------------------------------------------------------------

def _update_row_loadings(
    Y: np.ndarray, F: np.ndarray, C: np.ndarray, R: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    Tn, p1, p2 = Y.shape
    R_new = np.zeros_like(R)
    for i in range(p1):
        X_stack = []
        y_stack = []
        for t in range(Tn):
            m = mask[t, i, :]
            if not m.any():
                continue
            X = (F[t] @ C.T).T[m, :]
            y_stack.append(Y[t, i, m])
            X_stack.append(X)
        if X_stack:
            Xmat = np.vstack(X_stack)
            yvec = np.concatenate(y_stack)
            R_new[i] = np.linalg.lstsq(Xmat, yvec, rcond=None)[0]
        else:
            R_new[i] = R[i]
    return R_new


def _update_col_loadings(
    Y: np.ndarray, F: np.ndarray, R_new: np.ndarray, C: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    Tn, p1, p2 = Y.shape
    C_new = np.zeros_like(C)
    for j in range(p2):
        X_stack = []
        y_stack = []
        for t in range(Tn):
            m = mask[t, :, j]
            if not m.any():
                continue
            X = (R_new @ F[t])[m, :]
            y_stack.append(Y[t, m, j])
            X_stack.append(X)
        if X_stack:
            Xmat = np.vstack(X_stack)
            yvec = np.concatenate(y_stack)
            C_new[j] = np.linalg.lstsq(Xmat, yvec, rcond=None)[0]
        else:
            C_new[j] = C[j]
    return C_new


def _update_dynamics(F, A, B, Pord, k1, k2, nonstationary, kronecker_only):
    if nonstationary:
        return A, B, [np.kron(B[l], A[l]) for l in range(Pord)]
    if kronecker_only:
        r_vec = k1 * k2
        X_rows = []
        Y_rows = []
        for t in range(Pord, F.shape[0]):
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
            Phi_new = [coeff[l * r_vec : (l + 1) * r_vec, :].T for l in range(Pord)]
        else:
            Phi_new = [np.kron(B[l], A[l]) for l in range(Pord)]
        return A, B, Phi_new
    A_new = [np.zeros_like(A[0]) for _ in range(Pord)]
    B_new = [np.zeros_like(B[0]) for _ in range(Pord)]
    for ell in range(Pord):
        A_num = np.zeros((k1, k1))
        A_den = np.zeros((k1, k1))
        B_num = np.zeros((k2, k2))
        B_den = np.zeros((k2, k2))
        for t in range(ell + 1, F.shape[0]):
            F_pred_other = np.zeros((k1, k2))
            for j in range(Pord):
                if j == ell or t - j - 1 < 0:
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
        A_new[ell] = np.clip(A_est, -0.99, 0.99)
        B_new[ell] = np.clip(B_est, -0.99, 0.99)
    Phi_new = [np.kron(B_new[l], A_new[l]) for l in range(Pord)]
    return A_new, B_new, Phi_new


def _update_innovations(F, Vs, Vss, A_new, B_new, Phi_new, i1_factors, kronecker_only):
    Tn, k1, k2 = F.shape
    Pord = len(A_new)
    if i1_factors:
        r = k1 * k2
    else:
        if kronecker_only:
            r = k1 * k2
            Tmat_new = _construct_state_matrices(
                None, None, Phi=Phi_new, kronecker_only=True
            )
        else:
            Tmat_new = _construct_state_matrices(A_new, B_new)
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
    return P_new, Q_new


def _update_idiosyncratic(Y, F, R_new, C_new, mask, diagonal_idiosyncratic):
    Tn, p1, p2 = Y.shape
    H_new = np.zeros((p1, p1))
    K_new = np.zeros((p2, p2))
    resid = np.zeros_like(Y)
    for t in range(Tn):
        resid[t] = Y[t] - R_new @ F[t] @ C_new.T
    if mask is not None:
        resid = np.where(mask, resid, np.nan)
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
    if diagonal_idiosyncratic:
        H_new = np.diag(np.diag(H_new))
        K_new = np.diag(np.diag(K_new))
    tr_H = np.trace(H_new)
    tr_K = np.trace(K_new)
    if tr_H > 0:
        H_new *= float(p1) / tr_H
    if tr_K > 0:
        K_new *= float(p2) / tr_K
    return H_new, K_new


def _compute_param_diff(params, new_params, Pord, kronecker_only):
    diff = 0.0
    for key in ["R", "C"]:
        diff += np.linalg.norm(params[key] - new_params[key])
    for l in range(Pord):
        if kronecker_only:
            diff += np.linalg.norm(params["Phi"][l] - new_params["Phi"][l])
        else:
            diff += np.linalg.norm(params["A"][l] - new_params["A"][l])
            diff += np.linalg.norm(params["B"][l] - new_params["B"][l])
    diff /= max(1.0, np.linalg.norm(params["R"]))
    return diff


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