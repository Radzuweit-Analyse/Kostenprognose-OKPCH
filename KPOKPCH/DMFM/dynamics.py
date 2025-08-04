"""Dynamics of the DMFM latent factors."""

from __future__ import annotations

import numpy as np


class DMFMDynamics:
    """MAR(P) dynamics for the latent factor process."""

    def __init__(self, A: list[np.ndarray], B: list[np.ndarray], Pmat: np.ndarray, Qmat: np.ndarray) -> None:
        self.A = A
        self.B = B
        self.Pmat = Pmat
        self.Qmat = Qmat

    # ------------------------------------------------------------------
    def evolve(self, F_history: list[np.ndarray]) -> np.ndarray:
        """Return the next factor given past ``F`` values.

        Parameters
        ----------
        F_history : list of arrays
            Contains the most recent ``P`` factor matrices ordered from
            lag 1 to ``P``.
        """

        F_next = np.zeros_like(F_history[0])
        for l, F_lag in enumerate(F_history):
            F_next += self.A[l] @ F_lag @ self.B[l].T
        return F_next

    # ------------------------------------------------------------------
    def estimate(self, F: np.ndarray) -> None:
        """Estimate MAR coefficients from a factor sequence ``F``.

        The update is performed via least squares separately for the row
        and column transition matrices following Barigozzi and Trapin
        (2025).
        """

        P = len(self.A)
        k1, k2 = self.A[0].shape[0], self.B[0].shape[0]
        A_new = [np.zeros_like(self.A[0]) for _ in range(P)]
        B_new = [np.zeros_like(self.B[0]) for _ in range(P)]
        for ell in range(P):
            A_num = np.zeros((k1, k1))
            A_den = np.zeros((k1, k1))
            B_num = np.zeros((k2, k2))
            B_den = np.zeros((k2, k2))
            for t in range(ell + 1, F.shape[0]):
                F_pred_other = np.zeros((k1, k2))
                for j in range(P):
                    if j == ell or t - j - 1 < 0:
                        continue
                    F_pred_other += self.A[j] @ F[t - j - 1] @ self.B[j].T
                Y_res = F[t] - F_pred_other
                X_A = F[t - ell - 1] @ self.B[ell].T
                A_num += Y_res @ X_A.T
                A_den += X_A @ X_A.T
                X_B = F[t - ell - 1].T @ self.A[ell].T
                B_num += Y_res.T @ X_B.T
                B_den += X_B @ X_B.T
            if np.linalg.norm(A_den) > 1e-8:
                A_new[ell] = np.clip(A_num @ np.linalg.pinv(A_den), -0.99, 0.99)
            else:
                A_new[ell] = self.A[ell]
            if np.linalg.norm(B_den) > 1e-8:
                B_new[ell] = np.clip(B_num @ np.linalg.pinv(B_den), -0.99, 0.99)
            else:
                B_new[ell] = self.B[ell]
        self.A = A_new
        self.B = B_new
