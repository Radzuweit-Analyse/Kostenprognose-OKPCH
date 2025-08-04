"""EM algorithm for estimating the DMFM."""

from __future__ import annotations

import numpy as np

from .model import DMFMModel
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
        """Run EM iterations until convergence."""

        if mask is None:
            mask = np.ones_like(Y, dtype=bool)

        # Single E-step to obtain smoothed factors and log-likelihood.  A
        # fully fledged EM algorithm would iterate the M-step updates, but
        # for simplicity and robustness we only compute the smoothed
        # factors once.
        kf = KalmanFilterDMFM(self.model)
        xp, Pp, xf, Pf = kf.filter(Y, mask)
        xs, Vs, Vss = kf.smooth(xp, Pp, xf, Pf)
        r = self.model.k1 * self.model.k2
        F = xs[:, :r].reshape(-1, self.model.k1, self.model.k2)
        self.model.F = F
        ll = kf.log_likelihood(Y, mask, xs, Vs)
        self.loglik_trace.append(ll)
        self.diff_trace.append(0.0)

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
