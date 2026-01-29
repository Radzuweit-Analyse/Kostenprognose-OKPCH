"""EM algorithm for estimating the DMFM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from numpy.linalg import inv
import numpy as np

from .model import DMFMModel
from .dynamics import DMFMDynamics
from .kalman import KalmanFilterDMFM


@dataclass
class EMConfig:
    """Configuration for EM algorithm.

    Parameters
    ----------
    max_iter : int, default 100
        Maximum number of EM iterations.
    tol : float, default 1e-4
        Convergence tolerance for parameter differences.
    verbose : bool, default False
        Whether to print iteration progress.
    check_loglik_increase : bool, default True
        Whether to check that log-likelihood is non-decreasing.
    """

    max_iter: int = 100
    tol: float = 1e-4
    verbose: bool = False
    check_loglik_increase: bool = True


@dataclass
class EMResult:
    """Results from EM algorithm fitting.

    Attributes
    ----------
    converged : bool
        Whether EM converged within max_iter.
    num_iter : int
        Number of iterations performed.
    final_loglik : float
        Final log-likelihood value.
    loglik_trace : list[float]
        Log-likelihood at each iteration.
    diff_trace : list[float]
        Parameter difference at each iteration.
    """

    converged: bool
    num_iter: int
    final_loglik: float
    loglik_trace: list[float]
    diff_trace: list[float]


class EMEstimatorDMFM:
    """Estimate DMFM parameters via the EM algorithm.

    This class handles the iterative parameter estimation for a Dynamic
    Matrix Factor Model using the Expectation-Maximization algorithm.

    Parameters
    ----------
    model : DMFMModel
        The model to estimate. Must be initialized before calling fit().
    config : EMConfig, optional
        EM algorithm configuration. If None, uses default settings.

    Attributes
    ----------
    model : DMFMModel
        Reference to the underlying model being estimated.
    config : EMConfig
        EM algorithm configuration.
    result : EMResult or None
        Fitting results, available after calling fit().
    """

    def __init__(self, model: DMFMModel, config: EMConfig | None = None) -> None:
        """Initialize EM estimator.

        Parameters
        ----------
        model : DMFMModel
            Model to estimate (must be initialized).
        config : EMConfig, optional
            EM configuration. Uses defaults if None.

        Raises
        ------
        ValueError
            If model is not initialized.
        """
        if not model.is_initialized():
            raise ValueError(
                "Model must be initialized before creating EMEstimator. "
                "Call model.initialize() first."
            )

        self.model = model
        self.config = config if config is not None else EMConfig()
        self.result: EMResult | None = None

        # Private state during fitting
        self._loglik_trace: list[float] = []
        self._diff_trace: list[float] = []

    # ------------------------------------------------------------------
    # Main fitting interface
    # ------------------------------------------------------------------

    def fit(
        self,
        Y: np.ndarray,
        mask: np.ndarray | None = None,
        checkpoint_callback: Callable[[int, dict], None] | None = None,
    ) -> EMResult:
        """Run EM algorithm until convergence.

        Parameters
        ----------
        Y : np.ndarray
            Observed data of shape (T, p1, p2).
        mask : np.ndarray, optional
            Boolean mask for missing values (True = observed).
        checkpoint_callback : callable, optional
            Function called after each iteration with signature
            ``callback(iteration: int, state: dict)``.

        Returns
        -------
        EMResult
            Fitting results including convergence status and traces.

        Raises
        ------
        ValueError
            If data dimensions don't match model configuration.
        RuntimeError
            If EM algorithm encounters numerical issues.
        """
        # Validate inputs
        self._validate_data(Y)
        if mask is None:
            mask = np.ones_like(Y, dtype=bool)

        # Reset state
        self._loglik_trace = []
        self._diff_trace = []

        # Extract dimensions
        k1, k2 = self.model.k1, self.model.k2
        Pord = self.model.P
        r = k1 * k2
        T = Y.shape[0]

        last_ll = -np.inf

        # EM iterations
        for it in range(self.config.max_iter):
            if self.config.verbose:
                print(f"EM iteration {it + 1}/{self.config.max_iter}")

            # ---------- E-STEP ----------
            kf = KalmanFilterDMFM(self.model)

            # Filter
            state = kf.filter(Y, mask)

            # Smooth
            state = kf.smooth(state)

            # Extract factors from smoothed states
            F = kf.extract_factors(state, smoothed=True)

            # ---------- M-STEP ----------
            # Store previous parameters for convergence check
            prev_params = self._extract_params()

            # Update all parameters
            new_params = self._m_step(Y, F, state.P_smooth, state.P_smooth_lag, mask)

            # Update model with new parameters
            self._update_model_params(new_params, F)

            # ---------- CONVERGENCE CHECK ----------
            # Evaluate log-likelihood under updated parameters
            kf_eval = KalmanFilterDMFM(self.model)
            state_eval = kf_eval.filter(Y, mask)
            state_eval = kf_eval.smooth(state_eval)

            # Update model factors with smoothed estimates
            self.model._F = kf_eval.extract_factors(state_eval, smoothed=True)

            # Compute log-likelihood
            ll = kf_eval.log_likelihood(Y, mask, state_eval)

            # Compute parameter difference
            diff = _compute_param_diff(
                prev_params,
                new_params,
                Pord,
                self.model.dynamics is not None
                and self.model.dynamics.config.kronecker_only,
            )

            # Check for log-likelihood decrease
            if self.config.check_loglik_increase and it > 0 and (ll - last_ll) < -1e-6:
                if self.config.verbose:
                    print(
                        f"⚠️  Warning: Log-likelihood decreased at iteration {it + 1}"
                    )
                    print(f"    Previous: {last_ll:.6f}, Current: {ll:.6f}")

                # Revert to previous parameters
                self._update_model_params(prev_params, self.model.F)
                ll = last_ll
                diff = 0.0
                self._loglik_trace.append(ll)
                self._diff_trace.append(diff)
                break

            # Store traces
            self._loglik_trace.append(ll)
            self._diff_trace.append(diff)

            # Checkpoint callback
            if checkpoint_callback is not None:
                checkpoint_state = {
                    "iteration": it + 1,
                    "loglik": ll,
                    "diff": diff,
                    "params": new_params,
                    "factors": self.model.F,
                    "state": state_eval,  # Include full Kalman state
                }
                checkpoint_callback(it + 1, checkpoint_state)

            # Check convergence
            if diff < self.config.tol:
                if self.config.verbose:
                    print(f"✓ Converged at iteration {it + 1} (diff={diff:.2e})")
                break

            last_ll = ll

        else:
            if self.config.verbose:
                print(f"⚠️  Maximum iterations ({self.config.max_iter}) reached")

        # Mark model as fitted
        self.model._is_fitted = True

        # Create result object
        self.result = EMResult(
            converged=(diff < self.config.tol),
            num_iter=it + 1,
            final_loglik=ll,
            loglik_trace=self._loglik_trace.copy(),
            diff_trace=self._diff_trace.copy(),
        )

        return self.result

    # ------------------------------------------------------------------
    # M-step: update all parameters
    # ------------------------------------------------------------------

    def _m_step(
        self,
        Y: np.ndarray,
        F: np.ndarray,
        Vs: np.ndarray,
        Vss: np.ndarray,
        mask: np.ndarray,
    ) -> dict[str, Any]:
        """Perform M-step: update all model parameters.

        Parameters
        ----------
        Y : np.ndarray
            Observed data (T, p1, p2).
        F : np.ndarray
            Smoothed factors (T, k1, k2).
        Vs : np.ndarray
            Smoothed state covariances.
        Vss : np.ndarray
            Smoothed cross-covariances.
        mask : np.ndarray
            Missing data mask.

        Returns
        -------
        dict
            Dictionary of updated parameters.
        """
        # Update loadings
        R_new = _update_row_loadings(Y, F, self.model.C, self.model.R, mask)
        C_new = _update_col_loadings(Y, F, R_new, self.model.C, mask)

        # Orthonormalize loadings
        R_new, R_fac = np.linalg.qr(R_new)
        C_new, C_fac = np.linalg.qr(C_new)

        # Rotate factors accordingly
        for t in range(F.shape[0]):
            F[t] = R_fac @ F[t] @ C_fac.T

        # Check if we're in I(1) mode (skip dynamics and drift updates)
        is_i1 = self.model.dynamics is not None and self.model.dynamics.i1_factors
        is_kronecker = (
            self.model.dynamics is not None and self.model.dynamics.kronecker_only
        )

        # Update dynamics (skipped for I(1) factors per Barigozzi & Trapin Section 6)
        A_new, B_new, Phi_new = _update_dynamics(
            F,
            self.model.A,
            self.model.B,
            self.model.P,
            self.model.k1,
            self.model.k2,
            is_i1,
            is_kronecker,
        )

        # Update drift (skipped for I(1) factors - random walk has no drift)
        C_drift_new = _update_drift(F, A_new, B_new, self.model.P, skip=is_i1)

        # Update innovations
        P_new, Q_new = _update_innovations(
            F,
            Vs,
            Vss,
            A_new,
            B_new,
            Phi_new,
            is_i1,
            is_kronecker,
        )

        # Update idiosyncratic covariances
        H_new, K_new = _update_idiosyncratic(
            Y, F, R_new, C_new, mask, self.model.diagonal_idiosyncratic
        )

        return {
            "R": R_new,
            "C": C_new,
            "A": A_new,
            "B": B_new,
            "Phi": Phi_new,
            "C_drift": C_drift_new,
            "H": H_new,
            "K": K_new,
            "P": P_new,
            "Q": Q_new,
        }

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def _extract_params(self) -> dict[str, Any]:
        """Extract current parameters from model."""
        return {
            "R": self.model.R.copy(),
            "C": self.model.C.copy(),
            "A": [A.copy() for A in self.model.A],
            "B": [B.copy() for B in self.model.B],
            "Phi": (
                [
                    np.kron(self.model.B[l], self.model.A[l])
                    for l in range(len(self.model.A))
                ]
                if self.model.dynamics
                else None
            ),
            "C_drift": (
                self.model.dynamics.C_drift.copy()
                if self.model.dynamics
                else np.zeros((self.model.k1, self.model.k2))
            ),
            "H": self.model.H.copy(),
            "K": self.model.K.copy(),
            "P": self.model.Pmat.copy(),
            "Q": self.model.Qmat.copy(),
        }

    def _update_model_params(self, new_params: dict[str, Any], F: np.ndarray) -> None:
        """Update model with new parameters.

        Parameters
        ----------
        new_params : dict
            Dictionary of new parameter values.
        F : np.ndarray
            Updated factors.
        """
        # Store dynamics flags
        dyn_flags = {}
        if self.model.dynamics is not None:
            dyn_flags["nonstationary"] = self.model.dynamics.nonstationary
            dyn_flags["kronecker_only"] = self.model.dynamics.kronecker_only
            dyn_flags["i1_factors"] = self.model.dynamics.i1_factors

        # Update parameters (use private attributes)
        self.model._R = new_params["R"]
        self.model._C = new_params["C"]
        self.model._A = new_params["A"]
        self.model._B = new_params["B"]
        self.model._Pmat = new_params["P"]
        self.model._Qmat = new_params["Q"]
        self.model._H = new_params["H"]
        self.model._K = new_params["K"]
        self.model._F = F

        # Recreate dynamics object with drift
        self.model._dynamics = DMFMDynamics(
            self.model.A,
            self.model.B,
            self.model.Pmat,
            self.model.Qmat,
            C=new_params["C_drift"],
        )

        # Restore flags
        if dyn_flags:
            self.model._dynamics.nonstationary = dyn_flags.get("nonstationary")
            self.model._dynamics.kronecker_only = dyn_flags.get("kronecker_only")
            self.model._dynamics.i1_factors = dyn_flags.get("i1_factors")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_data(self, Y: np.ndarray) -> None:
        """Validate data dimensions match model configuration.

        Parameters
        ----------
        Y : np.ndarray
            Observed data.

        Raises
        ------
        ValueError
            If dimensions don't match.
        """
        if Y.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {Y.shape}")
        if Y.shape[1] != self.model.p1 or Y.shape[2] != self.model.p2:
            raise ValueError(
                f"Data dimensions {Y.shape[1:]}, expected "
                f"({self.model.p1}, {self.model.p2})"
            )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_factors(self) -> np.ndarray:
        """Return the smoothed factor sequence.

        Returns
        -------
        np.ndarray
            Smoothed factors of shape (T, k1, k2).

        Raises
        ------
        RuntimeError
            If estimator has not been fitted yet.
        """
        self.model._check_fitted()
        return self.model.F

    def get_loglik_trace(self) -> list[float]:
        """Return the log-likelihood values across EM iterations.

        Returns
        -------
        list[float]
            Log-likelihood at each iteration.

        Raises
        ------
        RuntimeError
            If estimator has not been fitted yet.
        """
        if self.result is None:
            raise RuntimeError("Estimator has not been fitted yet")
        return self.result.loglik_trace

    def get_diff_trace(self) -> list[float]:
        """Return parameter difference across EM iterations.

        Returns
        -------
        list[float]
            Parameter difference at each iteration.

        Raises
        ------
        RuntimeError
            If estimator has not been fitted yet.
        """
        if self.result is None:
            raise RuntimeError("Estimator has not been fitted yet")
        return self.result.diff_trace


# ----------------------------------------------------------------------
# Helper routines for M-step updates
# ----------------------------------------------------------------------


def _update_row_loadings(
    Y: np.ndarray,
    F: np.ndarray,
    C: np.ndarray,
    R: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Update row loadings via least squares."""
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
            # Use regularization for numerical stability
            # rcond=1e-6 helps avoid SVD convergence issues with ill-conditioned matrices
            try:
                R_new[i] = np.linalg.lstsq(Xmat, yvec, rcond=1e-6)[0]
            except np.linalg.LinAlgError:
                # If lstsq still fails, keep previous value
                R_new[i] = R[i]
        else:
            R_new[i] = R[i]
    return R_new


def _update_col_loadings(
    Y: np.ndarray,
    F: np.ndarray,
    R_new: np.ndarray,
    C: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Update column loadings via least squares."""
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
            # Use regularization for numerical stability
            try:
                C_new[j] = np.linalg.lstsq(Xmat, yvec, rcond=1e-6)[0]
            except np.linalg.LinAlgError:
                # If lstsq still fails, keep previous value
                C_new[j] = C[j]
        else:
            C_new[j] = C[j]
    return C_new


def _update_dynamics(F, A, B, Pord, k1, k2, nonstationary, kronecker_only):
    """Update dynamics matrices."""
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
        A_new[ell] = _enforce_stability_spectral(A_est, threshold=0.99)
        B_new[ell] = _enforce_stability_spectral(B_est, threshold=0.99)
    Phi_new = [np.kron(B_new[l], A_new[l]) for l in range(Pord)]
    return A_new, B_new, Phi_new


def _enforce_stability_spectral(
    mat: np.ndarray, threshold: float = 0.99
) -> np.ndarray:
    """Enforce stability via spectral radius constraint.

    For MAR dynamics Φ = B ⊗ A, eigenvalues are products λ_i(A) · μ_j(B).
    To ensure ρ(Φ) < threshold, we enforce ρ(A), ρ(B) ≤ √threshold.

    Parameters
    ----------
    mat : np.ndarray
        Transition matrix to stabilize.
    threshold : float, default 0.99
        Maximum allowed spectral radius for the Kronecker product.

    Returns
    -------
    np.ndarray
        Stabilized matrix with spectral radius ≤ √threshold.
    """
    target_radius = np.sqrt(threshold)

    eigenvalues = np.linalg.eigvals(mat)
    spectral_radius = np.max(np.abs(eigenvalues))

    if spectral_radius <= target_radius or spectral_radius < 1e-10:
        return mat

    # Scale matrix uniformly
    scale_factor = target_radius / spectral_radius
    return mat * scale_factor


def _update_drift(
    F: np.ndarray, A: list, B: list, P: int, skip: bool = False
) -> np.ndarray:
    """Estimate drift matrix from smoothed factors.

    The drift C is estimated as the mean residual:
    C = (1/T) * Σ_t [F_t - Σ_l A_l @ F_{t-l} @ B_l^T]

    Parameters
    ----------
    F : np.ndarray
        Smoothed factors (T, k1, k2).
    A : list[np.ndarray]
        Row transition matrices.
    B : list[np.ndarray]
        Column transition matrices.
    P : int
        MAR order.
    skip : bool, default False
        If True, return zeros (used for I(1) factors where random walk
        has no drift per Barigozzi & Trapin 2025 Section 6).

    Returns
    -------
    np.ndarray
        Estimated drift matrix of shape (k1, k2).
    """
    Tn, k1, k2 = F.shape

    # For I(1) factors, random walk has no drift
    if skip:
        return np.zeros((k1, k2))

    # Compute residuals
    residuals = []
    for t in range(P, Tn):
        # Compute MAR prediction without drift
        F_pred = np.zeros((k1, k2))
        for l in range(P):
            F_pred += A[l] @ F[t - l - 1] @ B[l].T

        # Residual = F_t - prediction
        residual = F[t] - F_pred
        residuals.append(residual)

    if len(residuals) == 0:
        return np.zeros((k1, k2))

    # Drift is the mean residual
    C_drift = np.mean(residuals, axis=0)

    return C_drift


def _update_innovations(F, Vs, Vss, A_new, B_new, Phi_new, i1_factors, kronecker_only):
    """Update innovation covariances."""
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
    """Update idiosyncratic covariances."""
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
    """Compute normalized parameter difference."""
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
