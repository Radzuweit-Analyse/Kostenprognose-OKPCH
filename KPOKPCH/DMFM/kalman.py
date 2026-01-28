"""Kalman filtering and smoothing for the DMFM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from numpy.linalg import inv, LinAlgError


@dataclass
class KalmanState:
    """State from Kalman filter/smoother.

    Attributes
    ----------
    x_pred : np.ndarray
        Predicted states (T, d) where d is state dimension.
    P_pred : np.ndarray
        Predicted state covariances (T, d, d).
    x_filt : np.ndarray
        Filtered states (T, d).
    P_filt : np.ndarray
        Filtered state covariances (T, d, d).
    x_smooth : np.ndarray or None
        Smoothed states (T, d), available after smoothing.
    P_smooth : np.ndarray or None
        Smoothed state covariances (T, d, d).
    P_smooth_lag : np.ndarray or None
        Lag-1 smoothed cross-covariances (T-1, d, d).
    loglik : float or None
        Log-likelihood of observed data.
    """

    x_pred: np.ndarray
    P_pred: np.ndarray
    x_filt: np.ndarray
    P_filt: np.ndarray
    x_smooth: np.ndarray | None = None
    P_smooth: np.ndarray | None = None
    P_smooth_lag: np.ndarray | None = None
    loglik: float | None = None


@dataclass
class KalmanConfig:
    """Configuration for Kalman filter.

    Parameters
    ----------
    initial_state_variance : float, default 1e2
        Variance for initial state covariance (standard dynamics).
    initial_state_variance_i1 : float, default 1e4
        Variance for initial state covariance (I(1) dynamics).
    regularization : float, default 1e-8
        Regularization added to covariance matrices for stability.
    use_woodbury : bool, default False
        Whether to use Woodbury identity for high-dimensional observations.
    check_symmetry : bool, default True
        Whether to enforce symmetry of covariance matrices.
    """

    initial_state_variance: float = 1e2
    initial_state_variance_i1: float = 1e4
    regularization: float = 1e-8
    use_woodbury: bool = False
    check_symmetry: bool = True


class KalmanFilterDMFM:
    """Kalman filter and RTS smoother for the DMFM.

    This class implements the Kalman filter and Rauch-Tung-Striebel (RTS)
    smoother for a Dynamic Matrix Factor Model cast in state-space form:

        State equation:  z_t = T @ z_{t-1} + η_t,  η_t ~ N(0, Q)
        Observation:     y_t = Z @ z_t + ε_t,      ε_t ~ N(0, R)

    where z_t stacks the vectorized factors and their lags.

    Parameters
    ----------
    model : DMFMModel
        Fitted DMFM model with initialized parameters.
    config : KalmanConfig, optional
        Kalman filter configuration. Uses defaults if None.

    Attributes
    ----------
    model : DMFMModel
        Reference to the underlying model.
    config : KalmanConfig
        Filter configuration.
    state : KalmanState or None
        Most recent filter/smoother state, available after calling filter().
    """

    def __init__(
        self, model: "DMFMModel", config: KalmanConfig | None = None
    ) -> None:
        """Initialize Kalman filter.

        Parameters
        ----------
        model : DMFMModel
            Model to filter (must be initialized).
        config : KalmanConfig, optional
            Filter configuration.

        Raises
        ------
        ValueError
            If model is not initialized.
        """
        if not model.is_initialized():
            raise ValueError(
                "Model must be initialized before creating KalmanFilter. "
                "Call model.initialize() first."
            )

        self.model = model
        self.config = config if config is not None else KalmanConfig()
        self.state: KalmanState | None = None

        # Cache state-space matrices (computed once)
        self._T: np.ndarray | None = None
        self._mu: np.ndarray | None = None
        self._Q: np.ndarray | None = None
        self._Z: np.ndarray | None = None
        self._R: np.ndarray | None = None

    # ------------------------------------------------------------------
    # State-space matrix construction
    # ------------------------------------------------------------------

    def _construct_matrices(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Construct state-space matrices from model parameters with drift.

        Returns
        -------
        T : np.ndarray
            State transition matrix (d, d) where d = k1*k2*P.
        mu : np.ndarray
            Drift vector (d,).
        Q : np.ndarray
            State innovation covariance (d, d).
        Z : np.ndarray
            Observation matrix (p1*p2, d).
        R : np.ndarray
            Observation noise covariance (p1*p2, p1*p2).

        Notes
        -----
        The state z_t = [vec(F_t), vec(F_{t-1}), ..., vec(F_{t-P+1})]^T
        has dimension d = k1*k2*P (companion form).
        State equation with drift: z_t = mu + T @ z_{t-1} + η_t
        """
        if self._T is not None:
            # Return cached matrices
            return self._T, self._mu, self._Q, self._Z, self._R

        m = self.model
        k1, k2, P = m.k1, m.k2, m.P
        r = k1 * k2

        # Transition matrix T (companion form)
        Phi = [np.kron(m.B[l], m.A[l]) for l in range(P)]
        T = np.zeros((r * P, r * P))
        T[:r, : r * P] = np.hstack(Phi)
        if P > 1:
            T[r:, :-r] = np.eye(r * (P - 1))

        # Drift vector (only first block)
        mu = np.zeros(r * P)
        mu[:r] = m.dynamics.C_drift.ravel()

        # State innovation covariance Q
        Q_block = np.kron(m.Qmat, m.Pmat)
        Q = np.zeros((r * P, r * P))
        Q[:r, :r] = Q_block

        # Observation matrix Z
        Z = np.zeros((m.p1 * m.p2, r * P))
        Z[:, :r] = np.kron(m.C, m.R)

        # Observation noise covariance R
        if m.diagonal_idiosyncratic:
            R = np.kron(
                np.diag(np.diag(m.K)), np.diag(np.diag(m.H))
            )
        else:
            R = np.kron(m.K, m.H)

        # Cache for reuse
        self._T, self._mu, self._Q, self._Z, self._R = T, mu, Q, Z, R

        return T, mu, Q, Z, R

    def _clear_cache(self) -> None:
        """Clear cached state-space matrices."""
        self._T = None
        self._mu = None
        self._Q = None
        self._Z = None
        self._R = None

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter(
        self, Y: np.ndarray, mask: np.ndarray | None = None
    ) -> KalmanState:
        """Run Kalman filter on observed data.

        Parameters
        ----------
        Y : np.ndarray
            Observed data of shape (T, p1, p2).
        mask : np.ndarray, optional
            Boolean mask for missing values (True = observed).
            If None, assumes all data is observed.

        Returns
        -------
        KalmanState
            Filter state with predicted and filtered estimates.

        Raises
        ------
        ValueError
            If data dimensions don't match model.
        """
        self._validate_data(Y)

        if mask is None:
            mask = np.ones_like(Y, dtype=bool)

        T, mu, Q, Z, R = self._construct_matrices()

        # Run filter
        x_pred, P_pred, x_filt, P_filt = self._kalman_filter(
            Y, mask, T, mu, Q, Z, R
        )

        # Store state
        self.state = KalmanState(
            x_pred=x_pred, P_pred=P_pred, x_filt=x_filt, P_filt=P_filt
        )

        return self.state

    def _kalman_filter(
        self,
        Y: np.ndarray,
        mask: np.ndarray,
        T: np.ndarray,
        mu: np.ndarray,
        Q: np.ndarray,
        Z: np.ndarray,
        R: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Kalman filter implementation with drift.

        Returns
        -------
        x_pred : np.ndarray
            Predicted states (T, d).
        P_pred : np.ndarray
            Predicted covariances (T, d, d).
        x_filt : np.ndarray
            Filtered states (T, d).
        P_filt : np.ndarray
            Filtered covariances (T, d, d).
        """
        Tn = Y.shape[0]
        d = T.shape[0]

        # Storage
        x_pred = np.zeros((Tn, d))
        P_pred = np.zeros((Tn, d, d))
        x_filt = np.zeros((Tn, d))
        P_filt = np.zeros((Tn, d, d))

        # Initial conditions
        init_var = (
            self.config.initial_state_variance_i1
            if getattr(self.model.dynamics, "i1_factors", False)
            else self.config.initial_state_variance
        )
        x_post = np.zeros(d)
        P_post = np.eye(d) * init_var

        # Filter loop
        for t in range(Tn):
            # Prediction step with drift
            if t == 0:
                x_prior = mu.copy()  # Start with drift
                P_prior = np.eye(d) * init_var
            else:
                x_prior = mu + T @ x_post  # Add drift to prediction
                P_prior = T @ P_post @ T.T + Q
                if self.config.check_symmetry:
                    P_prior = 0.5 * (P_prior + P_prior.T)

            # Update step
            x_post, P_post = self._kalman_update(
                Y[t], mask[t], x_prior, P_prior, Z, R
            )

            # Store
            x_pred[t] = x_prior
            P_pred[t] = P_prior
            x_filt[t] = x_post
            P_filt[t] = P_post

        return x_pred, P_pred, x_filt, P_filt

    def _kalman_update(
        self,
        y: np.ndarray,
        m: np.ndarray,
        x_prior: np.ndarray,
        P_prior: np.ndarray,
        Z: np.ndarray,
        R: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Kalman update step (measurement update).

        Parameters
        ----------
        y : np.ndarray
            Observation at time t, shape (p1, p2).
        m : np.ndarray
            Mask at time t, shape (p1, p2).
        x_prior : np.ndarray
            Prior state mean.
        P_prior : np.ndarray
            Prior state covariance.
        Z : np.ndarray
            Full observation matrix.
        R : np.ndarray
            Full observation noise covariance.

        Returns
        -------
        x_post : np.ndarray
            Posterior state mean.
        P_post : np.ndarray
            Posterior state covariance.
        """
        y_vec = y.reshape(-1)
        m_vec = m.reshape(-1)
        idx = np.where(m_vec)[0]

        if idx.size == 0:
            # No observations, posterior = prior
            return x_prior, P_prior

        # Extract observed components
        Z_t = Z[idx, :]
        R_t = R[np.ix_(idx, idx)]
        y_obs = y_vec[idx]

        # Innovation covariance S = Z @ P @ Z^T + R
        S = Z_t @ P_prior @ Z_t.T + R_t
        S += self.config.regularization * np.eye(S.shape[0])

        # Kalman gain K = P @ Z^T @ S^{-1}
        try:
            S_inv = inv(S)
        except LinAlgError:
            S_inv = np.linalg.pinv(S)

        K = P_prior @ Z_t.T @ S_inv

        # Update
        innovation = y_obs - Z_t @ x_prior
        x_post = x_prior + K @ innovation
        P_post = P_prior - K @ Z_t @ P_prior

        if self.config.check_symmetry:
            P_post = 0.5 * (P_post + P_post.T)

        return x_post, P_post

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    def smooth(self, state: KalmanState | None = None) -> KalmanState:
        """Run RTS smoother on filtered state.

        Parameters
        ----------
        state : KalmanState, optional
            Filtered state to smooth. If None, uses self.state.

        Returns
        -------
        KalmanState
            State with smoothed estimates added.

        Raises
        ------
        ValueError
            If no filtered state is available.
        """
        if state is None:
            state = self.state

        if state is None:
            raise ValueError(
                "No filtered state available. Call filter() first."
            )

        T, mu, _, _, _ = self._construct_matrices()

        x_smooth, P_smooth, P_smooth_lag = self._rts_smoother(
            state.x_pred, state.P_pred, state.x_filt, state.P_filt, T, mu
        )

        # Update state
        state.x_smooth = x_smooth
        state.P_smooth = P_smooth
        state.P_smooth_lag = P_smooth_lag

        return state

    def _rts_smoother(
        self,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        x_filt: np.ndarray,
        P_filt: np.ndarray,
        T: np.ndarray,
        mu: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rauch-Tung-Striebel smoother implementation with drift.

        The drift is already incorporated in x_pred, so the backward
        recursion doesn't need modification.

        Returns
        -------
        x_smooth : np.ndarray
            Smoothed states (T, d).
        P_smooth : np.ndarray
            Smoothed covariances (T, d, d).
        P_smooth_lag : np.ndarray
            Lag-1 cross-covariances (T-1, d, d).
        """
        Tn, d = x_filt.shape

        # Storage
        x_smooth = np.zeros_like(x_filt)
        P_smooth = np.zeros_like(P_filt)
        J = np.zeros((Tn - 1, d, d))

        # Initialize with filtered values at final time
        x_smooth[-1] = x_filt[-1]
        P_smooth[-1] = P_filt[-1]

        # Backward pass
        for t in range(Tn - 2, -1, -1):
            # Smoother gain J_t = P_filt[t] @ T^T @ P_pred[t+1]^{-1}
            try:
                P_pred_inv = inv(
                    P_pred[t + 1]
                    + self.config.regularization * np.eye(d)
                )
            except LinAlgError:
                P_pred_inv = np.linalg.pinv(
                    P_pred[t + 1]
                    + self.config.regularization * np.eye(d)
                )

            J[t] = P_filt[t] @ T.T @ P_pred_inv

            # Smooth
            x_smooth[t] = x_filt[t] + J[t] @ (
                x_smooth[t + 1] - x_pred[t + 1]
            )
            P_smooth[t] = P_filt[t] + J[t] @ (
                P_smooth[t + 1] - P_pred[t + 1]
            ) @ J[t].T

            if self.config.check_symmetry:
                P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)

        # Lag-1 cross-covariances
        P_smooth_lag = np.zeros((Tn - 1, d, d))
        for t in range(Tn - 1):
            P_smooth_lag[t] = J[t] @ P_smooth[t + 1]

        return x_smooth, P_smooth, P_smooth_lag

    # ------------------------------------------------------------------
    # Log-likelihood computation
    # ------------------------------------------------------------------

    def log_likelihood(
        self,
        Y: np.ndarray,
        mask: np.ndarray | None = None,
        state: KalmanState | None = None,
    ) -> float:
        """Compute log-likelihood of observed data.

        Parameters
        ----------
        Y : np.ndarray
            Observed data of shape (T, p1, p2).
        mask : np.ndarray, optional
            Boolean mask for missing values.
        state : KalmanState, optional
            Smoothed state to use. If None, uses self.state.

        Returns
        -------
        float
            Log-likelihood value.

        Raises
        ------
        ValueError
            If no smoothed state is available.

        Notes
        -----
        Computes the exact log-likelihood accounting for uncertainty
        in the latent states via the smoothed covariances.
        """
        if mask is None:
            mask = np.ones_like(Y, dtype=bool)

        if state is None:
            state = self.state

        if state is None or state.x_smooth is None:
            raise ValueError(
                "No smoothed state available. Call smooth() first."
            )

        return self._compute_loglik(Y, mask, state)

    def _compute_loglik(
        self, Y: np.ndarray, mask: np.ndarray, state: KalmanState
    ) -> float:
        """Compute log-likelihood using smoothed estimates."""
        Tn = Y.shape[0]
        k1, k2 = self.model.k1, self.model.k2
        r = k1 * k2

        loglik = 0.0
        Z_base = np.kron(self.model.C, self.model.R)

        if self.model.diagonal_idiosyncratic:
            R_base = np.kron(
                np.diag(np.diag(self.model.K)),
                np.diag(np.diag(self.model.H)),
            )
        else:
            R_base = np.kron(self.model.K, self.model.H)

        for t in range(Tn):
            f_t = state.x_smooth[t, :r]
            V_t = state.P_smooth[t, :r, :r]

            y_vec = Y[t].reshape(-1)
            m_vec = mask[t].reshape(-1)
            idx = np.where(m_vec)[0]

            if idx.size == 0:
                continue

            # Extract observed components
            Z_t = Z_base[idx, :]
            R_t = R_base[np.ix_(idx, idx)]

            # Predictive covariance Σ_y = Z @ V @ Z^T + R
            Sigma_y = Z_t @ V_t @ Z_t.T + R_t
            Sigma_y += self.config.regularization * np.eye(idx.size)

            # Innovation
            innovation = y_vec[idx] - Z_t @ f_t

            # Log-likelihood contribution
            sign, logdet = np.linalg.slogdet(Sigma_y)
            try:
                Sigma_inv = inv(Sigma_y)
            except LinAlgError:
                Sigma_inv = np.linalg.pinv(Sigma_y)

            loglik -= 0.5 * (
                logdet
                + innovation.T @ Sigma_inv @ innovation
                + idx.size * np.log(2 * np.pi)
            )

        return float(loglik)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _validate_data(self, Y: np.ndarray) -> None:
        """Validate data dimensions."""
        if Y.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {Y.shape}")
        if Y.shape[1] != self.model.p1 or Y.shape[2] != self.model.p2:
            raise ValueError(
                f"Data dimensions {Y.shape[1:]}, expected "
                f"({self.model.p1}, {self.model.p2})"
            )

    def extract_factors(
        self, state: KalmanState | None = None, smoothed: bool = True
    ) -> np.ndarray:
        """Extract factor matrices from state.

        Parameters
        ----------
        state : KalmanState, optional
            State to extract from. If None, uses self.state.
        smoothed : bool, default True
            Whether to use smoothed (True) or filtered (False) states.

        Returns
        -------
        np.ndarray
            Factors of shape (T, k1, k2).

        Raises
        ------
        ValueError
            If requested state type is not available.
        """
        if state is None:
            state = self.state

        if state is None:
            raise ValueError("No state available. Call filter() first.")

        k1, k2 = self.model.k1, self.model.k2
        r = k1 * k2

        if smoothed:
            if state.x_smooth is None:
                raise ValueError(
                    "No smoothed state available. Call smooth() first."
                )
            x = state.x_smooth
        else:
            x = state.x_filt

        # Extract first r elements and reshape
        T = x.shape[0]
        F = x[:, :r].reshape(T, k1, k2)

        return F

    def __repr__(self) -> str:
        """String representation."""
        has_filtered = self.state is not None
        has_smoothed = (
            self.state is not None and self.state.x_smooth is not None
        )
        status = []
        if has_filtered:
            status.append("filtered")
        if has_smoothed:
            status.append("smoothed")
        status_str = ", ".join(status) if status else "not run"

        return (
            f"KalmanFilterDMFM(model={self.model.k1}×{self.model.k2}, "
            f"P={self.model.P}, {status_str})"
        )
