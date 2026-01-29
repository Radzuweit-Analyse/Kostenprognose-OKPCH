"""Dynamics of the DMFM latent factors."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class DynamicsConfig:
    """Configuration for factor dynamics estimation.

    Parameters
    ----------
    stability_threshold : float, default 0.99
        Maximum absolute eigenvalue for A and B matrices (enforces stability).
    regularization : float, default 1e-8
        Regularization parameter for pseudo-inverse computation.
    min_denominator_norm : float, default 1e-8
        Minimum norm for denominator matrices to avoid division issues.
    kronecker_only : bool, default False
        Whether to estimate dynamics in vectorized Kronecker form only.
    i1_factors : bool, default False
        Whether factors are integrated of order 1 (I(1)).
        When True, implements Barigozzi & Trapin (2025) Section 6:
        - Dynamics A, B are fixed at identity (random walk)
        - No drift is estimated
        - Kalman filter uses diffuse initial state
        - Estimation is done in levels (no differencing needed)

    Notes
    -----
    The `nonstationary` attribute is kept as an alias for `i1_factors`
    for backward compatibility but `i1_factors` is the preferred name.
    """

    stability_threshold: float = 0.99
    regularization: float = 1e-8
    min_denominator_norm: float = 1e-8
    kronecker_only: bool = False
    i1_factors: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0 < self.stability_threshold <= 1:
            raise ValueError(
                f"stability_threshold must be in (0, 1], got {self.stability_threshold}"
            )
        if self.regularization < 0:
            raise ValueError(
                f"regularization must be non-negative, got {self.regularization}"
            )

    @property
    def nonstationary(self) -> bool:
        """Alias for i1_factors (deprecated, use i1_factors instead)."""
        return self.i1_factors

    @nonstationary.setter
    def nonstationary(self, value: bool) -> None:
        """Set i1_factors via nonstationary alias."""
        self.i1_factors = value


class DMFMDynamics:
    """MAR(P) dynamics for the latent factor process.

    This class represents Matrix AutoRegressive dynamics of order P as
    specified in Barigozzi and Trapin (2025):

        F_t = A_1 @ F_{t-1} @ B_1^T + ... + A_P @ F_{t-P} @ B_P^T + E_t

    where:
        - F_t is a (k1 × k2) matrix of factors at time t
        - A_l are (k1 × k1) row transition matrices
        - B_l are (k2 × k2) column transition matrices
        - E_t ~ MN(0, P, Q) is matrix normal innovation

    Parameters
    ----------
    A : list[np.ndarray]
        Row transition matrices, each of shape (k1, k1), length P.
    B : list[np.ndarray]
        Column transition matrices, each of shape (k2, k2), length P.
    Pmat : np.ndarray
        Row innovation covariance, shape (k1, k1).
    Qmat : np.ndarray
        Column innovation covariance, shape (k2, k2).
    config : DynamicsConfig, optional
        Configuration for dynamics behavior. Uses defaults if None.

    Attributes
    ----------
    A, B : list[np.ndarray]
        Transition matrices for rows and columns.
    Pmat, Qmat : np.ndarray
        Innovation covariance matrices.
    config : DynamicsConfig
        Configuration settings.
    P : int
        Order of the MAR process.
    k1, k2 : int
        Dimensions of the factor matrices.
    """

    def __init__(
        self,
        A: list[np.ndarray],
        B: list[np.ndarray],
        Pmat: np.ndarray,
        Qmat: np.ndarray,
        C: np.ndarray | None = None,
        config: DynamicsConfig | None = None,
    ) -> None:
        """Initialize MAR(P) dynamics with optional drift.

        Parameters
        ----------
        A : list[np.ndarray]
            Row transition matrices.
        B : list[np.ndarray]
            Column transition matrices.
        Pmat : np.ndarray
            Row innovation covariance.
        Qmat : np.ndarray
            Column innovation covariance.
        C : np.ndarray, optional
            Drift matrix of shape (k1, k2). If None, initialized to zeros.
        config : DynamicsConfig, optional
            Dynamics configuration.

        Raises
        ------
        ValueError
            If A and B have different lengths or incompatible dimensions.
        """
        self._validate_inputs(A, B, Pmat, Qmat)

        self.A = A
        self.B = B
        self.Pmat = Pmat
        self.Qmat = Qmat
        self.config = config if config is not None else DynamicsConfig()

        # Inferred properties
        self.P = len(A)
        self.k1 = A[0].shape[0]
        self.k2 = B[0].shape[0]

        # Drift term
        if C is None:
            self.C_drift = np.zeros((self.k1, self.k2))
        else:
            if C.shape != (self.k1, self.k2):
                raise ValueError(
                    f"Drift C has shape {C.shape}, expected ({self.k1}, {self.k2})"
                )
            self.C_drift = C

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        A: list[np.ndarray],
        B: list[np.ndarray],
        Pmat: np.ndarray,
        Qmat: np.ndarray,
    ) -> None:
        """Validate input dimensions and structure."""
        if len(A) != len(B):
            raise ValueError(
                f"A and B must have same length, got {len(A)} and {len(B)}"
            )
        if len(A) == 0:
            raise ValueError("A and B must contain at least one matrix")

        k1, k2 = A[0].shape[0], B[0].shape[0]

        # Check A matrices
        for i, A_i in enumerate(A):
            if A_i.shape != (k1, k1):
                raise ValueError(f"A[{i}] has shape {A_i.shape}, expected ({k1}, {k1})")

        # Check B matrices
        for i, B_i in enumerate(B):
            if B_i.shape != (k2, k2):
                raise ValueError(f"B[{i}] has shape {B_i.shape}, expected ({k2}, {k2})")

        # Check covariances
        if Pmat.shape != (k1, k1):
            raise ValueError(f"Pmat has shape {Pmat.shape}, expected ({k1}, {k1})")
        if Qmat.shape != (k2, k2):
            raise ValueError(f"Qmat has shape {Qmat.shape}, expected ({k2}, {k2})")

    # ------------------------------------------------------------------
    # Convenience properties for backward compatibility
    # ------------------------------------------------------------------

    @property
    def i1_factors(self) -> bool:
        """Whether factors are integrated of order 1 (I(1) / random walk)."""
        return self.config.i1_factors

    @i1_factors.setter
    def i1_factors(self, value: bool) -> None:
        """Set i1_factors flag."""
        self.config.i1_factors = value

    @property
    def nonstationary(self) -> bool:
        """Alias for i1_factors (deprecated)."""
        return self.config.i1_factors

    @nonstationary.setter
    def nonstationary(self, value: bool) -> None:
        """Set i1_factors via nonstationary alias."""
        self.config.i1_factors = value

    @property
    def kronecker_only(self) -> bool:
        """Whether to use Kronecker-only form for dynamics."""
        return self.config.kronecker_only

    @kronecker_only.setter
    def kronecker_only(self, value: bool) -> None:
        """Set kronecker_only flag."""
        self.config.kronecker_only = value

    # ------------------------------------------------------------------
    # Forward simulation
    # ------------------------------------------------------------------

    def evolve(
        self, F_history: list[np.ndarray], add_noise: bool = False
    ) -> np.ndarray:
        """Compute next factor matrix given history with drift.

        Applies the MAR(P) recursion with drift:
            F_t = C + Σ_{l=1}^P A_l @ F_{t-l} @ B_l^T + E_t

        Parameters
        ----------
        F_history : list[np.ndarray]
            Past factor matrices ordered from lag 1 to P.
            Each array has shape (k1, k2).
        add_noise : bool, default False
            Whether to add matrix normal noise E_t ~ MN(0, Pmat, Qmat).

        Returns
        -------
        np.ndarray
            Predicted factor matrix of shape (k1, k2).

        Raises
        ------
        ValueError
            If F_history has incorrect length or dimensions.

        Examples
        --------
        >>> dynamics = DMFMDynamics(A=[A1], B=[B1], Pmat=P, Qmat=Q, C=drift)
        >>> F_history = [F_tm1]  # Just one lag for MAR(1)
        >>> F_t = dynamics.evolve(F_history)
        """
        if len(F_history) != self.P:
            raise ValueError(
                f"F_history must have length P={self.P}, got {len(F_history)}"
            )

        # Start with drift
        F_next = self.C_drift.copy()

        for l, F_lag in enumerate(F_history):
            if F_lag.shape != (self.k1, self.k2):
                raise ValueError(
                    f"F_history[{l}] has shape {F_lag.shape}, "
                    f"expected ({self.k1}, {self.k2})"
                )
            F_next += self.A[l] @ F_lag @ self.B[l].T

        if add_noise:
            # Sample from matrix normal MN(0, Pmat, Qmat)
            # If X ~ MN(M, Pmat, Qmat), then vec(X) ~ N(vec(M), Qmat ⊗ Pmat)
            E = self._sample_matrix_normal()
            F_next += E

        return F_next

    def _sample_matrix_normal(self) -> np.ndarray:
        """Sample from matrix normal MN(0, Pmat, Qmat)."""
        # Standard approach: X = Pmat^{1/2} @ Z @ Qmat^{1/2}
        # where Z has i.i.d. N(0,1) entries
        Z = np.random.randn(self.k1, self.k2)
        P_sqrt = np.linalg.cholesky(self.Pmat + 1e-6 * np.eye(self.k1))
        Q_sqrt = np.linalg.cholesky(self.Qmat + 1e-6 * np.eye(self.k2))
        return P_sqrt @ Z @ Q_sqrt.T

    # ------------------------------------------------------------------
    # Parameter estimation
    # ------------------------------------------------------------------

    def estimate(self, F: np.ndarray) -> None:
        """Estimate MAR coefficients from factor sequence.

        Updates A and B matrices via least squares as described in
        Barigozzi and Trapin (2025). For each lag ℓ:

            A_ℓ = argmin_A Σ_t ||F_t - A @ F_{t-ℓ} @ B_ℓ^T - residual||²_F
            B_ℓ = argmin_B Σ_t ||F_t - A_ℓ @ F_{t-ℓ} @ B^T - residual||²_F

        where residual includes contributions from other lags.

        Parameters
        ----------
        F : np.ndarray
            Factor sequence of shape (T, k1, k2).

        Raises
        ------
        ValueError
            If F has incorrect dimensions or insufficient time steps.

        Notes
        -----
        - Enforces stability by clipping eigenvalues to config.stability_threshold
        - Uses regularized pseudo-inverse for numerical stability
        - Preserves previous estimates if denominator is too small
        """
        if F.ndim != 3:
            raise ValueError(f"F must be 3D array, got shape {F.shape}")
        if F.shape[1] != self.k1 or F.shape[2] != self.k2:
            raise ValueError(
                f"F has dimensions ({F.shape[1]}, {F.shape[2]}), "
                f"expected ({self.k1}, {self.k2})"
            )
        if F.shape[0] <= self.P:
            raise ValueError(f"Need at least {self.P + 1} time steps, got {F.shape[0]}")

        A_new = [np.zeros_like(self.A[0]) for _ in range(self.P)]
        B_new = [np.zeros_like(self.B[0]) for _ in range(self.P)]

        for ell in range(self.P):
            A_num = np.zeros((self.k1, self.k1))
            A_den = np.zeros((self.k1, self.k1))
            B_num = np.zeros((self.k2, self.k2))
            B_den = np.zeros((self.k2, self.k2))

            for t in range(ell + 1, F.shape[0]):
                # Compute residual from other lags
                F_pred_other = np.zeros((self.k1, self.k2))
                for j in range(self.P):
                    if j == ell or t - j - 1 < 0:
                        continue
                    F_pred_other += self.A[j] @ F[t - j - 1] @ self.B[j].T

                Y_res = F[t] - F_pred_other

                # Update A numerator and denominator
                X_A = F[t - ell - 1] @ self.B[ell].T
                A_num += Y_res @ X_A.T
                A_den += X_A @ X_A.T

                # Update B numerator and denominator
                X_B = F[t - ell - 1].T @ self.A[ell].T
                B_num += Y_res.T @ X_B.T
                B_den += X_B @ X_B.T

            # Solve for A_ell with regularization and NaN protection
            if (
                np.linalg.norm(A_den) > self.config.min_denominator_norm
                and not np.isnan(A_den).any()
            ):
                A_est = A_num @ np.linalg.pinv(A_den, rcond=self.config.regularization)
                # Check for NaNs in the estimate
                if not np.isnan(A_est).any():
                    A_new[ell] = self._enforce_stability(A_est)
                else:
                    A_new[ell] = self.A[ell]  # Keep previous value if NaN
            else:
                A_new[ell] = self.A[ell]

            # Solve for B_ell with regularization and NaN protection
            if (
                np.linalg.norm(B_den) > self.config.min_denominator_norm
                and not np.isnan(B_den).any()
            ):
                B_est = B_num @ np.linalg.pinv(B_den, rcond=self.config.regularization)
                # Check for NaNs in the estimate
                if not np.isnan(B_est).any():
                    B_new[ell] = self._enforce_stability(B_est)
                else:
                    B_new[ell] = self.B[ell]  # Keep previous value if NaN
            else:
                B_new[ell] = self.B[ell]

        self.A = A_new
        self.B = B_new

    def _enforce_stability(self, mat: np.ndarray) -> np.ndarray:
        """Enforce stability by clipping element-wise.

        Parameters
        ----------
        mat : np.ndarray
            Transition matrix to stabilize.

        Returns
        -------
        np.ndarray
            Stabilized matrix with entries in [-threshold, threshold].

        Notes
        -----
        This is a simple approach. More sophisticated methods would
        clip eigenvalues or use constrained optimization.
        """
        threshold = self.config.stability_threshold
        return np.clip(mat, -threshold, threshold)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def to_var1(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert MAR(P) to VAR(1) companion form with drift.

        Returns
        -------
        T : np.ndarray
            Companion transition matrix of shape (k1*k2*P, k1*k2*P).
        mu : np.ndarray
            Drift vector of shape (k1*k2*P,), with drift only in first block.
        Sigma : np.ndarray
            Companion innovation covariance, shape (k1*k2*P, k1*k2*P).

        Notes
        -----
        The companion form with drift:
            z_t = [vec(F_t), vec(F_{t-1}), ..., vec(F_{t-P+1})]^T

        Then: z_t = mu + T @ z_{t-1} + η_t, where η_t ~ N(0, Σ).
        """
        r = self.k1 * self.k2
        d = r * self.P

        # Compute Kronecker products Φ_l = B_l ⊗ A_l
        Phi = [np.kron(self.B[l], self.A[l]) for l in range(self.P)]

        # Build companion matrix
        T = np.zeros((d, d))
        T[:r, : r * self.P] = np.hstack(Phi)
        if self.P > 1:
            T[r:, :-r] = np.eye(r * (self.P - 1))

        # Build drift vector (only affects first block)
        mu = np.zeros(d)
        mu[:r] = self.C_drift.ravel()

        # Build innovation covariance (only affects first block)
        Sigma = np.zeros((d, d))
        Sigma[:r, :r] = np.kron(self.Qmat, self.Pmat)

        return T, mu, Sigma

    def check_stability(self) -> tuple[bool, float]:
        """Check if dynamics are stable.

        Returns
        -------
        is_stable : bool
            True if all eigenvalues of companion form are inside unit circle.
        max_eigenvalue : float
            Absolute value of largest eigenvalue.

        Notes
        -----
        For MAR(P) to be stable, the companion VAR(1) form must have
        all eigenvalues with modulus < 1.
        """
        T, _, _ = self.to_var1()  # Unpack T, mu, Sigma
        eigenvalues = np.linalg.eigvals(T)
        max_abs_eval = np.max(np.abs(eigenvalues))
        is_stable = max_abs_eval < 1.0
        return is_stable, max_abs_eval

    def __repr__(self) -> str:
        """String representation."""
        is_stable, max_eval = self.check_stability()
        stability_str = "stable" if is_stable else "unstable"
        return (
            f"DMFMDynamics(P={self.P}, k1={self.k1}, k2={self.k2}, "
            f"{stability_str}, max_eval={max_eval:.3f})"
        )
