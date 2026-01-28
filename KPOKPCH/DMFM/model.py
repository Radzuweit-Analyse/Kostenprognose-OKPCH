"""Model representation for the dynamic matrix factor model."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .dynamics import DMFMDynamics
from . import utils


@dataclass
class DMFMConfig:
    """Configuration for a Dynamic Matrix Factor Model.

    The model follows the specification of Barigozzi and Trapin (2025)
    where the observed matrix ``Y_t`` of dimension ``(p1, p2)`` is driven
    by latent matrix factors ``F_t`` with loadings ``R`` and ``C``.

    Parameters
    ----------
    p1, p2 : int
        Cross-sectional dimensions of the observed matrices.
    k1, k2 : int
        Number of row and column factors.
    P : int, default 1
        Order of the MAR dynamics for ``F_t``.
    diagonal_idiosyncratic : bool, default False
        Whether to use diagonal idiosyncratic covariance matrices.
    """

    p1: int
    p2: int
    k1: int
    k2: int
    P: int = 1
    diagonal_idiosyncratic: bool = False

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.p1 <= 0 or self.p2 <= 0:
            raise ValueError(f"Dimensions must be positive: p1={self.p1}, p2={self.p2}")
        if self.k1 <= 0 or self.k2 <= 0:
            raise ValueError(f"Number of factors must be positive: k1={self.k1}, k2={self.k2}")
        if self.k1 > self.p1 or self.k2 > self.p2:
            raise ValueError(
                f"Factors cannot exceed dimensions: k1={self.k1} > p1={self.p1} "
                f"or k2={self.k2} > p2={self.p2}"
            )
        if self.P <= 0:
            raise ValueError(f"MAR order must be positive: P={self.P}")


@dataclass
class DMFMModel:
    """Fitted Dynamic Matrix Factor Model.

    This class holds both the model configuration and learned parameters.
    Models must be initialized before fitting.

    Attributes
    ----------
    config : DMFMConfig
        Model configuration and dimensions.
    R, C : np.ndarray or None
        Factor loadings (row and column), set after initialization.
    H, K : np.ndarray or None
        Idiosyncratic covariance matrices, set after initialization.
    A, B : list[np.ndarray]
        MAR dynamics coefficient matrices, set after initialization.
    Pmat, Qmat : np.ndarray or None
        Initial state and innovation covariance matrices.
    F : np.ndarray or None
        Estimated factors (T x k1 x k2).
    dynamics : DMFMDynamics or None
        Dynamics object for state-space operations.
    """

    config: DMFMConfig

    # Learned parameters (private, set during initialization/fitting)
    _R: np.ndarray | None = field(default=None, repr=False, init=False)
    _C: np.ndarray | None = field(default=None, repr=False, init=False)
    _H: np.ndarray | None = field(default=None, repr=False, init=False)
    _K: np.ndarray | None = field(default=None, repr=False, init=False)
    _A: list[np.ndarray] = field(default_factory=list, repr=False, init=False)
    _B: list[np.ndarray] = field(default_factory=list, repr=False, init=False)
    _Pmat: np.ndarray | None = field(default=None, repr=False, init=False)
    _Qmat: np.ndarray | None = field(default=None, repr=False, init=False)
    _F: np.ndarray | None = field(default=None, repr=False, init=False)
    _dynamics: DMFMDynamics | None = field(default=None, repr=False, init=False)

    # Training state
    _init_method: str | None = field(default=None, repr=False, init=False)
    _is_fitted: bool = field(default=False, repr=False, init=False)

    @classmethod
    def from_dims(
        cls,
        p1: int,
        p2: int,
        k1: int,
        k2: int,
        P: int = 1,
        diagonal_idiosyncratic: bool = False,
    ) -> DMFMModel:
        """Convenience constructor from dimensions.

        Parameters
        ----------
        p1, p2 : int
            Cross-sectional dimensions.
        k1, k2 : int
            Number of factors.
        P : int, default 1
            MAR order.
        diagonal_idiosyncratic : bool, default False
            Whether to use diagonal idiosyncratic covariance.

        Returns
        -------
        DMFMModel
            Uninitialized model instance.
        """
        config = DMFMConfig(p1, p2, k1, k2, P, diagonal_idiosyncratic)
        return cls(config)

    # Properties for read access to learned parameters
    @property
    def R(self) -> np.ndarray:
        """Row factor loadings (p1 x k1)."""
        if self._R is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        return self._R

    @property
    def C(self) -> np.ndarray:
        """Column factor loadings (p2 x k2)."""
        if self._C is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        return self._C

    @property
    def H(self) -> np.ndarray:
        """Row idiosyncratic covariance (p1 x p1)."""
        if self._H is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        return self._H

    @property
    def K(self) -> np.ndarray:
        """Column idiosyncratic covariance (p2 x p2)."""
        if self._K is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        return self._K

    @property
    def A(self) -> list[np.ndarray]:
        """Row MAR coefficient matrices (length P)."""
        if not self._A:
            raise ValueError("Model not initialized. Call initialize() first.")
        return self._A

    @property
    def B(self) -> list[np.ndarray]:
        """Column MAR coefficient matrices (length P)."""
        if not self._B:
            raise ValueError("Model not initialized. Call initialize() first.")
        return self._B

    @property
    def Pmat(self) -> np.ndarray:
        """Initial state covariance."""
        if self._Pmat is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        return self._Pmat

    @property
    def Qmat(self) -> np.ndarray:
        """Innovation covariance."""
        if self._Qmat is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        return self._Qmat

    @property
    def F(self) -> np.ndarray:
        """Estimated factors (T x k1 x k2)."""
        if self._F is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._F

    @property
    def dynamics(self) -> DMFMDynamics:
        """Dynamics object for state-space operations."""
        if self._dynamics is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        return self._dynamics

    @property
    def p1(self) -> int:
        """First cross-sectional dimension."""
        return self.config.p1

    @property
    def p2(self) -> int:
        """Second cross-sectional dimension."""
        return self.config.p2

    @property
    def k1(self) -> int:
        """Number of row factors."""
        return self.config.k1

    @property
    def k2(self) -> int:
        """Number of column factors."""
        return self.config.k2

    @property
    def P(self) -> int:
        """MAR order."""
        return self.config.P

    @property
    def diagonal_idiosyncratic(self) -> bool:
        """Whether idiosyncratic covariances are diagonal."""
        return self.config.diagonal_idiosyncratic

    def is_initialized(self) -> bool:
        """Check if model has been initialized with starting values."""
        return self._R is not None

    def is_fitted(self) -> bool:
        """Check if model has been fitted to data."""
        return self._is_fitted

    def initialize(
        self, Y: np.ndarray, mask: np.ndarray | None = None, method: str = "svd"
    ) -> None:
        """Initialize parameters from data.

        This routine computes starting values for the factor loadings,
        factors and covariance matrices which are subsequently refined by
        the EM algorithm.

        Parameters
        ----------
        Y : np.ndarray
            Observed data array of shape (T, p1, p2).
        mask : np.ndarray or None, optional
            Boolean mask indicating missing values.
        method : str, default "svd"
            Initialization method for factor loadings.

        Raises
        ------
        ValueError
            If data dimensions don't match config.
        """
        # Validate dimensions
        if Y.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {Y.shape}")
        if Y.shape[1] != self.p1 or Y.shape[2] != self.p2:
            raise ValueError(
                f"Data dimensions {Y.shape[1:]}, expected ({self.p1}, {self.p2})"
            )

        # Initialize parameters
        R, C, F = utils.init_factor_loadings(Y, mask, self.k1, self.k2, method)
        H, K = utils.init_idiosyncratic(Y, R, C, F)
        A, B, Pmat, Qmat = utils.init_dynamics(self.k1, self.k2, self.P)

        self._R = R
        self._C = C
        self._F = F
        self._H = H
        self._K = K
        self._A = A
        self._B = B
        self._Pmat = Pmat
        self._Qmat = Qmat
        self._init_method = method

        # Initialize dynamics with zero drift (will be estimated during EM)
        self._dynamics = DMFMDynamics(
            A, B, Pmat, Qmat, C=np.zeros((self.k1, self.k2))
        )

    def _check_initialized(self) -> None:
        """Raise error if model not initialized."""
        if not self.is_initialized():
            raise ValueError(
                "Model not initialized. Call initialize() before fitting."
            )

    def _check_fitted(self) -> None:
        """Raise error if model not fitted."""
        if not self.is_fitted():
            raise ValueError("Model not fitted. Call fit() before prediction.")

    def __repr__(self) -> str:
        """String representation showing config and status."""
        status = []
        if self.is_initialized():
            status.append(f"initialized(method={self._init_method})")
        if self.is_fitted():
            status.append("fitted")
        status_str = ", ".join(status) if status else "not initialized"

        return (
            f"DMFMModel(p1={self.p1}, p2={self.p2}, k1={self.k1}, k2={self.k2}, "
            f"P={self.P}, {status_str})"
        )