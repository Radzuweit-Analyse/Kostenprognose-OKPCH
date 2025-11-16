"""Model representation for the dynamic matrix factor model."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .dynamics import DMFMDynamics
from . import utils


@dataclass
class DMFMModel:
    """Parameters of a Dynamic Matrix Factor Model.

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
    """

    p1: int
    p2: int
    k1: int
    k2: int
    P: int = 1
    diagonal_idiosyncratic: bool = False

    R: np.ndarray | None = None
    C: np.ndarray | None = None
    H: np.ndarray | None = None
    K: np.ndarray | None = None
    A: list[np.ndarray] = field(default_factory=list)
    B: list[np.ndarray] = field(default_factory=list)
    Pmat: np.ndarray | None = None
    Qmat: np.ndarray | None = None
    F: np.ndarray | None = None
    dynamics: DMFMDynamics | None = None

    def initialize(
        self, Y: np.ndarray, mask: np.ndarray | None = None, method: str = "svd"
    ) -> None:
        """Initialise parameters from data.

        This routine computes starting values for the factor loadings,
        factors and covariance matrices which are subsequently refined by
        the EM algorithm.
        """

        R, C, F = utils.init_factor_loadings(Y, mask, self.k1, self.k2, method)
        H, K = utils.init_idiosyncratic(Y, R, C, F)
        A, B, Pmat, Qmat = utils.init_dynamics(self.k1, self.k2, self.P)

        self.R = R
        self.C = C
        self.F = F
        self.H = H
        self.K = K
        self.A = A
        self.B = B
        self.Pmat = Pmat
        self.Qmat = Qmat

        self.dynamics = DMFMDynamics(A, B, Pmat, Qmat)
