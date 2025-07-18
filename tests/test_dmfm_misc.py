import numpy as np
import pytest

import KPOKPCH


def test_factor_number_sensitivity():
    rng = np.random.default_rng(4)
    Y = rng.normal(size=(6, 3, 4))
    params = KPOKPCH.fit_dmfm_em(Y, 2, 2, 1, max_iter=2)
    assert params["F"].shape == (6, 2, 2)
    assert np.isfinite(params["F"]).all()


def test_fit_with_nontrivial_P_Q():
    rng = np.random.default_rng(5)
    Y = rng.normal(size=(5, 2, 2))
    params = KPOKPCH.initialize_dmfm(Y, 1, 1, 1)
    res = KPOKPCH.kalman_smoother_dmfm(
        Y,
        params["R"],
        params["C"],
        params["A"],
        params["B"],
        params["H"],
        params["K"],
        Pmat=2 * np.eye(1),
        Qmat=3 * np.eye(1),
    )
    assert res["F_smooth"].shape == (5, 1, 1)
