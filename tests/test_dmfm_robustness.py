import numpy as np
import pytest

import KPOKPCH


def test_initialize_with_zero_data():
    Y = np.zeros((4, 2, 3))
    params = KPOKPCH.initialize_dmfm(Y, 1, 1, 1)
    assert not np.isnan(params["F"]).any()
    assert np.allclose(params["F"], 0)


def test_initialize_with_low_rank_Y():
    rng = np.random.default_rng(0)
    T, p1, p2 = 5, 3, 3
    base = rng.normal(size=(p1, p2))
    Y = np.array([base * (t + 1) for t in range(T)])  # rank one across time
    params = KPOKPCH.initialize_dmfm(Y, 1, 1, 1)
    assert np.isfinite(params["F"]).all()


def test_initialize_with_nans():
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(4, 2, 2))
    mask = np.ones_like(Y, dtype=bool)
    Y[0, 0, 0] = np.nan
    mask[0, 0, 0] = False
    params = KPOKPCH.initialize_dmfm(Y, 1, 1, 1, mask=mask)
    assert np.isfinite(params["F"]).all()
