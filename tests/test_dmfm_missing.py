import numpy as np
import pytest

import KPOKPCH


def test_fit_with_high_missingness():
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(5, 3, 2))
    mask = rng.random(size=Y.shape) < 0.2
    mask[0] = True  # ensure at least one fully observed time point
    params = KPOKPCH.fit_dmfm_em(Y, 1, 1, 1, max_iter=2, mask=mask)
    assert np.isfinite(params["F"]).all()


def test_fit_with_block_missing_pattern():
    rng = np.random.default_rng(2)
    Y = rng.normal(size=(6, 4, 3))
    mask = np.ones_like(Y, dtype=bool)
    mask[:3, :2, :] = False
    mask[0] = True  # at least one fully observed
    params = KPOKPCH.fit_dmfm_em(Y, 1, 1, 1, max_iter=2, mask=mask)
    assert np.isfinite(params["F"]).all()


def test_missing_with_nan_and_mask():
    rng = np.random.default_rng(3)
    Y = rng.normal(size=(4, 2, 2))
    mask = np.ones_like(Y, dtype=bool)
    Y[1, 0, 1] = np.nan
    mask[1, 0, 1] = False
    params = KPOKPCH.initialize_dmfm(Y, 1, 1, 1, mask=mask)
    assert np.isfinite(params["F"]).all()
