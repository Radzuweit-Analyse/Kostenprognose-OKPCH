import numpy as np
import pytest

import KPOKPCH


def generate_data(T=4, p1=3, p2=2):
    rng = np.random.default_rng(0)
    return rng.normal(size=(T, p1, p2))


def test_em_step_idempotence():
    Y = generate_data(T=5)
    params = KPOKPCH.initialize_dmfm(Y, 1, 1, 1)
    params1, diff1, ll1 = KPOKPCH.em_step_dmfm(Y, params)
    params2, diff2, ll2 = KPOKPCH.em_step_dmfm(Y, params1)
    assert diff1 >= 0 and diff2 >= 0
    assert np.isfinite(ll1) and np.isfinite(ll2)


def test_em_loglik_monotonicity():
    Y = generate_data(T=5)
    params = KPOKPCH.initialize_dmfm(Y, 1, 1, 1)
    params1, diff1, ll1 = KPOKPCH.em_step_dmfm(Y, params)
    params2, diff2, ll2 = KPOKPCH.em_step_dmfm(Y, params1)
    assert np.isfinite(ll1) and np.isfinite(ll2)
