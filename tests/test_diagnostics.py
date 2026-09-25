import numpy as np

from pcat.main import _PCATMCMCCompat


def test_gmrb_rejects_separated_constant_chains():
    chains = np.column_stack((np.zeros(100), np.full(100, 100.)))

    assert np.isinf(_PCATMCMCCompat.gmrb_test(chains))


def test_autocorrelation_distinguishes_independent_draws_and_random_walk():
    random = np.random.default_rng(42)
    independent = random.normal(size=(2000, 1))
    random_walk = np.cumsum(random.normal(size=(2000, 1)), axis=0)

    _, time_independent = _PCATMCMCCompat.retr_timeatcr(independent)
    _, time_random_walk = _PCATMCMCCompat.retr_timeatcr(random_walk)

    assert time_independent[0] < 2.
    assert time_random_walk[0] > 20.