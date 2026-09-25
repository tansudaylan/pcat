import numpy as np

from pcat.main import retr_indxelemfree, retr_unitrefl


def test_retr_unitrefl_handles_both_boundaries_and_large_overshoots():
    values = np.array([-3.2, -1.2, -0.2, 0.2, 1.2, 2.2, 3.2])
    expected = np.array([0.8, 0.8, 0.2, 0.2, 0.8, 0.2, 0.8])

    np.testing.assert_allclose(retr_unitrefl(values), expected)


def test_retr_unitrefl_preserves_uniform_distribution():
    random = np.random.default_rng(42)
    values = random.uniform(size=200_000) + random.normal(scale=0.2, size=200_000)
    reflected = retr_unitrefl(values)

    assert abs(np.mean(reflected) - 0.5) < 0.003
    assert abs(np.mean(reflected < 0.1) - 0.1) < 0.003
    assert abs(np.mean(reflected > 0.9) - 0.1) < 0.003


def test_retr_indxelemfree_uses_allocated_capacity():
    assert retr_indxelemfree([], 3) == 0
    assert retr_indxelemfree([0, 1], 3) == 2
    assert retr_indxelemfree([0, 2], 3) == 1
    assert retr_indxelemfree([0, 1, 2], 3) is None


def test_prior_components_accumulate_without_overwriting():
    lpri = np.zeros(4)
    lpri[0] = -1.
    lpri[3] += -np.log(2.)
    lpri[3] += -np.log(4.)

    np.testing.assert_allclose(np.sum(lpri), -3.0794415416798357)