import numpy as np
import pytest
from scipy.special import gammaln

from pcat.transdimensional import sample_catalog


def test_flat_likelihood_recovers_normalized_truncated_poisson_count_prior():
    mean = 1.4
    result = sample_catalog(lambda base, catalog: 0.0, [[-2.0, 2.0]], [[0.0, 5.0], [-3.0, 3.0]],
                            maximum_components=3, count_prior_mean=mean,
                            number_steps=40000, burn_steps=2000, thinning=2, seed=7)
    counts = np.arange(4)
    expected = np.exp(counts * np.log(mean) - gammaln(counts + 1))
    expected /= expected.sum()
    measured = np.bincount(result['component_counts'], minlength=4) / len(result['component_counts'])

    np.testing.assert_allclose(measured, expected, atol=0.025, rtol=0.0)
    assert result['acceptance_counts']['birth'] > 0
    assert result['acceptance_counts']['death'] > 0
    assert all(catalog.shape == (count, 2) for count, catalog in zip(result['component_counts'], result['catalog_samples']))


def test_catalog_sampler_reproducibility_and_zero_component_limit():
    arguments = dict(maximum_components=0, number_steps=100, burn_steps=20, thinning=4, seed=2)
    first = sample_catalog(lambda base, catalog: -0.5 * base[0]**2, [[-1.0, 1.0]], [[0.0, 1.0]], **arguments)
    second = sample_catalog(lambda base, catalog: -0.5 * base[0]**2, [[-1.0, 1.0]], [[0.0, 1.0]], **arguments)

    np.testing.assert_array_equal(first['base_samples'], second['base_samples'])
    assert np.all(first['component_counts'] == 0)
    assert first['proposal_counts']['birth'] == first['proposal_counts']['death'] == 0


def test_catalog_sampler_rejects_invalid_parameter_bounds():
    with pytest.raises(ValueError, match='bounds'):
        sample_catalog(lambda base, catalog: 0.0, [[1.0, 0.0]], [[0.0, 1.0]])


def test_nonuniform_catalog_dependent_selection_preserves_count_prior():
    def draw(base, catalog, generator):
        return generator.beta(len(catalog) + 1, 1, size=1)

    def density(base, catalog, component):
        return np.log(len(catalog) + 1) + len(catalog) * np.log(component[0])

    result = sample_catalog(
        lambda base, catalog: 0.0, [[0.0, 1.0]], [[0.0, 1.0]],
        draw_component=draw, log_component_density=density,
        maximum_components=2, number_steps=16000, burn_steps=2000, thinning=2, seed=13,
    )
    measured = np.bincount(result['component_counts'], minlength=3) / len(result['component_counts'])
    np.testing.assert_allclose(measured, [0.4, 0.4, 0.2], atol=0.035, rtol=0.0)


def test_catalog_selection_functions_are_paired_and_inputs_are_protected():
    with pytest.raises(ValueError, match='both'):
        sample_catalog(lambda base, catalog: 0.0, [[0.0, 1.0]], [[0.0, 1.0]], draw_component=lambda: None)

    def draw(base, catalog, generator):
        base[:] = -100.0
        catalog[:] = -100.0
        return np.array([0.5])

    result = sample_catalog(
        lambda base, catalog: 0.0, [[0.0, 1.0]], [[0.0, 1.0]],
        draw_component=draw, log_component_density=lambda base, catalog, component: 0.0,
        number_steps=100, burn_steps=20, seed=5,
    )
    assert np.all((result['base_samples'] >= 0.0) & (result['base_samples'] <= 1.0))
    assert all(np.all((catalog >= 0.0) & (catalog <= 1.0)) for catalog in result['catalog_samples'])


def test_complete_catalog_selection_corrects_count_and_parameter_probabilities():
    def draw(generator):
        count = int(generator.integers(3))
        return generator.random(1), generator.beta(2.0, 1.0, size=(count, 1))

    def density(base, catalog):
        return -np.log(3.0) + np.sum(np.log(2.0 * catalog[:, 0]))

    result = sample_catalog(
        lambda base, catalog: 0.0, [[0.0, 1.0]], [[0.0, 1.0]],
        maximum_components=2, number_steps=16000, burn_steps=2000, thinning=2,
        draw_catalog=draw, log_catalog_density=density, catalog_move_probability=1.0, seed=41,
    )
    measured = np.bincount(result['component_counts'], minlength=3) / len(result['component_counts'])
    np.testing.assert_allclose(measured, [0.4, 0.4, 0.2], atol=0.03, rtol=0.0)
    components = np.concatenate([catalog[:, 0] for catalog in result['catalog_samples'] if len(catalog)])
    assert components.mean() == pytest.approx(0.5, abs=0.03)
    assert result['acceptance_counts']['catalog'] > 0


def test_complete_catalog_selection_requires_both_functions():
    with pytest.raises(ValueError, match='both'):
        sample_catalog(lambda base, catalog: 0.0, [[0.0, 1.0]], [[0.0, 1.0]], draw_catalog=lambda generator: None)


def test_complete_catalog_density_is_cached_until_the_state_changes():
    calls = []

    def density(base, catalog):
        calls.append(base.copy())
        return 0.0

    sample_catalog(
        lambda base, catalog: 0.0 if base[0] < 0.1 else -np.inf,
        [[0.0, 1.0]], [[0.0, 1.0]], maximum_components=0,
        draw_catalog=lambda generator: (generator.random(1), np.empty((0, 1))),
        log_catalog_density=density, catalog_move_probability=1.0,
        initial_base=[0.05], number_steps=100, burn_steps=20, seed=61,
    )
    assert len(calls) == 101