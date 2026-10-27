from pathlib import Path

import numpy as np
import pytest

from pcat.diagnostics import posterior_convergence
from pcat.fixed import sample_allesfitter_pcat, sample_fixed_chains
from pcat.main import readfile, sample
from pcat.main import readfile, retr_levipost, sample
from pcat.time_series import (
    log_likelihood_transit_times,
    make_transit_time_likelihood,
)


def test_transit_timing_likelihood_uses_uncertainty_weighted_residuals():
    observed_days = np.array([1.0, 3.0])  # [day]
    predicted_days = np.array([2.0, 1.0])  # [day]
    uncertainties_days = np.array([1.0, 2.0])  # [day]

    assert log_likelihood_transit_times(observed_days, predicted_days, uncertainties_days) == -1.0
    assert log_likelihood_transit_times([1.0], [np.nan], [1.0]) == -np.inf
    with pytest.raises(ValueError, match="positive"):
        log_likelihood_transit_times([1.0], [1.0], [0.0])
    with pytest.raises(ValueError, match="matching"):
        log_likelihood_transit_times([1.0], [1.0, 2.0], [1.0])


def test_transit_time_likelihood_factory_uses_forward_predictor():
    transit_epochs = np.array([0.0, 1.0, 2.0])
    observed_times = np.array([1.0, 3.1, 5.0])
    timing_errors = np.array([0.1, 0.2, 0.1])
    predictor_calls = []

    def predictor(parameters, epochs):
        predictor_calls.append(tuple(parameters))
        return parameters[0] + parameters[1] * epochs

    likelihood = make_transit_time_likelihood(predictor)
    parameters = np.array([1.0, 2.0])

    actual = likelihood(parameters, (transit_epochs, observed_times, timing_errors))
    expected = log_likelihood_transit_times(
        observed_times,
        predictor(parameters, transit_epochs),
        timing_errors,
    )

    assert actual == expected
    assert predictor_calls[0] == predictor_calls[1] == tuple(parameters)


def test_retr_levipost_matches_harmonic_mean_for_unequal_likelihoods():
    assert retr_levipost(np.log([1.0, 3.0])) == pytest.approx(np.log(1.5))


def test_retr_levipost_remains_stable_for_extreme_log_likelihoods():
    with np.errstate(all="raise"):
        actual = retr_levipost([-1000.0, -1001.0])
    expected = -1001.0 + np.log(2.0 / (1.0 + np.exp(-1.0)))
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("samples", [[], [np.nan], [np.inf], [-np.inf]])
def test_retr_levipost_rejects_empty_or_nonfinite_samples(samples):
    with pytest.raises(ValueError, match="must not be empty|must be finite"):
        retr_levipost(samples)


def gaussian_log_likelihood(gdat, strgmodl, values):
    gdat.generic_callback_count = getattr(gdat, 'generic_callback_count', 0) + 1
    covariance = np.array([[1.0, 0.6], [0.6, 2.0]])
    return -0.5 * values @ np.linalg.inv(covariance) @ values


def scalar_log_likelihood(gdat, strgmodl, values):
    return -0.5 * values[0] ** 2


def test_tempered_burn_history_is_persisted_and_finishes_before_sampling(tmp_path):
    result = sample(
        typeexpr="gener",
        retr_llik=scalar_log_likelihood,
        parameter_names=("x",),
        prior_types=("self",),
        prior_minima=(-4.0,),
        prior_maxima=(4.0,),
        initial_values=(3.0,),
        proposal_scales=(0.2,),
        pathbase=tmp_path,
        strgcnfg="tempered_burn",
        numbswep=12,
        numbburn=4,
        numbsamp=8,
        boolburntmpr=True,
        factburntmpr=0.75,
        boolmakeplot=False,
        makeanim=False,
        typeverb=0,
    )

    inverse_temperature = np.asarray(result.listpostfacttmpr).reshape(-1)
    assert inverse_temperature[:3] == pytest.approx([(1. / 3.) ** 4, (2. / 3.) ** 4, 1.0])
    assert np.all(inverse_temperature[3:] == 1.0)


def constant_legacy_likelihood(values, state):
    return np.log(3.)


@pytest.mark.parametrize('bounds', [(1.0, 1.0), (2.0, 1.0)])
@pytest.mark.parametrize('container', [list, np.array])
@pytest.mark.parametrize('scale', ['self', 'logt'])
def test_sample_posterior_rejects_invalid_bounds_before_outputs(tmp_path, monkeypatch, bounds, container, scale):
    from unittest.mock import Mock
    import pcat.fixed as fixed

    minima = container([bounds[0]])
    maxima = container([bounds[1]])
    original_minima = np.array(minima, copy=True)
    original_maxima = np.array(maxima, copy=True)
    output = tmp_path / 'invalid_bounds'
    sampler = Mock(side_effect=AssertionError('Sampling must not start for invalid bounds'))
    monkeypatch.setattr(fixed, 'sample_fixed_chains', sampler)

    with np.errstate(all='raise'):
        with pytest.raises(ValueError, match='minimum.*less.*maximum'):
            fixed.sample_posterior(
                None, 4, constant_legacy_likelihood,
                ['x'], [['x', '']], [scale], minima, maxima,
                pathbase=str(output) + '/', boolplot=False,
                booltqdm=False, typeverb=0,
            )

    sampler.assert_not_called()
    assert not output.exists()
    np.testing.assert_array_equal(minima, original_minima)
    np.testing.assert_array_equal(maxima, original_maxima)


@pytest.mark.parametrize('bound_name', ['minmpara', 'maxmpara'])
@pytest.mark.parametrize('kind', ['masked', 'complex'])
def test_sample_posterior_rejects_masked_or_complex_bounds_before_outputs(tmp_path, monkeypatch, bound_name, kind):
    import pcat.fixed as fixed

    bounds = {'minmpara': np.array([-1.0]), 'maxmpara': np.array([1.0])}
    if kind == 'masked':
        invalid = np.ma.array(bounds[bound_name], mask=[True])
    else:
        invalid = bounds[bound_name].astype(complex) + 0.1j
    bounds[bound_name] = invalid
    original_data = np.ma.getdata(invalid).copy()
    original_mask = np.ma.getmaskarray(invalid).copy()
    output = tmp_path / 'invalid_physical_bounds'
    monkeypatch.setattr(fixed, 'retr_limtpara',
                        lambda *args: pytest.fail('Invalid bounds reached plotting preparation'))
    monkeypatch.setattr(fixed, 'sample_fixed_chains',
                        lambda *args, **kwargs: pytest.fail('Invalid bounds reached the sampler'))

    with pytest.raises(ValueError, match='unmasked real'):
        fixed.sample_posterior(
            None, 4, constant_legacy_likelihood,
            ['x'], [['x', '']], ['self'], **bounds,
            pathbase=str(output) + '/', boolplot=False,
            booltqdm=False, typeverb=0,
        )

    assert not output.exists()
    np.testing.assert_array_equal(np.ma.getdata(invalid), original_data)
    np.testing.assert_array_equal(np.ma.getmaskarray(invalid), original_mask)


@pytest.mark.parametrize('container', ['list', 'array', 'unmasked'])
def test_sample_posterior_accepts_real_bounds_without_modifying_inputs(tmp_path, container):
    import pcat.fixed as fixed

    minima, maxima = np.array([-1.0]), np.array([1.0])
    if container == 'list':
        minima, maxima = minima.tolist(), maxima.tolist()
    elif container == 'unmasked':
        minima, maxima = np.ma.array(minima, mask=False), np.ma.array(maxima, mask=False)
    original_minima, original_maxima = np.asarray(minima).copy(), np.asarray(maxima).copy()
    random_state = np.random.get_state()
    try:
        np.random.seed(123)
        samples = fixed.sample_posterior(
            None, 8, constant_legacy_likelihood,
            ['x'], [['x', '']], ['self'], minima, maxima,
            pathbase=str(tmp_path) + '/', boolforcrepr=True, boolplot=False,
            numbsamppostwalk=4, numbsampburnwalk=2, booltqdm=False,
            booldiag=False, typeverb=0,
        )
    finally:
        np.random.set_state(random_state)

    assert samples['x'].size > 0
    assert np.isfinite(samples['x']).all()
    assert np.all((samples['x'] >= -1.0) & (samples['x'] <= 1.0))
    np.testing.assert_array_equal(minima, original_minima)
    np.testing.assert_array_equal(maxima, original_maxima)
    assert not np.ma.getmaskarray(minima).any()
    assert not np.ma.getmaskarray(maxima).any()


@pytest.mark.parametrize('bound_name', ['minmpara', 'maxmpara'])
@pytest.mark.parametrize('kind', ['matrix', 'nan', 'wrong_length'])
def test_sample_posterior_rejects_malformed_bounds_before_outputs(tmp_path, monkeypatch, bound_name, kind):
    import pcat.fixed as fixed

    bounds = {'minmpara': np.array([-1.0, -2.0]), 'maxmpara': np.array([1.0, 2.0])}
    if kind == 'matrix':
        invalid = bounds[bound_name].reshape(1, 2)
    elif kind == 'nan':
        invalid = bounds[bound_name].copy()
        invalid[0] = np.nan
    else:
        invalid = bounds[bound_name][:1]
    bounds[bound_name] = invalid
    original = invalid.copy()
    output = tmp_path / 'malformed_bounds'
    monkeypatch.setattr(fixed, 'retr_limtpara',
                        lambda *args: pytest.fail('Malformed bounds reached plotting preparation'))
    monkeypatch.setattr(fixed, 'sample_fixed_chains',
                        lambda *args, **kwargs: pytest.fail('Malformed bounds reached the sampler'))

    with pytest.raises(ValueError, match='one-dimensional|length|NaN'):
        fixed.sample_posterior(
            None, 4, constant_legacy_likelihood,
            ['x', 'y'], [['x', ''], ['y', '']], ['self', 'self'], **bounds,
            pathbase=str(output) + '/', boolplot=False,
            booltqdm=False, typeverb=0,
        )

    assert not output.exists()
    np.testing.assert_array_equal(invalid, original)


@pytest.mark.parametrize('lower,upper', [(0.0, 1.0), (-1.0, 1.0), (-2.0, -1.0), (1.0, np.inf)])
def test_sample_posterior_rejects_invalid_log_bounds_before_outputs(tmp_path, monkeypatch, lower, upper):
    import pcat.fixed as fixed

    minima, maxima = np.array([lower]), np.array([upper])
    original_minima, original_maxima = minima.copy(), maxima.copy()
    output = tmp_path / 'invalid_log_bounds'
    monkeypatch.setattr(fixed, 'retr_limtpara',
                        lambda *args: pytest.fail('Invalid logarithmic bounds reached plotting preparation'))
    monkeypatch.setattr(fixed, 'sample_fixed_chains',
                        lambda *args, **kwargs: pytest.fail('Invalid logarithmic bounds reached sampling'))

    with pytest.raises(ValueError, match='[Ll]og.*finite.*positive'):
        fixed.sample_posterior(
            None, 4, constant_legacy_likelihood,
            ['x'], [['x', '']], ['logt'], minima, maxima,
            pathbase=str(output) + '/', boolplot=False, booltqdm=False, typeverb=0,
        )

    assert not output.exists()
    np.testing.assert_array_equal(minima, original_minima)
    np.testing.assert_array_equal(maxima, original_maxima)


def test_sample_posterior_runs_with_valid_positive_log_bounds(tmp_path):
    import pcat.fixed as fixed

    minima, maxima = np.array([1.0]), np.array([10.0])
    random_state = np.random.get_state()
    try:
        np.random.seed(123)
        samples = fixed.sample_posterior(
            None, 8, constant_legacy_likelihood,
            ['x'], [['x', '']], ['logt'], minima, maxima,
            pathbase=str(tmp_path) + '/', boolforcrepr=True, boolplot=False,
            numbsamppostwalk=4, numbsampburnwalk=2, booltqdm=False,
            booldiag=False, typeverb=0,
        )
    finally:
        np.random.set_state(random_state)

    assert samples['x'].size > 0
    assert np.isfinite(samples['x']).all()
    assert np.all((samples['x'] >= 1.0) & (samples['x'] <= 10.0))
    np.testing.assert_array_equal(minima, [1.0])
    np.testing.assert_array_equal(maxima, [10.0])


@pytest.mark.parametrize('field,invalid', [
    ('listnamepara', ['x']),
    ('listnamepara', ['x', 'y', 'z']),
    ('scalpara', ['self']),
    ('scalpara', ['self', 'self', 'self']),
    ('listnamepara', ['x', 'x']),
])
def test_sample_posterior_rejects_misaligned_metadata_before_outputs(tmp_path, monkeypatch, field, invalid):
    from copy import deepcopy
    import pcat.fixed as fixed

    arguments = dict(
        listnamepara=['x', 'y'], listlablpara=[['x', ''], ['y', '']],
        scalpara=['self', 'self'],
    )
    arguments[field] = invalid
    original = deepcopy(arguments)
    output = tmp_path / 'invalid_metadata'
    monkeypatch.setattr(fixed, 'retr_limtpara',
                        lambda *args: pytest.fail('Invalid metadata reached plotting preparation'))
    monkeypatch.setattr(fixed, 'sample_fixed_chains',
                        lambda *args, **kwargs: pytest.fail('Invalid metadata reached sampling'))

    with pytest.raises(ValueError, match='length|unique'):
        fixed.sample_posterior(
            None, 4, constant_legacy_likelihood, **arguments,
            minmpara=np.array([-1.0, -2.0]), maxmpara=np.array([1.0, 2.0]),
            pathbase=str(output) + '/', boolplot=False, booltqdm=False, typeverb=0,
        )

    assert not output.exists()
    assert arguments == original


@pytest.mark.parametrize('scaling', ['unknown', 'linear', '', None, 1, ['self']])
def test_sample_posterior_rejects_unsupported_scaling_before_outputs(tmp_path, monkeypatch, scaling):
    from copy import deepcopy
    import pcat.fixed as fixed

    scalings = [scaling]
    original = deepcopy(scalings)
    output = tmp_path / 'unsupported_scaling'
    monkeypatch.setattr(fixed, 'retr_limtpara',
                        lambda *args: pytest.fail('Unsupported scaling reached uninitialized plot limits'))
    monkeypatch.setattr(fixed, 'sample_fixed_chains',
                        lambda *args, **kwargs: pytest.fail('Unsupported scaling reached sampling'))

    with pytest.raises(ValueError, match='scalings.*self.*logt.*gaus'):
        fixed.sample_posterior(
            None, 4, constant_legacy_likelihood,
            ['x'], [['x', '']], scalings, np.array([1.0]), np.array([2.0]),
            pathbase=str(output) + '/', boolplot=False, booltqdm=False, typeverb=0,
        )

    assert not output.exists()
    assert scalings == original


@pytest.mark.parametrize('field,invalid', [
    ('meangauspara', None), ('stdvgauspara', None),
    ('meangauspara', np.array([np.nan])), ('meangauspara', np.array([np.inf])),
    ('stdvgauspara', np.array([0.0])), ('stdvgauspara', np.array([-1.0])),
    ('stdvgauspara', np.array([np.nan])), ('stdvgauspara', np.array([np.inf])),
    ('meangauspara', np.array([[0.0]])), ('stdvgauspara', np.array([[1.0]])),
    ('meangauspara', np.array([0.0, 1.0])), ('stdvgauspara', np.array([1.0, 2.0])),
    ('meangauspara', np.ma.array([0.0], mask=[True])),
    ('stdvgauspara', np.ma.array([1.0], mask=[True])),
    ('meangauspara', np.array([0.0 + 0.1j])),
    ('stdvgauspara', np.array([1.0 + 0.1j])),
])
def test_sample_posterior_rejects_invalid_gaussian_parameters_before_outputs(tmp_path, monkeypatch, field, invalid):
    import pcat.fixed as fixed

    parameters = {'meangauspara': np.array([0.0]), 'stdvgauspara': np.array([1.0])}
    parameters[field] = invalid
    original_data = None if invalid is None else np.ma.getdata(invalid).copy()
    original_mask = None if invalid is None else np.ma.getmaskarray(invalid).copy()
    output = tmp_path / 'invalid_gaussian_parameters'
    monkeypatch.setattr(fixed, 'retr_limtpara',
                        lambda *args: pytest.fail('Invalid Gaussian parameters reached plotting preparation'))
    monkeypatch.setattr(fixed, 'sample_fixed_chains',
                        lambda *args, **kwargs: pytest.fail('Invalid Gaussian parameters reached sampling'))

    with pytest.raises(ValueError, match='Gaussian'):
        fixed.sample_posterior(
            None, 4, constant_legacy_likelihood,
            ['x'], [['x', '']], ['gaus'], np.array([-5.0]), np.array([5.0]),
            **parameters, pathbase=str(output) + '/', boolplot=False, booltqdm=False, typeverb=0,
        )

    assert not output.exists()
    if invalid is not None:
        np.testing.assert_array_equal(np.ma.getdata(invalid), original_data)
        np.testing.assert_array_equal(np.ma.getmaskarray(invalid), original_mask)


def test_sample_posterior_runs_with_valid_gaussian_parameters(tmp_path):
    import pcat.fixed as fixed

    means, standard_deviations = np.array([0.0]), np.array([0.2])
    random_state = np.random.get_state()
    try:
        np.random.seed(123)
        samples = fixed.sample_posterior(
            None, 8, constant_legacy_likelihood,
            ['x'], [['x', '']], ['gaus'], np.array([-1.0]), np.array([1.0]),
            meangauspara=means, stdvgauspara=standard_deviations,
            pathbase=str(tmp_path) + '/', boolforcrepr=True, boolplot=False,
            numbsamppostwalk=4, numbsampburnwalk=2, booltqdm=False, booldiag=False, typeverb=0,
        )
    finally:
        np.random.set_state(random_state)

    assert samples['x'].size > 0
    assert np.isfinite(samples['x']).all()
    np.testing.assert_array_equal(means, [0.0])
    np.testing.assert_array_equal(standard_deviations, [0.2])


@pytest.mark.parametrize('field,invalid', [
    ('numbsampwalk', 8.5), ('numbsampwalk', True),
    ('numbsampwalk', 0), ('numbsampwalk', np.nan),
    ('numbsampburnwalk', -1), ('numbsampburnwalk', 1.5), ('numbsampburnwalk', False),
    ('numbsampburnwalkinit', -1), ('numbsampburnwalkinit', 1.5), ('numbsampburnwalkinit', True),
    ('numbsamppostwalk', 0), ('numbsamppostwalk', -1),
    ('numbsamppostwalk', 2.5), ('numbsamppostwalk', True),
])
def test_sample_posterior_rejects_invalid_counts_before_outputs(tmp_path, monkeypatch, field, invalid):
    import pcat.fixed as fixed

    counts = dict(numbsampwalk=8, numbsampburnwalk=2, numbsampburnwalkinit=0, numbsamppostwalk=4)
    counts[field] = invalid
    output = tmp_path / 'invalid_counts'
    monkeypatch.setattr(fixed, 'retr_limtpara',
                        lambda *args: pytest.fail('Invalid count reached plotting preparation'))
    monkeypatch.setattr(fixed, 'sample_fixed_chains',
                        lambda *args, **kwargs: pytest.fail('Invalid count reached sampling'))

    with pytest.raises(ValueError, match='integer'):
        fixed.sample_posterior(
            None, retr_llik=constant_legacy_likelihood, **counts,
            listnamepara=['x'], listlablpara=[['x', '']], scalpara=['self'],
            minmpara=np.array([-1.0]), maxmpara=np.array([1.0]),
            pathbase=str(output) + '/', boolplot=False, booltqdm=False, typeverb=0,
        )

    assert not output.exists()


def test_sample_posterior_accepts_numpy_counts_and_default_retention(tmp_path):
    import pcat.fixed as fixed

    random_state = np.random.get_state()
    try:
        np.random.seed(123)
        samples = fixed.sample_posterior(
            None, np.int64(8), constant_legacy_likelihood,
            ['x'], [['x', '']], ['self'], np.array([-1.0]), np.array([1.0]),
            numbsampburnwalk=np.int64(0), numbsampburnwalkinit=np.int64(0),
            numbsamppostwalk=None, pathbase=str(tmp_path) + '/', boolforcrepr=True,
            boolplot=False, booltqdm=False, booldiag=False, typeverb=0,
        )
    finally:
        np.random.set_state(random_state)

    assert samples['x'].size > 0
    assert np.isfinite(samples['x']).all()
    assert np.all((samples['x'] >= -1.0) & (samples['x'] <= 1.0))


@pytest.mark.parametrize('parameter_count,cache_rows', [(1, 1), (3, 3), (3, 1)])
@pytest.mark.parametrize('verbosity', ['0', '1'])
def test_sample_posterior_reads_warm_start_rows_once(tmp_path, monkeypatch, capsys, parameter_count, cache_rows, verbosity):
    import pcat.fixed as fixed

    class ReachedSampler(Exception):
        pass

    data = tmp_path / 'mcmc' / 'data'
    data.mkdir(parents=True)
    cache = data / 'postpara.csv'
    medians = np.linspace(3.0, 4.0, cache_rows)
    np.savetxt(cache, np.column_stack((medians, np.full(cache_rows, 0.1),
                                     np.full(cache_rows, 0.2))), delimiter=',')
    original_cache = cache.read_bytes()
    read_calls = []
    reader = np.loadtxt
    initial_values = []

    def read_table(*args, **kwargs):
        read_calls.append(args[0])
        return reader(*args, **kwargs)

    def capture_sampler(*args, **kwargs):
        initial_values.append(np.asarray(args[9]))
        raise ReachedSampler

    monkeypatch.setattr(fixed.np, 'loadtxt', read_table)
    monkeypatch.setattr(fixed.scipy.stats.norm, 'rvs', lambda: 0.0)
    monkeypatch.setattr(fixed, 'sample_fixed_chains', capture_sampler)
    monkeypatch.setenv('TDPY_VERBOSITY', verbosity)
    with pytest.raises(ReachedSampler):
        fixed.sample_posterior(
            None, 4, constant_legacy_likelihood,
            [f'x{index}' for index in range(parameter_count)],
            [[f'x{index}', ''] for index in range(parameter_count)],
            ['self'] * parameter_count,
            np.zeros(parameter_count), np.full(parameter_count, 10.0),
            pathbase=str(tmp_path) + '/', boolforcrepr=True,
            boolplot=False, booltqdm=False, typeverb=0,
        )

    expected = medians if parameter_count == cache_rows else np.full(parameter_count, 5.0)
    np.testing.assert_array_equal(initial_values[0], np.tile(expected, (20, 1)))
    assert read_calls == [str(cache)]
    assert capsys.readouterr().out == (f'Reading from {cache}...\n' if verbosity == '1' else '')
    assert cache.read_bytes() == original_cache


@pytest.mark.parametrize('scaling,median', [
    ('self', np.nan), ('self', np.inf), ('self', -np.inf),
    ('logt', np.nan), ('logt', np.inf), ('logt', -np.inf),
    ('gaus', np.nan), ('gaus', np.inf), ('gaus', -np.inf),
    ('self', 0.0), ('self', 11.0), ('logt', 0.0), ('logt', 11.0),
    ('gaus', -40.0), ('gaus', 40.0),
])
def test_sample_posterior_rejects_invalid_warm_start_medians(tmp_path, monkeypatch, scaling, median):
    import pcat.fixed as fixed

    data = tmp_path / 'mcmc' / 'data'
    data.mkdir(parents=True)
    cache = data / 'postpara.csv'
    np.savetxt(cache, np.array([[median, 0.1, 0.2]]), delimiter=',')
    original_cache = cache.read_bytes()
    monkeypatch.setattr(fixed, 'sample_fixed_chains',
                        lambda *args, **kwargs: pytest.fail('Invalid cached median reached the sampler'))
    monkeypatch.setattr(fixed.scipy.stats.norm, 'rvs', lambda: 0.0)
    gaussian = {'meangauspara': np.array([3.0]), 'stdvgauspara': np.array([1.0])} if scaling == 'gaus' else {}

    with np.errstate(all='raise'):
        with pytest.raises(ValueError, match='Cached.*median'):
            fixed.sample_posterior(
                None, 4, constant_legacy_likelihood,
                ['x'], [['x', '']], [scaling], np.array([1.0]), np.array([10.0]),
                **gaussian, pathbase=str(tmp_path) + '/', boolforcrepr=True,
                boolplot=False, booltqdm=False, booldiag=False, typeverb=0,
            )

    assert cache.read_bytes() == original_cache
    assert not (data / 'samppostpara.csv').exists()


def test_sample_posterior_allows_finite_gaussian_warm_start_outside_bounds(tmp_path, monkeypatch):
    import pcat.fixed as fixed

    class ReachedSampler(Exception):
        pass

    data = tmp_path / 'mcmc' / 'data'
    data.mkdir(parents=True)
    cache = data / 'postpara.csv'
    np.savetxt(cache, np.array([[3.0, 0.1, 0.2]]), delimiter=',')
    original_cache = cache.read_bytes()
    captured = []

    def capture_sampler(*args, **kwargs):
        captured.append(np.asarray(args[9]))
        raise ReachedSampler

    monkeypatch.setattr(fixed, 'sample_fixed_chains', capture_sampler)
    monkeypatch.setattr(fixed.scipy.stats.norm, 'rvs', lambda: 0.0)
    with pytest.raises(ReachedSampler):
        fixed.sample_posterior(
            None, 4, constant_legacy_likelihood,
            ['x'], [['x', '']], ['gaus'], np.array([-1.0]), np.array([1.0]),
            meangauspara=np.array([0.0]), stdvgauspara=np.array([1.0]),
            pathbase=str(tmp_path) + '/', boolforcrepr=True,
            boolplot=False, booltqdm=False, booldiag=False, typeverb=0,
        )

    np.testing.assert_allclose(captured[0], np.full((20, 1), 3.0), rtol=0, atol=1e-12)
    assert cache.read_bytes() == original_cache


@pytest.mark.parametrize('seed,native_seed,expected', [
    (None, None, None), (7, None, 7), (None, 17, 17), (7, 17, 17),
])
def test_fixed_chains_forward_seed_and_preserve_explicit_native_option(monkeypatch, seed, native_seed, expected):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import pcat.fixed as fixed

    sampler = Mock(return_value=SimpleNamespace(
        listpostparagenrscalbase=np.zeros((2, 1, 1)),
        listpostlpostotl=np.zeros((2, 1)),
    ))
    monkeypatch.setattr(fixed, 'sample_fixed', sampler)
    options = {} if native_seed is None else {'typeseed': native_seed}

    sample_fixed_chains(
        None, constant_legacy_likelihood, None, ('x',), ('self',),
        (-1.,), (1.,), None, None, np.zeros((1, 1)), 1, 2, 0,
        seed=seed, **options,
    )

    assert sampler.call_args.kwargs.get('typeseed') == expected


@pytest.mark.parametrize('second_seed', [7, 8])
def test_fixed_chains_native_seed_controls_reproducibility(tmp_path, second_seed):
    outputs = [sample_fixed_chains(
        None, constant_legacy_likelihood, None, ('x',), ('self',),
        (-1.,), (1.,), None, None, None, 1, 12, 4,
        pathbase=tmp_path / str(index), seed=seed, booldiag=False,
        booladaptstdp=False, proposal_scales=(0.2,),
    ) for index, seed in enumerate((7, second_seed))]

    if second_seed == 7:
        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    else:
        assert not np.array_equal(outputs[0][0], outputs[1][0])


def test_fixed_chains_estimate_constant_likelihood_evidence(tmp_path):
    chain, logprob, evidence = sample_fixed_chains(
        None, constant_legacy_likelihood, None, ('x',), ('self',),
        (-1.,), (1.,), None, None, np.array([[0.]]),
        1, 8, 2, pathbase=str(tmp_path), typeverb=-1,
        estimate_log_evidence=True, evidence_samples=300, seed=7,
    )
    assert chain.shape == (1, 8, 1)
    assert logprob.shape == (1, 8)
    assert abs(evidence['log_evidence'] - np.log(3.)) < 4 * evidence['relative_error']


@pytest.mark.parametrize('requested_count', [3, 8])
def test_fixed_chains_preserve_worker_and_sweep_order(tmp_path, monkeypatch, requested_count):
    import types
    import pcat.fixed as fixed

    parameter_samples = np.array([[10., 20.], [11., 21.], [12., 22.]])[:, :, None]
    log_probability = np.array([[100., 200.], [101., 201.], [102., 202.]])
    monkeypatch.setattr(fixed, 'sample_fixed', lambda **kwargs: types.SimpleNamespace(
        listpostparagenrscalbase=parameter_samples,
        listpostlpostotl=log_probability,
    ))

    chains, logprob = sample_fixed_chains(
        None, constant_legacy_likelihood, None, ('x',), ('self',),
        (0.,), (30.,), None, None, np.array([[0.]]),
        2, requested_count, 0, pathbase=str(tmp_path),
    )

    np.testing.assert_array_equal(chains[:, :, 0], [[10., 11., 12.], [20., 21., 22.]])
    np.testing.assert_array_equal(logprob, [[100., 101., 102.], [200., 201., 202.]])


def test_fixed_chains_reject_mismatched_completed_arrays(tmp_path, monkeypatch):
    import types
    import pcat.fixed as fixed

    monkeypatch.setattr(fixed, 'sample_fixed', lambda **kwargs: types.SimpleNamespace(
        listpostparagenrscalbase=np.zeros((3, 2, 1)),
        listpostlpostotl=np.zeros((2, 2)),
    ))
    with pytest.raises(ValueError, match='same nonzero number of draws'):
        sample_fixed_chains(
            None, constant_legacy_likelihood, None, ('x',), ('self',),
            (0.,), (30.,), None, None, np.array([[0.]]),
            2, 8, 0, pathbase=str(tmp_path),
        )


@pytest.mark.parametrize('initial_value', [0.0, 1.0])
@pytest.mark.parametrize('retained_count', [4, 8])
def test_allesfitter_adapter_runs_native_pcat(tmp_path, monkeypatch, initial_value, retained_count):
    import sys
    import types
    import h5py
    import pcat.fixed as fixed

    original_sample_fixed_chains = fixed.sample_fixed_chains
    recorded_acceptances = {}

    def capture_acceptances(*args, **kwargs):
        chain, logprob, result = original_sample_fixed_chains(*args, **kwargs)
        result.listpostboolpropaccp = np.asarray(result.listpostboolpropaccp)[:retained_count]
        recorded_acceptances['counts'] = np.asarray(
            result.listpostboolpropaccp, dtype=bool
        ).reshape(-1, 1).sum(axis=0)
        return chain[:, :retained_count], logprob[:, :retained_count], result

    monkeypatch.setattr(fixed, 'sample_fixed_chains', capture_acceptances)

    config = types.ModuleType('allesfitter.config')
    config.init = lambda path: setattr(config, 'BASEMENT', types.SimpleNamespace(
        datadir=path, bounds=[('uniform', 0., 1.)], theta_0=np.array([initial_value]),
        outdir=str(tmp_path / 'results'),
        settings={'mcmc_nwalkers': 1, 'mcmc_total_steps': 8, 'mcmc_thin_by': 1},
    ))
    likelihood = types.ModuleType('allesfitter.mcmc')
    likelihood.mcmc_lnlike = lambda values: -0.5 * (values[0] - 0.5)**2
    allesfitter = types.ModuleType('allesfitter')
    allesfitter.config = config
    monkeypatch.setitem(sys.modules, 'allesfitter', allesfitter)
    monkeypatch.setitem(sys.modules, 'allesfitter.config', config)
    monkeypatch.setitem(sys.modules, 'allesfitter.mcmc', likelihood)

    path = sample_allesfitter_pcat(str(tmp_path))
    print('Reading from %s...' % path)
    with h5py.File(path, 'r') as saved:
        assert saved['mcmc/chain'].shape == (retained_count, 1, 1)
        assert saved['mcmc'].attrs['iteration'] == retained_count
        assert np.all(np.isfinite(saved['mcmc/log_prob'][:]))
        np.testing.assert_array_equal(saved['mcmc/accepted'][:], recorded_acceptances['counts'])


def test_generic_model_uses_main_sampling_pipeline(tmp_path):
    result = sample(
        typeexpr='gener',
        retr_llik=gaussian_log_likelihood,
        parameter_names=('x', 'y'),
        prior_types=('self', 'self'),
        prior_minima=(-5.0, -7.0),
        prior_maxima=(5.0, 7.0),
        initial_values=(0.0, 0.0),
        proposal_scales=(0.08, 0.08),
        proposal_correlation=((1.0, 0.6 / np.sqrt(2.0)), (0.6 / np.sqrt(2.0), 1.0)),
        propwithsing=False,
        pathbase=str(tmp_path),
        strgcnfg='fixed_gaussian',
        numbproc=4,
        numbswep=1500,
        numbburn=500,
        numbsamp=1000,
        boolcheckconv=False,
        booladaptstdp=True,
        typeverb=-1,
    )

    assert result.typeexpr == 'gener'
    assert result.fitt.numbpopl == 0
    assert result.probtran == 0.0
    assert result.probspmr == 0.0
    assert not result.propwithsing
    assert np.all(result.listpostindxproptype == 0)
    assert result.listpostparagenrscalbase.shape == (4000, 2)
    assert np.all(np.isfinite(result.listpostparagenrscalbase))
    np.testing.assert_allclose(
        np.mean(result.listpostparagenrscalbase, axis=0),
        np.zeros(2),
        atol=0.15,
    )
    np.testing.assert_allclose(
        np.cov(result.listpostparagenrscalbase, rowvar=False),
        np.array([[1.0, 0.6], [0.6, 2.0]]),
        atol=0.25,
    )
    assert np.max(result.gmrbparagenrscalbase) < 1.1


def test_native_sampler_preserves_unrelated_empty_output_directories(tmp_path):
    sibling_visuals = tmp_path / 'mcmc' / 'visuals'
    sibling_visuals.mkdir(parents=True)

    sample(
        typeexpr='gener',
        retr_llik=scalar_log_likelihood,
        parameter_names=('x',),
        prior_types=('self',),
        prior_minima=(-2.0,),
        prior_maxima=(2.0,),
        initial_values=(0.0,),
        pathbase=str(tmp_path),
        strgcnfg='preserve_sibling_outputs',
        numbproc=1,
        numbswep=8,
        numbburn=2,
        numbsamp=6,
        typeverb=-1,
    )

    assert sibling_visuals.is_dir()


def test_single_parameter_proposals_adapt_only_selected_scale(tmp_path):
    sample(
        typeexpr='gener',
        retr_llik=gaussian_log_likelihood,
        parameter_names=('x', 'y'),
        prior_types=('self', 'self'),
        prior_minima=(-5.0, -7.0),
        prior_maxima=(5.0, 7.0),
        initial_values=(0.0, 0.0),
        proposal_scales=(0.08, 0.08),
        propwithsing=True,
        probpropblock=0.0,
        pathbase=str(tmp_path),
        strgcnfg='single_parameter_adaptation',
        numbproc=1,
        numbswep=300,
        numbburn=200,
        numbsamp=100,
        booladaptstdp=True,
        typeseed=7,
        typeverb=-1,
    )

    worker_path = (
        Path(tmp_path)
        / 'pcat_runs/single_parameter_adaptation/data/outp/'
        'single_parameter_adaptation/gdatmodi0000post'
    )
    worker = readfile(str(worker_path))
    assert worker.numbpropstdp.sum() == 200
    assert np.all(worker.numbpropstdp > 50)


def test_dynamic_convergence_stops_sampling_early_single_process(tmp_path):
    """boolcheckconv should stop a single-process run well before the requested numbswep."""
    result = sample(
        typeexpr='gener',
        retr_llik=scalar_log_likelihood,
        parameter_names=('x',),
        prior_types=('self',),
        prior_minima=(-3.0,),
        prior_maxima=(3.0,),
        initial_values=(0.0,),
        proposal_scales=(0.5,),
        pathbase=str(tmp_path),
        strgcnfg='dynamic_stop_single',
        numbproc=1,
        numbswep=200_000,
        numbburn=100,
        numbsampconvcheck=50,
        maxmconvrhat=1.2,
        numbsampconveffc=20.,
        numbconvpass=2,
        typeverb=-1,
    )

    assert result.boolconv is True
    assert result.numbswep < 200_000
    assert np.all(np.isfinite(result.listpostparagenrscalbase))


def test_dynamic_convergence_stops_sampling_early_multiple_processes(tmp_path):
    """boolcheckconv should also stop a multi-process run early via the shared cross-worker check."""
    result = sample(
        typeexpr='gener',
        retr_llik=scalar_log_likelihood,
        parameter_names=('x',),
        prior_types=('self',),
        prior_minima=(-3.0,),
        prior_maxima=(3.0,),
        initial_values=(0.0,),
        proposal_scales=(0.5,),
        pathbase=str(tmp_path),
        strgcnfg='dynamic_stop_multi',
        numbproc=2,
        numbswep=200_000,
        numbburn=100,
        numbsampconvmin=200,
        numbsampconvcheck=50,
        maxmconvrhat=1.2,
        numbsampconveffc=20.,
        numbconvpass=2,
        boolcheckconv=True,
        typeverb=-1,
    )

    assert result.boolconv is True
    assert result.numbswep < 200_000
    assert np.all(np.isfinite(result.listpostparagenrscalbase))


def test_demc_proposal_recovers_gaussian_posterior_without_bias(tmp_path):
    """The optional DE-MC within-model jump (probdemc) should not bias the recovered posterior."""
    result = sample(
        typeexpr='gener',
        retr_llik=gaussian_log_likelihood,
        parameter_names=('x', 'y'),
        prior_types=('self', 'self'),
        prior_minima=(-5.0, -7.0),
        prior_maxima=(5.0, 7.0),
        initial_values=(0.0, 0.0),
        proposal_scales=(0.08, 0.08),
        proposal_correlation=((1.0, 0.6 / np.sqrt(2.0)), (0.6 / np.sqrt(2.0), 1.0)),
        propwithsing=False,
        probdemc=0.5,
        pathbase=str(tmp_path),
        strgcnfg='demc_gaussian',
        numbproc=4,
        numbswep=1500,
        numbburn=500,
        numbsamp=1000,
        booladaptstdp=True,
        typeverb=-1,
    )

    assert np.all(result.listpostindxproptype == 0)
    assert np.all(np.isfinite(result.listpostparagenrscalbase))
    np.testing.assert_allclose(
        np.mean(result.listpostparagenrscalbase, axis=0),
        np.zeros(2),
        atol=0.2,
    )
    np.testing.assert_allclose(
        np.cov(result.listpostparagenrscalbase, rowvar=False),
        np.array([[1.0, 0.6], [0.6, 2.0]]),
        atol=0.3,
    )


def test_student_t_within_model_mix_recovers_gaussian_posterior(tmp_path):
    result = sample(
        typeexpr='gener',
        retr_llik=gaussian_log_likelihood,
        parameter_names=('x', 'y'),
        prior_types=('self', 'self'),
        prior_minima=(-5.0, -7.0),
        prior_maxima=(5.0, 7.0),
        initial_values=(0.0, 0.0),
        proposal_scales=(0.08, 0.08),
        proposal_correlation=((1.0, 0.6 / np.sqrt(2.0)), (0.6 / np.sqrt(2.0), 1.0)),
        propwithsing=False,
        probstud=0.5,
        pathbase=str(tmp_path),
        strgcnfg='student_t_gaussian',
        numbproc=2,
        numbswep=1800,
        numbburn=600,
        numbsamp=1200,
        typeverb=-1,
    )

    samples = np.asarray(result.listpostparagenrscalbase)
    assert np.all(result.listpostindxproptype == 0)
    np.testing.assert_allclose(np.mean(samples, axis=0), np.zeros(2), atol=0.2)
    np.testing.assert_allclose(
        np.cov(samples, rowvar=False), np.array([[1.0, 0.6], [0.6, 2.0]]), atol=0.35
    )


def test_generic_model_honors_explicit_plot_settings(tmp_path, monkeypatch):
    import tdpy

    monkeypatch.setattr(tdpy, 'mcmc', object(), raising=False)
    sample(
        typeexpr='gener',
        retr_llik=gaussian_log_likelihood,
        parameter_names=('x', 'y'),
        prior_types=('self', 'self'),
        prior_minima=(-5.0, -7.0),
        prior_maxima=(5.0, 7.0),
        initial_values=(0.0, 0.0),
        pathbase=str(tmp_path),
        strgcnfg='plot_enabled',
        numbproc=1,
        numbswep=20,
        numbburn=10,
        numbsamp=10,
        numbswepplot=5,
        boolmakeplot=True,
        boolmakeplotinit=True,
        boolmakeplotfram=True,
        boolmakeplotfinlpost=True,
        makeanim=True,
        typefileplot='png',
        typeverb=-1,
    )

    visual_root = tmp_path / 'pcat_runs' / 'plot_enabled' / 'visuals'
    assert list(visual_root.rglob('*.png'))


def test_posterior_convergence_applies_rhat_and_effective_sample_thresholds():
    state = type(
        'PosteriorState',
        (),
        {
            'gmrbparagenrscalbase': np.array([1.01, 1.04]),
            'timeatcrpara': np.array([[20.0, 25.0], [16.0, 20.0]]),
            'numbproc': 4,
            'numbsamp': 2000,
        },
    )()

    summary = posterior_convergence(state)
    assert summary['converged']
    assert summary['max_rhat'] == 1.04
    assert summary['min_effective_sample_size'] == 320.0

    state.gmrbparagenrscalbase[1] = 1.06
    assert not posterior_convergence(state)['converged']