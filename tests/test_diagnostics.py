import numpy as np
import pytest
import importlib
from pathlib import Path
from types import SimpleNamespace

import pcat.main as pcat_main
from PIL import Image

from pcat.diagnostics import (
    autocorrelation_time,
    binomial_wilson_interval,
    catalog_count_transitions,
    estimate_kl_divergence,
    gelman_rubin,
)
from pcat.main import (
    _PCATMCMCCompat,
    _configure_proposal_types,
    _retr_adapted_proposal_scale,
    _retr_burn_inverse_temperature,
    _retr_chain_convergence,
    _retr_multichain_convergence,
    _retr_persistent_element_parameter_indices,
    _retr_population_legend_label,
    _retr_parameter_label,
    _retr_posterior_summary_channels,
    _retr_proposal_sweep_label,
    _retr_proposal_type_labels,
    _retr_representative_atcr,
    _retr_true_parameter_value,
    _retr_tempered_log_target_difference,
    _set_element_amplitude_indices,
    _write_proposal_activity_animation,
    _write_proposal_candidate_frame,
)
from pcat.plotting import plot_posterior_convergence


def _capture_figures(monkeypatch):
    figures = []
    make_figure = pcat_main.plt.figure

    def capture_figure(*args, **kwargs):
        figure = make_figure(*args, **kwargs)
        figures.append(figure)
        return figure

    monkeypatch.setattr(pcat_main.plt, "figure", capture_figure)
    return figures


@pytest.mark.parametrize('kind', ['plot_plot', 'plot_trac', 'plot_hist', 'plot_atcr'])
@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
@pytest.mark.parametrize('inherit', [False, True])
def test_compatibility_plot_helpers_honor_background(tmp_path, monkeypatch, kind, typeplotback, inherit):
    from matplotlib.colors import to_rgba
    from tdpy.plotting import plot_background_colors, plot_style

    figures = _capture_figures(monkeypatch)
    options = {'typefileplot': 'png'}
    if not inherit:
        options['typeplotback'] = typeplotback
    outer_theme = typeplotback if inherit else ('dark' if typeplotback == 'white' else 'white')
    path = str(tmp_path / kind)
    with plot_style(outer_theme):
        original_settings = dict(pcat_main.plt.rcParams)
        if kind == 'plot_plot':
            _PCATMCMCCompat.plot_plot(path, [0, 1], np.array([1, 0]), **options)
        elif kind == 'plot_trac':
            _PCATMCMCCompat.plot_trac(path, [0, 1], 'value', **options)
        elif kind == 'plot_hist':
            _PCATMCMCCompat.plot_hist(path, [0, 1], 'value', **options)
        else:
            _PCATMCMCCompat.plot_atcr(path, [1, 0.5], 1.5, **options)
            path += 'atcr'
        assert dict(pcat_main.plt.rcParams) == original_settings
    background, foreground = plot_background_colors(typeplotback)
    figure = figures[0]
    axis = figure.axes[0]
    assert figure.get_facecolor() == to_rgba(background)
    assert axis.get_facecolor() == to_rgba(background)
    assert axis.xaxis.label.get_color() == foreground
    if kind == 'plot_hist':
        assert axis.patches[0].get_edgecolor() == to_rgba(foreground)
    elif kind != 'plot_plot':
        assert axis.lines[0].get_color() == foreground
    with Image.open(path + '.png') as image:
        expected = tuple(int(channel * 255) for channel in to_rgba(background)[:3])
        assert image.convert('RGB').getpixel((0, 0)) == expected


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
@pytest.mark.parametrize('inherit', [False, True])
def test_sample_scopes_plot_theme_and_forwards_selection(tmp_path, monkeypatch, typeplotback, inherit):
    from matplotlib.colors import to_rgba
    from tdpy.plotting import plot_background_colors, plot_style

    sentinel = SimpleNamespace(boolmakeplot=False)
    background, foreground = plot_background_colors(typeplotback)
    path = tmp_path / 'sampler.png'

    def initialize(configuration):
        assert configuration['typeplotback'] == typeplotback
        figure, axis = pcat_main.plt.subplots()
        assert axis.xaxis.label.get_color() == foreground
        figure.savefig(path)
        pcat_main.plt.close(figure)
        return sentinel

    monkeypatch.setattr(pcat_main, 'init', initialize)
    options = {} if inherit else {'typeplotback': typeplotback}
    outer_theme = typeplotback if inherit else ('dark' if typeplotback == 'white' else 'white')
    with plot_style(outer_theme):
        original = dict(pcat_main.plt.rcParams)
        assert pcat_main.sample(**options) is sentinel
        assert dict(pcat_main.plt.rcParams) == original
        with Image.open(path) as image:
            expected = tuple(int(channel * 255) for channel in to_rgba(background)[:3])
            assert image.convert('RGB').getpixel((0, 0)) == expected


def test_sample_restores_theme_after_initialization_failure(monkeypatch):
    from tdpy.plotting import plot_background_colors, plot_style

    def fail(configuration):
        raise RuntimeError('Original initialization failure')

    monkeypatch.setattr(pcat_main, 'init', fail)
    with plot_style('white'):
        original = dict(pcat_main.plt.rcParams)
        with pytest.raises(RuntimeError, match='Original initialization failure'):
            pcat_main.sample(typeplotback='dark')
        assert dict(pcat_main.plt.rcParams) == original
        assert plot_background_colors() == ('white', 'black')


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_posterior_convergence_plots_fixed_and_transdimensional_chains(tmp_path, typeplotback):
    from tdpy.plotting import plot_style

    rng = np.random.default_rng(31)
    state = SimpleNamespace(
        listpostparagenrscalbase=rng.normal(size=(240, 2)),
        listpostnumbelem=rng.integers(0, 4, size=(240, 1)),
        numbproc=2,
        numbsamp=120,
    )

    with plot_style(typeplotback):
        paths = plot_posterior_convergence(state, tmp_path, typefileplot="png")

    assert set(paths) == {
        "fixed_parameter_trace", "fixed_parameter_autocorrelation",
        "fixed_parameter_mixing", "element_count_trace", "element_count_autocorrelation",
        "element_count_occupancy", "element_count_transitions", "element_count_mixing",
    }
    for path in paths.values():
        print(f"Reading from {path}...")
        with Image.open(path) as image:
            assert image.width > 200 and image.height > 200
            expected = (0, 0, 0) if typeplotback == 'dark' else (255, 255, 255)
            assert image.convert('RGB').getpixel((0, 0)) == expected


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_sampler_view_export_overrides_hardcoded_figure_background(tmp_path, typeplotback):
    from matplotlib.colors import to_rgba
    from pcat.plotting import _save_view
    from tdpy.plotting import plot_background_colors, plot_style

    with plot_style(typeplotback):
        figure, axis = pcat_main.plt.subplots(facecolor='white')
        line, = axis.plot([0, 1], [1, 0], color='red', label='Data')
        legend = axis.legend(facecolor='white')
        path = _save_view(figure, tmp_path / 'operation', 'png')
        background, foreground = plot_background_colors()
        assert axis.get_facecolor() == to_rgba(background)
        assert legend.get_frame().get_facecolor() == to_rgba(background)
        assert legend.get_texts()[0].get_color() == foreground
        assert line.get_color() == 'red'
        with Image.open(path) as image:
            expected = tuple(int(channel * 255) for channel in to_rgba(background)[:3])
            assert image.convert('RGB').getpixel((0, 0)) == expected


def test_catalog_transitions_do_not_cross_independent_chains():
    counts = np.array([[0, 4], [0, 4], [0, 4]])
    transitions = catalog_count_transitions(counts)
    assert transitions.sum() == 4
    assert transitions[0, 0] == transitions[4, 4] == 2
    assert transitions[0, 4] == transitions[4, 0] == 0


def test_catalog_count_rhat_uses_sample_major_worker_order(monkeypatch, tmp_path):
    plotting = importlib.import_module("pcat.plotting")
    compared = []
    original = plotting.gelman_rubin

    def record_chains(chains):
        compared.append(np.asarray(chains).copy())
        return original(chains)

    monkeypatch.setattr(plotting, "gelman_rubin", record_chains)
    state = SimpleNamespace(
        listpostparagenrscalbase=np.arange(8, dtype=float)[:, None],
        listpostnumbelem=np.array([[0], [4], [1], [3], [0], [4], [1], [3]]),
        numbproc=2, numbsamp=4,
    )
    figures = plot_posterior_convergence(state, tmp_path)

    np.testing.assert_array_equal(compared[-1], [[0, 4], [1, 3], [0, 4], [1, 3]])
    assert figures["element_count_mixing"].is_file()


def test_element_distribution_compares_early_and_late_posterior(tmp_path):
    state = SimpleNamespace(
        listpostparagenrscalbase=np.arange(20, dtype=float)[:, None],
        listpostnumbelem=np.ones((20, 1), dtype=int),
        listpostdictelem=[[{"flux": np.array([1.0 + sample / 10])}] for sample in range(20)],
        numbproc=1,
        numbsamp=20,
    )
    figures = plot_posterior_convergence(state, tmp_path)
    assert figures["element_parameter_pop0_flux_stability"].is_file()


def test_completed_plot_enabled_sample_writes_convergence_suite(monkeypatch, tmp_path):
    sampler = importlib.import_module("pcat.main")
    plots = importlib.import_module("pcat.plotting")
    state = SimpleNamespace(boolmakeplot=True, boolmakeplotfinlpost=True,
                            listpostparagenrscalbase=np.ones((4, 1)), typefileplot="png")
    calls = []
    monkeypatch.setattr(sampler, "init_image", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(sampler, "init", lambda config: state)
    monkeypatch.setattr(plots, "plot_posterior_convergence",
                        lambda result, path, typefileplot: calls.append((result, path, typefileplot)))

    assert sampler.sample(typeexpr="gmix", pathbase=str(tmp_path), strgcnfg="test") is state
    assert calls == [(state, Path(sampler.retr_pathplotcnfg(str(tmp_path), "test"))
                      / "post" / "convergence", "png")]

    state.boolmakeplotfinlpost = False
    sampler.sample(typeexpr="gmix", pathbase=str(tmp_path), strgcnfg="test")
    assert len(calls) == 1


def test_real_fixed_sample_writes_convergence_figures(tmp_path):
    from pcat.fixed import sample_fixed_chains

    sample_fixed_chains(
        None, lambda values, state: -0.5 * values[0] ** 2, None,
        ("location",), ("self",), (-2.0,), (2.0,), None, None,
        np.array([[0.0]]), 1, 8, 2, pathbase=str(tmp_path), typeverb=0,
        boolmakeplot=True, boolmakeplotinit=False, boolmakeplotfram=False,
        boolmakeplotfinlpost=True, makeanim=False,
    )
    figures = list(tmp_path.glob("pcat_runs/*/visuals/post/convergence/fixed_parameter_*.png"))
    assert len(figures) == 3


def test_gmrb_rejects_separated_constant_chains():
    chains = np.column_stack((np.zeros(100), np.full(100, 100.0)))

    assert np.isinf(_PCATMCMCCompat.gmrb_test(chains))
    assert np.isinf(gelman_rubin(chains))


@pytest.mark.parametrize('kind', ['masked', 'nan', 'positive_infinity', 'negative_infinity', 'complex'])
@pytest.mark.parametrize('legacy', [False, True])
def test_gelman_rubin_marks_invalid_traces_undefined_without_warnings(kind, legacy):
    import warnings

    chains = np.column_stack((np.arange(8.0), np.arange(8.0)))
    if kind == 'masked':
        chains = np.ma.array(chains, mask=False)
        chains.data[0, :] = 1e6
        chains.mask[0, :] = True
    elif kind == 'complex':
        chains = chains.astype(complex)
        chains[0, 0] += 0.5j
    else:
        chains[0, 0] = {'nan': np.nan, 'positive_infinity': np.inf, 'negative_infinity': -np.inf}[kind]
    original = chains.copy()
    calculate = _PCATMCMCCompat.gmrb_test if legacy else gelman_rubin
    with warnings.catch_warnings(), np.errstate(all='raise'):
        warnings.simplefilter('error')
        assert np.isnan(calculate(chains))
    np.testing.assert_array_equal(np.ma.getdata(chains), np.ma.getdata(original))
    np.testing.assert_array_equal(np.ma.getmaskarray(chains), np.ma.getmaskarray(original))


@pytest.mark.parametrize('kind', ['varied', 'identical_constant', 'separated_constant'])
@pytest.mark.parametrize('unmasked_container', [False, True])
@pytest.mark.parametrize('legacy', [False, True])
def test_gelman_rubin_preserves_valid_trace_behavior(kind, unmasked_container, legacy):
    if kind == 'varied':
        chains = np.random.default_rng(614).normal(size=(12, 2))
        within = np.mean(np.var(chains, axis=0, ddof=1))
        between = 12 * np.var(np.mean(chains, axis=0), ddof=1)
        expected = float(np.sqrt(((11 / 12) * within + between / 12) / within))
    elif kind == 'identical_constant':
        chains = np.ones((12, 2))
        expected = np.nan
    else:
        chains = np.column_stack((np.zeros(12), np.ones(12)))
        expected = np.inf
    if unmasked_container:
        chains = np.ma.array(chains, mask=False)
    original = chains.copy()
    calculate = _PCATMCMCCompat.gmrb_test if legacy else gelman_rubin
    actual = calculate(chains)
    if np.isnan(expected):
        assert np.isnan(actual)
    else:
        assert actual == expected
    np.testing.assert_array_equal(chains, original)


@pytest.mark.parametrize('exponent', [-1000, -900, 900, 1020])
@pytest.mark.parametrize('unmasked_container', [False, True])
@pytest.mark.parametrize('legacy', [False, True])
def test_gelman_rubin_preserves_estimate_at_extreme_scales(exponent, unmasked_container, legacy):
    import warnings

    chains = np.random.default_rng(616).normal(size=(16, 2))
    expected = gelman_rubin(chains)
    values = np.ldexp(chains, exponent)
    if unmasked_container:
        values = np.ma.array(values, mask=False)
    original = values.copy()
    calculate = _PCATMCMCCompat.gmrb_test if legacy else gelman_rubin
    with warnings.catch_warnings(), np.errstate(all='raise'):
        warnings.simplefilter('error')
        assert calculate(values) == expected
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize('value', [1e-200, 1e200, np.finfo(float).max])
@pytest.mark.parametrize('separated', [False, True])
@pytest.mark.parametrize('legacy', [False, True])
def test_gelman_rubin_extreme_constants_keep_existing_conventions(value, separated, legacy):
    import warnings

    values = np.column_stack((np.full(12, value), np.full(12, -value if separated else value)))
    calculate = _PCATMCMCCompat.gmrb_test if legacy else gelman_rubin
    with warnings.catch_warnings(), np.errstate(all='raise'):
        warnings.simplefilter('error')
        actual = calculate(values)
    assert np.isinf(actual) if separated else np.isnan(actual)


@pytest.mark.parametrize('statistics', [np.array([0.9, 1.1]), np.array([0.8, 0.9]),
                                      np.ones(3), np.array([1.1, 1.2]),
                                      np.array([[0.9, 1.1], [0.95, 1.2]])])
@pytest.mark.parametrize('typefileplot', ['png', 'pdf'])
def test_gelman_rubin_histogram_counts_every_finite_statistic(tmp_path, monkeypatch, statistics, typefileplot):
    from matplotlib import pyplot as plt
    from pcat import plotting

    original = statistics.copy()
    captured = []
    save = plotting.save_figure
    before = set(plt.get_fignums())

    def capture(figure, path, *arguments, **kwargs):
        axis = figure.axes[0]
        assert sum(bar.get_height() for bar in axis.patches) == statistics.size
        assert axis.get_xlabel() == 'PSRF'
        assert axis.get_ylabel() == '$N_p$'
        captured.append(figure)
        return save(figure, path, *arguments, **kwargs)

    monkeypatch.setattr(plotting, 'save_figure', capture)
    plotting.plot_gelman_rubin(str(tmp_path / 'complete_rhat_'), statistics, typefileplot)
    assert len(captured) == 1
    np.testing.assert_array_equal(statistics, original)
    assert set(plt.get_fignums()) == before


@pytest.mark.parametrize('statistics', [np.array([]), np.array([np.nan]), np.array([np.inf]),
                                      np.array([-np.inf]), np.ma.array([0.9, 1.1], mask=[True, False]),
                                      np.array([1.0 + 0.0j]), np.array(['1.0'])])
def test_gelman_rubin_histogram_rejects_undefined_inputs_before_rendering(tmp_path, monkeypatch, statistics):
    import warnings
    from unittest.mock import Mock
    from matplotlib import pyplot as plt
    from pcat.plotting import plot_gelman_rubin

    render = Mock(side_effect=AssertionError('Undefined R-hat input reached figure creation'))
    monkeypatch.setattr(plt, 'subplots', render)
    with warnings.catch_warnings(), np.errstate(all='raise'), pytest.raises(ValueError, match='R-hat statistics'):
        warnings.simplefilter('error')
        plot_gelman_rubin(str(tmp_path / 'undefined_rhat_'), statistics)
    render.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_tempered_burn_schedule_reaches_unit_temperature_before_sampling():
    schedule = [
        _retr_burn_inverse_temperature(sweep, 4, tempered_fraction=0.75, enabled=True)
        for sweep in range(6)
    ]

    assert schedule == pytest.approx([(1. / 3.) ** 4, (2. / 3.) ** 4, 1., 1., 1., 1.])
    assert _retr_burn_inverse_temperature(0, 4, enabled=False) == 1.


def test_tempered_target_scales_likelihood_but_not_prior():
    current = SimpleNamespace(lpritotl=-2.0, lliktotl=-10.0)
    candidate = SimpleNamespace(lpritotl=-3.0, lliktotl=-6.0)

    assert _retr_tempered_log_target_difference(current, candidate, 1.0) == pytest.approx(3.0)
    assert _retr_tempered_log_target_difference(current, candidate, 0.25) == pytest.approx(0.0)
    assert _retr_tempered_log_target_difference(current, candidate, 0.25, target="prio") == pytest.approx(-1.0)


@pytest.mark.parametrize('successes, trials, z_score', [
    (0.5, 2, 1.0), (1, 2.5, 1.0), (1, 2, np.nan), (1, 2, np.inf),
    (1, 2, -np.inf), (np.nan, 2, 1.0), (1, np.inf, 1.0),
    (np.ma.masked, 2, 1.0), (1, 2, np.ma.masked), ([1], 2, 1.0),
    (1, 2, [1.0]), (1.0 + 0.0j, 2, 1.0), (1, 2, 1.0 + 0.0j),
    ('1', 2, 1.0),
])
def test_binomial_wilson_interval_rejects_invalid_scalar_inputs(successes, trials, z_score):
    import warnings

    with warnings.catch_warnings(), np.errstate(all='raise'), pytest.raises(ValueError, match='binomial counts|z_score'):
        warnings.simplefilter('error')
        binomial_wilson_interval(successes, trials, z_score)


@pytest.mark.parametrize('successes, trials', [(0, 40), (20, 40), (40, 40)])
@pytest.mark.parametrize('convert', [int, float, np.int64, np.uint8, np.asarray,
                                    lambda value: np.ma.array(value, mask=False)])
def test_binomial_wilson_interval_preserves_integer_scalar_encodings(successes, trials, convert):
    import warnings

    expected = binomial_wilson_interval(successes, trials)
    with warnings.catch_warnings(), np.errstate(all='raise'):
        warnings.simplefilter('error')
        actual = binomial_wilson_interval(convert(successes), convert(trials), convert(1))
    assert actual == expected
    assert 0.0 <= actual[0] <= actual[1] <= 1.0


@pytest.mark.parametrize('trials', [1, 3, 5, 10, 40, 100])
@pytest.mark.parametrize('z_score', [0.1, 0.5, 1.0, 1.96, 3.0])
@pytest.mark.parametrize('all_successes', [False, True])
def test_binomial_wilson_boundary_limits_match_scipy(trials, z_score, all_successes):
    from scipy.stats import binomtest, norm

    successes = trials if all_successes else 0
    with np.errstate(all='raise'):
        lower, upper = binomial_wilson_interval(successes, trials, z_score)
    reference = binomtest(successes, trials).proportion_ci(
        confidence_level=2 * norm.cdf(z_score) - 1, method='wilson',
    )
    assert 0.0 <= lower <= upper <= 1.0
    assert upper == 1.0 if all_successes else lower == 0.0
    np.testing.assert_allclose((lower, upper), (reference.low, reference.high), rtol=5e-14, atol=3e-15)


@pytest.mark.parametrize('successes, trials', [(0, 40), (1, 40), (20, 40), (39, 40), (40, 40),
                                             (2**62, 2**63 - 1), (5e307, 1e308),
                                             (1, 1e160), (1, 1e308), (0, 1e308),
                                             (1e308, 1e308), (40, 1e308)])
@pytest.mark.parametrize('z_score', [0.5, 1.0, 3.0, 1e155, 1e160, np.finfo(float).max])
def test_binomial_wilson_extreme_values_match_decimal(successes, trials, z_score):
    from decimal import Decimal, localcontext
    import warnings

    with localcontext() as context:
        context.prec = 800
        count = Decimal(int(successes))
        total = Decimal(int(trials))
        score = Decimal.from_float(float(z_score))
        fraction = count / total
        squared = score**2
        denominator = 1 + squared / total
        center = (fraction + squared / (2 * total)) / denominator
        half_width = score * (fraction * (1 - fraction) / total + squared / (4 * total**2)).sqrt() / denominator
        expected = float(center - half_width), float(center + half_width)
    with warnings.catch_warnings(), np.errstate(all='raise'):
        warnings.simplefilter('error')
        actual = binomial_wilson_interval(successes, trials, z_score)
    assert 0.0 <= actual[0] <= actual[1] <= 1.0
    if expected[0] > 0.0:
        assert actual[0] > 0.0
    np.testing.assert_array_max_ulp(actual, expected, maxulp=8)


def test_binomial_wilson_interval_has_finite_boundary_uncertainty():
    empty_lower, empty_upper = binomial_wilson_interval(0, 40)
    full_lower, full_upper = binomial_wilson_interval(40, 40)
    half_lower, half_upper = binomial_wilson_interval(20, 40)

    assert type(half_lower) is float
    assert type(half_upper) is float
    assert empty_lower == 0.0
    assert empty_upper > 0.0
    assert full_lower < 1.0
    assert full_upper == 1.0
    assert np.isclose(half_lower, 1.0 - half_upper)


def test_kl_divergence_matches_analytic_gaussian_forward_value():
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(412)
    mean_p = np.array([0.6, -0.3])
    covariance_p = np.array([[0.8, 0.15], [0.15, 1.1]])
    mean_q = np.array([-0.2, 0.5])
    covariance_q = np.array([[1.4, 0.25], [0.25, 0.7]])
    chains = rng.multivariate_normal(mean_p, covariance_p, size=(3, 2500))
    density_p = multivariate_normal(mean=mean_p, cov=covariance_p)
    density_q = multivariate_normal(mean=mean_q, cov=covariance_q)

    result = estimate_kl_divergence(chains, density_p.logpdf, density_q.logpdf)
    difference = mean_q - mean_p
    sign_p, logdet_p = np.linalg.slogdet(covariance_p)
    sign_q, logdet_q = np.linalg.slogdet(covariance_q)
    analytic = 0.5 * (
        np.trace(np.linalg.solve(covariance_q, covariance_p))
        + difference @ np.linalg.solve(covariance_q, difference)
        - mean_p.size + logdet_q - logdet_p
    )

    assert sign_p == sign_q == 1
    assert abs(result["kl_divergence"] - analytic) < 4.0 * result["standard_error"]
    assert result["standard_error"] > 0.0
    assert 0.0 < result["effective_sample_size"] <= chains.shape[0] * chains.shape[1]


def test_kl_divergence_does_not_clip_negative_finite_sample_estimates():
    from scipy.stats import norm

    samples = np.ones((1, 2, 1))
    result = estimate_kl_divergence(
        samples,
        lambda values: norm.logpdf(values[:, 0], loc=0.0, scale=1.0),
        lambda values: norm.logpdf(values[:, 0], loc=1.0, scale=1.0),
    )

    assert result["kl_divergence"] == pytest.approx(-0.5)


def test_kl_divergence_reports_infinity_when_reference_has_zero_support():
    samples = np.zeros((2, 4, 1))
    result = estimate_kl_divergence(
        samples,
        lambda values: np.zeros(values.shape[0]),
        lambda values: np.full(values.shape[0], -np.inf),
    )

    assert result == {
        "kl_divergence": np.inf,
        "standard_error": np.inf,
        "effective_sample_size": 0.0,
    }


@pytest.mark.parametrize('scale', [1e-200, 1e200])
@pytest.mark.parametrize('chain_count', [1, 2])
def test_kl_divergence_preserves_finite_log_ratio_scale(scale, chain_count):
    samples = np.tile(np.array([0.5, 1.0, 0.5, 1.0])[None, :, None], (chain_count, 1, 1))
    samples += 0.5 * np.arange(chain_count)[:, None, None]
    original = samples.copy()
    baseline = estimate_kl_divergence(
        samples, lambda values: np.zeros(len(values)), lambda values: -values[:, 0],
    )
    with np.errstate(all='raise'):
        actual = estimate_kl_divergence(
            samples, lambda values: np.zeros(len(values)),
            lambda values: -scale * values[:, 0],
        )
    assert actual['kl_divergence'] / scale == pytest.approx(baseline['kl_divergence'])
    assert actual['standard_error'] / scale == pytest.approx(baseline['standard_error'])
    assert actual['effective_sample_size'] == baseline['effective_sample_size']
    np.testing.assert_array_equal(samples, original)


@pytest.mark.parametrize('magnitude', [1e308, np.finfo(float).max])
@pytest.mark.parametrize('chain_count', [1, 2])
def test_kl_divergence_scales_before_overflowing_log_density_subtraction(magnitude, chain_count):
    samples = np.arange(4.0 * chain_count).reshape(chain_count, 4, 1)
    log_density = np.array([[1.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])[:chain_count].ravel()
    original_samples = samples.copy()
    original_density = log_density.copy()
    baseline = estimate_kl_divergence(
        samples, lambda draws: log_density, lambda draws: -log_density,
    )
    with np.errstate(all='raise'):
        actual = estimate_kl_divergence(
            samples, lambda draws: magnitude * log_density,
            lambda draws: -magnitude * log_density,
        )
    assert actual['kl_divergence'] / magnitude == pytest.approx(baseline['kl_divergence'])
    assert actual['standard_error'] / magnitude == pytest.approx(baseline['standard_error'])
    assert actual['effective_sample_size'] == baseline['effective_sample_size']
    np.testing.assert_array_equal(samples, original_samples)
    np.testing.assert_array_equal(log_density, original_density)


def test_kl_divergence_representable_constant_near_float_limit():
    samples = np.zeros((2, 4, 1))
    magnitude = np.finfo(float).max
    with np.errstate(all='raise'):
        actual = estimate_kl_divergence(
            samples, lambda values: np.zeros(len(values)),
            lambda values: np.full(len(values), -magnitude),
        )
    assert actual == {
        'kl_divergence': magnitude,
        'standard_error': 0.0,
        'effective_sample_size': 8.0,
    }


@pytest.mark.parametrize('correlation', [-0.7, 0.0, 0.7])
def test_mutual_information_matches_correlated_gaussian(correlation):
    from scipy.stats import multivariate_normal, norm
    from pcat.diagnostics import estimate_mutual_information

    distribution = multivariate_normal(cov=[[1.0, correlation], [correlation, 1.0]])
    samples = np.random.default_rng(601).multivariate_normal(
        [0.0, 0.0], distribution.cov, size=(3, 2500),
    )
    result = estimate_mutual_information(
        samples, distribution.logpdf,
        lambda values: norm.logpdf(values[:, 0]),
        lambda values: norm.logpdf(values[:, 0]),
    )
    analytic = -0.5 * np.log1p(-correlation**2)
    assert abs(result['mutual_information'] - analytic) < 4 * result['standard_error'] + 1e-14
    assert 0 < result['effective_sample_size'] <= samples.shape[0] * samples.shape[1]


def test_mutual_information_handles_vector_groups_and_independence():
    from scipy.stats import norm
    from pcat.diagnostics import estimate_mutual_information

    samples = np.random.default_rng(602).normal(size=(100, 4))
    result = estimate_mutual_information(
        samples,
        lambda values: norm.logpdf(values[:, :2]).sum(axis=1) + norm.logpdf(values[:, 2:]).sum(axis=1),
        lambda values: norm.logpdf(values).sum(axis=1),
        lambda values: norm.logpdf(values).sum(axis=1),
        split_index=2,
    )
    assert result == {'mutual_information': 0.0, 'standard_error': 0.0, 'effective_sample_size': 100.0}


@pytest.mark.parametrize('split_index', [0, 2, -1, 0.5, True, None])
def test_mutual_information_requires_two_nonempty_column_groups(split_index):
    from pcat.diagnostics import estimate_mutual_information

    with pytest.raises(ValueError, match='split_index'):
        estimate_mutual_information(np.zeros((10, 2)), None, None, None, split_index)


@pytest.mark.parametrize('field', ['samples', 'joint', 'x', 'y'])
@pytest.mark.parametrize('kind', ['masked', 'complex'])
def test_information_estimates_reject_discarded_masks_and_complex_values(field, kind):
    from pcat.diagnostics import estimate_mutual_information

    def invalid(values):
        if kind == 'complex':
            return values.astype(complex) + 1j
        mask = np.zeros(values.shape, dtype=bool)
        mask.flat[0] = True
        return np.ma.array(values, mask=mask)

    samples = np.zeros((10, 2))
    callbacks = {name: (lambda values: np.zeros(len(values))) for name in ('joint', 'x', 'y')}
    if field == 'samples':
        samples = invalid(samples)
    else:
        callbacks[field] = lambda values: invalid(np.zeros(len(values)))
    with pytest.raises(ValueError, match='unmasked real'):
        estimate_mutual_information(samples, callbacks['joint'], callbacks['x'], callbacks['y'])


@pytest.mark.parametrize('invalid_marginal', [np.nan, np.inf])
def test_mutual_information_rejects_invalid_marginal_densities(invalid_marginal):
    from pcat.diagnostics import estimate_mutual_information

    with pytest.raises(ValueError, match='marginal log densities'):
        estimate_mutual_information(
            np.zeros((10, 2)), lambda values: np.zeros(len(values)),
            lambda values: np.full(len(values), invalid_marginal),
            lambda values: np.zeros(len(values)),
        )


@pytest.mark.parametrize('field', ['joint', 'x', 'y'])
def test_mutual_information_rejects_wrong_callback_shapes(field):
    from pcat.diagnostics import estimate_mutual_information

    callbacks = {name: (lambda values: np.zeros(len(values))) for name in ('joint', 'x', 'y')}
    callbacks[field] = lambda values: np.zeros((len(values), 1))
    with pytest.raises(ValueError, match='one value per draw'):
        estimate_mutual_information(np.zeros((10, 2)), callbacks['joint'], callbacks['x'], callbacks['y'])


def test_mutual_information_preserves_negative_finite_sample_estimates():
    from scipy.stats import multivariate_normal, norm
    from pcat.diagnostics import estimate_mutual_information

    samples = np.array([[1.0, -1.0], [1.0, -1.0]])
    joint = multivariate_normal(cov=[[1.0, 0.5], [0.5, 1.0]])
    result = estimate_mutual_information(
        samples, joint.logpdf, lambda values: norm.logpdf(values[:, 0]),
        lambda values: norm.logpdf(values[:, 0]),
    )
    expected = np.mean(joint.logpdf(samples) - norm.logpdf(samples).sum(axis=1))
    assert expected < 0
    assert result['mutual_information'] == pytest.approx(expected)


def test_autocorrelation_distinguishes_independent_draws_and_random_walk():
    random = np.random.default_rng(42)
    independent = random.normal(size=(2000, 1))
    random_walk = np.cumsum(random.normal(size=(2000, 1)), axis=0)

    _, time_independent = _PCATMCMCCompat.retr_timeatcr(independent)
    _, time_random_walk = _PCATMCMCCompat.retr_timeatcr(random_walk)
    _, direct_time = autocorrelation_time(independent)

    assert time_independent[0] < 2.0
    assert time_random_walk[0] > 20.0
    assert direct_time[0] == pytest.approx(time_independent[0])


@pytest.mark.parametrize('kind', ['masked', 'nan', 'positive_infinity', 'negative_infinity'])
@pytest.mark.parametrize('legacy', [False, True])
def test_autocorrelation_marks_only_invalid_series_undefined(kind, legacy):
    import warnings

    values = np.column_stack((np.arange(12.0), np.sin(np.arange(12.0))))
    expected_correlations, expected_time = autocorrelation_time(values[:, 1])
    if kind == 'masked':
        values = np.ma.array(values, mask=False)
        values.data[0, 0] = 1e6
        values.mask[0, 0] = True
    else:
        values[0, 0] = {'nan': np.nan, 'positive_infinity': np.inf, 'negative_infinity': -np.inf}[kind]
    original = values.copy()
    calculate = _PCATMCMCCompat.retr_timeatcr if legacy else autocorrelation_time
    with warnings.catch_warnings(), np.errstate(all='raise'):
        warnings.simplefilter('error')
        correlations, times = calculate(values)
    assert correlations.shape == (2, 6)
    assert times.shape == (2,)
    assert np.isnan(correlations[0]).all()
    assert np.isnan(times[0])
    np.testing.assert_array_equal(correlations[1], expected_correlations)
    assert times[1] == expected_time
    np.testing.assert_array_equal(np.ma.getdata(values), np.ma.getdata(original))
    np.testing.assert_array_equal(np.ma.getmaskarray(values), np.ma.getmaskarray(original))


@pytest.mark.parametrize('shape', [(12,), (12, 2), (12, 2, 2)])
@pytest.mark.parametrize('unmasked_container', [False, True])
@pytest.mark.parametrize('legacy', [False, True])
def test_autocorrelation_preserves_valid_series_shapes_and_constants(shape, unmasked_container, legacy):
    values = np.random.default_rng(615).normal(size=shape)
    expected = autocorrelation_time(values)
    if unmasked_container:
        values = np.ma.array(values, mask=False)
    original = values.copy()
    calculate = _PCATMCMCCompat.retr_timeatcr if legacy else autocorrelation_time
    actual = calculate(values)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(values, original)
    correlations, times = calculate(np.ones(shape))
    assert np.isnan(correlations).all()
    assert np.isnan(times).all()


@pytest.mark.parametrize('legacy', [False, True])
def test_autocorrelation_complex_series_are_undefined_without_warnings(legacy):
    import warnings

    values = np.arange(12.0).astype(complex) + 0.5j
    calculate = _PCATMCMCCompat.retr_timeatcr if legacy else autocorrelation_time
    with warnings.catch_warnings(), np.errstate(all='raise'):
        warnings.simplefilter('error')
        correlations, times = calculate(values)
    assert np.isnan(correlations).all()
    assert np.isnan(times).all()


@pytest.mark.parametrize('exponent', [-1000, -900, 900, 1020])
@pytest.mark.parametrize('unmasked_container', [False, True])
@pytest.mark.parametrize('legacy', [False, True])
def test_autocorrelation_preserves_estimates_at_extreme_scales(exponent, unmasked_container, legacy):
    import warnings

    series = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 1.0,
                       -1.0, -2.0, 2.0, 0.0, 1.0, -1.0, 3.0, -3.0])
    expected = autocorrelation_time(np.column_stack((series, np.sin(np.arange(16.0)))))
    values = np.column_stack((np.ldexp(series, exponent), np.sin(np.arange(16.0))))
    if unmasked_container:
        values = np.ma.array(values, mask=False)
    original = values.copy()
    calculate = _PCATMCMCCompat.retr_timeatcr if legacy else autocorrelation_time
    with warnings.catch_warnings(), np.errstate(all='raise'):
        warnings.simplefilter('error')
        actual = calculate(values)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize('value', [1e-200, 1e200, np.finfo(float).max])
@pytest.mark.parametrize('legacy', [False, True])
def test_autocorrelation_extreme_constants_are_undefined_without_warnings(value, legacy):
    import warnings

    calculate = _PCATMCMCCompat.retr_timeatcr if legacy else autocorrelation_time
    with warnings.catch_warnings(), np.errstate(all='raise'):
        warnings.simplefilter('error')
        correlations, times = calculate(np.full(16, value))
    assert np.isnan(correlations).all()
    assert np.isnan(times).all()


def test_autocorrelation_plot_selects_finite_nonconstant_series():
    autocorrelation = np.array([[[np.nan, np.nan, np.nan], [1.0, 0.5, 0.1]]])
    correlation_time = np.array([[np.nan, 2.2]])

    series, time = _retr_representative_atcr(autocorrelation, correlation_time)

    assert series == pytest.approx([1.0, 0.5, 0.1])
    assert time == pytest.approx(2.2)


@pytest.mark.parametrize('masked_field', ['correlation', 'time', 'both'])
@pytest.mark.parametrize('all_invalid', [False, True])
def test_autocorrelation_plot_excludes_explicitly_masked_candidates(masked_field, all_invalid):
    import warnings

    correlations = np.ma.array([[1.0, 0.9, 0.8], [1.0, 0.5, 0.1]], mask=False)
    times = np.ma.array([1000.0, 2.2], mask=False)
    if masked_field in ('correlation', 'both'):
        correlations.mask[0, 1] = True
        if all_invalid:
            correlations.mask[1, 1] = True
    if masked_field in ('time', 'both'):
        times.mask[0] = True
        if all_invalid:
            times.mask[1] = True
    originals = correlations.copy(), times.copy()
    with warnings.catch_warnings(), np.errstate(all='raise'):
        warnings.simplefilter('error')
        series, time = _retr_representative_atcr(correlations, times)
    if all_invalid:
        assert series is None and time is None
    else:
        assert isinstance(series, np.ndarray) and not np.ma.isMaskedArray(series)
        np.testing.assert_array_equal(series, [1.0, 0.5, 0.1])
        assert time == 2.2
    for actual, original in zip((correlations, times), originals):
        np.testing.assert_array_equal(actual.data, original.data)
        np.testing.assert_array_equal(actual.mask, original.mask)


@pytest.mark.parametrize('unmasked_container', [False, True])
def test_autocorrelation_plot_preserves_longest_valid_candidate(unmasked_container):
    correlations = np.array([[[1.0, 0.5, 0.1], [1.0, 0.8, 0.4]],
                             [[1.0, 0.9, 0.8], [np.nan, np.nan, np.nan]]])
    times = np.array([[2.2, 4.0], [8.0, np.nan]])
    if unmasked_container:
        correlations = np.ma.array(correlations, mask=False)
        times = np.ma.array(times, mask=False)
    series, time = _retr_representative_atcr(correlations, times)
    assert isinstance(series, np.ndarray) and not np.ma.isMaskedArray(series)
    np.testing.assert_array_equal(series, [1.0, 0.9, 0.8])
    assert time == 8.0


@pytest.mark.parametrize('curves, times', [
    (np.ones((2, 3)), np.ones(1)), (np.ones((2, 3)), np.ones(3)),
    (np.ones((2, 3)), np.array([])), (np.ones((0, 3)), np.ones(1)),
    (np.ones((2, 0)), np.ones(2)), (np.array(1.0), np.array(1.0)),
])
def test_autocorrelation_plot_rejects_unpaired_series_metadata(curves, times):
    import warnings

    originals = curves.copy(), times.copy()
    with warnings.catch_warnings(), np.errstate(all='raise'), pytest.raises(ValueError, match='lag axis|one correlation time'):
        warnings.simplefilter('error')
        _retr_representative_atcr(curves, times)
    np.testing.assert_array_equal(curves, originals[0])
    np.testing.assert_array_equal(times, originals[1])


@pytest.mark.parametrize('times', [np.array(2.2), np.array([2.2]), np.array([[2.2]])])
def test_autocorrelation_plot_preserves_single_series_time_layouts(times):
    series, time = _retr_representative_atcr(np.array([1.0, 0.5, 0.1]), times)
    np.testing.assert_array_equal(series, [1.0, 0.5, 0.1])
    assert time == 2.2


def test_autocorrelation_plot_preserves_flattened_time_layout():
    curves = np.array([[[1.0, 0.5, 0.1], [1.0, 0.8, 0.4]],
                       [[1.0, 0.9, 0.8], [1.0, 0.1, 0.0]]])
    series, time = _retr_representative_atcr(curves, np.array([2.2, 4.0, 8.0, 1.0]))
    np.testing.assert_array_equal(series, [1.0, 0.9, 0.8])
    assert time == 8.0


@pytest.mark.parametrize('shape', [(0,), (0, 3), (2, 0)])
def test_autocorrelation_plot_skips_two_empty_inputs(shape):
    assert _retr_representative_atcr(np.empty(shape), np.array([])) == (None, None)


def test_chain_convergence_accepts_stationary_independent_draws():
    random = np.random.default_rng(7)
    samples = random.normal(size=(4000, 3))

    converged, maximum_rhat, minimum_ess = _retr_chain_convergence(
        samples, maxmrhat=1.05, minmess=500
    )

    assert converged
    assert maximum_rhat < 1.05
    assert minimum_ess > 500


def test_chain_convergence_rejects_drifting_chain():
    random = np.random.default_rng(8)
    samples = np.cumsum(random.normal(size=(1000, 2)), axis=0)

    converged, maximum_rhat, minimum_ess = _retr_chain_convergence(
        samples, maxmrhat=1.05, minmess=100
    )

    assert not converged
    assert maximum_rhat > 1.05 or minimum_ess < 100


def test_multichain_convergence_accepts_independent_chains_with_the_same_target():
    random = np.random.default_rng(9)
    chains = [random.normal(size=(2000, 3)) for _ in range(4)]

    converged, maximum_rhat, minimum_ess = _retr_multichain_convergence(
        chains, maxmrhat=1.05, minmess=500
    )

    assert converged
    assert maximum_rhat < 1.05
    assert minimum_ess > 500


def test_multichain_convergence_rejects_chains_stuck_at_different_means():
    random = np.random.default_rng(10)
    chains = [random.normal(loc=offset, size=(2000, 2)) for offset in (0.0, 5.0, 10.0, 15.0)]

    converged, maximum_rhat, minimum_ess = _retr_multichain_convergence(
        chains, maxmrhat=1.05, minmess=100
    )

    assert not converged
    assert maximum_rhat > 1.05


def test_multichain_convergence_requires_at_least_two_chains():
    random = np.random.default_rng(11)
    converged, maximum_rhat, minimum_ess = _retr_multichain_convergence(
        [random.normal(size=(2000, 2))], maxmrhat=1.05, minmess=100
    )

    assert not converged
    assert np.isinf(maximum_rhat)


def test_proposal_scale_adaptation_tracks_acceptance():
    scale = 0.01

    assert _retr_adapted_proposal_scale(scale, True, 10) > scale
    assert _retr_adapted_proposal_scale(scale, False, 10) < scale


def test_posterior_summary_channels_include_saved_histograms():
    final = SimpleNamespace(listposthistelinpop0=np.ones((4, 3)))
    worker = SimpleNamespace(
        liststrgchan=["lliktotl"],
        liststrgvarbarrysamp=["lliktotl", "histelinpop0", "missinghist"],
    )

    channels = _retr_posterior_summary_channels(final, worker, "post")

    assert channels == ["lliktotl", "histelinpop0"]


def test_persistent_element_indices_require_slot_in_every_sample():
    model = SimpleNamespace(
        indxpopl=[0],
        namepara=SimpleNamespace(
            genr=[
                "base",
                "fluxpop00000",
                "elinpop00000",
                "fluxpop00001",
                "elinpop00001",
            ],
            genrelem=[["flux", "elin"]],
        ),
    )
    sample_slots = [[[0, 1]], [[0]], [[0, 2]]]

    indices = _retr_persistent_element_parameter_indices(model, sample_slots)

    np.testing.assert_array_equal(indices, [1, 2])


def _make_truth_state(typedata="simu"):
    return SimpleNamespace(
        typedata=typedata,
        true=SimpleNamespace(
            namepara=SimpleNamespace(genr=["base", "fluxpop00000", "fluxpop00001"]),
            indxpara=SimpleNamespace(genrbase=np.array([0])),
            this=SimpleNamespace(
                paragenrscalfull=np.array([2.5, 7.0, 0.0]),
                indxparagenrelemfull=[{"full": np.array([1])}],
            ),
        ),
    )


def test_true_parameter_value_matches_active_simulated_parameters_by_name():
    state = _make_truth_state()

    assert _retr_true_parameter_value(state, "base") == pytest.approx(2.5)
    assert _retr_true_parameter_value(state, "fluxpop00000") == pytest.approx(7.0)


def test_true_parameter_value_rejects_inactive_missing_and_real_parameters():
    state = _make_truth_state()

    assert _retr_true_parameter_value(state, "fluxpop00001") is None
    assert _retr_true_parameter_value(state, "fitting_only") is None
    assert _retr_true_parameter_value(_make_truth_state(typedata="inpt"), "base") is None


def test_proposal_defaults_enable_transdimensional_moves():
    state = SimpleNamespace(probtran=None, probspmr=0.0)
    model = SimpleNamespace(numbpopl=1)

    _configure_proposal_types(state, model)

    assert state.probtran == pytest.approx(0.4)
    assert state.probstud == 0.0
    assert state.nameproptype.tolist() == [
        "with",
        "brth",
        "deth",
        "splt",
        "merg",
        "jump",
    ]
    assert state.lablproptype[1] == "Birth proposal"


def test_differential_evolution_and_student_proposal_mixture_is_validated():
    state = SimpleNamespace(probtran=0.0, probspmr=0.0, probdemc=0.6, probstud=0.4)
    _configure_proposal_types(state, SimpleNamespace(numbpopl=0))

    assert state.probdemc + state.probstud == pytest.approx(1.0)
    with pytest.raises(ValueError, match="Proposal probabilities"):
        _configure_proposal_types(
            SimpleNamespace(probtran=0.0, probspmr=0.0, probdemc=0.7, probstud=0.4),
            SimpleNamespace(numbpopl=0),
        )


def test_proposal_labels_support_legacy_saved_identifiers():
    assert _retr_proposal_type_labels(["with", "brth", "deth", "splt", "merg"]) == [
        "Within-model proposal",
        "Birth proposal",
        "Death proposal",
        "Split proposal",
        "Merge proposal",
    ]


def test_proposal_animation_uses_recorded_types_and_acceptance(tmp_path):
    labels = [
        "Within-model proposal",
        "Birth proposal",
        "Death proposal",
        "Split proposal",
        "Merge proposal",
    ]
    proposals = np.arange(5)[:, None]
    accepted = np.array([True, True, False, True, False])[:, None]

    assert _retr_proposal_sweep_label(3, proposals, accepted, labels) == (
        "Sweep 4 | Split proposal | accepted"
    )
    output_path = tmp_path / "visuals" / "proposal_activity.gif"
    assert _write_proposal_activity_animation(proposals, accepted, labels, str(output_path))
    with Image.open(output_path) as animation:
        assert animation.n_frames == 5
        assert animation.size[0] > 800


def test_candidate_residual_scale_contains_missed_observed_signal_and_stays_fixed():
    from pcat.main import _proposal_residual_limit

    state = SimpleNamespace()
    data = np.array([100.0, 20000.0, 150.0])
    limit = _proposal_residual_limit(state, data)
    assert limit >= float(data.max())
    assert _proposal_residual_limit(state, data) == limit


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_rejected_proposal_candidate_frame_shows_data_and_residual_panels(
    tmp_path, monkeypatch, typeplotback
):
    current = SimpleNamespace(
        boolpropfilt=True,
        cntpmodl=np.array([[[1.], [2.], [3.], [4.]]]),
        paragenrunitfull=np.array([0.2, 0.7]),
        indxproptype=np.array([1]),
        accpprob=np.array([0.25]),
        lpostotl=-10.0,
        ltrp=np.array([0.3]),
        ljcb=np.array([0.0]),
    )
    candidate = SimpleNamespace(
        cntpmodl=np.array([[[1.5], [2.5], [3.5], [4.5]]]),
        paragenrunitfull=np.array([0.25, 0.9]),
        lpostotl=-11.0,
    )
    state = SimpleNamespace(
        cntpdata=np.array([[[1.], [2.], [3.], [4.]]]),
        numbsidecart=2,
        lablproptype=["Within-model proposal", "Birth proposal"],
        typeplotback=typeplotback,
    )
    worker = SimpleNamespace(this=current, next=candidate, cntrswep=3)
    output_path = tmp_path / "proposal_candidates_swep000000003.png"
    figures = _capture_figures(monkeypatch)

    assert _write_proposal_candidate_frame(state, worker, False, output_path) == str(output_path)
    assert len(figures) == 1
    assert [axis.get_title() for axis in figures[0].axes] == [
        "Observed data and model states",
        "Residuals (data − model)",
    ]
    assert figures[0].axes[0].images
    assert figures[0].axes[1].images
    assert figures[0]._suptitle.get_color() == '#A51C30'
    assert figures[0]._suptitle.get_ha() == 'left'
    assert figures[0]._suptitle.get_fontfamily() == ['monospace']
    assert figures[0]._suptitle.get_position()[0] == 0.04
    assert figures[0].texts[-1].get_position() == (0.04, 0.025)
    assert all(axis.xaxis.label.get_color() == ('white' if typeplotback == 'dark' else 'black')
               for axis in figures[0].axes)
    assert any(
        getattr(text, "arrow_patch", None) is not None
        for text in figures[0].axes[0].texts
    )
    assert {text.get_text() for text in figures[0].axes[0].get_legend().get_texts()} == {
        "Current state", "Proposed state", "State move"
    }
    with Image.open(output_path) as image:
        assert image.width > image.height
        assert image.width > 1000
        assert image.convert('RGB').getpixel((0, 0)) == ((0, 0, 0) if typeplotback == 'dark' else (255, 255, 255))


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_proposal_candidate_frame_shows_one_dimensional_data_and_residuals(tmp_path, typeplotback):
    current = SimpleNamespace(
        boolpropfilt=True,
        cntpmodl=np.array([1., 3., 2., 5., 3.]),
        indxproptype=np.array([0]),
        accpprob=np.array([0.4]),
        lpostotl=-3.0,
        ltrp=np.array([0.1]),
        ljcb=np.array([0.0]),
    )
    candidate = SimpleNamespace(
        cntpmodl=np.array([1., 2., 4., 4., 3.]),
        lpostotl=-3.2,
    )
    state = SimpleNamespace(
        cntpdata=np.array([[[1.], [4.], [2.], [5.], [3.]]]),
        lablproptype=["Within-model proposal"],
        typeplotback=typeplotback,
    )
    worker = SimpleNamespace(this=current, next=candidate, cntrswep=0)
    output_path = tmp_path / "proposal_candidates_swep000000000.png"

    _write_proposal_candidate_frame(state, worker, True, output_path)

    with Image.open(output_path) as image:
        assert image.width > image.height
        assert image.convert('RGB').getpixel((0, 0)) == ((0, 0, 0) if typeplotback == 'dark' else (255, 255, 255))
    assert output_path.is_file()


def test_candidate_residual_scales_are_fixed_for_1d_and_2d_frames(tmp_path, monkeypatch):
    figures = _capture_figures(monkeypatch)
    data_series = np.array([10., 16., 24., 35.])
    variance_series = np.array([1., 4., 9., 16.])
    state_series = SimpleNamespace(
        cntpdata=data_series,
        varidata=variance_series,
        lablproptype=["Within-model proposal"],
        numbswep=2,
    )
    for sweep, candidate_series in enumerate((data_series + 1., data_series + 20.)):
        current = SimpleNamespace(
            boolpropfilt=True, cntpmodl=data_series, indxproptype=np.array([0]),
            accpprob=np.array([0.5]), lpostotl=-1., ltrp=np.array([0.]), ljcb=np.array([0.]),
        )
        candidate = SimpleNamespace(cntpmodl=candidate_series, lpostotl=-1.1)
        pcat_main._write_proposal_candidate_frame(
            state_series, SimpleNamespace(this=current, next=candidate, cntrswep=sweep),
            True, tmp_path / f"series_{sweep}.png",
        )
    expected_series_limit = max(1.2 * data_series.max(),
                                5. * np.percentile(np.sqrt(variance_series), 99.5))
    np.testing.assert_allclose(figures[0].axes[1].get_ylim(),
                               (-expected_series_limit, expected_series_limit))
    assert figures[0].axes[1].get_ylim() == figures[1].axes[1].get_ylim()

    figures.clear()
    data_image = np.full((4, 4), 100.)
    variance_image = np.full((4, 4), 25.)
    state_image = SimpleNamespace(
        cntpdata=data_image,
        varidata=variance_image,
        lablproptype=["Within-model proposal"],
        numbswep=2,
    )
    for sweep, candidate_image in enumerate((data_image + 1., data_image + 50.)):
        current = SimpleNamespace(
            boolpropfilt=True, cntpmodl=data_image, indxproptype=np.array([0]),
            accpprob=np.array([0.5]), lpostotl=-1., ltrp=np.array([0.]), ljcb=np.array([0.]),
        )
        candidate = SimpleNamespace(cntpmodl=candidate_image, lpostotl=-1.1)
        pcat_main._write_proposal_candidate_frame(
            state_image, SimpleNamespace(this=current, next=candidate, cntrswep=sweep),
            True, tmp_path / f"image_{sweep}.png",
        )
    expected_image_limit = max(1.2 * data_image.max(),
                               5. * np.percentile(np.sqrt(variance_image), 99.5))
    np.testing.assert_allclose(figures[0].axes[1].images[0].get_clim(),
                               (-expected_image_limit, expected_image_limit))
    assert figures[0].axes[1].images[0].get_clim() == figures[1].axes[1].images[0].get_clim()


def test_proposal_frame_arrays_preserve_spectral_bins():
    from pcat.main import _proposal_data_array, _proposal_model_array

    state = SimpleNamespace(
        numbener=30,
        numbpixl=1,
        cntpdata=np.arange(30, dtype=float)[:, None, None],
    )
    model_state = SimpleNamespace(cntpmodl=(100. + np.arange(30))[:, None, None])

    np.testing.assert_array_equal(_proposal_data_array(state), np.arange(30))
    np.testing.assert_array_equal(
        _proposal_model_array(state, model_state), 100. + np.arange(30)
    )


def test_spectral_chain_frames_keep_y_limits_fixed_to_observed_data(tmp_path, monkeypatch):
    data = np.array([100., 130., 115., 150., 125.])[:, None, None]
    models = [
        np.array([90., 120., 110., 140., 120.]),
        np.array([80., 140., 100., 170., 115.]),
    ]
    state = SimpleNamespace(
        cntpdata=data,
        numbener=data.shape[0],
        numbpixl=1,
        boolbinsener=False,
        typeexpr="fire",
        typedata="inpt",
        plotsize=4.0,
        bctrpara=SimpleNamespace(ener=np.linspace(1.0, 5.0, data.shape[0])),
        lablener="Wavenumber",
        strgenerunit="m^-1",
        fitt=SimpleNamespace(colr="C0"),
    )
    figures = []
    make_subplots = pcat_main.plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = make_subplots(*args, **kwargs)
        figures.append(figure)
        return figure, axes

    monkeypatch.setattr(
        pcat_main,
        "retr_fromgdat",
        lambda *args, **kwargs: models.pop(0)[:, None, None],
    )
    monkeypatch.setattr(pcat_main, "retr_plotpath", lambda *args, **kwargs: str(tmp_path / "frame.png"))
    monkeypatch.setattr(pcat_main, "savefigr", lambda *args, **kwargs: None)
    monkeypatch.setattr(pcat_main.plt, "subplots", capture_subplots)

    for _ in range(len(models)):
        pcat_main.plot_scatcntp(state, None, "this", "fitt", "post", 0)

    expected_range = np.ptp(data[:, 0, 0])
    expected_limits = (
        min(0.0, np.min(data)) - 0.03 * expected_range,
        np.max(data) + 0.12 * expected_range,
    )
    assert figures[0].axes[0].get_ylim() == figures[1].axes[0].get_ylim()
    np.testing.assert_allclose(figures[0].axes[0].get_ylim(), expected_limits)


def test_image_chain_frames_keep_color_limits_fixed_for_one_run():
    figure, axis = pcat_main.plt.subplots()
    state = SimpleNamespace(
        frambndrmodl=1.0,
        frambndrdata=1.0,
        numbpixlfull=4,
        numbpixl=4,
        numbsidecart=2,
        exttrofi=(0.0, 2.0, 0.0, 2.0),
        alphmaps=1.0,
        alphbndr=0.0,
        typeexpr="gmix",
        typepixl="cart",
        scalpara=SimpleNamespace(),
        cmappara=SimpleNamespace(),
        minmpara=SimpleNamespace(),
        maxmpara=SimpleNamespace(),
        fitt=SimpleNamespace(),
    )
    first_map = np.array([[0.0, 1.0], [2.0, 3.0]])
    later_map = np.array([[0.0, 100.0], [200.0, 300.0]])

    image = pcat_main.retr_imag(
        state, axis, first_map, "this", "fitt", "cntpmodl"
    )
    fixed_limits = image.get_clim()
    pcat_main.retr_imag(
        state, axis, later_map, "this", "fitt", "cntpmodl", imag=image
    )

    assert image.get_clim() == fixed_limits
    pcat_main.plt.close(figure)


def test_candidate_animation_text_width_is_stable_across_sweeps(tmp_path, monkeypatch):
    labels = ["Within-model proposal", "A much longer transdimensional proposal"]
    figures = _capture_figures(monkeypatch)
    for sweep, proposal, subtype, accepted, posterior in (
        (0, 0, "Gaussian random walk", True, -10.0),
        (30, 0, "Differential evolution", False, -40.0),
        (60, 0, "Student-t(5)", True, -70.0),
        (90, 0, "Correlated Gaussian block", False, -100.0),
        (119, 1, None, False, -125.0),
    ):
        current = SimpleNamespace(
            cntpmodl=np.array([1.0, 2.0, 3.0]),
            indxproptype=np.array([proposal]),
            accpprob=np.array([0.35]),
            lpostotl=posterior,
            ltrp=np.array([0.1]),
            ljcb=np.array([0.0]),
            strgproposalsubtype=subtype,
        )
        candidate = SimpleNamespace(cntpmodl=np.array([1.1, 2.1, 2.9]), lpostotl=posterior - 1)
        state = SimpleNamespace(cntpdata=np.array([[[1.0], [2.0], [3.0]]]), lablproptype=labels,
                                numbswep=120)
        worker = SimpleNamespace(this=current, next=candidate, cntrswep=sweep)
        _write_proposal_candidate_frame(
            state, worker, accepted, tmp_path / f"candidate_{sweep}.png"
        )

    titles = [figure._suptitle.get_text() for figure in figures]
    footers = [figure.texts[-1].get_text() for figure in figures]
    for title, subtype in zip(
        titles,
        (
            "Gaussian random walk", "Differential evolution", "Student-t(5)",
            "Correlated Gaussian block", "A much longer transdimensional proposal",
        ),
    ):
        assert subtype in title
    assert len({len(title) for title in titles}) == 1
    assert len({len(footer) for footer in footers}) == 1


def test_element_parameter_labels_are_descriptive():
    model = SimpleNamespace(
        namepara=SimpleNamespace(genr=["fluxpop00001", "sigmpop10002"]),
        labltotlpara=SimpleNamespace(),
    )

    assert _retr_parameter_label(model, 0) == "Line flux, population 1, element 2"
    assert _retr_parameter_label(model, 1) == "Gaussian line width, population 2, element 3"


def test_single_population_legend_omits_redundant_index():
    single_population = SimpleNamespace(numbpopl=1)
    named_population = SimpleNamespace(numbpopl=1, legdpopl=["Point sources"])
    multiple_populations = SimpleNamespace(numbpopl=2)

    assert _retr_population_legend_label(single_population, 0, "Sample") == "Sample"
    assert _retr_population_legend_label(named_population, 0, "Sample") == "Sample Point sources"
    assert (
        _retr_population_legend_label(multiple_populations, 0, "Condensed")
        == "Condensed Population 0"
    )


def test_gaussian_cluster_amplitude_index_uses_object_count():
    model = SimpleNamespace(
        indxpopl=np.array([0]),
        nameparagenrelemampl=["nobj"],
        namepara=SimpleNamespace(genrelem=[["xpos", "ypos", "nobj", "gwdt"]]),
        indxpara=SimpleNamespace(),
    )

    _set_element_amplitude_indices(model)

    np.testing.assert_array_equal(model.indxpara.genrelemampl, [2])
