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


def test_posterior_convergence_plots_fixed_and_transdimensional_chains(tmp_path):
    rng = np.random.default_rng(31)
    state = SimpleNamespace(
        listpostparagenrscalbase=rng.normal(size=(240, 2)),
        listpostnumbelem=rng.integers(0, 4, size=(240, 1)),
        numbproc=2,
        numbsamp=120,
    )

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


def test_autocorrelation_plot_selects_finite_nonconstant_series():
    autocorrelation = np.array([[[np.nan, np.nan, np.nan], [1.0, 0.5, 0.1]]])
    correlation_time = np.array([[np.nan, 2.2]])

    series, time = _retr_representative_atcr(autocorrelation, correlation_time)

    assert series == pytest.approx([1.0, 0.5, 0.1])
    assert time == pytest.approx(2.2)


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
    assert state.nameproptype.tolist() == [
        "with",
        "brth",
        "deth",
        "splt",
        "merg",
        "jump",
    ]
    assert state.lablproptype[1] == "Birth proposal"


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


def test_rejected_proposal_candidate_frame_shows_data_and_residual_panels(
    tmp_path, monkeypatch
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
    )
    worker = SimpleNamespace(this=current, next=candidate, cntrswep=3)
    output_path = tmp_path / "proposal_candidates_swep000000003.png"
    figures = []
    make_figure = pcat_main.plt.figure

    def capture_figure(*args, **kwargs):
        figure = make_figure(*args, **kwargs)
        figures.append(figure)
        return figure

    monkeypatch.setattr(pcat_main.plt, "figure", capture_figure)

    assert _write_proposal_candidate_frame(state, worker, False, output_path) == str(output_path)
    assert len(figures) == 1
    assert [axis.get_title() for axis in figures[0].axes] == [
        "Observed data and model states",
        "Residuals (data − model)",
    ]
    assert figures[0].axes[0].images
    assert figures[0].axes[1].images
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


def test_proposal_candidate_frame_shows_one_dimensional_data_and_residuals(tmp_path):
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
    )
    worker = SimpleNamespace(this=current, next=candidate, cntrswep=0)
    output_path = tmp_path / "proposal_candidates_swep000000000.png"

    _write_proposal_candidate_frame(state, worker, True, output_path)

    with Image.open(output_path) as image:
        assert image.width > image.height
    assert output_path.is_file()


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
