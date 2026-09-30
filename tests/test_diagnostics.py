import numpy as np
import pytest
from types import SimpleNamespace

from PIL import Image

from pcat.diagnostics import (
    autocorrelation_time,
    binomial_wilson_interval,
    gelman_rubin,
)
from pcat.main import (
    _PCATMCMCCompat,
    _configure_proposal_types,
    _retr_adapted_proposal_scale,
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
    _set_element_amplitude_indices,
    _write_proposal_activity_animation,
    _write_proposal_candidate_frame,
)


def test_gmrb_rejects_separated_constant_chains():
    chains = np.column_stack((np.zeros(100), np.full(100, 100.0)))

    assert np.isinf(_PCATMCMCCompat.gmrb_test(chains))
    assert np.isinf(gelman_rubin(chains))


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


def test_rejected_proposal_candidate_frame_shows_both_states(tmp_path):
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

    assert _write_proposal_candidate_frame(state, worker, False, output_path) == str(output_path)
    with Image.open(output_path) as image:
        assert image.width > image.height
        assert image.width > 1000


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
