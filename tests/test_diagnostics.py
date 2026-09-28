import numpy as np
import pytest
from types import SimpleNamespace

from pcat.main import (
    _PCATMCMCCompat,
    _configure_proposal_types,
    _retr_adapted_proposal_scale,
    _retr_chain_convergence,
    _retr_persistent_element_parameter_indices,
    _retr_parameter_label,
    _retr_posterior_summary_channels,
    _retr_proposal_type_labels,
    _retr_representative_atcr,
    _retr_true_parameter_value,
    _set_element_amplitude_indices,
)


def test_gmrb_rejects_separated_constant_chains():
    chains = np.column_stack((np.zeros(100), np.full(100, 100.0)))

    assert np.isinf(_PCATMCMCCompat.gmrb_test(chains))


def test_autocorrelation_distinguishes_independent_draws_and_random_walk():
    random = np.random.default_rng(42)
    independent = random.normal(size=(2000, 1))
    random_walk = np.cumsum(random.normal(size=(2000, 1)), axis=0)

    _, time_independent = _PCATMCMCCompat.retr_timeatcr(independent)
    _, time_random_walk = _PCATMCMCCompat.retr_timeatcr(random_walk)

    assert time_independent[0] < 2.0
    assert time_random_walk[0] > 20.0


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


def test_element_parameter_labels_are_descriptive():
    model = SimpleNamespace(
        namepara=SimpleNamespace(genr=["fluxpop00001", "sigmpop10002"]),
        labltotlpara=SimpleNamespace(),
    )

    assert _retr_parameter_label(model, 0) == "Line flux, population 1, element 2"
    assert _retr_parameter_label(model, 1) == "Gaussian line width, population 2, element 3"


def test_gaussian_cluster_amplitude_index_uses_object_count():
    model = SimpleNamespace(
        indxpopl=np.array([0]),
        nameparagenrelemampl=["nobj"],
        namepara=SimpleNamespace(genrelem=[["xpos", "ypos", "nobj", "gwdt"]]),
        indxpara=SimpleNamespace(),
    )

    _set_element_amplitude_indices(model)

    np.testing.assert_array_equal(model.indxpara.genrelemampl, [2])
