import numpy as np
import pytest
from types import SimpleNamespace

from pcat.main import (
    _PCATMCMCCompat,
    _retr_adapted_proposal_scale,
    _retr_chain_convergence,
    _retr_persistent_element_parameter_indices,
    _retr_posterior_summary_channels,
    _retr_representative_atcr,
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
