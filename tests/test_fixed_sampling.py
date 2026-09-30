from pathlib import Path

import numpy as np

from pcat.diagnostics import estimate_evidence, posterior_convergence
from pcat.main import readfile, sample


def gaussian_log_likelihood(gdat, strgmodl, values):
    gdat.generic_callback_count = getattr(gdat, 'generic_callback_count', 0) + 1
    covariance = np.array([[1.0, 0.6], [0.6, 2.0]])
    return -0.5 * values @ np.linalg.inv(covariance) @ values


def test_importance_evidence_matches_normal_convolution():
    from scipy.stats import norm

    observation = 0.7
    error = 0.5
    posterior = np.linspace(-1.5, 2.5, 200)[:, None]
    result = estimate_evidence(
        posterior, lambda values: norm.logpdf(observation, values[0], error),
        ('gaus',), (-5.,), (5.,), (0.,), (1.,), sample_count=4000, seed=7,
    )
    expected = norm.logpdf(observation, 0., np.sqrt(1. + error**2))
    assert abs(result['log_evidence'] - expected) < 0.08
    assert result['relative_error'] < 0.1


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
        / 'data/outp/single_parameter_adaptation/gdatmodi0000post'
    )
    worker = readfile(str(worker_path))
    assert worker.numbpropstdp.sum() == 200
    assert np.all(worker.numbpropstdp > 50)


def test_selected_parameter_block_proposals_are_used(tmp_path):
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
        proposal_blocks=((0, 1),),
        propwithsing=True,
        probpropblock=1.0,
        pathbase=str(tmp_path),
        strgcnfg='selected_block',
        numbproc=2,
        numbswep=800,
        numbburn=300,
        numbsamp=500,
        booladaptstdp=True,
        typeseed=8,
        typeverb=-1,
    )

    assert result.proposal_blocks == ((0, 1),)
    assert np.max(result.gmrbparagenrscalbase) < 1.1


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