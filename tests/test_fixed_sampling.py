from pathlib import Path

import numpy as np

from pcat.diagnostics import posterior_convergence
from pcat.fixed import sample_allesfitter_pcat, sample_fixed_chains
from pcat.main import readfile, sample


def gaussian_log_likelihood(gdat, strgmodl, values):
    gdat.generic_callback_count = getattr(gdat, 'generic_callback_count', 0) + 1
    covariance = np.array([[1.0, 0.6], [0.6, 2.0]])
    return -0.5 * values @ np.linalg.inv(covariance) @ values


def constant_legacy_likelihood(values, state):
    return np.log(3.)


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


def test_allesfitter_adapter_runs_native_pcat(tmp_path, monkeypatch):
    import sys
    import types
    import h5py

    config = types.ModuleType('allesfitter.config')
    config.init = lambda path: setattr(config, 'BASEMENT', types.SimpleNamespace(
        datadir=path, bounds=[('uniform', 0., 1.)], theta_0=np.array([0.5]),
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
        assert saved['mcmc/chain'].shape == (8, 1, 1)
        assert np.all(np.isfinite(saved['mcmc/log_prob'][:]))


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
        / 'pcat_runs/single_parameter_adaptation/data/outp/'
        'single_parameter_adaptation/gdatmodi0000post'
    )
    worker = readfile(str(worker_path))
    assert worker.numbpropstdp.sum() == 200
    assert np.all(worker.numbpropstdp > 50)


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