from pathlib import Path

import numpy as np

from pcat.diagnostics import posterior_convergence
from pcat.fixed import sample_allesfitter_pcat, sample_fixed_chains
from pcat.main import readfile, sample


def gaussian_log_likelihood(gdat, strgmodl, values):
    gdat.generic_callback_count = getattr(gdat, 'generic_callback_count', 0) + 1
    covariance = np.array([[1.0, 0.6], [0.6, 2.0]])
    return -0.5 * values @ np.linalg.inv(covariance) @ values


def scalar_log_likelihood(gdat, strgmodl, values):
    return -0.5 * values[0] ** 2


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