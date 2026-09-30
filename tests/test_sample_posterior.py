import pickle

import numpy as np

import pcat.fixed as fixed
from pcat.fixed import retr_lpos, sample_posterior
from pcat.plotting import plot_autocorrelation, plot_gelman_rubin


def test_retr_lpos_penalizes_both_gaussian_tails():
    def retr_llik(para, gdat):
        return 0.

    args = [None, np.arange(1), ['gaus'], np.array([-10.]), np.array([10.]),
            np.array([0.]), np.array([1.]), retr_llik, None]
    values = [retr_lpos(np.array([value]), *args) for value in (-1., 0., 1.)]

    assert np.allclose(values, [-0.5, 0., -0.5])



def test_pcat_evidence_preserves_constant_likelihood(tmp_path):
    def retr_llik(para, gdat):
        return np.log(3.)

    chain, logprob, evidence = fixed.sample_fixed_chains(
        None, retr_llik, None, ['x'], ['self'], np.array([0.]), np.array([1.]),
        None, None, np.array([[0.5]]), 1, 8, 2, str(tmp_path), -1,
        estimate_log_evidence=True, evidence_samples=300, seed=7,
    )
    assert chain.shape == (1, 8, 1)
    assert logprob.shape == (1, 8)
    assert abs(evidence['log_evidence'] - np.log(3.)) < 4 * evidence['relative_error']
    assert evidence['relative_error'] < 0.1



def test_sample_posterior_clips_retained_post_burn_in_samples_to_available_data(monkeypatch):
    def fake_pcat_chains(*args):
        initial, numbwalk, numbsampwalk = args[9:12]
        assert np.all(np.isfinite(initial))
        chain = np.arange(numbwalk)[:, None, None] + np.arange(numbsampwalk)[None, :, None]
        return chain.astype(float), np.zeros((numbwalk, numbsampwalk))

    monkeypatch.setattr(fixed, 'sample_fixed_chains', fake_pcat_chains)

    def retr_llik(para, gdat):
        return 0.

    result = sample_posterior(
        None, 5, retr_llik, ['x'], [['X', '']], ['self'], np.array([0.]), np.array([1.]),
        numbsampburnwalk=3, numbsamppostwalk=20, booltqdm=False, typeverb=0,
    )

    assert result['x'].size == 20 * (5 - 3)
    assert result['x'][0] == 3.
    assert result['x'][-1] == 23.



def test_sample_posterior_aligns_derived_results_and_summarizes_each_parameter(monkeypatch, tmp_path):
    def fake_pcat_chains(*args):
        initial, numbwalk, numbsampwalk = args[9:12]
        initial = np.asarray(initial)
        assert np.all(np.isfinite(initial))
        numbpara = initial.shape[1]
        if numbpara == 2:
            assert np.any(initial[:, 0] < 0.)
        chain = np.empty((numbwalk, numbsampwalk, numbpara))
        chain[:, :, 0] = np.arange(numbwalk)[:, None]
        if numbpara == 2:
            chain[:, :, 1] = 100. + np.arange(numbsampwalk)[None, :]
        return chain, np.zeros((numbwalk, numbsampwalk))

    monkeypatch.setattr(fixed, 'sample_fixed_chains', fake_pcat_chains)
    plot_calls = []

    def check_plot_call(*args, **kwargs):
        plot_calls.append(kwargs)

    monkeypatch.setattr(fixed.plt, 'savefig', lambda path: None)

    def retr_llik(para, gdat):
        return 0.

    def retr_dictderi(para, gdat):
        return {'sum': para.sum(), 'pair': para.copy()}

    output_path = tmp_path / 'output with spaces'
    pathbase = str(output_path) + '/'
    result = sample_posterior(
        None, 3, retr_llik, ['gaussian', 'uniform'], [['Gaussian', ''], ['Uniform', '']],
        ['gaus', 'self'], np.array([-10., 0.]), np.array([10., 200.]), pathbase=pathbase,
        numbsamppostwalk=3, meangauspara=np.array([0., 0.]), stdvgauspara=np.array([1., 1.]),
        retr_dictderi=retr_dictderi, dictlablscalparaderi={'sum': ['Sum', '']}, boolplot=True,
        booltqdm=False, typeverb=0, plot_posterior=check_plot_call,
    )

    assert result['sum'].shape == (60,)
    assert result['pair'].shape == (60, 2)
    assert np.allclose(result['sum'], result['pair'].sum(axis=1))

    assert any(call.get('listnamepara') == ['sum'] for call in plot_calls)

    pathdata = output_path / 'mcmc' / 'data'
    summary = np.loadtxt(pathdata / 'postpara.csv', delimiter=',')
    samples = np.column_stack((result['gaussian'], result['uniform'], result['sum']))
    assert np.allclose(summary[:, 0], np.median(samples, axis=0))
    assert np.allclose(summary[:, 1], np.percentile(samples, 84., axis=0) - summary[:, 0])
    assert np.allclose(summary[:, 2], summary[:, 0] - np.percentile(samples, 16., axis=0))

    with (pathdata / 'postderi.pickle').open('rb') as fileobj:
        saved = pickle.load(fileobj)
    assert saved['sum'].shape == (60,)
    assert saved['pair'].shape == (60, 2)

    result_memory = sample_posterior(
        None, 3, retr_llik, ['uniform'], [['Uniform', '']], ['self'], np.array([0.]),
        np.array([1.]), numbsamppostwalk=3, booltqdm=False, typeverb=0,
    )
    assert result_memory['uniform'].shape == (60,)


def test_plot_autocorrelation_writes_one_figure(tmp_path):
    path = plot_autocorrelation(str(tmp_path) + '/', np.array([1., 0.5, 0.]), 2., strgextn='para',
                                typefileplot='png', typeplotback='dark')
    assert path == str(tmp_path / 'atcrpara.png')
    assert (tmp_path / 'atcrpara.png').is_file()


def test_plot_gelman_rubin_writes_one_figure(tmp_path):
    path = plot_gelman_rubin(str(tmp_path) + '/', np.array([1.01, 1.05, 1.1]), typefileplot='png')
    assert path == str(tmp_path / 'gmrb.png')
    assert (tmp_path / 'gmrb.png').is_file()
