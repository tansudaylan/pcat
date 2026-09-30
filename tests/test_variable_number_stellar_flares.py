import astropy.io.fits
import h5py
import numpy as np

from pcat import sampling


def test_variable_number_stellar_flares_uses_input_counts_bins_and_baseline_template(tmp_path):
    # a flat quiescent baseline of 1000 counts per bin with one injected burst, on time bins [day]
    edges = 1.0 + np.linspace(0.0, 1.0, 41)
    width = np.diff(edges)
    centers = 0.5 * (edges[1:] + edges[:-1])
    counts = np.round(1000. + 800. * np.exp(-0.5 * ((centers - 1.5) / 0.02)**2))
    exposure = 1. / width
    path_inpt = tmp_path / 'variable_number_stellar_flares' / 'data' / 'inpt'
    path_inpt.mkdir(parents=True)
    astropy.io.fits.writeto(path_inpt / 'sbrt.fits', counts[:, None, None, None].astype(float))
    astropy.io.fits.writeto(path_inpt / 'expo.fits', exposure[:, None, None])

    sampling.sample(
        typeexpr='fire', typedata='inpt', strgexprsbrt='sbrt.fits', typeexpo='file', strgexpo='expo.fits',
        binsenerfull=edges, spectype=['voig'], spatdisttype=['line'], typeelem=['lghtlinevoig'],
        dictfitt={'typeelem': ['lghtlinevoig'], 'spectype': ['voig'],
                  'sbrtbacknorm': [np.full((edges.size - 1, 1, 1), 1000.)], 'listnamediff': ['back0000']},
        limtparaelem={'flux': (1e1, 1e4), 'sigm': (5e-3, 5e-2), 'gamm': (1e-3, 2e-2)},
        maxmgangdata=1e-4, anlytype='spec', fittminmnumbelempop0=0, fittmaxmnumbelempop0=3,
        inittype='rand', numbswep=400, numbsamp=40, boolmakeplot=False, boolmakeplotinit=False,
        booldiag=False, typeverb=0, pathbase=str(tmp_path / 'variable_number_stellar_flares'),
        strgcnfg='variable_number_stellar_flares',
    )

    path_outp = tmp_path / 'variable_number_stellar_flares' / 'data' / 'outp' / 'variable_number_stellar_flares'
    with h5py.File(path_outp / 'gdatinit.h5', 'r') as file:
        np.testing.assert_allclose(file['cntpdata'][()].ravel(), counts)
    with h5py.File(path_outp / 'gdatfinlpost.h5', 'r') as file:
        background = file['listpostcntpback0000'][()]
        llik = file['listpostlliktotl'][()]
    # the baseline template enters the model with its input normalization
    np.testing.assert_allclose(background[-1].ravel(), 1000., rtol=1e-6)
    assert np.all(np.isfinite(llik))
