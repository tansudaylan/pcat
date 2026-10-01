import astropy.io.fits
import h5py
import numpy as np

from pcat import sampling


def test_jump_proposal_redraws_element_without_changing_dimension(tmp_path):
    """The dimension-preserving 'jump' proposal type (indxproptype 5) should be usable,
    should never carry a nonzero Jacobian or transition-probability correction (since it
    does not change the number of elements), and should be accepted some of the time."""
    edges = np.linspace(1.8e5, 1.82e5, 41)
    width = np.diff(edges)
    centers = 0.5 * (edges[1:] + edges[:-1])
    counts = np.round(1000. + 800. * np.exp(-0.5 * ((centers - 1.81e5) / 60.)**2))
    exposure = 1. / width
    path_inpt = tmp_path / 'jump_proposal' / 'data' / 'inpt'
    path_inpt.mkdir(parents=True)
    astropy.io.fits.writeto(path_inpt / 'sbrt.fits', counts[:, None, None, None].astype(float))
    astropy.io.fits.writeto(path_inpt / 'expo.fits', exposure[:, None, None])

    sampling.sample(
        typeexpr='fire', typedata='inpt', strgexprsbrt='sbrt.fits', typeexpo='file', strgexpo='expo.fits',
        binsenerfull=edges, spectype=['voig'], spatdisttype=['line'], typeelem=['lghtlinevoig'],
        dictfitt={'typeelem': ['lghtlinevoig'], 'spectype': ['voig'],
                  'sbrtbacknorm': [np.full((edges.size - 1, 1, 1), 1000.)], 'listnamediff': ['back0000']},
        limtparaelem={'flux': (1e3, 1e6), 'sigm': (20., 200.), 'gamm': (1., 50.)},
        maxmgangdata=1e-4, anlytype='spec', fittminmnumbelempop0=0, fittmaxmnumbelempop0=3,
        inittype='rand', typeseed=0, numbproc=1, probjump=0.3,
        numbswep=3000, numbsamp=300, boolmakeplot=False, boolmakeplotinit=False,
        booldiag=False, typeverb=0, pathbase=str(tmp_path / 'jump_proposal'), strgcnfg='jump_proposal',
    )

    path_outp = tmp_path / 'jump_proposal' / 'data' / 'outp' / 'jump_proposal'
    with h5py.File(path_outp / 'gdatmodi0000post.h5', 'r') as file:
        indxproptype = file['listpostindxproptype'][()].ravel()
        ljcb = file['listpostljcb'][()].ravel()
        ltrp = file['listpostltrp'][()].ravel()
        boolpropaccp = file['listpostboolpropaccp'][()].ravel()
        boolpropfilt = file['listpostboolpropfilt'][()].ravel()
        lliktotl = file['listpostlliktotl'][()].ravel()

    indxjump = np.where(indxproptype == 5)[0]
    # the jump proposal should actually get selected at the requested rate
    assert indxjump.size > 0.15 * indxproptype.size
    # a dimension-preserving independence proposal carries no Jacobian or transition-probability term
    np.testing.assert_array_equal(ljcb[indxjump], 0.)
    np.testing.assert_array_equal(ltrp[indxjump], 0.)
    # some jumps should pass the prior-support filter and get accepted
    assert boolpropfilt[indxjump].any()
    assert boolpropaccp[indxjump].any()
    assert np.all(np.isfinite(lliktotl))


def test_jump_proposal_is_on_for_catalogs_and_off_for_fixed_models():
    from pcat.main import _configure_proposal_types
    from types import SimpleNamespace

    state = SimpleNamespace(probtran=None, probspmr=None)
    _configure_proposal_types(state, SimpleNamespace(numbpopl=1))
    assert state.probjump == 0.1
    assert 'jump' in state.nameproptype.tolist()

    state = SimpleNamespace(probtran=None, probspmr=None)
    _configure_proposal_types(state, SimpleNamespace(numbpopl=0))
    assert state.probjump == 0.
