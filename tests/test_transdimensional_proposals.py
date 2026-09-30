import h5py
import numpy as np

from pcat import sampling


def test_voigt_split_merge_run_records_proposal_durations(tmp_path):
    sampling.sample(
        typeexpr='fire', spectype=['voig'], strgexpo=1.0e5, spatdisttype=['line'],
        typeelem=['lghtlinevoig'], maxmgangdata=100.0 / (3600.0 * 180.0 / np.pi),
        numbsidecart=1, anlytype='spec', typeseed=0, typeseedelem=17, inittype='refr',
        truenumbelempop0=2, fittminmnumbelempop0=1, fittmaxmnumbelempop0=3,
        dicttrue={'typeelem': ['lghtlinevoig'], 'spectype': ['voig']},
        dictfitt={'typeelem': ['lghtlinevoig'], 'spectype': ['voig']},
        stdvpropelemfire=[5.0e-4, 5.0e-4, 2.0e-3, 2.0e-3], probspmr=0.3,
        numbswep=600, numbsamp=60, boolmakeplot=False, boolmakeplotinit=False,
        booldiag=False, typeverb=0, pathbase=str(tmp_path), strgcnfg='voigt_spmr',
    )

    path = tmp_path / 'pcat_runs' / 'voigt_spmr' / 'data' / 'outp' / 'voigt_spmr' / 'gdatmodi0000post.h5'
    with h5py.File(path, 'r') as file:
        proposal_types = file['listpostindxproptype'][()].ravel()
        durations = file['listpostchrototl'][()].ravel()
    # all five proposal families are exercised: within, birth, death, split, merge
    assert set(np.unique(proposal_types)) == {0, 1, 2, 3, 4}
    # per-sweep totals are durations [s], not clock readings
    assert np.all(durations > 0.) and np.all(durations < 1.)
