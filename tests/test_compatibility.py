from types import SimpleNamespace

import numpy as np

from pcat.main import retr_fromgdat, show_paragenrscalfull


def test_retr_fromgdat_indexes_fallback_error_map_without_uncertainty_axis():
    gdat = SimpleNamespace(
        cntpdata=np.ones((1, 2, 1)),
        numbener=1,
        fitt=SimpleNamespace(),
    )

    result = retr_fromgdat(
        gdat,
        None,
        'pdfn',
        'fitt',
        'cntpmodl',
        'post',
        strgmome='errr',
        indxvarb=[slice(None), slice(None), 0],
    )

    assert result.shape == (1, 2)
    assert np.array_equal(result, np.zeros((1, 2)))


def test_show_paragenrscalfull_accepts_empty_element_indices(capsys):
    model_state = SimpleNamespace(
        paragenrunitfull=np.array([0.5]),
        paragenrscalfull=np.array([0.5]),
    )
    model = SimpleNamespace(
        indxpara=SimpleNamespace(genrfull=np.array([0])),
        indxpopl=np.array([0]),
        indxparagenrelemsing=[np.array([], dtype=int)],
        numbpopl=1,
        namepara=SimpleNamespace(genr=['value']),
        scalpara=SimpleNamespace(genr=['self']),
        this=model_state,
    )
    gdat = SimpleNamespace(fitt=model, booldiag=False)

    show_paragenrscalfull(gdat, None)

    assert 'value' in capsys.readouterr().out