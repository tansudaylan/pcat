import importlib
import io
from types import SimpleNamespace

import numpy as np
import pytest

from pcat.main import plot_genemaps, plot_scatcntp, retr_fromgdat, show_paragenrscalfull


pcat_main = importlib.import_module('pcat.main')


def test_retr_fromgdat_rejects_missing_uncertainty_map():
    gdat = SimpleNamespace(
        cntpdata=np.ones((1, 2, 1)),
        numbener=1,
        fitt=SimpleNamespace(),
    )

    with pytest.raises(AttributeError, match='cntpmodl is unavailable'):
        retr_fromgdat(
            gdat,
            None,
            'pdfn',
            'fitt',
            'cntpmodl',
            'post',
            strgmome='errr',
            indxvarb=[slice(None), slice(None), 0],
        )


def test_retr_fromgdat_rejects_missing_posterior_model():
    gdat = SimpleNamespace(
        cntpdata=np.array([4., 9.]),
        numbener=1,
        fitt=SimpleNamespace(),
    )

    with pytest.raises(AttributeError, match='cntpmodl is unavailable'):
        retr_fromgdat(gdat, None, 'pdfn', 'fitt', 'cntpmodl', 'post')


def test_count_scatter_skips_missing_optional_posterior_model(capsys):
    gdat = SimpleNamespace(
        cntpdata=np.ones((1, 2, 1)),
        numbener=1,
        fitt=SimpleNamespace(),
    )

    plot_scatcntp(gdat, None, 'pdfn', 'fitt', 'post', 0)

    assert 'skipping count scatter plot' in capsys.readouterr().out


def test_generic_map_skips_missing_optional_posterior_product(capsys):
    gdat = SimpleNamespace(
        cntpdata=np.ones((1, 2, 1)),
        numbener=1,
        fitt=SimpleNamespace(),
    )

    plot_genemaps(gdat, None, 'pdfn', 'fitt', 'post', 'cntpresi')

    assert 'skipping cntpresi map' in capsys.readouterr().out


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


def test_proc_anim_uses_explicit_output_root(tmp_path, monkeypatch):
    captured = {}

    def fake_output_path(pathbase, run_name):
        captured['pathbase'] = pathbase
        captured['run_name'] = run_name
        return str(tmp_path) + '/'

    monkeypatch.setattr(pcat_main, 'retr_pathoutpcnfg', fake_output_path)
    monkeypatch.setattr(
        pcat_main,
        'readfile',
        lambda path: SimpleNamespace(liststrgpdfn=[], liststrgfoldanim=[]),
    )
    monkeypatch.setattr(pcat_main, 'open_narr', lambda path, mode: io.StringIO())

    pcat_main.proc_anim('demo', pathbase=tmp_path)

    assert captured == {'pathbase': tmp_path, 'run_name': 'demo'}
