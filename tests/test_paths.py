import os

import pytest

import pcat


def test_setup_pcat_uses_pcater_data_path(monkeypatch, tmp_path):
    monkeypatch.delenv('PCAT_DATA_PATH', raising=False)
    monkeypatch.delenv('TDGU_DATA_PATH', raising=False)
    monkeypatch.setenv('PCAT_DATA_PATH', str(tmp_path / 'pcat-root'))

    gdat = type('Gdat', (), {})()
    gdat.pathbase = None
    gdat.liststrgfeatparalist = []
    gdat.liststrgfeatpara = []
    gdat.listscaltype = []
    gdat.numbstdvgaus = 4.0

    pcat.setup_pcat(gdat)

    assert os.path.normpath(gdat.pathbase) == os.path.normpath(str(tmp_path / 'pcat-root'))
    assert os.path.normpath(gdat.pathdata).endswith('pcat-root/data')
    assert os.path.normpath(gdat.pathvisu).endswith('pcat-root/visuals')


def test_setup_pcat_requires_a_data_path(monkeypatch):
    monkeypatch.delenv('PCAT_DATA_PATH', raising=False)
    monkeypatch.delenv('TDGU_DATA_PATH', raising=False)

    gdat = type('Gdat', (), {})()
    gdat.pathbase = None
    gdat.liststrgfeatparalist = []
    gdat.liststrgfeatpara = []
    gdat.listscaltype = []
    gdat.numbstdvgaus = 4.0

    with pytest.raises(RuntimeError, match='PCAT_DATA_PATH|TDGU_DATA_PATH'):
        pcat.setup_pcat(gdat)
