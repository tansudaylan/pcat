import importlib
import io


def test_console_tee_respects_verbosity_without_suppressing_file_log(monkeypatch):
    main = importlib.import_module('pcat.main')
    tee = main.logg.__new__(main.logg)
    tee.terminal = io.StringIO()
    tee.log = io.StringIO()

    monkeypatch.setenv('TDPY_VERBOSITY', '0')
    tee.write('hidden')
    assert tee.terminal.getvalue() == ''
    assert tee.log.getvalue() == 'hidden'

    monkeypatch.setenv('TDPY_VERBOSITY', '1')
    tee.write('shown')
    assert tee.terminal.getvalue() == 'shown'
    assert tee.log.getvalue() == 'hiddenshown'


def test_pcat_package_imports():
    pcat = importlib.import_module('pcat')
    assert hasattr(pcat, '__file__')
    assert 'pcat' in pcat.__file__


def test_pcat_main_module_imports():
    main = importlib.import_module('pcat.main')
    assert hasattr(main, 'init')


def test_sampling_api_preserves_main_compatibility():
    main = importlib.import_module('pcat.main')
    sampling = importlib.import_module('pcat.sampling')

    assert sampling.init is main.init
    assert sampling.init_image is main.init_image
    assert sampling.sample is main.sample
    assert sampling.sample_parallel is main.sample_parallel
    from pcat.fixed import sample_allesfitter_pcat, sample_fixed, sample_fixed_chains

    assert sampling.sample_fixed is sample_fixed
    assert sampling.sample_fixed_chains is sample_fixed_chains
    assert sampling.sample_allesfitter_pcat is sample_allesfitter_pcat
