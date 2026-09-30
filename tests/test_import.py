import importlib


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
