import importlib


def test_pcat_package_imports():
    pcat = importlib.import_module('pcat')
    assert hasattr(pcat, '__file__')
    assert 'pcat' in pcat.__file__


def test_pcat_main_module_imports():
    main = importlib.import_module('pcat.main')
    assert hasattr(main, 'init')
