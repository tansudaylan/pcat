import numpy as np

from pcat import plot_population_grid


def test_population_grid_writes_corner_histogram_and_pair_plots(tmp_path):
    rng = np.random.default_rng(1)
    listpara = [rng.standard_normal((50, 3)), rng.standard_normal((200, 3)) + 3.0]
    plot_population_grid(
        [['a', ''], ['b', ''], ['c', '']], listpara=listpara, listlablpopl=['one', 'two'],
        pathbase=f'{tmp_path}/', strgextn='test', boolplottria=True, boolplothistodim=True,
        boolplotpair=True, typeverb=0,
    )

    names = {path.name for path in tmp_path.glob('*.png')}
    assert 'pmar_test_scatscat.png' in names
    assert {f'hist_p00{k}_test.png' for k in range(3)} <= names
    assert any(name.startswith('pmar_p001_p000_test') for name in names)
