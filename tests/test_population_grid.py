import numpy as np
import pytest

from pcat import plot_population_grid


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
@pytest.mark.parametrize('inherit', [False, True])
def test_population_grid_writes_corner_histogram_and_pair_plots(tmp_path, monkeypatch, typeplotback, inherit):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from PIL import Image
    from tdpy.plotting import plot_background_colors, plot_style

    rng = np.random.default_rng(1)
    listpara = [rng.standard_normal((50, 3)), rng.standard_normal((200, 3)) + 3.0]
    background, foreground = plot_background_colors(typeplotback)
    save = plt.Figure.savefig

    def capture_and_save(figure, path, **kwargs):
        assert figure.get_facecolor() == to_rgba(background)
        for axis in figure.axes:
            assert axis.get_facecolor() == to_rgba(background)
            assert axis.xaxis.label.get_color() == foreground
            assert axis.yaxis.label.get_color() == foreground
        return save(figure, path, **kwargs)

    monkeypatch.setattr(plt.Figure, 'savefig', capture_and_save)
    options = {} if inherit else {'typeplotback': typeplotback}
    outer_theme = typeplotback if inherit else ('dark' if typeplotback == 'white' else 'white')
    with plot_style(outer_theme):
        original = dict(plt.rcParams)
        plot_population_grid(
            [['a', ''], ['b', ''], ['c', '']], listpara=listpara, listlablpopl=['one', 'two'],
            pathbase=f'{tmp_path}/', strgextn='test', boolplottria=True, boolplothistodim=True,
            boolplotpair=True, typeverb=0, **options,
        )
        assert dict(plt.rcParams) == original

    names = {path.name for path in tmp_path.glob('*.png')}
    extension = 'test_dark' if typeplotback == 'dark' else 'test'
    assert f'pmar_{extension}_scatscat.png' in names
    assert {f'hist_p00{k}_{extension}.png' for k in range(3)} <= names
    assert any(name.startswith(f'pmar_p001_p000_{extension}') for name in names)
    expected = tuple(int(channel * 255) for channel in to_rgba(background)[:3])
    for name in names:
        with Image.open(tmp_path / name) as image:
            pixels = np.asarray(image.convert('RGB'))
            assert tuple(pixels[0, 0]) == expected
            assert np.any(pixels != expected)


def test_population_grid_lof_annotations_label_outliers(tmp_path, monkeypatch):
    import matplotlib.axes
    import sklearn.neighbors

    generator = np.random.default_rng(4)
    samples = generator.normal(size=(60, 2))
    samples[-1] = [12.0, 12.0]
    labels = np.array([f'sample_{index}' for index in range(samples.shape[0])])
    annotations = []
    estimators = []
    original_annotate = matplotlib.axes.Axes.annotate

    class DeterministicLocalOutlierFactor:
        def __init__(self, n_neighbors):
            self.n_neighbors = n_neighbors
            estimators.append(self)

        def fit(self, values):
            self.negative_outlier_factor_ = np.arange(values.shape[0], dtype=float)
            self.negative_outlier_factor_[-1] = -100.0
            return self

    def record_annotation(axis, text, *args, **kwargs):
        annotations.append(text)
        return original_annotate(axis, text, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, 'annotate', record_annotation)
    monkeypatch.setattr(sklearn.neighbors, 'LocalOutlierFactor', DeterministicLocalOutlierFactor)
    plot_population_grid(
        [['x', ''], ['y', '']],
        listpara=samples,
        listlablsamp=[labels],
        typeannosamp='LOF',
        pathbase=f'{tmp_path}/',
        strgextn='lof',
        boolplottria=False,
        boolplothistodim=True,
        boolplotpair=True,
        typeplottdim='scat',
        typeverb=0,
    )

    assert len(annotations) == 10
    assert 'sample_59' in annotations
    assert estimators[0].n_neighbors == samples.shape[0] - 1
