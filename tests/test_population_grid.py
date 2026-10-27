import numpy as np
import pytest

from pcat import plot_population_grid


@pytest.mark.parametrize('plot_kind', ['corner', 'histogram', 'pair'])
@pytest.mark.parametrize('container', [list, np.asarray])
def test_population_grid_preserves_reference_draw_coordinates(tmp_path, monkeypatch, plot_kind, container):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    samples = np.random.default_rng(6).normal(size=(40, 2))
    original_samples = samples.copy()
    reference_values = np.array([[-0.7, 0.2], [0.8, -0.6]])
    draws = container(reference_values)
    figures = []
    save = plt.Figure.savefig

    def capture_and_save(figure, path, **kwargs):
        FigureCanvasAgg(figure)
        figures.append(figure)
        return save(figure, path, **kwargs)

    monkeypatch.setattr(plt.Figure, 'savefig', capture_and_save)
    plot_population_grid(
        [['x', ''], ['y', '']], listpara=samples,
        listparadraw=draws, listnamepara=['x', 'y'],
        pathbase=f'{tmp_path}/', strgextn='reference',
        boolplottria=plot_kind == 'corner',
        boolplothistodim=plot_kind == 'histogram',
        boolplotpair=plot_kind == 'pair', boolplotpies=False,
        typeplottdim='scat', typeplotback='white', typeverb=0,
    )

    assert figures
    assert all(path.stat().st_size > 0 for path in tmp_path.glob('*.png'))
    histogram_axes = [axis for figure in figures for axis in figure.axes if axis.patches]
    pair_axes = [axis for figure in figures for axis in figure.axes if axis.collections]
    if plot_kind != 'pair':
        assert len(histogram_axes) == 2
        actual_lines = []
        for axis in histogram_axes:
            assert len(axis.lines) == len(reference_values)
            for line in axis.lines:
                assert line.get_xdata()[0] == line.get_xdata()[1]
            actual_lines.append(tuple(sorted(line.get_xdata()[0] for line in axis.lines)))
        expected_lines = [tuple(sorted(reference_values[:, parameter])) for parameter in range(2)]
        assert sorted(actual_lines) == sorted(expected_lines)
    if plot_kind != 'histogram':
        assert len(pair_axes) == 1
        overlays = [collection for collection in pair_axes[0].collections
                    if np.array_equal(collection.get_sizes(), [350])]
        assert len(overlays) == len(reference_values)
        actual_positions = np.vstack([collection.get_offsets() for collection in overlays])
        np.testing.assert_array_equal(actual_positions, reference_values)
        np.testing.assert_array_equal(pair_axes[0].collections[0].get_offsets(), samples)
    np.testing.assert_array_equal(samples, original_samples)
    np.testing.assert_array_equal(draws, reference_values)


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
