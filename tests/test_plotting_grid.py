import numpy as np
import pytest

from pcat.plotting import plot_grid


def test_posterior_grid_is_owned_by_pcat():
    import tdpy

    assert plot_grid.__module__ == "pcat.plotting"
    assert not hasattr(tdpy, "plot_grid")


def test_plot_grid_writes_one_and_three_parameter_pdfs(tmp_path):
    random = np.random.default_rng(123)
    one_path = plot_grid(
        tmp_path / "posterior",
        "one",
        random.normal(size=100),
        ("x",),
    )
    three_path = plot_grid(
        tmp_path / "posterior",
        "three",
        random.normal(size=(100, 3)),
        ("x", "y", "z"),
        truepara=(0.0, 0.0, 0.0),
        listvarbdraw=[(0.1, -0.1, 0.2)],
    )
    assert one_path.is_file()
    assert three_path.is_file()


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
@pytest.mark.parametrize('inherit', [False, True])
def test_plot_grid_keeps_samples_visible_in_both_themes(tmp_path, monkeypatch, typeplotback, inherit):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from PIL import Image
    from tdpy.plotting import plot_background_colors, plot_style

    background, foreground = plot_background_colors(typeplotback)
    samples = np.random.default_rng(123).normal(size=(100, 2))
    captured = []
    save = plt.Figure.savefig

    def capture_and_save(figure, path, **kwargs):
        captured.append(figure)
        assert figure.get_facecolor() == to_rgba(background)
        for axis in figure.axes:
            assert axis.get_facecolor() == to_rgba(background)
        np.testing.assert_array_equal(figure.axes[2].collections[0].get_offsets(), samples)
        assert figure.axes[0].patches[0].get_edgecolor() == to_rgba(foreground)
        return save(figure, path, **kwargs)

    monkeypatch.setattr(plt.Figure, 'savefig', capture_and_save)
    options = {} if inherit else {'typeplotback': typeplotback}
    outer_theme = typeplotback if inherit else ('dark' if typeplotback == 'white' else 'white')
    with plot_style(outer_theme):
        path = plot_grid(tmp_path / 'posterior', 'theme', samples, ('x', 'y'), typefileplot='png', **options)
    assert len(captured) == 1
    with Image.open(path) as image:
        expected = tuple(int(channel * 255) for channel in to_rgba(background)[:3])
        assert image.convert('RGB').getpixel((0, 0)) == expected


@pytest.mark.parametrize('shape', [(2, 4), (4, 2), (4, 4)])
@pytest.mark.parametrize('pixel_scale', [0.1, 0.5])
@pytest.mark.parametrize('theme', ['white', 'dark'])
def test_lens_image_fit_preserves_cutout_shape_and_sky_scale(tmp_path, monkeypatch, shape, pixel_scale, theme):
    from PIL import Image
    from matplotlib import pyplot as plt
    from pcat import plotting
    from tdpy.plotting import plot_style

    observed = np.arange(1.0, np.prod(shape) + 1.0).reshape(shape)
    model = observed - 0.25
    variance = np.full(shape, 4.0)
    original = tuple(array.copy() for array in (observed, model, variance))
    captured = []
    save = plotting.save_figure
    before = set(plt.get_fignums())

    def capture(figure, path, **kwargs):
        captured.append(figure)
        axes = [axis for axis in figure.axes if axis.images]
        assert len(axes) == 3
        half_width = shape[1] * pixel_scale / 2.0  # [arcsec]
        half_height = shape[0] * pixel_scale / 2.0  # [arcsec]
        extent = (-half_width, half_width, -half_height, half_height)
        for axis, expected_values in zip(axes, (observed, model, (observed - model) / np.sqrt(variance))):
            assert tuple(axis.images[0].get_extent()) == extent
            np.testing.assert_array_equal(axis.images[0].get_array(), expected_values)
            assert axis.get_xlabel() == axis.get_ylabel() == 'Offset [arcsec]'
        return save(figure, path, **kwargs)

    monkeypatch.setattr(plotting, 'save_figure', capture)
    with plot_style(theme):
        path = plotting.plot_lens_image_fit(tmp_path / 'cutout', observed, model, variance, pixel_scale)

    assert len(captured) == 1
    for current, expected in zip((observed, model, variance), original):
        np.testing.assert_array_equal(current, expected)
    print(f'Reading from {path}...')
    with Image.open(path) as image:
        pixels = np.asarray(image.convert('RGB'))
        assert pixels.shape[0] > 100 and pixels.shape[1] > 100
        assert np.ptp(pixels) > 0
        assert tuple(pixels[0, 0]) == ((255, 255, 255) if theme == 'white' else (0, 0, 0))
    assert set(plt.get_fignums()) == before


@pytest.mark.parametrize('value', [0.0, -1.0, np.nan, np.inf, -np.inf, True, 1.0 + 0.0j, '1.0', None])
def test_lens_image_fit_rejects_invalid_variance_before_rendering(tmp_path, monkeypatch, value):
    from unittest.mock import Mock
    from matplotlib import pyplot as plt
    from pcat import plotting

    render = Mock(side_effect=AssertionError('Invalid variance reached figure creation'))
    monkeypatch.setattr(plt, 'subplots', render)
    with np.errstate(all='raise'), pytest.raises(ValueError, match='variance'):
        plotting.plot_lens_image_fit(tmp_path / 'invalid', np.ones((2, 4)), np.zeros((2, 4)), np.full((2, 4), value), 0.1)
    render.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('pixel_scale', [
    0.0, -0.1, np.nan, np.inf, -np.inf, True, 0.1 + 0.0j,
    '0.1', None, [0.1], np.ma.masked, np.ma.array(0.1, mask=True),
])
def test_lens_image_fit_rejects_invalid_pixel_scale_before_rendering(tmp_path, monkeypatch, pixel_scale):
    from unittest.mock import Mock
    from matplotlib import pyplot as plt
    from pcat import plotting

    render = Mock(side_effect=AssertionError('Invalid scale reached figure creation'))
    monkeypatch.setattr(plt, 'subplots', render)
    with np.errstate(all='raise'), pytest.raises(ValueError, match='pixel_scale_arcsec'):
        plotting.plot_lens_image_fit(tmp_path / 'invalid_scale', np.ones((2, 4)), np.zeros((2, 4)), np.ones((2, 4)), pixel_scale)
    render.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('pixel_scale', [1, 0.1, np.int64(1), np.float64(0.1), np.array(0.1), np.ma.array(0.1, mask=False)])
def test_lens_image_fit_preserves_positive_scalar_pixel_scales(tmp_path, monkeypatch, pixel_scale):
    from pcat import plotting

    captured = []
    save = plotting.save_figure

    def capture(figure, path, **kwargs):
        captured.extend(tuple(axis.images[0].get_extent()) for axis in figure.axes if axis.images)
        return save(figure, path, **kwargs)

    monkeypatch.setattr(plotting, 'save_figure', capture)
    plotting.plot_lens_image_fit(tmp_path / 'valid_scale', np.ones((2, 4)), np.zeros((2, 4)), np.ones((2, 4)), pixel_scale)

    expected = (-2 * pixel_scale, 2 * pixel_scale, -pixel_scale, pixel_scale)  # [arcsec]
    assert len(captured) == 3
    assert all(extent == expected for extent in captured)


@pytest.mark.parametrize('shape', [(), (1, 4), (2, 1), (2, 4)])
def test_lens_image_fit_preserves_variance_broadcasting(tmp_path, monkeypatch, shape):
    from pcat import plotting

    observed = np.arange(1.0, 9.0).reshape(2, 4)
    model = observed - 0.5
    variance = np.full(shape, 4.0)
    captured = []
    save = plotting.save_figure

    def capture(figure, path, **kwargs):
        captured.append(figure.axes[2].images[0].get_array().copy())
        return save(figure, path, **kwargs)

    monkeypatch.setattr(plotting, 'save_figure', capture)
    plotting.plot_lens_image_fit(tmp_path / 'broadcast', observed, model, variance, 0.1)
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], np.full((2, 4), 0.25))


@pytest.mark.parametrize('observed_shape, model_shape', [
    ((1, 4), (2, 1)), ((2, 1), (1, 4)), ((2, 4), (3, 4)),
    ((2, 4), (1, 2, 4)), ((2, 4), ()), ((2, 4), (4,)),
    ((4,), (4,)), ((0, 4), (0, 4)), ((2, 0), (2, 0)),
])
def test_lens_image_fit_rejects_invalid_image_geometry_before_rendering(tmp_path, monkeypatch, observed_shape, model_shape):
    from unittest.mock import Mock
    from matplotlib import pyplot as plt
    from pcat import plotting

    render = Mock(side_effect=AssertionError('Invalid image geometry reached figure creation'))
    monkeypatch.setattr(plt, 'subplots', render)
    with pytest.raises(ValueError, match='shape|two-dimensional'):
        plotting.plot_lens_image_fit(tmp_path / 'invalid_geometry', np.ones(observed_shape), np.zeros(model_shape), 1.0, 0.1)
    render.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('model_shape', [(1, 4), (2, 1), (1, 1), (2, 4)])
@pytest.mark.parametrize('masked_model', [False, True])
def test_lens_image_fit_preserves_model_broadcasting_within_observed_shape(tmp_path, monkeypatch, model_shape, masked_model):
    from pcat import plotting

    observed = np.arange(1.0, 9.0).reshape(2, 4)
    model = np.ones(model_shape)
    if masked_model:
        mask = np.zeros(model_shape, dtype=bool)
        mask.flat[0] = True
        model = np.ma.array(model, mask=mask)
    originals = observed.copy(), model.copy()
    expected_mask = np.broadcast_to(np.ma.getmaskarray(model), observed.shape)
    expected = (observed - np.broadcast_to(np.ma.getdata(model), observed.shape)) / 2.0
    captured = []
    save = plotting.save_figure

    def capture(figure, path, **kwargs):
        axes = [axis for axis in figure.axes if axis.images]
        np.testing.assert_array_equal(axes[0].images[0].get_array(), observed)
        np.testing.assert_array_equal(axes[1].images[0].get_array(), model)
        captured.append(axes[2].images[0].get_array().copy())
        return save(figure, path, **kwargs)

    monkeypatch.setattr(plotting, 'save_figure', capture)
    plotting.plot_lens_image_fit(tmp_path / 'model_broadcast', observed, model, 4.0, 0.1)
    assert len(captured) == 1
    assert captured[0].shape == observed.shape
    np.testing.assert_array_equal(np.ma.getmaskarray(captured[0]), expected_mask)
    np.testing.assert_array_equal(captured[0].compressed(), expected[~expected_mask])
    np.testing.assert_array_equal(observed, originals[0])
    np.testing.assert_array_equal(model, originals[1])


@pytest.mark.parametrize('shape', [(3,), (1, 2, 4)])
def test_lens_image_fit_rejects_variance_shape_expansion_before_rendering(tmp_path, monkeypatch, shape):
    from unittest.mock import Mock
    from matplotlib import pyplot as plt
    from pcat import plotting

    render = Mock(side_effect=AssertionError('Invalid variance shape reached figure creation'))
    monkeypatch.setattr(plt, 'subplots', render)
    with pytest.raises(ValueError, match='shape'):
        plotting.plot_lens_image_fit(tmp_path / 'invalid_shape', np.ones((2, 4)), np.zeros((2, 4)), np.ones(shape), 0.1)
    render.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('all_masked', [False, True])
def test_lens_image_fit_preserves_explicit_variance_masks(tmp_path, monkeypatch, all_masked):
    from pcat import plotting

    mask = np.ones((2, 4), dtype=bool) if all_masked else np.array([[True, False, False, False], [False, False, False, False]])
    variance = np.ma.array(np.where(mask, 0.0, 4.0), mask=mask)
    captured = []
    save = plotting.save_figure

    def capture(figure, path, **kwargs):
        captured.append(figure.axes[2].images[0].get_array().copy())
        return save(figure, path, **kwargs)

    monkeypatch.setattr(plotting, 'save_figure', capture)
    with np.errstate(all='raise'):
        plotting.plot_lens_image_fit(tmp_path / 'masked', np.ones((2, 4)), np.zeros((2, 4)), variance, 0.1)
    assert len(captured) == 1
    np.testing.assert_array_equal(np.ma.getmaskarray(captured[0]), mask)
    np.testing.assert_array_equal(captured[0].compressed(), np.full(np.count_nonzero(~mask), 0.5))


@pytest.mark.parametrize('dtype', [np.uint8, np.int8, np.int64, np.uint64])
@pytest.mark.parametrize('masked_model', [False, True])
def test_lens_image_fit_subtracts_integer_counts_without_wrapping_or_rounding(tmp_path, monkeypatch, dtype, masked_model):
    from pcat import plotting

    if dtype == np.uint64:
        base = 2 ** 63
        observed = np.array([[base + 1, base + 2], [base + 3, base + 4]], dtype=dtype)
        model = np.array([[base, base + 4], [base + 2, base + 6]], dtype=dtype)
    elif dtype == np.int64:
        limits = np.iinfo(dtype)
        observed = np.full((2, 2), limits.max, dtype=dtype)
        model = np.array([[limits.min, limits.max - 1], [limits.min + 2, limits.max - 2]], dtype=dtype)
    else:
        observed = np.array([[2, 4], [6, 8]], dtype=dtype)
        model = np.array([[3, 2], [255 if dtype == np.uint8 else -128, 127 if dtype == np.int8 else 10]], dtype=dtype)
    expected = np.array([[int(first) - int(second) for first, second in zip(first_row, second_row)] for first_row, second_row in zip(observed, model)], dtype=float) / 2.0
    mask = np.array([[True, False], [False, False]]) if masked_model else np.zeros((2, 2), dtype=bool)
    if masked_model:
        model = np.ma.array(model, mask=mask)
    originals = observed.copy(), model.copy()
    captured = []
    save = plotting.save_figure

    def capture(figure, path, **kwargs):
        axes = [axis for axis in figure.axes if axis.images]
        np.testing.assert_array_equal(axes[0].images[0].get_array(), observed)
        np.testing.assert_array_equal(axes[1].images[0].get_array(), model)
        captured.append(axes[2].images[0].get_array().copy())
        return save(figure, path, **kwargs)

    monkeypatch.setattr(plotting, 'save_figure', capture)
    with np.errstate(all='raise'):
        plotting.plot_lens_image_fit(tmp_path / 'integer', observed, model, np.full((2, 2), 4.0), 0.1)

    assert len(captured) == 1
    np.testing.assert_array_equal(np.ma.getmaskarray(captured[0]), mask)
    np.testing.assert_array_equal(captured[0].compressed(), expected[~mask])
    np.testing.assert_array_equal(observed, originals[0])
    np.testing.assert_array_equal(model, originals[1])


@pytest.mark.parametrize('hidden_value', [1e12, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('mask_position', [0, 3])
def test_lens_image_fit_uses_only_visible_observations_for_count_limits(tmp_path, monkeypatch, hidden_value, mask_position):
    import warnings
    from pcat import plotting

    observed = np.ma.array(np.arange(1.0, 5.0).reshape(2, 2), mask=False)  # [electron pixel^-1]
    observed.data.flat[mask_position] = hidden_value
    observed.mask.flat[mask_position] = True
    original = observed.copy()
    visible = np.delete(np.arange(1.0, 5.0), mask_position)  # [electron pixel^-1]
    expected_limit = np.nanpercentile(visible, 99.5)  # [electron pixel^-1]
    captured = []
    save = plotting.save_figure

    def capture(figure, path, **kwargs):
        axes = [axis for axis in figure.axes if axis.images]
        captured.extend(axis.images[0].get_clim() for axis in axes)
        np.testing.assert_array_equal(np.ma.getmaskarray(axes[0].images[0].get_array()), observed.mask)
        np.testing.assert_array_equal(np.ma.getmaskarray(axes[2].images[0].get_array()), observed.mask)
        return save(figure, path, **kwargs)

    monkeypatch.setattr(plotting, 'save_figure', capture)
    with warnings.catch_warnings(), np.errstate(all='raise'):
        warnings.simplefilter('error')
        plotting.plot_lens_image_fit(tmp_path / 'visible_counts', observed, np.ones((2, 2)), 1.0, 0.1)
    assert captured == [(0.0, expected_limit), (0.0, expected_limit), (-5.0, 5.0)]
    np.testing.assert_array_equal(observed.data, original.data)
    np.testing.assert_array_equal(observed.mask, original.mask)


@pytest.mark.parametrize('masked_container', [False, True])
def test_lens_image_fit_preserves_unmasked_nan_percentile(tmp_path, monkeypatch, masked_container):
    import warnings
    from pcat import plotting

    observed = np.array([[1.0, 2.0], [np.nan, 4.0]])  # [electron pixel^-1]
    expected_limit = np.nanpercentile(observed, 99.5)  # [electron pixel^-1]
    if masked_container:
        observed = np.ma.array(observed, mask=False)
    original = observed.copy()
    captured = []
    save = plotting.save_figure

    def capture(figure, path, **kwargs):
        captured.extend(axis.images[0].get_clim() for axis in figure.axes if axis.images)
        return save(figure, path, **kwargs)

    monkeypatch.setattr(plotting, 'save_figure', capture)
    with warnings.catch_warnings(), np.errstate(all='raise'):
        warnings.simplefilter('error')
        plotting.plot_lens_image_fit(tmp_path / 'unmasked_counts', observed, np.ones((2, 2)), 1.0, 0.1)
    assert captured == [(0.0, expected_limit), (0.0, expected_limit), (-5.0, 5.0)]
    np.testing.assert_array_equal(observed, original)


def test_lens_image_fit_rejects_fully_masked_observations_before_rendering(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from matplotlib import pyplot as plt
    from pcat import plotting

    render = Mock(side_effect=AssertionError('Fully masked observations reached figure creation'))
    monkeypatch.setattr(plt, 'subplots', render)
    with pytest.raises(ValueError, match='unmasked pixel'):
        plotting.plot_lens_image_fit(tmp_path / 'masked_observation', np.ma.masked_all((2, 2)), np.ones((2, 2)), 1.0, 0.1)
    render.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('draw_shape, truth_shape, number_labels', [
    ((20, 4), (4,), 3), ((20, 3), (3,), 4),
    ((20, 3), (2,), 3), ((20, 3), (4,), 3), ((20, 3), (1, 3), 3),
    ((0, 3), (3,), 3), ((20, 0), (0,), 0), ((20,), (1,), 1),
    ((), (1,), 1), ((2, 10, 3), (3,), 3),
])
def test_lens_parameter_recovery_rejects_misaligned_inputs_before_rendering(tmp_path, monkeypatch, draw_shape, truth_shape, number_labels):
    import warnings
    from unittest.mock import Mock
    from matplotlib import pyplot as plt
    from pcat import plotting

    render = Mock(side_effect=AssertionError('Invalid parameter metadata reached figure creation'))
    monkeypatch.setattr(plt, 'subplots', render)
    labels = tuple(f'Parameter {index}' for index in range(number_labels))
    with warnings.catch_warnings(), pytest.raises(ValueError, match='two-dimensional|per parameter column'):
        warnings.simplefilter('error')
        plotting.plot_lens_parameter_recovery(tmp_path / 'invalid_parameters', np.zeros(draw_shape), np.ones(truth_shape), labels=labels)
    render.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('number_parameters', [1, 3, 4, 6])
@pytest.mark.parametrize('typefileplot', ['png', 'pdf'])
def test_lens_parameter_recovery_keeps_every_parameter_panel(tmp_path, monkeypatch, number_parameters, typefileplot):
    from matplotlib import pyplot as plt
    from pcat import plotting

    draws = np.random.default_rng(51).normal(size=(64, number_parameters))
    truth = np.arange(number_parameters) / 10.0
    labels = tuple(f'Parameter {index}' for index in range(number_parameters))
    originals = draws.copy(), truth.copy()
    captured = []
    save = plotting.save_figure
    before = set(plt.get_fignums())

    def capture(figure, path, **kwargs):
        visible = [axis for axis in figure.axes if axis.get_visible()]
        assert len(figure.axes) == 3 * ((number_parameters + 2) // 3)
        assert len(visible) == number_parameters
        for index, axis in enumerate(visible):
            assert axis.get_xlabel() == labels[index]
            assert len(axis.patches) == 22
            np.testing.assert_array_equal(axis.lines[0].get_xdata(), [truth[index], truth[index]])
            median = np.median(draws[:, index])
            np.testing.assert_array_equal(axis.lines[1].get_xdata(), [median, median])
        captured.append(figure)
        return save(figure, path, **kwargs)

    monkeypatch.setattr(plotting, 'save_figure', capture)
    path = plotting.plot_lens_parameter_recovery(tmp_path / 'all_parameters', draws, truth, typefileplot, labels=labels)
    assert len(captured) == 1
    print(f'Reading from {path}...')
    contents = path.read_bytes()
    assert contents.startswith(b'\x89PNG\r\n\x1a\n' if typefileplot == 'png' else b'%PDF-')
    np.testing.assert_array_equal(draws, originals[0])
    np.testing.assert_array_equal(truth, originals[1])
    assert set(plt.get_fignums()) == before


@pytest.mark.parametrize('hidden_value', [1e8, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('typefileplot', ['png', 'pdf'])
def test_lens_parameter_recovery_uses_matching_visible_samples(tmp_path, monkeypatch, hidden_value, typefileplot):
    import warnings
    from pcat import plotting

    draws = np.ma.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0], [hidden_value, 40.0]],
                        mask=[[False, True], [False, False], [False, False], [True, False]])
    truth = np.array([0.5, 25.0])
    originals = draws.copy(), truth.copy()
    visible_columns = (np.array([0.0, 1.0, 2.0]), np.array([20.0, 30.0, 40.0]))
    captured = []
    save = plotting.save_figure

    def capture(figure, path, **kwargs):
        axes = [axis for axis in figure.axes if axis.get_visible()]
        assert len(axes) == 2
        for index, (axis, values) in enumerate(zip(axes, visible_columns)):
            heights, edges = np.histogram(values, bins=22, density=True)
            np.testing.assert_allclose([patch.get_height() for patch in axis.patches], heights)
            np.testing.assert_allclose([patch.get_x() for patch in axis.patches], edges[:-1], atol=np.finfo(float).eps)
            np.testing.assert_allclose([patch.get_width() for patch in axis.patches], np.diff(edges))
            np.testing.assert_array_equal(axis.lines[0].get_xdata(), [truth[index], truth[index]])
            median = np.median(values)
            np.testing.assert_array_equal(axis.lines[1].get_xdata(), [median, median])
        captured.append(figure)
        return save(figure, path, **kwargs)

    monkeypatch.setattr(plotting, 'save_figure', capture)
    with warnings.catch_warnings(), np.errstate(all='raise'):
        warnings.simplefilter('error')
        plotting.plot_lens_parameter_recovery(tmp_path / 'masked_parameters', draws, truth, typefileplot,
                                             labels=('First parameter', 'Second parameter'))
    assert len(captured) == 1
    np.testing.assert_array_equal(draws.data, originals[0].data)
    np.testing.assert_array_equal(draws.mask, originals[0].mask)
    np.testing.assert_array_equal(truth, originals[1])


@pytest.mark.parametrize('draws', [
    np.full((4, 2), np.nan), np.full((4, 2), np.inf), np.full((4, 2), -np.inf),
    np.ones((4, 2), dtype=complex), np.full((4, 2), '1.0'), np.ones((4, 2), dtype=bool),
    np.ma.masked_all((4, 2)), np.ma.array(np.ones((4, 2)), mask=[[True, False]] * 4),
])
def test_lens_parameter_recovery_rejects_invalid_visible_samples_before_rendering(tmp_path, monkeypatch, draws):
    import warnings
    from unittest.mock import Mock
    from matplotlib import pyplot as plt
    from pcat import plotting

    render = Mock(side_effect=AssertionError('Invalid visible samples reached figure creation'))
    monkeypatch.setattr(plt, 'subplots', render)
    with warnings.catch_warnings(), np.errstate(all='raise'), pytest.raises(ValueError, match='finite real unmasked samples'):
        warnings.simplefilter('error')
        plotting.plot_lens_parameter_recovery(tmp_path / 'invalid_visible_parameters', draws, np.ones(2),
                                             labels=('First parameter', 'Second parameter'))
    render.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('truth', [
    np.array([np.nan, 0.0]), np.array([np.inf, 0.0]), np.array([-np.inf, 0.0]),
    np.array([1.0 + 0.0j, 0.0]), np.array(['1.0', '0.0']), [None, 0.0],
    np.array([True, False]), np.ma.array([0.0, 1.0], mask=[True, False]),
    np.ma.array([np.nan, 1.0], mask=[True, False]),
])
def test_lens_parameter_recovery_rejects_invalid_injected_values_before_rendering(tmp_path, monkeypatch, truth):
    import warnings
    from unittest.mock import Mock
    from matplotlib import pyplot as plt
    from pcat import plotting

    render = Mock(side_effect=AssertionError('Invalid injected values reached figure creation'))
    monkeypatch.setattr(plt, 'subplots', render)
    with warnings.catch_warnings(), np.errstate(all='raise'), pytest.raises(ValueError, match='Injected lens parameters'):
        warnings.simplefilter('error')
        plotting.plot_lens_parameter_recovery(tmp_path / 'invalid_truth', np.ones((4, 2)), truth,
                                             labels=('First parameter', 'Second parameter'))
    render.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('truth', [np.array([0.25, -0.5]), [0.25, -0.5],
                                 np.array([0, 1]), np.ma.array([0.25, -0.5], mask=False)])
@pytest.mark.parametrize('typefileplot', ['png', 'pdf'])
def test_lens_parameter_recovery_preserves_valid_injected_values(tmp_path, monkeypatch, truth, typefileplot):
    import warnings
    from pcat import plotting

    draws = np.random.default_rng(67).normal(size=(32, 2))
    original_draws = draws.copy()
    original_truth = np.ma.asarray(truth).copy()
    captured = []
    save = plotting.save_figure

    def capture(figure, path, **kwargs):
        axes = [axis for axis in figure.axes if axis.get_visible()]
        assert len(axes) == 2
        for index, axis in enumerate(axes):
            assert axis.lines[0].get_label() == 'Injected value'
            np.testing.assert_array_equal(axis.lines[0].get_xdata(), [truth[index], truth[index]])
        captured.append(figure)
        return save(figure, path, **kwargs)

    monkeypatch.setattr(plotting, 'save_figure', capture)
    with warnings.catch_warnings(), np.errstate(all='raise'):
        warnings.simplefilter('error')
        plotting.plot_lens_parameter_recovery(tmp_path / 'finite_truth', draws, truth, typefileplot,
                                             labels=('First parameter', 'Second parameter'))
    assert len(captured) == 1
    np.testing.assert_array_equal(draws, original_draws)
    np.testing.assert_array_equal(np.ma.getdata(truth), original_truth.data)
    np.testing.assert_array_equal(np.ma.getmaskarray(truth), np.ma.getmaskarray(original_truth))
