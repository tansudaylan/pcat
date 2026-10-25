from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pytest

from pcat.demo import load_example_namespace
from pcat.plotting import (
    DEFAULT_POSTERIOR_COLLAGE,
    EXAMPLES_ROOT,
    POSTERIOR_ANIMATION_PANELS,
    _animation_frame_paths,
    _quantize_shared_palette,
    histogram_frame_limits,
    make_image_sequence_animation,
    make_posterior_animation_collage,
    _pad_animation_text,
)


def _write_frame(path: Path, color: str, marker: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (80, 80), color)
    ImageDraw.Draw(image).ellipse((marker, marker, marker + 20, marker + 20), fill="white")
    image.save(path)


def test_make_collage_combines_posterior_sequences(tmp_path, monkeypatch):
    labels = []
    original_text = ImageDraw.ImageDraw.text

    def capture_text(draw, position, text, *arguments, **options):
        labels.append(text)
        return original_text(draw, position, text, *arguments, **options)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture_text)
    examples_root = tmp_path / "examples"
    for panel_index, panel in enumerate(POSTERIOR_ANIMATION_PANELS):
        if panel.static_image is not None:
            _write_frame(
                examples_root.parent / panel.static_image,
                "#A51C30",
                panel_index * 5,
            )
            continue
        assert panel.pattern is not None
        relative_pattern = Path(panel.pattern)
        frame_root = examples_root / relative_pattern.parent
        stem = relative_pattern.name.replace("*.png", "")
        _write_frame(frame_root / f"{stem}000.png", "#254f5b", panel_index * 5)
        _write_frame(frame_root / f"{stem}001.png", "#b44b35", panel_index * 5 + 10)

    output_path = tmp_path / "collage.gif"
    make_posterior_animation_collage(
        output_path=output_path,
        examples_root=examples_root,
        frame_count=4,
        duration_ms=100,
        panel_size=120,
    )

    with Image.open(output_path) as animation:
        assert animation.n_frames == 4
        assert animation.size == (440, 632)
        assert animation.info["duration"] == 100
    assert "PCAT inference examples" in labels
    assert not any("burn-in" in label or "posterior samples" in label for label in labels)


def test_collage_rejects_frozen_source_sequence(tmp_path):
    panel = POSTERIOR_ANIMATION_PANELS[0]
    frame_root = tmp_path / Path(panel.pattern).parent
    stem = Path(panel.pattern).name.replace("*.png", "")
    _write_frame(frame_root / f"{stem}000.png", "#254f5b", 5)
    _write_frame(frame_root / f"{stem}001.png", "#254f5b", 5)

    with pytest.raises(RuntimeError, match="no visual evolution"):
        _animation_frame_paths(panel, tmp_path)


def test_miri_collage_uses_only_hash_verified_recorded_native_frames(tmp_path):
    import hashlib
    import json

    panel = POSTERIOR_ANIMATION_PANELS[2]
    directory = tmp_path / "jwst_miri_ngc7027_line_catalog/visuals"
    first = directory / "jwst_line_hist_native_swep000.png"
    second = directory / "jwst_line_hist_native_swep001.png"
    stray = directory / "jwst_line_hist_test_swep002.png"
    for path, color in ((first, "#254f5b"), (second, "#b44b35"), (stray, "black")):
        _write_frame(path, color, 5)
    record = {"spectral_frame_paths": [first.name, second.name],
              "spectral_frame_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in (first, second)}}
    (directory / "jwst_miri_ngc7027_run_20261025_120000.json").write_text(json.dumps(record))
    assert _animation_frame_paths(panel, tmp_path) == [first, second]
    second.write_bytes(first.read_bytes())
    with pytest.raises(RuntimeError, match="recorded provenance"):
        _animation_frame_paths(panel, tmp_path)


def test_collage_places_spectral_fit_logo_and_variable_starspots_in_requested_panels():
    panels = POSTERIOR_ANIMATION_PANELS

    assert len(panels) == 9
    assert "Unbinned Gaussian mixture" in panels[0].label
    assert "Daylan+2017 NG cap mock" in panels[1].label
    assert "Fermi-LAT point sources" in panels[1].label
    assert "JWST/MIRI NGC 7027" in panels[2].label
    assert panels[2].pattern == "jwst_miri_ngc7027_line_catalog/visuals/jwst_spectrum_fit_*_swep*.png"
    assert "variable-count rotating starspots" in panels[3].label
    assert panels[4].static_image == "docs/_static/pcat_logo.png"
    assert panels[4].pattern is None
    assert "Roman/WFI detector image" in panels[5].label
    assert "strong-lens arc and compact counterimage" in panels[5].label
    assert "PCAT flare catalog" in panels[6].label
    assert "variable-number exoplanets" in panels[7].label
    assert "transit-time series" in panels[8].label
    assert panels[8].pattern == "transit_timing_variations/visuals/ttv_posterior_*_swep*.png"
    assert all("deflection" not in panel.label.lower() for panel in panels)


def test_miri_posterior_line_histogram_uses_blue_model(tmp_path, monkeypatch):
    from matplotlib import pyplot as plt
    from types import SimpleNamespace

    example = load_example_namespace(
        "jwst_miri_ngc7027_line_catalog/jwst_miri_ngc7027_line_catalog.py"
    )
    example["EXAMPLE_PATH"] = tmp_path
    colors = []
    axes = []
    original_bar = plt.Axes.bar
    original_subplots = plt.subplots

    def capture_bar(axis, *args, **kwargs):
        colors.append(kwargs.get("color"))
        return original_bar(axis, *args, **kwargs)

    def capture_subplots(*args, **kwargs):
        figure, axis = original_subplots(*args, **kwargs)
        axes.append(axis)
        return figure, axis

    monkeypatch.setattr(plt.Axes, "bar", capture_bar)
    monkeypatch.setattr(plt, "subplots", capture_subplots)
    wavenumber = 1e6 / np.array([5.34, 5.5112])
    posterior = SimpleNamespace(listpostdictelem=[
        [{"elin": wavenumber}],
        [{"elin": np.array([wavenumber[0], wavenumber[0], wavenumber[1]])}],
    ])

    paths = example["render_line_histogram_frames"](posterior)

    assert len(paths) == 2
    assert colors == ["#176B87", "#176B87"]
    assert all(path.is_file() for path in paths)
    assert [axis.get_ylim() for axis in axes] == [pytest.approx((0.0, 2.3))] * 2

    second_paths = example["render_line_histogram_frames"](posterior)

    assert len(second_paths) == 2
    assert set(paths).isdisjoint(second_paths)
    assert all(path.is_file() for path in paths + second_paths)
    assert [axis.get_ylim() for axis in axes[2:]] == [pytest.approx((0.0, 2.3))] * 2


def test_flare_posterior_frames_keep_one_data_based_y_scale(tmp_path, monkeypatch):
    from matplotlib import pyplot as plt
    from types import SimpleNamespace

    example = load_example_namespace(
        "variable_number_stellar_flares/variable_number_stellar_flares.py"
    )
    example["EXAMPLE_PATH"] = tmp_path
    axes = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axis = original_subplots(*args, **kwargs)
        axes.append(axis)
        return figure, axis

    monkeypatch.setattr(plt, "subplots", capture_subplots)
    snapshots = [
        {
            "cntrswep": sweep,
            "boolburn": sweep == 0,
            "numbelem": np.array([1]),
            "cntpmodl": np.full((4, 1, 1), level),
        }
        for sweep, level in ((0, 80.0), (10, 180.0))
    ]
    paths = example["render_flare_posterior_frames"](
        np.linspace(1.0, 2.0, 4),
        np.array([90.0, 110.0, 100.0, 120.0]),
        SimpleNamespace(listanimstate=snapshots),
    )

    assert len(paths) == len(axes) == 2
    for axis in axes:
        assert axis.get_ylim() == pytest.approx((75.0, 205.0))


def test_animation_frames_use_one_shared_palette():
    frames = [
        Image.new("RGB", (32, 32), color)
        for color in ("#A51C30", "#007360", "#202020")
    ]
    quantized = _quantize_shared_palette(frames)

    assert all(frame.mode == "P" for frame in quantized)
    assert all(frame.getpalette() == quantized[0].getpalette() for frame in quantized[1:])


def test_shared_palette_preserves_thin_blue_models_and_exact_white_background():
    frame = Image.new("RGB", (400, 400), "white")
    ImageDraw.Draw(frame).line((200, 5, 200, 395), fill="#176B87", width=1)
    quantized = _quantize_shared_palette([frame, frame.copy()])
    for image in quantized:
        decoded = image.convert("RGB")
        assert decoded.getpixel((200, 200)) == (23, 107, 135)
        assert decoded.getpixel((0, 0)) == (255, 255, 255)


def test_changing_animation_text_is_padded_to_a_fixed_width():
    labels = ["burn-in", "posterior samples"]
    width = max(map(len, labels))
    padded = [_pad_animation_text(label, width) for label in labels]

    assert len(padded[0]) == len(padded[1]) == width
    assert [label.rstrip() for label in padded] == labels


def test_animation_states_can_omit_empty_catalogs():
    from types import SimpleNamespace
    from pcat.plotting import animation_states

    state = SimpleNamespace(listanimstate=[
        {"cntrswep": 4, "numbelem": np.array([0])},
        {"cntrswep": 0, "numbelem": np.array([0])},
        {"cntrswep": 8, "numbelem": np.array([2])},
        {"cntrswep": 12, "numbelem": np.array([1])},
    ])

    snapshots = animation_states(state, require_nonzero_catalog=True)

    assert [snapshot["cntrswep"] for snapshot in snapshots] == [8, 12]


def test_histogram_frame_limits_cover_full_fitted_catalog():
    lower, upper = histogram_frame_limits(reference_count=40, maximum_model_count=600)
    assert lower > 0
    assert upper > 600
    assert histogram_frame_limits(reference_count=2, maximum_model_count=3)[1] < 4


def test_unbinned_mixture_scores_individual_events_without_bins():
    example = load_example_namespace("unbinned_gaussian_mixture/unbinned_gaussian_mixture.py")
    events = example["simulate_events"]()
    likelihood = example["event_log_likelihood"]
    assert events.shape == (140, 2)
    assert likelihood([-0.9, -0.35, 0.8, 0.65], events) > likelihood([1.4, 1.4, 1.5, 1.5], events)


def test_ttv_example_scores_simulated_transit_times():
    pytest.importorskip("ephesos.models")
    example = load_example_namespace("transit_timing_variations/transit_timing_variations.py")
    data = example["simulate_transits"]()
    likelihood = example["timing_likelihood"]
    parameters = (0.2, 3.0, 4.0 / 1440.0, 0.3)
    assert data[0].size == 24
    assert likelihood(parameters, data) > likelihood((0.2, 3.0, 0.0, 0.3), data)


def test_make_image_sequence_animation_includes_every_cutout(tmp_path):
    images = [
        np.full((12, 10), value, dtype=float)
        for value in (1.0, 2.0, 3.0)
    ]
    output_path = tmp_path / "rubin_cutouts.gif"

    make_image_sequence_animation(
        images,
        ["lens A i-band", "lens B r-band", "lens C z-band"],
        output_path,
        duration_ms=120,
        image_size=64,
    )

    with Image.open(output_path) as animation:
        assert animation.n_frames == len(images)
        assert animation.info["duration"] == 120


def test_make_image_sequence_animation_uses_one_shared_intensity_stretch(tmp_path):
    low_range = np.tile(np.linspace(0.0, 100.0, 100), (100, 1))
    high_range = np.tile(np.linspace(0.0, 1000.0, 100), (100, 1))
    low_range[50, 50] = 50.0
    high_range[50, 50] = 50.0
    output_path = tmp_path / "shared_stretch.gif"

    make_image_sequence_animation(
        [low_range, high_range],
        ["low range", "high range"],
        output_path,
        image_size=100,
        title="",
    )

    with Image.open(output_path) as animation:
        animation.seek(0)
        first_pixel = animation.convert("RGB").getpixel((50, 108))
        animation.seek(1)
        second_pixel = animation.convert("RGB").getpixel((50, 108))
    assert first_pixel == second_pixel


def test_readme_embeds_multiframe_collage():
    readme_path = EXAMPLES_ROOT.parent / "README.md"
    readme = readme_path.read_text()
    assert DEFAULT_POSTERIOR_COLLAGE.is_file()
    assert "![Eight changing PCAT inference views and the centered PCAT logo]" in readme
    assert str(DEFAULT_POSTERIOR_COLLAGE.relative_to(EXAMPLES_ROOT.parent)) in readme
    with Image.open(DEFAULT_POSTERIOR_COLLAGE) as animation:
        assert animation.n_frames >= 24
        assert animation.info["duration"] <= 80
        assert animation.width >= 2200
        assert animation.height >= 2400
        frame_signatures = [[] for _ in POSTERIOR_ANIMATION_PANELS]
        panel_size = 720
        margin = 20
        title_height = 66
        label_height = 42
        palettes = []
        for frame_index in range(animation.n_frames):
            animation.seek(frame_index)
            palettes.append(tuple(animation.getpalette() or ()))
            frame = animation.convert("RGB")
            for panel_index, panel in enumerate(POSTERIOR_ANIMATION_PANELS):
                row, column = divmod(panel_index, 3)
                x = margin + column * (panel_size + margin)
                y = title_height + margin + row * (panel_size + label_height + margin)
                crop = frame.crop((x, y, x + panel_size, y + panel_size))
                frame_signatures[panel_index].append(crop.resize((32, 32)).tobytes())
        for panel, signatures in zip(POSTERIOR_ANIMATION_PANELS, frame_signatures):
            if panel.static_image is not None:
                assert len(set(signatures)) == 1, panel.label
                assert signatures[0] != Image.new("RGB", (32, 32), "white").tobytes(), panel.label
            else:
                assert len(set(signatures)) > 1, panel.label
        assert palettes[0]
        assert all(not palette or palette == palettes[0] for palette in palettes)
