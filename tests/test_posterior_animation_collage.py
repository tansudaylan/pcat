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
)


def _write_frame(path: Path, color: str, marker: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (80, 80), color)
    ImageDraw.Draw(image).ellipse((marker, marker, marker + 20, marker + 20), fill="white")
    image.save(path)


def test_make_collage_combines_posterior_sequences(tmp_path):
    examples_root = tmp_path / "examples"
    for panel_index, panel in enumerate(POSTERIOR_ANIMATION_PANELS):
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


def test_collage_rejects_frozen_source_sequence(tmp_path):
    panel = POSTERIOR_ANIMATION_PANELS[0]
    frame_root = tmp_path / Path(panel.pattern).parent
    stem = Path(panel.pattern).name.replace("*.png", "")
    _write_frame(frame_root / f"{stem}000.png", "#254f5b", 5)
    _write_frame(frame_root / f"{stem}001.png", "#254f5b", 5)

    with pytest.raises(RuntimeError, match="no visual evolution"):
        _animation_frame_paths(panel, tmp_path)


def test_animation_frames_use_one_shared_palette():
    frames = [
        Image.new("RGB", (32, 32), color)
        for color in ("#A51C30", "#007360", "#202020")
    ]
    quantized = _quantize_shared_palette(frames)

    assert all(frame.mode == "P" for frame in quantized)
    assert all(frame.getpalette() == quantized[0].getpalette() for frame in quantized[1:])


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


def test_readme_embeds_multiframe_collage():
    readme_path = EXAMPLES_ROOT.parent / "README.md"
    readme = readme_path.read_text()
    assert DEFAULT_POSTERIOR_COLLAGE.is_file()
    assert "![Nine PCAT posterior views across unbinned, image, spectral, and time-series inference]" in readme
    assert str(DEFAULT_POSTERIOR_COLLAGE.relative_to(EXAMPLES_ROOT.parent)) in readme
    with Image.open(DEFAULT_POSTERIOR_COLLAGE) as animation:
        assert animation.n_frames >= 16
        assert animation.info["duration"] <= 150
        assert animation.width >= 1700
        frames = []
        for frame_index in range(animation.n_frames):
            animation.seek(frame_index)
            frames.append(animation.convert("RGB"))
    panel_size = 560
    margin = 20
    title_height = 66
    label_height = 42
    for panel_index in range(len(POSTERIOR_ANIMATION_PANELS)):
        row, column = divmod(panel_index, 3)
        x = margin + column * (panel_size + margin)
        y = title_height + margin + row * (panel_size + label_height + margin)
        crops = [frame.crop((x, y, x + panel_size, y + panel_size)).tobytes() for frame in frames]
        assert len(set(crops)) > 1, POSTERIOR_ANIMATION_PANELS[panel_index].label
