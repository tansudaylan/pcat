from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pcat.plotting import (
    DEFAULT_POSTERIOR_COLLAGE,
    EXAMPLES_ROOT,
    POSTERIOR_ANIMATION_PANELS,
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
        assert animation.size == (440, 814)
        assert animation.info["duration"] == 100


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
    assert "![Posterior samples from twelve PCAT example problems]" in readme
    assert str(DEFAULT_POSTERIOR_COLLAGE.relative_to(EXAMPLES_ROOT.parent)) in readme
    with Image.open(DEFAULT_POSTERIOR_COLLAGE) as animation:
        assert animation.n_frames >= 16
        assert animation.info["duration"] <= 150
        assert animation.width >= 1700
