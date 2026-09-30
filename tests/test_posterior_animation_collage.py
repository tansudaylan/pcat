from pathlib import Path

from PIL import Image, ImageDraw

from examples.make_posterior_animation_collage import (
    DEFAULT_OUTPUT,
    PANELS,
    ROOT,
    make_collage,
)


def _write_frame(path: Path, color: str, marker: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (80, 80), color)
    ImageDraw.Draw(image).ellipse((marker, marker, marker + 20, marker + 20), fill="white")
    image.save(path)


def test_make_collage_combines_posterior_sequences(tmp_path):
    examples_root = tmp_path / "examples"
    for panel_index, panel in enumerate(PANELS):
        relative_pattern = Path(panel.pattern)
        frame_root = examples_root / relative_pattern.parent
        stem = relative_pattern.name.replace("*.png", "")
        _write_frame(frame_root / f"{stem}000.png", "#254f5b", panel_index * 5)
        _write_frame(frame_root / f"{stem}001.png", "#b44b35", panel_index * 5 + 10)

    output_path = tmp_path / "collage.gif"
    make_collage(
        output_path=output_path,
        examples_root=examples_root,
        frame_count=4,
        duration_ms=100,
        panel_size=120,
    )

    with Image.open(output_path) as animation:
        assert animation.n_frames == 4
        assert animation.size == (300, 450)
        assert animation.info["duration"] == 100


def test_readme_embeds_multiframe_collage():
    readme_path = ROOT.parent / "README.md"
    readme = readme_path.read_text()
    assert DEFAULT_OUTPUT.is_file()
    assert "![Posterior samples from four PCAT example problems]" in readme
    assert str(DEFAULT_OUTPUT.relative_to(ROOT.parent)) in readme
    with Image.open(DEFAULT_OUTPUT) as animation:
        assert animation.n_frames >= 4