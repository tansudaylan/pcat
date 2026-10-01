import importlib.util
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = REPOSITORY_ROOT / "docs" / "_static"
LOGO_SCRIPT = REPOSITORY_ROOT / "docs" / "make_logo.py"
SPECIFICATION = importlib.util.spec_from_file_location("pcat_make_logo", LOGO_SCRIPT)
logo = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(logo)
CRIMSON = np.array((0xA5, 0x1C, 0x30), dtype=np.uint8)
BLACK = np.array((0x00, 0x00, 0x00), dtype=np.uint8)
PRIMARY_ASSETS = ("pcat_icon", "pcat_logo")
SITE_ASSETS = ("pcat_icon", "pcat_logo", "pcat_banner")


def _read_rgba(name):
    return np.asarray(Image.open(STATIC_ROOT / f"{name}.png").convert("RGBA"))


def _assert_circular_asset(name):
    image = _read_rgba(name)
    height, width = image.shape[:2]
    assert abs(height - width) <= 2
    assert max(image[0, 0, 3], image[0, -1, 3], image[-1, 0, 3], image[-1, -1, 3]) == 0
    assert image[height // 2, width // 2, 3] == 255
    opaque_rgb = image[..., :3][image[..., 3] == 255]
    assert np.any(np.all(opaque_rgb == CRIMSON, axis=1))
    assert np.any(np.all(opaque_rgb == BLACK, axis=1))
    white = np.all(image[..., :3] > 245, axis=-1) & (image[..., 3] > 0)
    coordinates = np.argwhere(white)
    center = 0.5 * (np.asarray(image.shape[:2]) - 1.0)
    radius = np.linalg.norm((coordinates - center) / (0.5 * height), axis=1)
    assert radius.max() < logo.DISC_RADIUS / 1.02


def test_primary_logo_exports_are_circular_and_use_brand_palette():
    for name in PRIMARY_ASSETS:
        _assert_circular_asset(name)


def test_only_the_primary_circular_logo_assets_remain():
    for name in SITE_ASSETS:
        svg = STATIC_ROOT / f"{name}.svg"
        assert svg.is_file()
        source = svg.read_text().lower()
        assert "#a51c30" in source
        assert "#000000" in source
        assert all(line == line.rstrip() for line in source.splitlines())
    assert not list(STATIC_ROOT.glob("pcat_logo_concept_*"))
    assert not (STATIC_ROOT / "pcat_logo_concepts.png").exists()


def test_logo_contains_wordmark_within_circle_and_uniform_cube_edges():
    figure, axis = plt.subplots(figsize=(3.0, 3.0))
    axis.set(xlim=(-1.02, 1.02), ylim=(-1.02, 1.02), aspect="equal")
    logo.draw_icon(axis)
    figure.canvas.draw()
    assert [text.get_text() for text in axis.texts] == ["PCAT"]
    text_bounds = axis.texts[0].get_window_extent(figure.canvas.get_renderer())
    text_corners = axis.transData.inverted().transform(
        ((text_bounds.x0, text_bounds.y0), (text_bounds.x1, text_bounds.y1))
    )
    assert np.max(np.hypot(text_corners[:, 0], text_corners[:, 1])) < logo.DISC_RADIUS
    assert axis.texts[0].get_fontsize() >= 40
    symbol_bottom = np.min(logo.CENTERS[:, 1]) - logo.SIZE / 2.0 - logo.DEPTH[1] / 2.0
    assert text_corners[:, 1].max() < symbol_bottom
    assert np.allclose(logo.CENTERS[:, 1], logo.CENTERS[0, 1])
    assert np.ptp(logo.CENTERS[:, 0]) >= 1.2
    cube_edges = [line for line in axis.lines if line.get_gid() == "pcat-cube-edge"]
    assert len(cube_edges) == 12
    assert {line.get_linewidth() for line in cube_edges} == {logo.CUBE_EDGE_WIDTH}
    plt.close(figure)

    image = _read_rgba("pcat_logo")
    white = np.all(image[..., :3] > 245, axis=-1) & (image[..., 3] > 0)
    coordinates = np.argwhere(white)
    center = 0.5 * (np.asarray(image.shape[:2]) - 1.0)
    radius = np.linalg.norm((coordinates - center) / (0.5 * image.shape[0]), axis=1)
    assert radius.max() < logo.DISC_RADIUS / 1.02


def test_brand_asset_guide_describes_only_the_primary_mark():
    guide = (STATIC_ROOT / "README.md").read_text()
    assert "There are no alternate logo concepts" in guide
    assert "PCAT" in guide