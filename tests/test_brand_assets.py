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
    figure = plt.figure(figsize=(3.0, 3.0))
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.set(xlim=(-1.02, 1.02), ylim=(-1.02, 1.02), aspect="equal")
    logo.draw_icon(axis)
    figure.canvas.draw()
    wordmarks = [patch for patch in axis.patches if patch.get_gid() == "pcat-wordmark"]
    assert len(wordmarks) == 1
    text = wordmarks[0].get_path().vertices
    assert np.max(np.hypot(*text.T)) < logo.CONTENT_RADIUS

    glyphs = [line.get_xydata() for line in axis.lines]
    glyphs += [collection.get_offsets() for collection in axis.collections]
    glyphs = np.concatenate(glyphs)
    assert text[:, 1].max() < glyphs[:, 1].min()
    assert np.max(np.hypot(*glyphs.T)) < logo.CONTENT_RADIUS
    assert np.max(np.hypot(*np.concatenate([text, glyphs]).T)) > 0.9 * logo.CONTENT_RADIUS

    centers = logo.CENTERS
    assert centers["point"][1] == centers["segment"][1] > centers["square"][1] == centers["cube"][1]
    assert centers["point"][0] < centers["segment"][0] and centers["square"][0] < centers["cube"][0]
    assert len([patch for patch in axis.patches if type(patch).__name__ == "FancyArrowPatch"]) == 3

    cube_edges = [line for line in axis.lines if line.get_gid() == "pcat-cube-edge"]
    assert len(cube_edges) == 12
    assert len({line.get_linewidth() for line in cube_edges}) == 1
    plt.close(figure)

    image = _read_rgba("pcat_logo")
    white = np.all(image[..., :3] > 245, axis=-1) & (image[..., 3] > 0)
    coordinates = np.argwhere(white)
    center = 0.5 * (np.asarray(image.shape[:2]) - 1.0)
    radius = np.linalg.norm((coordinates - center) / (0.5 * image.shape[0]), axis=1)
    assert radius.max() < logo.DISC_RADIUS / 1.02
