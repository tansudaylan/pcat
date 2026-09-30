from pathlib import Path

import numpy as np
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = REPOSITORY_ROOT / "docs" / "_static"
CRIMSON = np.array((0xA5, 0x1C, 0x30), dtype=np.uint8)
BLACK = np.array((0x00, 0x00, 0x00), dtype=np.uint8)
CONCEPT_NAMES = (
    "pcat_logo_concept_nested_spaces",
    "pcat_logo_concept_catalog_birth_death",
    "pcat_logo_concept_dimension_ladder",
    "pcat_logo_concept_model_orbit",
    "pcat_logo_concept_reversible_branching",
)


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


def test_primary_icon_is_circular_and_uses_brand_palette():
    _assert_circular_asset("pcat_icon")


def test_five_circular_logo_concepts_have_png_and_svg_assets():
    assert len(CONCEPT_NAMES) == 5
    for name in CONCEPT_NAMES:
        _assert_circular_asset(name)
        svg = STATIC_ROOT / f"{name}.svg"
        assert svg.is_file()
        source = svg.read_text().lower()
        assert "#a51c30" in source
        assert "#000000" in source
        assert all(line == line.rstrip() for line in source.splitlines())


def test_logo_comparison_sheet_and_asset_guide_exist():
    assert (STATIC_ROOT / "pcat_logo_concepts.png").is_file()
    guide = (STATIC_ROOT / "README.md").read_text()
    for name in CONCEPT_NAMES:
        assert name in guide