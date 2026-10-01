import importlib.util
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
SPECIFICATION = importlib.util.spec_from_file_location("pcat_make_diagrams", DOCS / "make_diagrams.py")
diagrams = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(diagrams)
NAMES = ("pcat_pipeline", "pcat_likelihoods", "pcat_extension_points")


def test_diagrams_render_without_overflow(tmp_path, monkeypatch):
    monkeypatch.setattr(diagrams, "PATH_DIAGRAMS", tmp_path)
    diagrams.draw_pipeline()
    diagrams.draw_likelihoods()
    diagrams.draw_extension_points()
    for name in NAMES:
        assert (tmp_path / f"{name}.png").is_file() and (tmp_path / f"{name}.svg").is_file()


def test_layout_check_rejects_overflowing_text():
    figure, axis = diagrams.new_figure(4.0, 3.0)
    box = diagrams.Box(2.0, 1.5, 1.0, 0.6, "Title", ("a label far too long for this box",))
    with pytest.raises(ValueError, match="overflows"):
        diagrams.check_layout(figure, [box], [diagrams.draw_box(axis, box)])


def test_architecture_page_embeds_every_committed_diagram():
    page = (DOCS / "architecture.rst").read_text()
    for name in NAMES:
        assert (DOCS / "_static" / "diagrams" / f"{name}.png").is_file()
        assert re.search(rf"image:: _static/diagrams/{name}\.png", page)
