import importlib.util
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


EXAMPLE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "catalog_association"
    / "catalog_association.py"
)
SPECIFICATION = importlib.util.spec_from_file_location(
    "pcat_catalog_association_example", EXAMPLE_SCRIPT
)
example = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(example)


def test_catalog_association_example_writes_nonblank_figure(tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(plt.rcParams, "text.usetex", False)
    output_path = tmp_path / "catalog_association.png"

    summary = example.run_example(output_path)

    print(f"Reading from {output_path}...")
    image = mpimg.imread(output_path)
    assert image.shape[0] > 100
    assert image.shape[1] > 100
    assert image[..., :3].min() < 0.8
    assert summary["number_sources"] == 80
    assert 0.7 < summary["completeness"] < 0.9
    assert 0.6 < summary["purity"] < 0.9
    output = capsys.readouterr().out
    assert f"Reading from {output_path}..." in output
    assert f"Writing to {output_path}..." in output