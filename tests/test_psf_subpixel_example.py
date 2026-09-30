import importlib.util
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


EXAMPLE_SCRIPT = (
    Path(__file__).resolve().parents[1] / "examples" / "psf_subpixel" / "psf_subpixel.py"
)
SPECIFICATION = importlib.util.spec_from_file_location("pcat_psf_subpixel_example", EXAMPLE_SCRIPT)
example = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(example)


def test_psf_subpixel_example_writes_accurate_nonblank_figure(tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(plt.rcParams, "text.usetex", False)
    output_path = tmp_path / "psf_subpixel.png"

    summary = example.run_example(output_path)

    print(f"Reading from {output_path}...")
    image = mpimg.imread(output_path)
    assert image.shape[0] > 100
    assert image.shape[1] > 100
    assert image[..., :3].min() < 0.8
    assert summary["number_evaluations"] == 328
    assert summary["maximum_residual_percent"] < 0.1
    output = capsys.readouterr().out
    assert f"Reading from {output_path}..." in output
    assert f"Writing to {output_path}..." in output