import importlib.util
from pathlib import Path

import matplotlib.image as mpimg


EXAMPLE_SCRIPT = (
    Path(__file__).resolve().parents[1] / "examples" / "roman_lens_catalog_diagnostic.py"
)
SPECIFICATION = importlib.util.spec_from_file_location(
    "pcat_roman_lens_catalog_diagnostic", EXAMPLE_SCRIPT
)
example = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(example)


def test_roman_lens_catalog_example_runs_pipeline(tmp_path, capsys):
    output_path = tmp_path / "roman_lens_catalog_diagnostic.png"

    summary = example.run_example(output_path, number_lenses=20)

    image = mpimg.imread(output_path)
    assert image.shape[0] > 100
    assert image.shape[1] > 100
    assert image[..., :3].min() < 0.8
    assert summary["number_lenses"] == 20
    assert summary["number_injected"] == 10
    assert f"Writing to {output_path}..." in capsys.readouterr().out