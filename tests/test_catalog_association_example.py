import importlib.util
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pergamon


EXAMPLE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "catalog_association_completeness_purity"
    / "catalog_association_completeness_purity.py"
)
SPECIFICATION = importlib.util.spec_from_file_location(
    "pcat_catalog_association_completeness_purity_example", EXAMPLE_SCRIPT
)
example = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(example)


def test_catalog_association_completeness_purity_example_writes_nonblank_figure(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    monkeypatch.setitem(plt.rcParams, "text.usetex", False)
    output_path = tmp_path / "catalog_association_completeness_purity.png"
    metric_sample_sizes = []
    original_estimator = pergamon.estimate_detection_performance

    def record_metric_sample_sizes(true_labels, recovered_labels):
        metric_sample_sizes.append(len(true_labels))
        return original_estimator(true_labels, recovered_labels)

    monkeypatch.setattr(pergamon, "estimate_detection_performance", record_metric_sample_sizes)

    summary = example.run_example(output_path)

    print(f"Reading from {output_path}...")
    image = mpimg.imread(output_path)
    assert image.shape[0] > 100
    assert image.shape[1] > 100
    assert image[..., :3].min() < 0.8
    assert summary["number_sources"] == 80
    assert summary["number_sources"] + 24 in metric_sample_sizes
    assert summary["number_targets"] in metric_sample_sizes
    assert 0.7 < summary["completeness"] < 0.9
    assert 0.8 < summary["purity"] <= 1.0
    assert summary["false_discovery_rate"] == 1.0 - summary["purity"]
    output = capsys.readouterr().out
    assert f"Reading from {output_path}..." in output
    assert f"Writing to {output_path}..." in output


def test_catalog_association_metrics_define_empty_radius_as_zero():
    catalogs = example.simulate_catalogs()

    completeness, purity, false_discovery_rate = example.association_rates(
        *catalogs, np.array([0.0])
    )

    assert completeness[0] == 0.0
    assert purity[0] == 1.0
    assert false_discovery_rate[0] == 0.0