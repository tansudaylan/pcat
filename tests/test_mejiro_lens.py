import importlib.util

import numpy as np
import pytest

pytest.importorskip("mejiro")
pytest.importorskip("roman_technical_information")

from pcat.mejiro_lens import (  # noqa: E402
    LABELS, PARAMETERS, mejiro_lens_log_likelihood, render_mejiro_counts,
    run_mejiro_lens_inference, simulate_mejiro_exposure,
)
from pcat.plotting import plot_lens_parameter_recovery  # noqa: E402


@pytest.fixture(scope="module")
def specification():
    return simulate_mejiro_exposure()


def test_pcat_model_reproduces_mejiro_image(specification):
    # the PCAT likelihood must evaluate exactly the image that mejiro rendered
    model = render_mejiro_counts(specification, specification["true_parameters"])
    np.testing.assert_allclose(model, specification["expected_counts"], rtol=1e-10)
    assert model.shape == specification["observed_counts"].shape
    assert specification["sky_counts"] > 0.0
    assert len(specification["true_parameters"]) == len(PARAMETERS) == len(LABELS)


def test_likelihood_prefers_injected_lens(specification):
    class State:
        mejiro_specification = specification

    truth = specification["true_parameters"]
    shifted = truth.copy()
    shifted[0] += 0.05  # [arcsec], Einstein radius
    assert mejiro_lens_log_likelihood(State, "fitt", truth) > mejiro_lens_log_likelihood(State, "fitt", shifted)


def test_pcat_samples_mejiro_exposure(specification, tmp_path):
    posterior = run_mejiro_lens_inference(
        specification, tmp_path, "mejiro_test", numbswep=300, numbburn=100, numbsamp=100, numbproc=1,
        numbframanim=4,
    )
    draws = np.asarray(posterior.listpostparagenrscalbase)
    assert draws.shape == (100, len(PARAMETERS))
    minima = np.array([row[4] for row in PARAMETERS])
    maxima = np.array([row[5] for row in PARAMETERS])
    assert np.all((draws >= minima) & (draws <= maxima))
    assert np.isfinite(np.asarray(posterior.listpostlliktotl)).all()
    assert len(posterior.listanimstate) == 4
    path = plot_lens_parameter_recovery(tmp_path / "recovery", draws, specification["true_parameters"],
                                        labels=LABELS)
    assert path.is_file()


def test_example_configuration_uses_mejiro_pipeline(monkeypatch, tmp_path):
    script = __import__("pathlib").Path(__file__).resolve().parents[1] / "examples" / \
        "mejiro_roman_strong_lens" / "mejiro_roman_strong_lens.py"
    module_specification = importlib.util.spec_from_file_location("mejiro_example", script)
    example = importlib.util.module_from_spec(module_specification)
    module_specification.loader.exec_module(example)
    captured = {}

    def fake_inference(specification, output_root, run_name, **configuration):
        captured.update(configuration, run_name=run_name)
        raise RuntimeError("stop after configuration")

    monkeypatch.setattr(example, "run_mejiro_lens_inference", fake_inference)
    # keep cache cleanup away from the example's own saved run
    monkeypatch.setattr(example, "EXAMPLE_PATH", tmp_path)
    with pytest.raises(RuntimeError, match="stop after configuration"):
        example.run_example()
    assert captured["numbproc"] == example.NUMBER_CHAINS == 4
    assert captured["numbswep"] == 60000 and captured["numbburn"] == 40000
    assert captured["numbframanim"] == 24
    # a run name equal to the folder name would place the run in, and let cleanup delete, the example folder
    assert captured["run_name"] != script.parent.name
