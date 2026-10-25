import runpy
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (Path(__file__).resolve().parents[1] / "examples" / "sampler_comparison_emcee_dynesty"
          / "sampler_comparison_emcee_dynesty.py")


@pytest.mark.parametrize("variable", [False, True])
def test_sampler_comparison_preserves_existing_pcat_results(tmp_path, variable):
    example = runpy.run_path(str(SCRIPT))
    namespace = example["run_pcat"].__globals__
    namespace["EXAMPLE_PATH"] = tmp_path
    namespace["RUN_NAME"] = "comparison_test"
    name = "comparison_test_variable_planet_count" if variable else "comparison_test"
    from pcat.main import retr_pathrun

    root = Path(retr_pathrun(tmp_path, name))
    cache = root / "data/outp"
    cache.mkdir(parents=True)
    marker = cache / "keep.txt"
    marker.write_text("previous comparison")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        if variable:
            example["run_pcat_planets"](np.arange(4.0), np.ones(4), np.ones(4), 100)
        else:
            example["run_pcat"](lambda values: 0.0, 100)
    assert marker.read_text() == "previous comparison"


def test_dynesty_comparison_forwards_the_budget(tmp_path, monkeypatch):
    example = runpy.run_path(str(SCRIPT))
    captured = {}

    class Results:
        samples = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        logwt = np.log([0.5, 0.5])
        logz = np.array([0.0])
        logzerr = np.array([0.1])

        def samples_equal(self):
            return self.samples

    class Sampler:
        results = Results()

        def __init__(self, *arguments, **options):
            captured.update(options)

        def run_nested(self, **options):
            captured.update(options)

    monkeypatch.setattr(example["dynesty"], "NestedSampler", Sampler)
    result = example["run_dynesty"](lambda values: 0.0, nlive=40, maxcall=200)
    assert captured["nlive"] == 40 and captured["maxcall"] == 200
    assert result[0].shape == (2, 3) and result[2] == pytest.approx(2.0)


def test_comparison_metrics_store_actual_arrays_and_preserve_output(tmp_path):
    import json
    from types import SimpleNamespace

    example = runpy.run_path(str(SCRIPT))
    namespace = example["write_comparison_metrics"].__globals__
    namespace["EXAMPLE_PATH"] = tmp_path
    namespace["DISPLAY_TAG"] = "20261025_120000"
    values = {"PCAT": {"samples": np.array([[1.0, 2.0, 3.0]]), "calls": np.int64(7)}}
    example["write_comparison_metrics"]("fixed", values, SimpleNamespace(part="fixed", maxcall=None))
    path = tmp_path / "visuals/sampler_comparison_fixed_20261025_120000.json"
    record = json.loads(path.read_text())
    assert record["results"]["PCAT"]["samples"] == [[1.0, 2.0, 3.0]]
    assert record["results"]["PCAT"]["calls"] == 7
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        example["write_comparison_metrics"]("fixed", values, SimpleNamespace(part="fixed", maxcall=None))
    assert path.read_bytes() == before


def test_unresolved_autocorrelation_is_not_reported_as_a_valid_ess(monkeypatch):
    example = runpy.run_path(str(SCRIPT))
    monkeypatch.setattr(example["emcee"].autocorr, "integrated_time", lambda *arguments, **options: np.array([-1.0, 2.0]))
    assert np.isnan(example["effective_sample_size"](np.ones((20, 1, 2))))


def test_planet_period_plot_handles_a_short_run_without_two_planet_draws(tmp_path):
    example = runpy.run_path(str(SCRIPT))
    example["plot_planet_periods"].__globals__["EXAMPLE_PATH"] = tmp_path
    example["plot_planet_periods"].__globals__["DISPLAY_TAG"] = "20261025_120000"
    example["plot_planet_periods"]({"peri": np.empty((0, 2))},
                                  {"peri": np.array([[9.0, 60.0], [9.2, 62.0]])}, "png")
    assert (tmp_path / "visuals/sampler_comparison_two_planet_periods_20261025_120000.png").is_file()