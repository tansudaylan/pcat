import importlib.util
from pathlib import Path

import pytest


EXAMPLE_SCRIPT = (
    Path(__file__).resolve().parents[1] / "examples" / "pcat_voigt_profile_detection.py"
)
SPECIFICATION = importlib.util.spec_from_file_location(
    "pcat_voigt_profile_detection", EXAMPLE_SCRIPT
)
example = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(example)


def test_voigt_profile_configuration_calls_pcat_pipeline(monkeypatch):
    calls = []

    def fake_sample(**configuration):
        calls.append(configuration)
        return {"configuration": configuration["strgcnfg"]}

    monkeypatch.setattr(example.pcat.main, "sample", fake_sample)

    result = example.run_voigt_profile_detection("nomi")

    configuration = calls[0]
    assert result == {"configuration": "voigt_nomi"}
    assert configuration["strgcnfg"] == "voigt_nomi"
    assert configuration["spectype"] == ["voig"]
    assert configuration["typeelem"] == ["lghtlinevoig"]
    assert configuration["strgexpo"] == pytest.approx(1.0e5)
    assert configuration["truenumbelempop0"] == 2
    assert configuration["typeseed"] == 0
    assert configuration["typeseedelem"] == 17
    assert configuration["dicttrue"]["typeelem"] == ["lghtlinevoig"]
    assert configuration["dictfitt"]["spectype"] == ["voig"]
