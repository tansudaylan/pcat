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

    def fake_sample_parallel(
        variations,
        names,
        dictpcatinpt=None,
        boolexecpara=True,
        strgcnfgextnexec=None,
    ):
        calls.append(
            (variations, dictpcatinpt, names, boolexecpara, strgcnfgextnexec)
        )
        return {"configuration": strgcnfgextnexec}

    monkeypatch.setattr(example.pcat.main, "sample_parallel", fake_sample_parallel)

    result = example.run_voigt_profile_detection("nomi")

    variations, common, names, parallel, selected = calls[0]
    assert result == {"configuration": "nomi"}
    assert selected == "nomi"
    assert parallel is False
    assert names == ["nomi", "s2nrhigh"]
    assert common["spectype"] == ["voig"]
    assert common["typeelem"] == ["lghtlinevoig"]
    assert variations["s2nrhigh"]["strgexpo"] == pytest.approx(1.0e5)