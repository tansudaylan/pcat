import importlib.util
import os
from pathlib import Path

import pytest

from pcat import demo


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_run_pipeline_demo_applies_defaults_and_overrides(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(demo.pcat_main, "sample", lambda **configuration: captured.update(configuration))
    output_root = tmp_path / "pcat-output"

    demo.run_pipeline_demo(output_root, typeexpr="gmix", numbswep=3)

    assert output_root.is_dir()
    assert os.environ["PCAT_DATA_PATH"] == str(output_root)
    assert captured["typeexpr"] == "gmix"
    assert captured["typedata"] == "simu"
    assert captured["boolmakeplot"] is True
    assert captured["numbswep"] == 3
    assert captured["pathbase"] == str(output_root)


@pytest.mark.parametrize(
    ("example_name", "expected"),
    [
        ("chandra_point_source", {"typeexpr": "chan", "elemtype": ["lghtpnts"]}),
        ("gaussian_mixture", {"typeexpr": "gmix", "numbspatdims": 2}),
        ("hst_lens", {"typeexpr": "HST_WFC3_IR", "typeelem": ["lens"]}),
    ],
)
def test_demo_entrypoint_preserves_scientific_configuration(
    example_name, expected, monkeypatch
):
    script_path = REPOSITORY_ROOT / "examples" / example_name / "generate_demo.py"
    specification = importlib.util.spec_from_file_location(f"pcat_{example_name}_demo", script_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    captured = {}
    monkeypatch.setattr(
        demo,
        "run_pipeline_demo",
        lambda output_root, **configuration: captured.update(
            output_root=output_root, **configuration
        ),
    )

    module.main()

    assert captured["output_root"] == script_path.parent / "pcat-output"
    for key, value in expected.items():
        assert captured[key] == value