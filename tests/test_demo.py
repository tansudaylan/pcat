import importlib.util
import os
from pathlib import Path

import pytest
from PIL import Image

from pcat import demo


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_example_module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_run_pipeline_demo_applies_defaults_and_overrides(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        demo.pcat_main, "sample", lambda **configuration: captured.update(configuration)
    )
    output_root = tmp_path / "pcat-output"

    demo.run_pipeline_demo(output_root, typeexpr="gmix", numbswep=3)

    assert output_root.is_dir()
    assert os.environ["PCAT_DATA_PATH"] == str(output_root)
    assert captured["typeexpr"] == "gmix"
    assert captured["typedata"] == "simu"
    assert captured["boolmakeplot"] is True
    assert captured["boolmakeplotinit"] is True
    assert captured["boolmakeplotfram"] is True
    assert captured["boolmakeplotfinlpost"] is True
    assert captured["makeanim"] is True
    assert captured["numbswepplot"] == 1
    assert captured["typefileplot"] == "png"
    assert captured["numbswep"] == 3
    assert captured["pathbase"] == str(output_root)


@pytest.mark.parametrize(
    ("example_name", "expected"),
    [
        ("chandra_point_source", {"typeexpr": "chan", "typeelem": ["lghtpnts"]}),
        (
            "gaussian_mixture",
            {"typeexpr": "gmix", "typeelem": ["clusvari"], "numbspatdims": 2},
        ),
        ("hst_lens", {"typeexpr": "HST_WFC3_IR", "typeelem": ["lens"]}),
    ],
)
def test_demo_entrypoint_preserves_scientific_configuration(
    example_name, expected, monkeypatch
):
    script_path = REPOSITORY_ROOT / "examples" / example_name / "generate_demo.py"
    module = load_example_module(f"pcat_{example_name}_demo", script_path)
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


def test_voigt_smoke_configuration_enables_animation(monkeypatch):
    script_path = REPOSITORY_ROOT / "examples" / "pcat_voigt_profile_detection.py"
    module = load_example_module("pcat_voigt_example", script_path)
    captured = {}
    monkeypatch.setattr(
        module.pcat.main,
        "sample",
        lambda **configuration: captured.update(configuration),
    )

    module.run_voigt_profile_detection(smoke=True)

    assert captured["numbswep"] == 150_000
    assert captured["numbsamp"] == 15_000
    assert captured["numbsampconvmin"] == 2_000
    assert captured["numbsampconveffc"] == pytest.approx(100.0)
    assert captured["boolcheckconv"] is True
    assert captured["numbswepplot"] == 10_000
    assert captured["makeanim"] is True
    assert captured["truenumbelempop0"] == 2
    assert captured["dictfitt"]["typeelem"] == ["lghtlinevoig"]


def test_daylan2016_configuration_preserves_published_mock_assumptions():
    script_path = (
        REPOSITORY_ROOT / "examples" / "daylan2016" / "generate_reproduction.py"
    )
    module = load_example_module("pcat_daylan2016_reproduction", script_path)

    full = module.build_configuration()
    smoke = module.build_configuration(smoke=True, typefileplot="pdf")

    assert full["truenumbelempop0"] == 300
    assert full["truemaxmnumbelempop0"] == 300
    assert full["numbelempop0reg0"] == 300
    assert full["truefluxdistslop"] == pytest.approx(-1.8)
    assert full["typeelem"] == ["lghtpnts"]
    assert full["typepixl"] == "cart"
    assert full["numbswep"] == 1_000_000
    assert full["numbsamp"] == 10_000
    assert full["boolcondcatl"] is True
    assert smoke["truenumbelempop0"] == 12
    assert smoke["numbelempop0reg0"] == 12
    assert smoke["numbswep"] == 10_000
    assert smoke["numbsamp"] == 1_000
    assert smoke["numbswepplot"] == 1_000
    assert smoke["boolcondcatl"] is True
    assert smoke["makeanim"] is True
    assert smoke["typefileplot"] == "pdf"


def test_example_output_verification_requires_multiframe_animation(tmp_path):
    script_path = REPOSITORY_ROOT / "examples" / "run_examples.py"
    module = load_example_module("pcat_example_runner", script_path)
    visual_root = tmp_path / "visuals" / "demo"
    for relative_path in ["init", "post/fram", "post/finl", "post/anim"]:
        (visual_root / relative_path).mkdir(parents=True)
    Image.new("RGB", (2, 2), "black").save(visual_root / "init" / "initial.png")
    Image.new("RGB", (2, 2), "red").save(visual_root / "post" / "fram" / "frame0.png")
    Image.new("RGB", (2, 2), "blue").save(visual_root / "post" / "fram" / "frame1.png")
    frames = [Image.new("RGB", (2, 2), color) for color in ["red", "blue"]]
    frames[0].save(
        visual_root / "post" / "anim" / "posterior.gif",
        save_all=True,
        append_images=frames[1:],
    )
    Image.new("RGB", (2, 2), "black").save(visual_root / "post" / "finl" / "final.png")

    module.verify_pipeline_outputs(tmp_path, "demo")
