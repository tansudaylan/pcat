import importlib.util
import os
from pathlib import Path

import numpy as np
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
    monkeypatch.setitem(demo.mpl.rcParams, "text.usetex", True)
    monkeypatch.setattr(
        demo.pcat_main, "sample", lambda **configuration: captured.update(configuration)
    )
    output_root = tmp_path / "pcat-output"

    demo.run_pipeline_demo(output_root, typeexpr="gmix", numbswep=3)

    assert demo.mpl.rcParams["text.usetex"] is False
    assert output_root.is_dir()
    assert os.environ["PCAT_DATA_PATH"] == str(output_root)
    assert captured["typeexpr"] == "gmix"
    assert captured["typedata"] == "simu"
    assert captured["boolmakeplot"] is True
    assert captured["boolmakeplotinit"] is True
    assert captured["boolmakeplotfram"] is True
    assert captured["boolmakeplotfinlpost"] is True
    assert captured["makeanim"] is True
    assert captured["numbswepplot"] == 10
    assert captured["probtran"] == pytest.approx(0.7)
    assert captured["probspmr"] == pytest.approx(0.4)
    assert captured["typefileplot"] == "png"
    assert captured["numbswep"] == 3
    assert captured["pathbase"] == str(output_root)


def test_publication_examples_use_proposal_rich_animated_runs(tmp_path, monkeypatch):
    from pcat import publication_examples

    calls = []
    monkeypatch.setattr(
        demo.pcat_main,
        "sample",
        lambda **configuration: calls.append(configuration),
    )

    publication_examples.run_optical_catalog_example(tmp_path / "optical", "optical", 10)
    publication_examples.run_spire_component_example(tmp_path / "spire", "spire", 10, 0.5)

    for configuration in calls:
        assert configuration["probtran"] == pytest.approx(0.7)
        assert configuration["probspmr"] == pytest.approx(0.4)
        assert configuration["makeanim"] is True
        assert configuration["numbswepplot"] > 1


def test_notebook_example_launcher_restores_arguments(monkeypatch):
    captured = {}
    original_arguments = list(os.sys.argv)

    def fake_run_path(path, run_name):
        captured.update(path=path, run_name=run_name, arguments=list(os.sys.argv))

    monkeypatch.setattr(demo.runpy, "run_path", fake_run_path)
    demo.run_example_script("chandra_point_source_catalog/chandra_point_source_catalog.py", "--smoke")

    assert Path(captured["path"]).is_file()
    assert captured["run_name"] == "__main__"
    assert captured["arguments"] == [captured["path"], "--smoke"]
    assert os.sys.argv == original_arguments


def test_notebook_visuals_display_saved_figures(tmp_path, monkeypatch):
    import IPython.display

    figure_path = tmp_path / "examples" / "mock" / "visuals" / "image.png"
    figure_path.parent.mkdir(parents=True)
    print(f"Writing to {figure_path}...")
    Image.new("RGB", (8, 8), "red").save(figure_path)
    animation_path = (
        tmp_path / "examples" / "mock" / "pcat_runs" / "mock" / "visuals"
        / "post" / "anim" / "proposal_activity.gif"
    )
    animation_path.parent.mkdir(parents=True)
    frames = [Image.new("RGB", (8, 8), color) for color in ("red", "blue")]
    frames[0].save(animation_path, save_all=True, append_images=frames[1:])
    shown = []
    monkeypatch.setattr(demo, "__file__", str(tmp_path / "pcat" / "demo.py"))
    monkeypatch.setattr(IPython.display, "display", shown.append)

    demo.display_example_visuals("mock", "visuals/*.png")

    assert len(shown) == 2
    assert shown[0].filename == str(figure_path)
    assert shown[1].filename == str(animation_path)
    with pytest.raises(FileNotFoundError, match="missing"):
        demo.display_example_visuals("mock", "visuals/missing*.png")


@pytest.mark.parametrize(
    ("example_name", "run_name", "expected"),
    [
        (
            "chandra_point_source_catalog",
            "chandra_point_source_catalog",
            {
                "typeexpr": "chan",
                "typeelem": ["lghtpnts"],
                "probspmr": 0.4,
                "numbswep": 250,
                "numbsamp": 50,
                "numbswepplot": 25,
            },
        ),
        (
            "gaussian_mixture_catalog",
            "gaussian_mixture_catalog",
            {"typeexpr": "gmix", "typeelem": ["clusvari"], "numbspatdims": 2},
        ),
        (
            "simulated_hst_strong_lens",
            "simulated_hst_strong_lens",
            {
                "typeexpr": "HST_WFC3_IR",
                "typeelem": ["lens"],
                "numbswep": 100,
                "numbsamp": 20,
                "numbswepplot": 10,
                "probtran": 0.7,
                "probspmr": 0.4,
            },
        ),
    ],
)
def test_demo_entrypoint_preserves_scientific_configuration(
    example_name, run_name, expected, monkeypatch
):
    script_path = REPOSITORY_ROOT / "examples" / example_name / f"{example_name}.py"
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

    expected_output_root = (
        script_path.parent
        if example_name == "simulated_hst_strong_lens"
        else script_path.parents[1] / run_name
    )
    assert captured["output_root"] == expected_output_root
    for key, value in expected.items():
        assert captured[key] == value
    if example_name == "gaussian_mixture_catalog":
        assert captured["dicttrue"]["typeelem"] == ["clusvari"]
        assert captured["dictfitt"]["typeelem"] == ["clusvari"]
        assert captured["strgexpo"] == pytest.approx(50.0)
        assert captured["typeseedelem"] == 2_293
        assert captured["probspmr"] == pytest.approx(0.4)
        assert captured["numbswep"] == 1_000
        assert captured["numbsamp"] == 100
        assert captured["numbswepplot"] == 100


def test_voigt_smoke_configuration_enables_animation(monkeypatch):
    script_path = (
        REPOSITORY_ROOT
        / "examples"
        / "voigt_spectral_line_catalog"
        / "voigt_spectral_line_catalog.py"
    )
    module = load_example_module("pcat_voigt_example", script_path)
    captured = {}
    monkeypatch.setattr(
        module.sampling,
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


def test_daylan2017_configuration_preserves_published_mock_assumptions():
    script_path = (
        REPOSITORY_ROOT / "examples" / "daylan+2017_fermi_point_sources"
        / "daylan+2017_fermi_point_sources.py"
    )
    module = load_example_module("pcat_daylan2017_reproduction", script_path)

    full = module.build_configuration()
    smoke = module.build_configuration(smoke=True, typefileplot="pdf")

    assert full["typeexpr"] == "ferm"
    assert full["truenumbelempop0"] == 300
    assert full["truemaxmnumbelempop0"] == 300
    assert full["numbelempop0reg0"] == 300
    assert full["truefluxdistslop"] == pytest.approx(-1.8)
    assert full["typeelem"] == ["lghtpnts"]
    assert full["typepixl"] == "cart"
    assert full["boolforccart"] is True
    assert np.rad2deg(full["maxmgangdata"]) == pytest.approx(20.0)
    assert len(full["indxenerincl"]) == 3
    assert len(full["indxdqltincl"]) == 2
    assert full["fermscalfact"].shape == (3, 2)
    assert full["dicttrue"]["psfpexpr"].shape == (30,)
    assert full["dicttrue"]["listnamediff"] == ["back0000", "back0001"]
    assert full["numbsidecart"] == 100
    assert full["numbswep"] == 1_000_000
    assert full["numbsamp"] == 10_000
    assert full["pathbase"] == str(
        REPOSITORY_ROOT / "examples" / "daylan+2017_fermi_point_sources"
    )
    assert full["boolcondcatl"] is True
    assert smoke["truenumbelempop0"] == 40
    assert smoke["numbelempop0reg0"] == 40
    assert smoke["numbsidecart"] == 48
    assert smoke["numbswep"] == 10_000
    assert smoke["numbsamp"] == 1_000
    assert smoke["numbswepplot"] == 1_000
    assert smoke["probtran"] == pytest.approx(0.7)
    assert smoke["probspmr"] == pytest.approx(0.4)
    assert smoke["boolcondcatl"] is True
    assert smoke["makeanim"] is True
    assert smoke["typefileplot"] == "pdf"

    diffuse_template = smoke["dicttrue"]["sbrtbacknorm"][1]
    assert diffuse_template.shape == (3, 48**2, 2)
    assert np.mean(diffuse_template, axis=1) == pytest.approx(np.ones((3, 2)))
    assert np.std(diffuse_template[1, :, 0]) > 0.15


def test_daylan2018_smoke_configuration_generates_proposal_animation(tmp_path, monkeypatch):
    script_path = (
        REPOSITORY_ROOT / "examples" / "daylan+2018_strong_lens_subhalos"
        / "daylan+2018_strong_lens_subhalos.py"
    )
    module = load_example_module("pcat_daylan2018_reproduction", script_path)

    assert module.build_configuration()["makeanim"] is True
    smoke_configuration = module.build_configuration(smoke=True)
    assert smoke_configuration["makeanim"] is True
    assert smoke_configuration["numbswep"] == 200
    assert smoke_configuration["numbswepplot"] == 10
    assert module.build_configuration()["boolmakeplotfinlpost"] is True
    assert module.build_configuration(smoke=True)["boolmakeplotfinlpost"] is True
    cached_output = tmp_path / "daylan2018_catalog"
    cached_output.mkdir()
    (cached_output / "stale.txt").touch()
    captured = {}
    monkeypatch.setattr(module, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(
        demo,
        "run_pipeline_demo",
        lambda output_root, **configuration: captured.update(
            output_root=output_root, **configuration
        ),
    )

    module.run_reproduction(smoke=True, fresh=True)

    assert not (cached_output / "stale.txt").exists()
    assert captured["output_root"] == cached_output
    assert captured["makeanim"] is True


def test_example_output_verification_requires_multiframe_animation(tmp_path):
    script_path = REPOSITORY_ROOT / "examples" / "run_all_examples.py"
    module = load_example_module("pcat_example_runner", script_path)
    visual_root = tmp_path / "visuals"
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
    frames[0].save(
        visual_root / "post" / "anim" / "proposal_activity.gif",
        save_all=True,
        append_images=frames[1:],
    )
    Image.new("RGB", (2, 2), "black").save(visual_root / "post" / "finl" / "final.png")

    module.verify_pipeline_outputs(tmp_path, "demo")
