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


@pytest.mark.parametrize("native_layout", [False, True])
def test_example_posterior_loader_uses_native_run_when_present(tmp_path, monkeypatch, native_layout):
    root = tmp_path / "example"
    native_root = root / "pcat_runs" / "selected"
    if native_layout:
        native_root.mkdir(parents=True)
    captured = []
    expected = object()
    monkeypatch.setattr(demo.pcat_main, "readfile", lambda path: captured.append(path) or expected)
    result = demo.load_example_posterior(
        "example/data/outp/selected/gdatfinlpost", output_root=root,
    )
    base = native_root if native_layout else root
    assert captured == [str(base / "data" / "outp" / "selected" / "gdatfinlpost")]
    assert result is expected


def test_example_posterior_loader_does_not_hide_incomplete_native_run(tmp_path, monkeypatch):
    root = tmp_path / "example"
    (root / "pcat_runs" / "selected").mkdir(parents=True)
    old = root / "data" / "outp" / "selected"
    old.mkdir(parents=True)
    (old / "gdatfinlpost.p").write_bytes(b"older posterior")

    def readfile(path):
        assert "pcat_runs" in Path(path).parts
        raise FileNotFoundError(path)

    monkeypatch.setattr(demo.pcat_main, "readfile", readfile)
    with pytest.raises(FileNotFoundError):
        demo.load_example_posterior("example/data/outp/selected/gdatfinlpost", output_root=root)


def test_proposal_profiling_preserves_existing_runs_and_forwards_new_name(tmp_path, monkeypatch):
    from pcat import sampling

    script = REPOSITORY_ROOT / "examples" / "proposal_profiling" / "proposal_profiling.py"
    example = load_example_module("profiling_preservation", script)
    monkeypatch.setattr(example, "EXAMPLE_PATH", tmp_path)
    cache = tmp_path / "data" / "outp" / "previous"
    cache.mkdir(parents=True)
    marker = cache / "posterior.txt"
    marker.write_text("earlier posterior", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        example.run_sampler(100, "previous")
    assert marker.read_text(encoding="utf-8") == "earlier posterior"
    native = tmp_path / "pcat_runs" / "native_previous" / "data" / "outp" / "native_previous"
    native.mkdir(parents=True)
    native_marker = native / "posterior.txt"
    native_marker.write_text("native posterior", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        example.run_sampler(100, "native_previous")
    assert native_marker.read_text(encoding="utf-8") == "native posterior"
    captured = {}
    monkeypatch.setattr(sampling, "sample", lambda **configuration: captured.update(configuration))
    assert example.run_sampler(100, "new_run") >= 0.0
    assert captured["strgcnfg"] == "new_run"
    assert captured["pathbase"] == str(tmp_path)
    assert captured["probtran"] == 0.7
    assert captured["probspmr"] == 0.3
    assert captured["probjump"] == 0.1


@pytest.mark.parametrize("name", ["", ".", "..", "nested/run"])
def test_proposal_profiling_requires_a_single_run_name(tmp_path, monkeypatch, name):
    script = REPOSITORY_ROOT / "examples" / "proposal_profiling" / "proposal_profiling.py"
    example = load_example_module("profiling_run_names", script)
    monkeypatch.setattr(example, "EXAMPLE_PATH", tmp_path)
    with pytest.raises(ValueError, match="single directory name"):
        example.run_sampler(100, name)


@pytest.mark.parametrize("native_layout", [False, True])
def test_proposal_profiling_reads_the_requested_chain_and_timing_units(tmp_path, monkeypatch, native_layout):
    import h5py

    script = REPOSITORY_ROOT / "examples" / "proposal_profiling" / "proposal_profiling.py"
    example = load_example_module("profiling_replay", script)
    monkeypatch.setattr(example, "EXAMPLE_PATH", tmp_path)
    root = tmp_path / "pcat_runs" / "selected" if native_layout else tmp_path
    cache = root / "data" / "outp" / "selected"
    cache.mkdir(parents=True)
    with h5py.File(cache / "gdatmodi0000post.h5", "w") as output:
        output["listpostindxproptype"] = [0, 1, 2]
        output["listpostboolpropaccp"] = [True, False, True]
        output["listpostboolpropfilt"] = [True, True, True]
        output["listpostchrototl"] = [0.001, 0.002, 0.003]  # [s]
        output["listpostchroprop"] = [0.0001, 0.0002, 0.0003]  # [s]
    chain = example.read_chain("selected")
    np.testing.assert_array_equal(chain["type"], [0, 1, 2])
    np.testing.assert_allclose(chain["time"], [1.0, 2.0, 3.0])  # [ms]
    np.testing.assert_allclose(chain["phase"]["prop"], [0.1, 0.2, 0.3])  # [ms]


def test_proposal_profiling_cli_uses_selected_output_and_run(tmp_path, monkeypatch):
    script = REPOSITORY_ROOT / "examples" / "proposal_profiling" / "proposal_profiling.py"
    example = load_example_module("profiling_cli", script)
    monkeypatch.setattr("sys.argv", [str(script), "--numbswep", "100", "--run-name", "fresh",
                                    "--output-root", str(tmp_path)])
    calls = []
    monkeypatch.setattr(example, "run_sampler", lambda sweeps, name: calls.append((sweeps, name)) or 0.1)
    chain = {"type": np.array([0]), "accp": np.array([True]), "filt": np.array([True]),
             "time": np.array([1.0]), "phase": {}}  # [ms]
    monkeypatch.setattr(example, "read_chain", lambda name: calls.append(name) or chain)
    monkeypatch.setattr(example, "configure_example_style", lambda theme: None)
    for name in ("plot_time_per_proposal", "plot_acceptance_and_cost", "plot_time_breakdown_by_phase"):
        monkeypatch.setattr(example, name, lambda *args: None)
    example.main()
    assert calls == [(100, "fresh"), "fresh"]
    assert example.EXAMPLE_PATH == tmp_path.resolve()


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
    assert os.environ["PCAT_PATH"] == str(output_root)
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


def test_shared_example_plot_helpers_apply_theme_and_save(tmp_path):
    import matplotlib.pyplot as plt

    with demo.mpl.rc_context():
        assert demo.configure_example_style("dark") == "white"
        assert demo.mpl.rcParams["figure.facecolor"] == "black"
        assert demo.configure_example_style("white") == "black"
        assert demo.mpl.rcParams["axes.grid"] is False

        figure, _ = plt.subplots()
        path = demo.save_example_figure(figure, tmp_path, "shared_helper", "png")

        assert path == tmp_path / "visuals" / "shared_helper.png"
        assert path.is_file()
        assert not plt.fignum_exists(figure.number)


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


def test_daylan_2017_mock_configuration_matches_paper_axes_and_population():
    script = (
        REPOSITORY_ROOT / "examples" / "daylan+2017_fermi_point_sources"
        / "daylan+2017_fermi_point_sources.py"
    )
    module = load_example_module("pcat_daylan_2017_configuration", script)
    configuration = module.build_configuration(quick=True)

    assert configuration["truenumbelempop0"] == 300
    assert configuration["truefluxdistslop"] == pytest.approx(-1.8)
    assert configuration["indxenerincl"].tolist() == [0, 1, 2]
    assert configuration["indxdqltfull"].tolist() == [0, 1]
    assert configuration["maxmgangdata"] == pytest.approx(np.deg2rad(20.0))
    assert configuration["typeexpr"] == "ferm"
    assert configuration["typedata"] == "simu"
    assert configuration["numbswep"] == 10_000


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
    candidate_path = animation_path.with_name("proposal_candidates.gif")
    frames[0].save(candidate_path, save_all=True, append_images=frames[1:])
    shown = []
    monkeypatch.setattr(demo, "__file__", str(tmp_path / "pcat" / "demo.py"))
    monkeypatch.setattr(IPython.display, "display", shown.append)

    demo.display_example_visuals("mock", "visuals/*.png")

    assert len(shown) == 3
    assert shown[0].filename == str(figure_path)
    assert shown[1].filename == str(animation_path)
    assert shown[2].filename == str(candidate_path)
    with pytest.raises(FileNotFoundError, match="missing"):
        demo.display_example_visuals("mock", "visuals/missing*.png")


def test_notebook_posterior_maps_use_state_and_write_to_example_visuals(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import IPython.display

    monkeypatch.setattr(demo, "__file__", str(tmp_path / "pcat" / "demo.py"))
    monkeypatch.setattr(IPython.display, "display", lambda figure: None)
    posterior = SimpleNamespace(
        cntpdata=np.ones((2, 16, 1)),
        listpostcntpmodl=np.ones((3, 2, 16, 1)),
        shapcart=(4, 4),
    )

    paths = demo.display_posterior_maps(posterior, "mock", "mock_posterior", channels=(0, 1))

    assert [path.name for path in paths] == [
        "mock_posterior_channel00.png",
        "mock_posterior_channel01.png",
    ]
    assert all(path.parent == tmp_path / "examples" / "mock" / "visuals" for path in paths)
    assert all(path.is_file() for path in paths)


def test_notebook_posterior_diagnostics_plot_loaded_catalog_samples(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import IPython.display

    monkeypatch.setattr(demo, "__file__", str(tmp_path / "pcat" / "demo.py"))
    monkeypatch.setattr(IPython.display, "display", lambda figure: None)
    posterior = SimpleNamespace(
        listpostnumbelem=np.array([[1], [2], [1]]),
        listpostlliktotl=np.array([[-12.0], [-9.0], [-10.0]]),
        listpostdictelem=[
            [{"flux": np.array([2.0])}],
            [{"flux": np.array([1.0, 3.0])}],
            [{"flux": np.array([2.5])}],
        ],
    )

    path = demo.display_posterior_diagnostics(posterior, "mock", "mock")

    assert path == tmp_path / "examples" / "mock" / "visuals" / "mock_posterior_diagnostics.png"
    assert path.is_file()


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

    monkeypatch.setattr("sys.argv", [str(script_path)])
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

    assert captured["numbswep"] == 120_000
    assert captured["numbburn"] == 60_000
    assert captured["numbsamp"] == 4_000
    assert captured["boolcheckconv"] is False
    assert captured["numbframanim"] == 24
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
    quick = module.build_configuration(quick=True)

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
    assert quick["truenumbelempop0"] == 300
    assert quick["numbelempop0reg0"] == 300
    assert quick["numbsidecart"] == 48
    assert quick["numbswep"] == 10_000
    assert quick["numbsamp"] == 1_000
    assert quick["truefluxdistslop"] == pytest.approx(-1.8)
    assert len(quick["indxenerincl"]) == 3
    assert len(quick["indxdqltincl"]) == 2

    diffuse_template = smoke["dicttrue"]["sbrtbacknorm"][1]
    assert diffuse_template.shape == (3, 48**2, 2)
    assert np.mean(diffuse_template, axis=1) == pytest.approx(np.ones((3, 2)))
    assert np.std(diffuse_template[1, :, 0]) > 0.15


def test_split_merge_jacobian_changes_counterfactual_acceptance_probability():
    from types import SimpleNamespace

    script = (
        REPOSITORY_ROOT / "examples" / "proposal_state_animation"
        / "proposal_state_animation.py"
    )
    module = load_example_module("pcat_jacobian_proposal_example", script)
    worker = SimpleNamespace(
        listpostindxproptype=np.array([3, 4, 3, 4]),
        listpostaccpprob=np.array([0.2, 0.2, 0.9, 1.0]),
        listpostaccplprb=np.log(np.array([0.2, 0.2, 0.9, 1.0])),
        listpostljcb=np.array([np.log(2.0), -np.log(2.0), 0.0, 0.0]),
        listpostboolpropaccp=np.array([False, False, True, True]),
        listpostboolpropfilt=np.ones(4),
    )

    summary = module.summarize_jacobian_acceptance(worker)

    assert summary["split"]["actual"].tolist() == pytest.approx([0.2, 0.9])
    assert summary["split"]["without_jacobian"].tolist() == pytest.approx([0.1, 0.9])
    assert summary["merge"]["actual"].tolist() == pytest.approx([0.2, 1.0])
    assert summary["merge"]["without_jacobian"].tolist() == pytest.approx([0.4, 1.0])


def test_same_dimension_parameterization_jumps_show_jacobian_acceptance_effect(tmp_path):
    script = (
        REPOSITORY_ROOT / "examples" / "proposal_state_animation"
        / "proposal_state_animation.py"
    )
    module = load_example_module("pcat_parameterization_jacobian_example", script)
    summary = module.simulate_parameterization_jumps(number_jumps=32, amplitude=2.0, seed=9)

    assert all(
        attempt["log_target_ratio"] + attempt["log_jacobian"] == pytest.approx(0.0)
        for attempt in summary["attempts"]
    )
    assert all(attempt["probability_with_jacobian"] == pytest.approx(1.0)
               for attempt in summary["attempts"])
    reverse_no_jacobian = [
        attempt["probability_without_jacobian"]
        for attempt in summary["attempts"]
        if attempt["from_without_jacobian"] == 1
    ]
    assert reverse_no_jacobian
    assert all(value == pytest.approx(0.5) for value in reverse_no_jacobian)
    assert summary["states_with_jacobian"].tolist() == [index % 2 for index in range(33)]
    assert len(set(summary["states_without_jacobian"])) == 2
    assert np.mean(summary["states_without_jacobian"]) > 0.5

    animation_path, _ = module.make_parameterization_jump_animation(
        tmp_path / "parameterization_jacobian.gif", number_jumps=8, seed=9
    )
    with Image.open(animation_path) as animation:
        assert animation.n_frames == 9
        assert animation.size == (1280, 720)


def test_proposal_state_animation_counts_every_categorical_move():
    from types import SimpleNamespace

    script = (
        REPOSITORY_ROOT / "examples" / "proposal_state_animation"
        / "proposal_state_animation.py"
    )
    module = load_example_module("pcat_proposal_coverage_example", script)
    worker = SimpleNamespace(
        listpostindxproptype=np.array([0, 1, 2, 3, 4, 5, 0, 3]),
    )

    counts = module.summarize_proposal_types(worker)

    assert counts == {
        "Within-model proposal": 2,
        "Birth proposal": 1,
        "Death proposal": 1,
        "Split proposal": 2,
        "Merge proposal": 1,
        "Jump proposal": 1,
    }


def test_log_acceptance_trace_is_not_summarized_as_a_posterior_parameter():
    from types import SimpleNamespace

    from pcat.main import _retr_posterior_summary_channels

    channels = _retr_posterior_summary_channels(
        SimpleNamespace(),
        SimpleNamespace(liststrgchan=["accplprb", "accpprob", "lliktotl"]),
        "post",
    )

    assert channels == ["accpprob", "lliktotl"]


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
