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


def test_example_namespace_output_root_is_the_writer_global(tmp_path):
    from types import SimpleNamespace

    namespace = demo.load_example_namespace(
        "jwst_miri_ngc7027_line_catalog/jwst_miri_ngc7027_line_catalog.py"
    )
    assert namespace is namespace["render_line_histogram_frames"].__globals__
    namespace["EXAMPLE_PATH"] = tmp_path
    wavenumber = 1e6 / np.array([5.34, 5.5112])
    posterior = SimpleNamespace(listpostdictelem=[
        [{"elin": wavenumber}], [{"elin": wavenumber[:1]}],
    ])
    paths = namespace["render_line_histogram_frames"](posterior)
    assert len(paths) == 2
    assert all(path.parent == tmp_path / "visuals" for path in paths)
    assert all(path.is_file() for path in paths)


def test_separate_example_loads_keep_independent_output_roots(tmp_path):
    import pickle

    relative = "jwst_miri_ngc7027_line_catalog/jwst_miri_ngc7027_line_catalog.py"
    first = demo.load_example_namespace(relative)
    second = demo.load_example_namespace(relative)
    first["EXAMPLE_PATH"] = tmp_path / "first"
    second["EXAMPLE_PATH"] = tmp_path / "second"
    assert first["render_line_histogram_frames"].__globals__["EXAMPLE_PATH"] == tmp_path / "first"
    assert second["render_line_histogram_frames"].__globals__["EXAMPLE_PATH"] == tmp_path / "second"
    assert pickle.loads(pickle.dumps(first["estimate_continuum"])) is first["estimate_continuum"]
    assert pickle.loads(pickle.dumps(second["estimate_continuum"])) is second["estimate_continuum"]


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
@pytest.mark.parametrize("encoding", ["bool", "numeric"])
def test_proposal_profiling_reads_the_requested_chain_and_timing_units(tmp_path, monkeypatch, native_layout, encoding):
    import h5py

    script = REPOSITORY_ROOT / "examples" / "proposal_profiling" / "proposal_profiling.py"
    example = load_example_module("profiling_replay", script)
    monkeypatch.setattr(example, "EXAMPLE_PATH", tmp_path)
    root = tmp_path / "pcat_runs" / "selected" if native_layout else tmp_path
    cache = root / "data" / "outp" / "selected"
    cache.mkdir(parents=True)
    with h5py.File(cache / "gdatmodi0000post.h5", "w") as output:
        output["listpostindxproptype"] = [0, 1, 2] if encoding == "bool" else [0.0, 1.0, 2.0]
        output["listpostboolpropaccp"] = [True, False, True] if encoding == "bool" else [1, 0, 1]
        output["listpostboolpropfilt"] = [True, True, True] if encoding == "bool" else [1.0, 1.0, 1.0]
        output["listpostchrototl"] = [0.001, 0.002, 0.003]  # [s]
        output["listpostchroprop"] = [0.0001, 0.0002, 0.0003]  # [s]
    if native_layout:
        legacy_cache = tmp_path / "data" / "outp" / "selected"
        legacy_cache.mkdir(parents=True)
        with h5py.File(legacy_cache / "gdatmodi0000post.h5", "w") as output:
            output["listpostindxproptype"] = [0, 1, 2]
            output["listpostboolpropaccp"] = [False, False, False]
            output["listpostboolpropfilt"] = [True, True, True]
            output["listpostchrototl"] = [0.010, 0.020, 0.030]  # [s]
    chain = example.read_chain("selected")
    np.testing.assert_array_equal(chain["type"], [0, 1, 2])
    np.testing.assert_array_equal(chain["accp"], [True, False, True])
    np.testing.assert_array_equal(chain["filt"], [True, True, True])
    np.testing.assert_allclose(chain["time"], [1.0, 2.0, 3.0])  # [ms]
    np.testing.assert_allclose(chain["phase"]["prop"], [0.1, 0.2, 0.3])  # [ms]


@pytest.mark.parametrize("name", ["", ".", "..", "nested/run", "../other"])
def test_proposal_profiling_reader_requires_a_single_run_name(tmp_path, monkeypatch, name):
    script = REPOSITORY_ROOT / "examples" / "proposal_profiling" / "proposal_profiling.py"
    example = load_example_module("profiling_reader_run_names", script)
    monkeypatch.setattr(example, "EXAMPLE_PATH", tmp_path)
    with pytest.raises(ValueError, match="Invalid run tag"):
        example.read_chain(name)


@pytest.mark.parametrize("native_kind", ["directory", "broken_symlink"])
def test_proposal_profiling_does_not_fall_back_from_incomplete_native_run(tmp_path, monkeypatch, native_kind):
    import h5py

    script = REPOSITORY_ROOT / "examples" / "proposal_profiling" / "proposal_profiling.py"
    example = load_example_module("profiling_incomplete_native_run", script)
    monkeypatch.setattr(example, "EXAMPLE_PATH", tmp_path)
    native_root = tmp_path / "pcat_runs" / "selected"
    native_root.parent.mkdir(parents=True)
    if native_kind == "directory":
        native_root.mkdir()
    else:
        native_root.symlink_to("missing_run", target_is_directory=True)
    legacy_cache = tmp_path / "data" / "outp" / "selected"
    legacy_cache.mkdir(parents=True)
    path = legacy_cache / "gdatmodi0000post.h5"
    with h5py.File(path, "w") as output:
        output["listpostindxproptype"] = [0, 1, 2]
        output["listpostboolpropaccp"] = [True, False, True]
        output["listpostboolpropfilt"] = [True, True, True]
        output["listpostchrototl"] = [0.001, 0.002, 0.003]  # [s]
    original = path.read_bytes()
    with pytest.raises(FileNotFoundError):
        example.read_chain("selected")
    assert path.read_bytes() == original


def test_shared_posterior_loader_does_not_fall_back_from_broken_native_symlink(tmp_path, monkeypatch):
    from pcat.demo import load_example_posterior
    import pcat.main

    native_root = tmp_path / "pcat_runs" / "selected"
    native_root.parent.mkdir()
    native_root.symlink_to("missing_run", target_is_directory=True)
    legacy_path = tmp_path / "data" / "outp" / "selected" / "gdatfinlpost"
    legacy_path.parent.mkdir(parents=True)
    print(f"Writing to {legacy_path}...")
    legacy_path.write_bytes(b"legacy artifact")

    def read_existing(path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        return path

    monkeypatch.setattr(pcat.main, "readfile", read_existing)
    with pytest.raises(FileNotFoundError):
        load_example_posterior("example/data/outp/selected/gdatfinlpost", output_root=tmp_path)
    print(f"Reading from {legacy_path}...")
    assert legacy_path.read_bytes() == b"legacy artifact"


@pytest.mark.parametrize("dataset,values", [
    ("listpostindxproptype", [0.0, 0.5, 2.0]),
    ("listpostindxproptype", [0, 6, 2]),
    ("listpostboolpropaccp", [0, 2, 1]),
    ("listpostboolpropaccp", [0.0, np.nan, 1.0]),
    ("listpostboolpropfilt", np.array([b"0", b"False", b"1"])),
    ("listpostboolpropfilt", [True, False]),
    ("listpostchrototl", [0.001, np.nan, 0.003]),  # [s]
    ("listpostchrototl", [0.001, -0.002, 0.003]),  # [s]
])
def test_proposal_profiling_rejects_invalid_saved_chain_metadata(tmp_path, monkeypatch, dataset, values):
    import h5py

    script = REPOSITORY_ROOT / "examples" / "proposal_profiling" / "proposal_profiling.py"
    example = load_example_module("profiling_invalid_metadata", script)
    monkeypatch.setattr(example, "EXAMPLE_PATH", tmp_path)
    cache = tmp_path / "data" / "outp" / "selected"
    cache.mkdir(parents=True)
    path = cache / "gdatmodi0000post.h5"
    with h5py.File(path, "w") as output:
        output["listpostindxproptype"] = [0, 1, 2]
        output["listpostboolpropaccp"] = [True, False, True]
        output["listpostboolpropfilt"] = [True, True, True]
        output["listpostchrototl"] = [0.001, 0.002, 0.003]  # [s]
        del output[dataset]
        output[dataset] = values
    original = path.read_bytes()
    with pytest.raises(ValueError, match="proposal|flag|length|shape|time"):
        example.read_chain("selected")
    assert path.read_bytes() == original


@pytest.mark.parametrize("times, expected_cost", [
    ([1.0, 3.0, 2.0, 2.0], [0.5, 0.5]),
    ([1e308, 1e308, 5e307, 5e307], [2.0 / 3.0, 1.0 / 3.0]),
])
def test_proposal_profiling_plot_shows_per_type_acceptance_ratio(tmp_path, monkeypatch, times, expected_cost):
    import matplotlib.pyplot as plt

    script = REPOSITORY_ROOT / "examples" / "proposal_profiling" / "proposal_profiling.py"
    example = load_example_module("profiling_acceptance", script)
    monkeypatch.setattr(example, "EXAMPLE_PATH", tmp_path)
    captured = {}

    def capture_figure(figure, example_root, name, typefileplot):
        captured.update(figure=figure, name=name)

    monkeypatch.setattr(example, "save_example_figure", capture_figure)
    chain = {
        "type": np.array([0, 0, 1, 1]),
        "accp": np.array([True, False, True, True]),
        "filt": np.array([True, True, False, True]),
        "time": np.array(times),  # [ms]
    }

    with np.errstate(over="raise", invalid="raise"):
        example.plot_acceptance_and_cost(chain, "png")

    axis = captured["figure"].axes[0]
    assert captured["name"] == "proposal_acceptance_and_cost"
    assert axis.containers[1].get_label() == "Acceptance ratio"
    np.testing.assert_allclose(
        [bar.get_height() for bar in axis.containers[1]], [0.5, 1.0]
    )
    np.testing.assert_allclose([bar.get_height() for bar in axis.containers[2]], expected_cost)
    plt.close(captured["figure"])


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


@pytest.mark.parametrize(("arguments", "smoke"), [([], False), (["--smoke"], True)])
def test_example_cli_uses_shared_smoke_option(monkeypatch, arguments, smoke):
    captured = []
    monkeypatch.setattr(demo.sys, "argv", ["example", *arguments])

    demo.run_example_cli(lambda **options: captured.append(options), description="Example")

    assert captured == [{"smoke": smoke}]


def test_add_smoke_argument_composes_with_custom_options():
    import argparse

    parser = demo.add_smoke_argument(
        argparse.ArgumentParser(), help="Run a small pipeline check."
    )
    parser.add_argument("--steps", type=int, default=10)
    arguments = parser.parse_args(["--smoke", "--steps", "3"])

    assert arguments.smoke is True
    assert arguments.steps == 3
    assert "Run a small pipeline check." in parser.format_help()


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
            {
                "typeexpr": "gmix",
                "typeelem": ["clusvari"],
                "numbspatdims": 2,
                "boolcondcatl": True,
                "strgcnfg": "gaussian_mixture_catalog_comparison",
            },
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
    comparisons = []
    if example_name == "gaussian_mixture_catalog":
        monkeypatch.setattr(
            module,
            "make_catalog_comparison",
            lambda output_root, comparison_run_name: comparisons.append(
                (output_root, comparison_run_name)
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
        assert comparisons == [
            (expected_output_root, "gaussian_mixture_catalog_comparison")
        ]


@pytest.mark.parametrize("empty_best", [False, True])
def test_gaussian_mixture_comparison_displays_completed_empty_catalogs(tmp_path, monkeypatch, empty_best):
    from types import SimpleNamespace
    import matplotlib.pyplot as plt

    script_path = REPOSITORY_ROOT / "examples" / "gaussian_mixture_catalog" / "gaussian_mixture_catalog.py"
    module = load_example_module("pcat_gaussian_empty_comparison", script_path)
    best = {"xpos": np.array([] if empty_best else [0.1]),
            "ypos": np.array([] if empty_best else [0.2])}
    state = SimpleNamespace(
        listpostlpostotl=np.array([-1.0]), listpostdictelem=[[best]],
        dictglob={"poststkscond": []}, prvl=np.array([]), numbstkscond=0,
    )
    figures = []
    original_close = plt.close

    def capture_close(figure=None):
        if hasattr(figure, "axes"):
            figures.append(figure)
        original_close(figure)

    monkeypatch.setattr(plt, "close", capture_close)
    result = module.plot_most_likely_vs_condensed(state, tmp_path / "empty_comparison.png")
    assert result["best_positions"].shape == (0 if empty_best else 1, 2)
    assert result["condensed_positions"].shape == (0, 2)
    assert result["association_frequencies"].shape == (0,)
    assert result["output_path"].is_file()
    assert len(figures[-1].axes) == 2
    for axis, count in zip(figures[-1].axes, (0 if empty_best else 1, 0)):
        assert len(axis.collections[0].get_offsets()) == count
        if count == 0:
            assert any(text.get_text() == "No catalog elements" for text in axis.texts)


@pytest.mark.parametrize("location", ["best", "condensed"])
@pytest.mark.parametrize("coordinate", ["xpos", "ypos"])
@pytest.mark.parametrize("kind", ["masked", "complex", "nonfinite"])
def test_gaussian_mixture_comparison_rejects_unavailable_positions_before_figure(tmp_path, monkeypatch, location, coordinate, kind):
    from types import SimpleNamespace
    import matplotlib.pyplot as plt

    script_path = REPOSITORY_ROOT / "examples" / "gaussian_mixture_catalog" / "gaussian_mixture_catalog.py"
    module = load_example_module("pcat_gaussian_invalid_position", script_path)
    best = {"xpos": np.array([0.1]), "ypos": np.array([0.2])}
    condensed = {"xpos": np.array([0.1, 0.1, 0.1]), "ypos": np.array([0.2, 0.2, 0.2])}
    target = best if location == "best" else condensed
    values = target[coordinate]
    if kind == "masked":
        target[coordinate] = np.ma.array(values, mask=True)
    elif kind == "complex":
        target[coordinate] = values.astype(complex) + 0.1j
    else:
        target[coordinate] = np.full(values.shape, np.nan)
    state = SimpleNamespace(
        listpostlpostotl=np.array([-1.0]), listpostdictelem=[[best]],
        dictglob={"poststkscond": [condensed]}, prvl=np.array([0.8]),
    )

    def unexpected_figure(*args, **kwargs):
        pytest.fail("Invalid positions reached figure creation")

    monkeypatch.setattr(plt, "subplots", unexpected_figure)
    output_path = tmp_path / "invalid_comparison.png"
    with pytest.raises(ValueError, match="position|coordinate"):
        module.plot_most_likely_vs_condensed(state, output_path)
    assert not output_path.exists()


def test_gaussian_mixture_comparison_uses_available_medians_with_missing_uncertainty_bounds(tmp_path):
    from types import SimpleNamespace

    script_path = REPOSITORY_ROOT / "examples" / "gaussian_mixture_catalog" / "gaussian_mixture_catalog.py"
    module = load_example_module("pcat_gaussian_available_position_medians", script_path)
    coordinates = np.ma.array([0.1, 0.0, 0.0], mask=[False, True, True])
    original_data = coordinates.data.copy()
    original_mask = coordinates.mask.copy()
    state = SimpleNamespace(
        listpostlpostotl=np.array([-1.0]),
        listpostdictelem=[[{"xpos": np.array([0.1]), "ypos": np.array([0.2])}]],
        dictglob={"poststkscond": [{"xpos": coordinates, "ypos": np.array([0.2, np.nan, np.nan])}]},
        prvl=np.array([0.8]),
    )
    result = module.plot_most_likely_vs_condensed(state, tmp_path / "comparison.png")
    np.testing.assert_array_equal(result["condensed_positions"], [[0.1, 0.2]])
    np.testing.assert_array_equal(coordinates.data, original_data)
    np.testing.assert_array_equal(coordinates.mask, original_mask)


@pytest.mark.parametrize("probabilities, expected_index", [
    (np.ma.array([-3.0, 99.0, -1.0], mask=[False, True, False]), 2),
    (np.ma.array([-3.0, -1.0, -2.0], mask=False), 1),
    (np.array([-3.0, np.nan, -1.0]), 2),
    (np.array([-3.0, np.inf, -1.0]), 2),
])
def test_gaussian_mixture_comparison_excludes_masked_log_probabilities(tmp_path, probabilities, expected_index):
    from types import SimpleNamespace

    script_path = REPOSITORY_ROOT / "examples" / "gaussian_mixture_catalog" / "gaussian_mixture_catalog.py"
    module = load_example_module("pcat_gaussian_masked_log_probability", script_path)
    original_data = np.ma.getdata(probabilities).copy()
    original_mask = np.ma.getmaskarray(probabilities).copy()
    state = SimpleNamespace(
        listpostlpostotl=probabilities,
        listpostdictelem=[
            [{"xpos": np.array([0.0]), "ypos": np.array([0.0])}],
            [{"xpos": np.array([9.0]), "ypos": np.array([9.0])}],
            [{"xpos": np.array([0.1]), "ypos": np.array([0.2])}],
        ],
        dictglob={"poststkscond": [{"xpos": np.array([0.1, 0.1, 0.1]), "ypos": np.array([0.2, 0.2, 0.2])}]},
        prvl=np.array([0.8]),
    )
    result = module.plot_most_likely_vs_condensed(state, tmp_path / "comparison.png")
    assert result["best_sample_index"] == expected_index
    best = state.listpostdictelem[expected_index][0]
    np.testing.assert_array_equal(result["best_positions"], np.column_stack((best["xpos"], best["ypos"])))
    np.testing.assert_array_equal(np.ma.getdata(probabilities), original_data)
    np.testing.assert_array_equal(np.ma.getmaskarray(probabilities), original_mask)


@pytest.mark.parametrize("probabilities", [
    np.ma.array([-1.0], mask=[True]), np.array([np.nan]), np.array([-1.0 + 0.1j]),
])
def test_gaussian_mixture_comparison_rejects_unusable_log_probabilities(tmp_path, probabilities):
    from types import SimpleNamespace

    script_path = REPOSITORY_ROOT / "examples" / "gaussian_mixture_catalog" / "gaussian_mixture_catalog.py"
    module = load_example_module("pcat_gaussian_unusable_log_probability", script_path)
    state = SimpleNamespace(
        listpostlpostotl=probabilities,
        listpostdictelem=[[{"xpos": np.array([0.1]), "ypos": np.array([0.2])}]],
        dictglob={"poststkscond": [{"xpos": np.array([0.1, 0.1, 0.1]), "ypos": np.array([0.2, 0.2, 0.2])}]},
        prvl=np.array([0.8]),
    )
    output_path = tmp_path / "invalid_comparison.png"
    with pytest.raises(ValueError, match="finite|real"):
        module.plot_most_likely_vs_condensed(state, output_path)
    assert not output_path.exists()


@pytest.mark.parametrize("frequencies", [
    np.array([-0.1]), np.array([1.1]), np.array([np.nan]), np.array([np.inf]),
    np.ma.array([0.8], mask=[True]), np.array([0.8], dtype=complex),
])
@pytest.mark.parametrize("existing_output", [False, True])
def test_gaussian_mixture_comparison_rejects_invalid_association_fraction(tmp_path, frequencies, existing_output):
    from types import SimpleNamespace

    script_path = REPOSITORY_ROOT / "examples" / "gaussian_mixture_catalog" / "gaussian_mixture_catalog.py"
    module = load_example_module("pcat_gaussian_invalid_frequency", script_path)
    state = SimpleNamespace(
        listpostlpostotl=np.array([-1.0]),
        listpostdictelem=[[{"xpos": np.array([0.1]), "ypos": np.array([0.2])}]],
        dictglob={"poststkscond": [{"xpos": np.array([0.1, 0.1, 0.1]), "ypos": np.array([0.2, 0.2, 0.2])}]},
        prvl=frequencies,
    )
    output_path = tmp_path / "invalid_comparison.png"
    original = b"existing comparison artifact"
    if existing_output:
        print(f"Writing to {output_path}...")
        output_path.write_bytes(original)
    with pytest.raises(ValueError, match="frequenc|fraction|zero|one"):
        module.plot_most_likely_vs_condensed(state, output_path)
    if existing_output:
        print(f"Reading from {output_path}...")
        assert output_path.read_bytes() == original
    else:
        assert not output_path.exists()


@pytest.mark.parametrize("frequencies", [[0.8, 0.55], [0.0, 1.0]])
def test_gaussian_mixture_catalog_comparison_uses_best_sample_and_condensed_positions(tmp_path, frequencies):
    from types import SimpleNamespace

    script_path = (
        REPOSITORY_ROOT
        / "examples"
        / "gaussian_mixture_catalog"
        / "gaussian_mixture_catalog.py"
    )
    module = load_example_module("pcat_gaussian_catalog_comparison", script_path)
    state = SimpleNamespace(
        listpostlpostotl=np.array([[-3.0], [-1.0], [-2.0]]),
        listpostdictelem=[
            [{"xpos": np.array([0.0]), "ypos": np.array([0.0])}],
            [{"xpos": np.array([0.2, 0.4]), "ypos": np.array([0.1, 0.3])}],
            [{"xpos": np.array([0.1]), "ypos": np.array([0.2])}],
        ],
        dictglob={"poststkscond": [
            {"xpos": np.array([0.25, 0.2, 0.3]), "ypos": np.array([0.2, 0.1, 0.3])},
            {"xpos": np.array([0.45, 0.4, 0.5]), "ypos": np.array([0.35, 0.3, 0.4])},
        ]},
        prvl=np.array(frequencies),
    )

    result = module.plot_most_likely_vs_condensed(
        state, tmp_path / "most_likely_vs_condensed_catalog.png",
    )

    assert result["best_sample_index"] == 1
    np.testing.assert_array_equal(result["best_positions"], [[0.2, 0.1], [0.4, 0.3]])
    np.testing.assert_array_equal(result["condensed_positions"], [[0.25, 0.2], [0.45, 0.35]])
    np.testing.assert_array_equal(result["association_frequencies"], frequencies)
    assert result["output_path"].is_file()
    with Image.open(result["output_path"]) as image:
        assert image.width > 400 and image.height > 300


def test_gaussian_mixture_comparison_reuses_completed_empty_summary(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pcat.main

    script_path = REPOSITORY_ROOT / "examples" / "gaussian_mixture_catalog" / "gaussian_mixture_catalog.py"
    module = load_example_module("pcat_gaussian_completed_empty_summary", script_path)
    state = SimpleNamespace(
        listpostlpostotl=np.array([-1.0]),
        listpostdictelem=[[{"xpos": np.array([]), "ypos": np.array([])}]],
        dictglob={"poststkscond": []}, prvl=np.array([]), numbstkscond=0,
    )
    monkeypatch.setattr(pcat.main, "readfile", lambda _: state)

    def unexpected_condensation(*args, **kwargs):
        pytest.fail("Completed empty summary was recomputed")

    monkeypatch.setattr(pcat.main, "retr_condcatl", unexpected_condensation)
    result = module.make_catalog_comparison(tmp_path, "completed_empty_run")
    assert result["condensed_positions"].shape == (0, 2)
    assert result["output_path"].is_file()


@pytest.mark.parametrize("summary_initialized", [False, True])
def test_gaussian_mixture_comparison_condenses_legacy_saved_state(tmp_path, monkeypatch, summary_initialized):
    from types import SimpleNamespace
    import pcat.main

    script_path = (
        REPOSITORY_ROOT
        / "examples"
        / "gaussian_mixture_catalog"
        / "gaussian_mixture_catalog.py"
    )
    module = load_example_module("pcat_gaussian_catalog_saved_comparison", script_path)
    state = SimpleNamespace(
        listpostlpostotl=np.array([[-2.0]]),
        listpostdictelem=[[{"xpos": np.array([0.1]), "ypos": np.array([0.2])}]],
        dictglob={"poststkscond": []} if summary_initialized else {},
    )
    condensation_calls = []

    def condense(posterior):
        condensation_calls.append(posterior)
        posterior.dictglob["poststkscond"] = [{
            "xpos": np.array([0.1, 0.1, 0.1]),
            "ypos": np.array([0.2, 0.2, 0.2]),
        }]
        posterior.prvl = np.array([1.0])

    monkeypatch.setattr(pcat.main, "readfile", lambda _: state)
    monkeypatch.setattr(pcat.main, "retr_condcatl", condense)

    result = module.make_catalog_comparison(tmp_path, "legacy_run")

    assert condensation_calls == [state]
    assert result["best_sample_index"] == 0
    assert result["output_path"].is_file()


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


def test_daylan2017_frames_sum_real_counts_and_preserve_earlier_files(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from pcat import plotting

    path = REPOSITORY_ROOT / "examples/daylan+2017_fermi_point_sources/daylan+2017_fermi_point_sources.py"
    module = load_example_module("pcat_fermi_frame_check", path)
    module.OUTPUT_ROOT = tmp_path
    directory = tmp_path / "visuals"
    directory.mkdir()
    old = directory / "fermi_ngpc_sources_swep000000001.png"
    old.write_bytes(b"preserved older frame")
    counts = np.arange(24, dtype=float).reshape(3, 4, 2) + 20
    original = counts.copy()
    sources = {"flux": np.array([1.0, 2.0]), "xpos": np.array([0.01, -0.02]),
               "ypos": np.array([0.02, 0.01])}
    snapshots = [{"cntrswep": 1, "dictelem": [sources]},
                 {"cntrswep": 2, "dictelem": [{**sources, "xpos": sources["xpos"] + 0.05}]}]
    monkeypatch.setattr(plotting, "animation_states", lambda *args, **kwargs: snapshots)
    monkeypatch.setattr(plotting, "animation_phase_label", lambda snapshot, **kwargs: "Posterior sample")
    import matplotlib.pyplot as plt
    save = plt.Figure.savefig
    scales = []

    def check_and_save(figure, destination, **kwargs):
        image = figure.axes[0].images[0]
        np.testing.assert_array_equal(image.get_array(), np.arcsinh(counts.sum(axis=(0, 2)).reshape(2, 2)).T)
        scales.append(image.get_clim())
        return save(figure, destination, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", check_and_save)
    paths = module.render_source_frames(SimpleNamespace(cntpdata=counts), provenance={"kind": "test fixture"})
    assert len(paths) == 2 and all(path.is_file() for path in paths)
    assert scales[0] == scales[1]
    assert scales[0] == (np.arcsinh(counts.sum(axis=(0, 2)).min()), np.arcsinh(counts.sum(axis=(0, 2)).max()))
    assert old.read_bytes() == b"preserved older frame"
    np.testing.assert_array_equal(counts, original)
    import hashlib
    import json
    manifest = json.loads(next(directory.glob("fermi_ngpc_frames_*.json")).read_text())
    assert manifest["native_frame_paths"] == [path.name for path in paths]
    assert manifest["native_frame_sha256"] == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert manifest["count_cube_shape"] == [3, 4, 2]
    assert manifest["projected_count_max"] == counts.sum(axis=(0, 2)).max()
    assert manifest["native_run_provenance"] == {"kind": "test fixture"}
    panel = plotting.PosteriorAnimationPanel(
        "Fermi fixture", "visuals/fermi_ngpc_sources_*_swep*.png",
        manifest_pattern="visuals/fermi_ngpc_frames_*.json", manifest_key="native")
    assert plotting._animation_frame_paths(panel, tmp_path) == paths


def test_daylan2017_reproduction_uses_new_run_names_without_deleting_cache(tmp_path, monkeypatch):
    from pcat import sampling
    path = REPOSITORY_ROOT / "examples/daylan+2017_fermi_point_sources/daylan+2017_fermi_point_sources.py"
    module = load_example_module("pcat_fermi_fresh_runs", path)
    module.OUTPUT_ROOT = tmp_path
    old = tmp_path / "earlier_posterior"
    old.write_bytes(b"preserved scientific result")
    calls = []
    monkeypatch.setattr(sampling, "sample", lambda **kwargs: calls.append(kwargs) or kwargs)
    module.run_reproduction(smoke=True)
    module.run_reproduction(smoke=True)
    assert calls[0]["strgcnfg"] != calls[1]["strgcnfg"]
    assert all(call["strgcnfg"].startswith(module.RUN_NAME + "_") for call in calls)
    assert old.read_bytes() == b"preserved scientific result"


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


def test_proposal_example_preserves_existing_run_before_sampling(tmp_path, monkeypatch):
    from pcat.main import retr_pathrun

    script = REPOSITORY_ROOT / "examples/proposal_state_animation/proposal_state_animation.py"
    module = load_example_module("pcat_preserved_proposal_example", script)
    monkeypatch.setattr(module, "EXAMPLE_PATH", tmp_path)
    tag = "20261025_120000"
    root = Path(retr_pathrun(tmp_path, f"{module.RUN_NAME}_{tag}"))
    cache = root / "data/outp"
    cache.mkdir(parents=True)
    marker = cache / "keep.txt"
    marker.write_text("existing chain")
    with pytest.raises(FileExistsError):
        module.run_example(120, time_tag=tag)
    assert marker.read_text() == "existing chain"


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
