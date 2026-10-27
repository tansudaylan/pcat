import runpy
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (Path(__file__).resolve().parents[1] / "examples" / "jwst_miri_ngc7027_line_catalog"
          / "jwst_miri_ngc7027_line_catalog.py")


@pytest.mark.parametrize("legacy", [False, True])
def test_jwst_sampling_preserves_existing_native_and_legacy_outputs(tmp_path, legacy):
    example = runpy.run_path(str(SCRIPT))
    run = example["run_pcat"]
    run.__globals__["EXAMPLE_PATH"] = tmp_path
    run.__globals__["RUN_NAME"] = "jwst_test"
    root = Path(example["retr_pathrun"](tmp_path, "jwst_test"))
    cache = tmp_path / "data/outp/jwst_test" if legacy else root / "data/outp"
    cache.mkdir(parents=True)
    marker = cache / "preserve.txt"
    marker.write_text("existing posterior")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run(np.array([1.0, 2.0]), np.ones((1, 1, 1)), 300)
    assert marker.read_text() == "existing posterior"


def test_jwst_main_rejects_existing_run_before_loading_or_writing_inputs(tmp_path, monkeypatch):
    import sys

    example = runpy.run_path(str(SCRIPT))
    namespace = example["main"].__globals__
    namespace["EXAMPLE_PATH"] = tmp_path
    namespace["read_spectrum"] = lambda: pytest.fail("must refuse before accessing input data")
    root = Path(example["retr_pathrun"](tmp_path, "existing"))
    (root / "data/outp").mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--run-name", "existing"])
    with pytest.raises(FileExistsError):
        namespace["main"]()


def test_jwst_plot_refuses_to_overwrite_existing_file(tmp_path):
    from matplotlib.figure import Figure

    example = runpy.run_path(str(SCRIPT))
    example["save_spectral_figure"].__globals__["EXAMPLE_PATH"] = tmp_path
    figure = Figure()
    path = example["save_spectral_figure"](figure, "jwst_test", "png", "20261025_120000")
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        example["save_spectral_figure"](figure, "jwst_test", "png", "20261025_120000")
    assert path.read_bytes() == before


def test_jwst_replot_never_rewrites_sampling_inputs(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    example = runpy.run_path(str(SCRIPT))
    namespace = example["main"].__globals__
    namespace["EXAMPLE_PATH"] = tmp_path
    namespace["PATH_INPT"] = tmp_path / "data/inpt"
    namespace["PATH_INPT"].mkdir(parents=True)
    source = namespace["PATH_INPT"] / namespace["NAME_X1D"]
    source.write_bytes(b"unchanged cached spectrum")
    namespace["read_spectrum"] = lambda: (np.array([5.3, 5.4]), np.ones(2), np.ones(2))
    namespace["estimate_continuum"] = lambda *arguments: np.ones(2)
    namespace["write_pcat_inputs"] = lambda *arguments: pytest.fail("replot must not write sampling inputs")
    namespace["run_pcat"] = lambda *arguments, **options: pytest.fail("replot must not sample")
    namespace["load_example_posterior"] = lambda *arguments: SimpleNamespace(
        listpostdictelem=[{}], numbswep=1200, numbproc=2,
    )
    namespace["configure_example_style"] = lambda *arguments: "black"
    namespace["render_line_histogram_frames"] = lambda *arguments: []
    namespace["render_spectrum_fit_frames"] = lambda *arguments: []
    (tmp_path / "visuals").mkdir()
    for name in ("plot_spectrum_fit", "plot_line_catalog", "plot_line_count"):
        namespace[name] = lambda *arguments: tmp_path / "visuals/figure.png"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--skip-sampling", "--time-tag", "20261025_120000"])
    namespace["main"]()
    assert source.read_bytes() == b"unchanged cached spectrum"
    assert not (namespace["PATH_INPT"] / "sbrt.fits").exists()
    metadata = json.loads((tmp_path / "visuals/jwst_miri_ngc7027_run_20261025_120000.json").read_text())
    assert metadata["sweeps_per_chain"] == 1200 and metadata["chains"] == 2


@pytest.mark.parametrize("container", [np.asarray, list, np.ma.asarray,
                                       lambda values: np.asarray(values, dtype=str)])
def test_spectral_frames_show_real_bins_and_exact_model_draws_with_fixed_limits(tmp_path, monkeypatch, container):
    from types import SimpleNamespace
    from matplotlib.figure import Figure
    from pcat.demo import load_example_namespace

    example = load_example_namespace("jwst_miri_ngc7027_line_catalog/jwst_miri_ngc7027_line_catalog.py")
    example["EXAMPLE_PATH"] = tmp_path
    wavelength = np.array([5.31, 5.40, 5.52])
    flux = np.array([2.0, -8.0, 3.0])
    error = np.array([0.1, 0.2, 0.1])
    continuum = np.full(3, 2.0)
    sigma = np.sqrt(error**2 + (example["FRACTIONAL_ERROR_FLOOR"] * continuum)**2)
    models = np.array([[1.0, 7.0, 2.0], [3.0, 9.0, 4.0]])
    posterior = SimpleNamespace(listpostcntpmodl=(models * (continuum / sigma**2))[..., None, None][:, ::-1])
    model_snapshot = posterior.listpostcntpmodl.copy()
    inputs = [container(values) for values in (wavelength, flux, error, continuum)]
    snapshots = [np.asanyarray(values).copy() for values in inputs]
    figures = []
    original_save = Figure.savefig

    def capture(figure, *arguments, **options):
        figures.append(figure)
        return original_save(figure, *arguments, **options)

    monkeypatch.setattr(Figure, "savefig", capture)
    paths = example["render_spectrum_fit_frames"](*inputs, posterior)
    assert len(paths) == len(figures) == 2
    assert all(path.parent == tmp_path / "visuals" and path.is_file() for path in paths)
    for index, figure in enumerate(figures):
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), flux)
        np.testing.assert_allclose(figure.axes[0].lines[1].get_ydata(), models[index])
        np.testing.assert_allclose(figure.axes[1].lines[0].get_ydata(), (flux - models[index]) / sigma)
        assert figure.axes[0].lines[0].get_linestyle() == "None"
        assert figure.axes[0].lines[1].get_color() == "#176B87"
    assert figures[0].axes[0].get_ylim() == figures[1].axes[0].get_ylim()
    assert figures[0].axes[1].get_ylim() == figures[1].axes[1].get_ylim()
    for values, snapshot in zip(inputs, snapshots):
        np.testing.assert_array_equal(values, snapshot)
    np.testing.assert_array_equal(posterior.listpostcntpmodl, model_snapshot)


@pytest.mark.parametrize("field,value", [
    ("wavelength", [5.4, 5.3, 5.5]),
    ("wavelength", [5.3, 5.3, 5.5]),
    ("wavelength", [0.0, 5.3, 5.5]),
    ("wavelength", [5.3, np.nan, 5.5]),
    ("flux", [2.0, np.inf, 3.0]),
    ("flux", [2.0, 3.0]),
    ("flux", [[2.0, 3.0, 4.0]]),
    ("flux", [2.0, "bad", 3.0]),
    ("flux", [2.0, 3.0j, 4.0]),
    ("error", [0.1, 0.0, 0.1]),
    ("error", [0.1, -0.2, 0.1]),
    ("continuum", [2.0, 0.0, 2.0]),
    ("continuum", [2.0, -1.0, 2.0]),
    ("flux", np.ma.array([2.0, 8.0, 3.0], mask=[False, True, False])),
    ("models", np.ma.array(np.ones((2, 3, 1, 1)), mask=True)),
    ("models", np.ones((2, 3, 1, 1), dtype=complex)),
    ("models", np.ones((2, 3))),
    ("models", np.ones((2, 4, 1, 1))),
    ("models", np.ones((1, 3, 1, 1))),
    ("models", np.full((2, 3, 1, 1), np.nan)),
])
def test_spectral_animation_rejects_invalid_inputs_before_creating_outputs(tmp_path, field, value):
    from types import SimpleNamespace
    from pcat.demo import load_example_namespace

    example = load_example_namespace("jwst_miri_ngc7027_line_catalog/jwst_miri_ngc7027_line_catalog.py")
    example["EXAMPLE_PATH"] = tmp_path
    inputs = dict(wavelength=np.array([5.31, 5.40, 5.52]), flux=np.array([2.0, 8.0, 3.0]),
                  error=np.full(3, 0.1), continuum=np.full(3, 2.0), models=np.ones((2, 3, 1, 1)))
    inputs[field] = value
    posterior = SimpleNamespace(listpostcntpmodl=inputs.pop("models"))
    with pytest.raises(ValueError, match="Spectral"):
        example["render_spectrum_fit_frames"](**inputs, posterior=posterior)
    assert not list(tmp_path.iterdir())