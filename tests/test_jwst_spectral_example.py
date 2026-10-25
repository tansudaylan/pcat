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
    (tmp_path / "visuals").mkdir()
    for name in ("plot_spectrum_fit", "plot_line_catalog", "plot_line_count"):
        namespace[name] = lambda *arguments: tmp_path / "visuals/figure.png"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--skip-sampling", "--time-tag", "20261025_120000"])
    namespace["main"]()
    assert source.read_bytes() == b"unchanged cached spectrum"
    assert not (namespace["PATH_INPT"] / "sbrt.fits").exists()
    metadata = json.loads((tmp_path / "visuals/jwst_miri_ngc7027_run_20261025_120000.json").read_text())
    assert metadata["sweeps_per_chain"] == 1200 and metadata["chains"] == 2