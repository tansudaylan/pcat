import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (Path(__file__).resolve().parents[1] / "examples" / "mismodeling_flare_catalog"
          / "mismodeling_flare_catalog.py")
SPECIFICATION = importlib.util.spec_from_file_location("mismodeling_flare_catalog", SCRIPT)
example = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(example)


def test_each_light_curve_has_a_correct_and_a_misspecified_fit():
    light_curves = example.simulate_light_curves()
    assert light_curves["counts"]["A"].shape == light_curves["counts"]["B"].shape == light_curves["time"].shape
    # light curve B differs from A only through the rotational modulation of the quiescent level
    modulation = light_curves["baselines"]["modulated"] / light_curves["baselines"]["constant"] - 1.0
    np.testing.assert_allclose(np.abs(modulation).max(), example.MODULATION_AMPLITUDE, rtol=1e-3)
    for light_curve in ("A", "B"):
        fits = [row for row in example.FITS if row[1] == light_curve]
        assert len(fits) == 2
        assert sum("misspecified" in row[4] for row in fits) == 1
        assert {(row[2], row[3]) for row in fits} != {fits[0][2:4]}
    # run names differ from the folder name, so cache cleanup cannot delete the example folder
    assert all(row[0] != SCRIPT.parent.name for row in example.FITS)


def test_misspecified_profile_fit_samples_gaussian_flares(tmp_path, monkeypatch):
    monkeypatch.setattr(example, "EXAMPLE_PATH", tmp_path)
    monkeypatch.setattr(example, "NUMBER_CHAINS", 1)
    light_curves = example.simulate_light_curves()
    posterior = example.run_fit(light_curves, "profile_wrong_test", "A", "flargauss", "constant", numbswep=300)
    catalog = posterior.listpostdictelem[-1][0]
    assert "fwhm" in catalog and "scalrise" not in catalog
    number = np.asarray(posterior.listpostnumbelem).ravel()
    assert np.all((number >= 0) & (number <= example.MAXIMUM_NUMBER))
    model = np.asarray(posterior.listpostcntpmodl)[:, :, 0, 0]
    assert model.shape[1] == light_curves["time"].size and np.isfinite(model).all()
