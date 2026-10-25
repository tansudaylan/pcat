import importlib.util

import numpy as np
import pytest

from pcat.mejiro_lens import (  # noqa: E402
    LABELS, PARAMETERS, mejiro_lens_log_likelihood, render_mejiro_counts,
    run_mejiro_lens_inference, simulate_mejiro_exposure,
)
from pcat import mejiro_lens as mejiro_lens_module  # noqa: E402
from pcat.plotting import plot_lens_parameter_recovery  # noqa: E402


@pytest.fixture(scope="module")
def specification():
    pytest.importorskip("mejiro")
    pytest.importorskip("roman_technical_information")
    return simulate_mejiro_exposure()


@pytest.mark.parametrize("halo_redshift", [0.3, 0.7])
def test_native_mejiro_bridge_preserves_single_and_multiple_lens_planes(monkeypatch, tmp_path, halo_redshift):
    import builtins
    from copy import deepcopy
    from astropy.cosmology import Planck18
    from chalcedon import ImageModel, LensModel, LightModel, PixelGrid, PSF
    from chalcedon.imaging import make_grid_with_coordtransform

    geometry = make_grid_with_coordtransform(31, 0.08)
    pixel = {"nx": 31, "ny": 31, "ra_at_xy_0": geometry[2],
             "dec_at_xy_0": geometry[3], "transform_pix2angle": geometry[6]}
    specification = {
        "backend": "chalcedon", "z_lens": 0.3, "z_source": 1.5,
        "lens_redshift_list": [0.3, 0.3, halo_redshift], "psf_fwhm": 0.08,
        "kwargs_pixel": pixel, "kwargs_numerics": {"supersampling_factor": 2},
        "lens_model_list": ["SIE", "SHEAR", "TNFW"],
        "source_light_model_list": ["GAUSSIAN"], "lens_light_model_list": [],
        "kwargs_lens": [
            {"theta_E": 0.7, "e1": 0.08, "e2": -0.04, "center_x": 0, "center_y": 0},
            {"gamma1": 0.02, "gamma2": 0.01},
            {"Rs": 0.12, "alpha_Rs": 0.04, "r_trunc": 0.5,
             "center_x": 0.6, "center_y": -0.2},
        ],
        "kwargs_source": [{"amp": 1.0, "sigma": 0.06, "center_x": 0.04, "center_y": 0.02}],
        "kwargs_lens_light": [], "exposure_time": 146.0, "sky_counts": 3.0,
        "cosmology_yaml": Planck18.to_format("yaml"),
    }
    original = deepcopy(specification)
    parameters = np.array([specification[group][index][key]
                           for _, group, index, key, *_ in PARAMETERS])
    lens = LensModel(specification["lens_model_list"], multi_plane=halo_redshift != 0.3,
                     z_lens=0.3, z_source=1.5,
                     lens_redshift_list=specification["lens_redshift_list"], cosmo=Planck18)
    renderer = ImageModel(PixelGrid(**pixel), PSF("GAUSSIAN", fwhm=0.08), lens,
                          LightModel(["GAUSSIAN"]), LightModel([]),
                          kwargs_numerics=specification["kwargs_numerics"])
    expected = renderer.image(specification["kwargs_lens"], specification["kwargs_source"], []) * 146 + 3
    original_import = builtins.__import__

    def forbid_lenstronomy(name, *args, **kwargs):
        if name == "lenstronomy" or name.startswith("lenstronomy."):
            pytest.fail("Native Mejiro specifications must stay independent of Lenstronomy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_lenstronomy)
    actual = render_mejiro_counts(specification, parameters)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert np.isfinite(actual).all() and np.all(actual >= 3)
    for group in ("kwargs_lens", "kwargs_source", "kwargs_lens_light"):
        assert specification[group] == original[group]
    if halo_redshift != 0.3:
        from astropy.cosmology import Cosmology

        renderer_before = mejiro_lens_module._image_model(specification)
        with monkeypatch.context() as cached:
            cached.setattr(Cosmology, "from_format", lambda *args, **kwargs: pytest.fail(
                "Cached likelihood evaluations must reuse the restored cosmology"))
            np.testing.assert_allclose(render_mejiro_counts(specification, parameters), expected)
        specification["cosmology_yaml"] = Planck18.clone(Om0=0.4, name="comparison").to_format("yaml")
        assert mejiro_lens_module._image_model(specification) is not renderer_before
        specification.pop("cosmology_yaml")
        with pytest.raises(ValueError, match="cosmology_yaml"):
            render_mejiro_counts(specification, parameters)
        sampling_specification = deepcopy(original)
        sampling_specification["observed_counts"] = np.random.default_rng(2026).poisson(expected)
        sampling_specification["true_parameters"] = parameters
        posterior = run_mejiro_lens_inference(
            sampling_specification, tmp_path, "native_multiplane", numbswep=60,
            numbburn=20, numbsamp=20, numbproc=1, numbframanim=4)
        draws = np.asarray(posterior.listpostparagenrscalbase)
        assert draws.shape == (20, len(PARAMETERS))
        assert np.isfinite(draws).all()
        assert np.isfinite(posterior.listpostlliktotl).all()
        assert posterior.mejiro_specification["cosmology_yaml"] == original["cosmology_yaml"]


def test_pcat_model_reproduces_mejiro_image(specification):
    # the PCAT likelihood must evaluate exactly the image that mejiro rendered
    model = render_mejiro_counts(specification, specification["true_parameters"])
    np.testing.assert_allclose(model, specification["expected_counts"], rtol=1e-10)
    assert model.shape == specification["observed_counts"].shape
    assert specification["sky_counts"] > 0.0
    assert len(specification["true_parameters"]) == len(PARAMETERS) == len(LABELS)


def test_likelihood_prefers_injected_lens(specification):
    class State:
        mejiro_specification = specification

    truth = specification["true_parameters"]
    shifted = truth.copy()
    shifted[0] += 0.05  # [arcsec], Einstein radius
    assert mejiro_lens_log_likelihood(State, "fitt", truth) > mejiro_lens_log_likelihood(State, "fitt", shifted)


def test_likelihood_keeps_zero_expected_counts_exact(monkeypatch):
    class State:
        mejiro_specification = {"observed_counts": np.array([0.0, 1.0])}

    monkeypatch.setattr(
        mejiro_lens_module,
        "render_mejiro_counts",
        lambda specification, parameters: np.array([0.0, 0.0]),
    )

    result = mejiro_lens_log_likelihood(State, "fitt", np.array([]))

    assert result == -np.inf


def test_likelihood_rejects_negative_expected_counts(monkeypatch):
    class State:
        mejiro_specification = {"observed_counts": np.array([1.0])}

    monkeypatch.setattr(
        mejiro_lens_module,
        "render_mejiro_counts",
        lambda specification, parameters: np.array([-1.0]),
    )

    with pytest.raises(ValueError, match="nonnegative"):
        mejiro_lens_log_likelihood(State, "fitt", np.array([]))


def test_pcat_samples_mejiro_exposure(specification, tmp_path):
    posterior = run_mejiro_lens_inference(
        specification, tmp_path, "mejiro_test", numbswep=300, numbburn=100, numbsamp=100, numbproc=1,
        numbframanim=4,
    )
    draws = np.asarray(posterior.listpostparagenrscalbase)
    assert draws.shape == (100, len(PARAMETERS))
    minima = np.array([row[4] for row in PARAMETERS])
    maxima = np.array([row[5] for row in PARAMETERS])
    assert np.all((draws >= minima) & (draws <= maxima))
    assert np.isfinite(np.asarray(posterior.listpostlliktotl)).all()
    assert len(posterior.listanimstate) == 4
    path = plot_lens_parameter_recovery(tmp_path / "recovery", draws, specification["true_parameters"],
                                        labels=LABELS)
    assert path.is_file()


def test_example_configuration_uses_mejiro_pipeline(monkeypatch, tmp_path):
    pytest.importorskip("mejiro")
    pytest.importorskip("roman_technical_information")
    script = __import__("pathlib").Path(__file__).resolve().parents[1] / "examples" / \
        "mejiro_roman_strong_lens" / "mejiro_roman_strong_lens.py"
    module_specification = importlib.util.spec_from_file_location("mejiro_example", script)
    example = importlib.util.module_from_spec(module_specification)
    module_specification.loader.exec_module(example)
    captured = {}

    def fake_inference(specification, output_root, run_name, **configuration):
        captured.update(configuration, run_name=run_name)
        raise RuntimeError("stop after configuration")

    monkeypatch.setattr(example, "run_mejiro_lens_inference", fake_inference)
    # keep cache cleanup away from the example's own saved run
    monkeypatch.setattr(example, "EXAMPLE_PATH", tmp_path)
    with pytest.raises(RuntimeError, match="stop after configuration"):
        example.run_example()
    assert captured["numbproc"] == example.NUMBER_CHAINS == 4
    assert captured["numbswep"] == 60000 and captured["numbburn"] == 40000
    assert captured["numbframanim"] == 24
    # a run name equal to the folder name would place the run in, and let cleanup delete, the example folder
    assert captured["run_name"] != script.parent.name
