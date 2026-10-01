"""Strong-lens inference with PCAT on images simulated by mejiro.

mejiro (https://github.com/AstroMusers/mejiro) describes a strong-lens system and an
instrument and renders the noiseless image with lenstronomy. This module turns that
image into a Poisson exposure and lets PCAT sample the macro-lens and source position
with the same lenstronomy model that mejiro used, so data and model share one definition.
mejiro and lenstronomy are imported lazily because they are optional dependencies.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

# sampled parameters, as (name, kwargs list, component index, kwargs key, prior minimum, prior maximum, proposal scale)
PARAMETERS = (
    ("theta_E", "kwargs_lens", 0, "theta_E", 0.5, 2.0, 0.02),  # [arcsec]
    ("lens_e1", "kwargs_lens", 0, "e1", -0.3, 0.3, 0.02),
    ("lens_e2", "kwargs_lens", 0, "e2", -0.3, 0.3, 0.02),
    ("lens_x", "kwargs_lens", 0, "center_x", -0.2, 0.2, 0.01),  # [arcsec]
    ("lens_y", "kwargs_lens", 0, "center_y", -0.2, 0.2, 0.01),  # [arcsec]
    ("gamma1", "kwargs_lens", 1, "gamma1", -0.2, 0.2, 0.01),
    ("gamma2", "kwargs_lens", 1, "gamma2", -0.2, 0.2, 0.01),
    ("source_x", "kwargs_source", 0, "center_x", -0.8, 0.8, 0.02),  # [arcsec]
    ("source_y", "kwargs_source", 0, "center_y", -0.8, 0.8, 0.02),  # [arcsec]
)
LABELS = (
    r"$\theta_{\rm E}$ [arcsec]", r"Lens $e_1$", r"Lens $e_2$", "Lens x [arcsec]", "Lens y [arcsec]",
    r"Shear $\gamma_1$", r"Shear $\gamma_2$", "Source x [arcsec]", "Source y [arcsec]",
)

# lenstronomy image models are rebuilt once per process and configuration, since they are not plain data
_IMAGE_MODELS: dict[tuple, object] = {}


def _plain(kwargs_list: list[dict]) -> list[dict]:
    """Return lenstronomy keyword dictionaries with every value cast to a float."""
    # mejiro converts magnitudes into amplitudes stored as length-one astropy columns
    return [{key: float(np.asarray(value).ravel()[0]) for key, value in kwargs.items()
             if value is not None} for kwargs in kwargs_list]


def simulate_mejiro_exposure(
    strong_lens: object = None,
    band: str = "F129",
    exposure_time: float = 146.0,  # [s]
    fov_arcsec: float = 5.0,  # [arcsec]
    supersampling_factor: int = 3,
    seed: int = 2026,
) -> dict:
    """Render a mejiro lens with Roman WFI and draw one Poisson exposure.

    The default lens is mejiro's ``SampleGG``, a galaxy-galaxy lens drawn from SLSim.
    The returned specification holds plain data only, so PCAT can store it.
    """
    from mejiro.galaxy_galaxy import SampleGG
    from mejiro.instruments.roman import Roman
    from mejiro.synthetic_image import SyntheticImage
    from mejiro.utils.lenstronomy_util import get_gaussian_psf_kwargs

    roman = Roman()
    psf_fwhm = float(roman.get_psf_fwhm(band)[0].value)  # [arcsec]
    kwargs_numerics = {"supersampling_factor": supersampling_factor, "compute_mode": "regular"}
    image = SyntheticImage(
        strong_lens or SampleGG(), roman, band, fov_arcsec=fov_arcsec,
        kwargs_psf=get_gaussian_psf_kwargs(psf_fwhm), kwargs_numerics=dict(kwargs_numerics),
    )
    # minimum zodiacal light plus internal thermal emission from mejiro's Roman tables
    sky_rate = float(roman.get_minimum_zodiacal_light(band)[0].value
                     + roman.get_thermal_background(band)[0].value)  # [counts s^-1 pixel^-1]
    lens = image.strong_lens
    specification = {
        "band": band,
        "exposure_time": exposure_time,  # [s]
        "sky_counts": sky_rate * exposure_time,  # [counts pixel^-1]
        "pixel_scale": float(image.pixel_scale),  # [arcsec pixel^-1]
        "psf_fwhm": psf_fwhm,  # [arcsec]
        "kwargs_numerics": kwargs_numerics,
        "kwargs_pixel": {
            "nx": image.num_pix, "ny": image.num_pix,
            "ra_at_xy_0": float(image.ra_at_xy_0), "dec_at_xy_0": float(image.dec_at_xy_0),
            "transform_pix2angle": np.asarray(image.Mpix2coord, dtype=float),
        },
        "lens_model_list": list(lens.kwargs_model["lens_model_list"]),
        "source_light_model_list": list(lens.kwargs_model["source_light_model_list"]),
        "lens_light_model_list": list(lens.kwargs_model["lens_light_model_list"]),
        "kwargs_lens": _plain(lens.kwargs_lens),
        "kwargs_source": _plain(lens.kwargs_source),
        "kwargs_lens_light": _plain(lens.kwargs_lens_light),
    }
    specification["true_parameters"] = np.array(
        [specification[group][index][key] for _, group, index, key, *_ in PARAMETERS]
    )
    specification["expected_counts"] = image.data * exposure_time + specification["sky_counts"]  # [counts pixel^-1]
    specification["observed_counts"] = np.random.default_rng(seed).poisson(
        specification["expected_counts"]).astype(float)  # [counts pixel^-1]
    specification["mejiro_counts"] = image.data * exposure_time  # [counts pixel^-1], mejiro's noiseless image
    return specification


def _image_model(specification: dict) -> object:
    """Return the lenstronomy image model described by a mejiro specification."""
    pixel = specification["kwargs_pixel"]
    key = (specification["psf_fwhm"], pixel["nx"], pixel["ny"], pixel["ra_at_xy_0"], pixel["dec_at_xy_0"],
           tuple(np.ravel(pixel["transform_pix2angle"])), tuple(sorted(specification["kwargs_numerics"].items())),
           tuple(specification["lens_model_list"]), tuple(specification["source_light_model_list"]),
           tuple(specification["lens_light_model_list"]))
    if key not in _IMAGE_MODELS:
        from lenstronomy.Data.pixel_grid import PixelGrid
        from lenstronomy.Data.psf import PSF
        from lenstronomy.ImSim.image_model import ImageModel
        from lenstronomy.LensModel.lens_model import LensModel
        from lenstronomy.LightModel.light_model import LightModel

        _IMAGE_MODELS[key] = ImageModel(
            PixelGrid(**specification["kwargs_pixel"]),
            PSF(psf_type="GAUSSIAN", fwhm=specification["psf_fwhm"]),
            lens_model_class=LensModel(specification["lens_model_list"]),
            source_model_class=LightModel(specification["source_light_model_list"]),
            lens_light_model_class=LightModel(specification["lens_light_model_list"]),
            kwargs_numerics=specification["kwargs_numerics"],
        )
    return _IMAGE_MODELS[key]


def render_mejiro_counts(specification: dict, parameters: np.ndarray) -> np.ndarray:
    """Return expected counts per pixel for sampled lens and source parameters."""
    kwargs = {group: deepcopy(specification[group])
              for group in ("kwargs_lens", "kwargs_source", "kwargs_lens_light")}
    for value, (_, group, index, key, *_) in zip(parameters, PARAMETERS):
        kwargs[group][index][key] = float(value)
    rate = _image_model(specification).image(
        kwargs["kwargs_lens"], kwargs["kwargs_source"], kwargs["kwargs_lens_light"])  # [counts s^-1 pixel^-1]
    return rate * specification["exposure_time"] + specification["sky_counts"]


def mejiro_lens_log_likelihood(state: object, model_name: str, parameters: np.ndarray) -> float:
    """Evaluate the Poisson log likelihood of the mejiro exposure, up to data-only terms."""
    model_counts = np.maximum(render_mejiro_counts(state.mejiro_specification, parameters), 1e-12)
    observed_counts = state.mejiro_specification["observed_counts"]
    return float(np.sum(observed_counts * np.log(model_counts) - model_counts))


def run_mejiro_lens_inference(
    specification: dict,
    output_root: str,
    run_name: str,
    numbswep: int = 60000,
    numbburn: int = 40000,
    numbsamp: int = 2000,
    numbproc: int = 4,
    numbframanim: int | None = None,
) -> object:
    """Sample the macro lens and source position with PCAT, starting from a prior draw."""
    from .main import sample

    return sample(
        typeexpr="gener",
        retr_llik=mejiro_lens_log_likelihood,
        mejiro_specification=specification,
        parameter_names=tuple(row[0] for row in PARAMETERS),
        prior_types=("self",) * len(PARAMETERS),
        prior_minima=tuple(row[4] for row in PARAMETERS),
        prior_maxima=tuple(row[5] for row in PARAMETERS),
        proposal_scales=tuple(row[6] for row in PARAMETERS),
        propwithsing=True,
        probpropblock=0.0,
        # a prior draw lands far from the arcs, so the likelihood is annealed slowly during burn-in
        booladaptstdp=True,
        boolburntmpr=True,
        factburntmpr=0.9,
        # differential-evolution jumps move the strongly correlated lens parameters together
        probdemc=0.2,
        pathbase=str(output_root),
        strgcnfg=run_name,
        numbproc=numbproc,
        numbswep=numbswep,
        numbburn=numbburn,
        numbsamp=numbsamp,
        numbframanim=numbframanim,
        typeseed=7,
        typeverb=-1,
    )
