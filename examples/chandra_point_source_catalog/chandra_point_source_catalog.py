#!/usr/bin/env python
"""Run a genuine PCAT Chandra-style source-detection pipeline example and let it produce the figures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_NAME = "chandra_point_source_catalog"
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / RUN_NAME


def make_faint_source_background_scene(seed: int = 173) -> dict[str, object]:
    """Make one synthetic Chandra-style image with a known faint population."""
    from pcat.image import forward_model_image, make_gaussian_scene

    image_shape = (24, 24)
    true_background = 2.0  # [count pixel^-1]
    psf_sigma_pixels = 0.8  # [pixel]
    intrinsic_sigma_pixels = 0.35  # [pixel]
    bright_sources = np.array(((6.0, 6.0, 650.0), (17.0, 18.0, 450.0)))
    source_coordinates = np.linspace(2.5, 20.5, 8)
    faint_xpos, faint_ypos = np.meshgrid(source_coordinates, source_coordinates)
    faint_sources = np.column_stack((
        faint_xpos.ravel(),
        faint_ypos.ravel(),
        np.full(faint_xpos.size, 4.0),
    ))
    source_parameters = np.vstack((bright_sources, faint_sources))
    true_source_image = make_gaussian_scene(
        image_shape, source_parameters, intrinsic_sigma_pixels
    )
    true_source_counts = forward_model_image(
        true_source_image, psf_sigma_pixels
    )["observed_image"]
    rng = np.random.default_rng(seed)
    observed_counts = rng.poisson(true_source_counts + true_background).astype(float)

    source_counts_by_cut = {}
    for flux_cut in (1.0, 10.0):
        retained_sources = source_parameters[source_parameters[:, 2] >= flux_cut]
        retained_image = make_gaussian_scene(
            image_shape, retained_sources, intrinsic_sigma_pixels
        )
        source_counts_by_cut[flux_cut] = forward_model_image(
            retained_image, psf_sigma_pixels
        )["observed_image"]

    omitted_surface_brightness = float(np.mean(
        source_counts_by_cut[1.0] - source_counts_by_cut[10.0]
    ))
    return {
        "image_shape": image_shape,
        "true_background": true_background,
        "psf_sigma_pixels": psf_sigma_pixels,
        "intrinsic_sigma_pixels": intrinsic_sigma_pixels,
        "source_parameters": source_parameters,
        "observed_counts": observed_counts,
        "true_source_counts": true_source_counts,
        "source_counts_by_cut": source_counts_by_cut,
        "omitted_surface_brightness": omitted_surface_brightness,
    }


def make_single_star_scene(seed: int = 173) -> dict[str, object]:
    """Simulate one centered point source in a 15-by-15 Poisson image."""
    from pcat.image import forward_model_image, make_gaussian_scene

    image_shape = (15, 15)
    source_position = (7.0, 7.0)  # [pixel]
    true_source_flux = 120.0  # [count]
    true_background = 0.6  # [count pixel^-1]
    psf_sigma_pixels = 0.8  # [pixel]
    intrinsic_sigma_pixels = 0.2  # [pixel]
    source_parameters = np.array([[*source_position, true_source_flux]])
    source_image = make_gaussian_scene(image_shape, source_parameters, intrinsic_sigma_pixels)
    source_counts = forward_model_image(source_image, psf_sigma_pixels)["observed_image"]
    unit_source_image = make_gaussian_scene(
        image_shape, [[*source_position, 1.0]], intrinsic_sigma_pixels
    )
    unit_source_template = forward_model_image(
        unit_source_image, psf_sigma_pixels
    )["observed_image"]
    rng = np.random.default_rng(seed)
    observed_counts = rng.poisson(source_counts + true_background).astype(float)
    return {
        "image_shape": image_shape,
        "source_position_pixels": source_position,
        "true_source_flux_counts": true_source_flux,
        "true_background_counts_per_pixel": true_background,
        "psf_sigma_pixels": psf_sigma_pixels,
        "intrinsic_sigma_pixels": intrinsic_sigma_pixels,
        "source_parameters": source_parameters,
        "unit_source_image": unit_source_image,
        "unit_source_template": unit_source_template,
        "source_counts": source_counts,
        "observed_counts": observed_counts,
    }


def _background_log_likelihood(gdat, strgmodl, values) -> float:
    """Score Poisson pixels with a fixed source image and fitted background."""
    background = float(values[0])
    if not np.isfinite(background) or background <= 0.0:
        return -np.inf
    expected_counts = np.asarray(gdat.source_counts, dtype=float) + background
    observed_counts = np.asarray(gdat.observed_counts, dtype=float)
    if expected_counts.shape != observed_counts.shape or np.any(expected_counts <= 0.0):
        return -np.inf
    return float(np.sum(observed_counts * np.log(expected_counts) - expected_counts))


def _single_star_log_likelihood(gdat, strgmodl, values) -> float:
    """Score a known-position source with sampled flux, background, and optional PSF width."""
    values = np.asarray(values, dtype=float)
    if values.shape not in {(2,), (3,)}:
        return -np.inf
    source_flux, background = values[:2]
    if not np.isfinite(source_flux) or source_flux <= 0.0:
        return -np.inf
    if not np.isfinite(background) or background <= 0.0:
        return -np.inf
    if values.size == 3:
        psf_sigma_pixels = values[2]
        if not np.isfinite(psf_sigma_pixels) or psf_sigma_pixels <= 0.0:
            return -np.inf
        from pcat.image import forward_model_image

        unit_source_template = forward_model_image(
            gdat.unit_source_image, psf_sigma_pixels
        )["observed_image"]
    else:
        unit_source_template = gdat.unit_source_template
    expected_counts = unit_source_template * source_flux + background
    observed_counts = np.asarray(gdat.observed_counts, dtype=float)
    if expected_counts.shape != observed_counts.shape or np.any(expected_counts <= 0.0):
        return -np.inf
    return float(np.sum(observed_counts * np.log(expected_counts) - expected_counts))


def _background_reference_quantiles(observed_counts, source_counts, grid):
    expected_counts = source_counts[None, :, :] + grid[:, None, None]
    log_likelihood = np.sum(
        observed_counts[None, :, :] * np.log(expected_counts) - expected_counts,
        axis=(1, 2),
    )
    weights = np.exp(log_likelihood - np.max(log_likelihood))
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return np.interp((0.16, 0.5, 0.84), cumulative, grid)


def run_background_cut_comparison(
    output_root: Path | None = None,
    *,
    seed: int = 173,
    numbswep: int = 6000,
    typefileplot: str = "png",
    example_root: Path = OUTPUT_ROOT,
) -> dict[str, object]:
    """Sample the background with and without a synthetic faint source population."""
    import matplotlib.pyplot as plt

    from pcat.fixed import sample_fixed
    from pcat.demo import configure_example_style, save_example_figure

    if numbswep < 500:
        raise ValueError("numbswep must be at least 500 for the background comparison")
    if typefileplot not in {"png", "pdf"}:
        raise ValueError("typefileplot must be 'png' or 'pdf'")
    scene = make_faint_source_background_scene(seed)
    output_root = (
        Path(output_root) if output_root is not None
        else Path(example_root) / "data" / "faint_source_background"
    )
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    run_directory = output_root / run_tag
    run_directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        run_directory / "synthetic_scene.npz",
        source_parameters=scene["source_parameters"],
        observed_counts=scene["observed_counts"],
        true_source_counts=scene["true_source_counts"],
        model_source_counts_fmin_1=scene["source_counts_by_cut"][1.0],
        model_source_counts_fmin_10=scene["source_counts_by_cut"][10.0],
    )

    prior_bounds = (0.01, 6.0)  # [count pixel^-1]
    background_grid = np.linspace(*prior_bounds, 6001)
    burn_count = numbswep // 5
    fit_results = {}
    for index, flux_cut in enumerate((1.0, 10.0)):
        source_counts = scene["source_counts_by_cut"][flux_cut]
        posterior = sample_fixed(
            retr_llik=_background_log_likelihood,
            observed_counts=scene["observed_counts"],
            source_counts=source_counts,
            parameter_names=("background",),
            prior_types=("self",),
            prior_minima=(prior_bounds[0],),
            prior_maxima=(prior_bounds[1],),
            initial_values=(scene["true_background"],),
            proposal_scales=(0.03,),
            pathbase=run_directory,
            strgcnfg=f"background_fmin_{flux_cut:g}",
            typeseed=seed + index,
            numbproc=1,
            numbswep=numbswep,
            numbburn=burn_count,
            numbsamp=numbswep - burn_count,
            booladaptstdp=True,
            boolmakeplot=False,
            makeanim=False,
            typeverb=0,
        )
        samples = np.asarray(posterior.listpostparagenrscalbase, dtype=float).reshape(-1)
        reference_quantiles = _background_reference_quantiles(
            scene["observed_counts"], source_counts, background_grid
        )
        sample_quantiles = np.quantile(samples, (0.16, 0.5, 0.84))
        fit_results[str(flux_cut)] = {
            "sample_quantiles": sample_quantiles.tolist(),
            "grid_reference_quantiles": reference_quantiles.tolist(),
        }

    summary = {
        "seed": seed,
        "numbswep": numbswep,
        "image_shape": list(scene["image_shape"]),
        "true_background_counts_per_pixel": scene["true_background"],
        "faint_source_count": 64,
        "faint_source_flux_counts": 4.0,
        "bright_source_fluxes_counts": [650.0, 450.0],
        "omitted_faint_surface_brightness_counts_per_pixel": scene["omitted_surface_brightness"],
        "interpretation": "Synthetic fixed-catalog experiment; source positions and retained-source fluxes are known, and only the constant background is sampled.",
        "fits": fit_results,
    }
    (run_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    configure_example_style("white")
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    colors = {"1.0": "#006D77", "10.0": "#C44536"}
    for flux_cut, result in fit_results.items():
        quantiles = result["sample_quantiles"]
        axis.errorbar(
            quantiles[1], float(flux_cut),
            xerr=[[quantiles[1] - quantiles[0]], [quantiles[2] - quantiles[1]]],
            fmt="o", color=colors[flux_cut], capsize=3,
            label=f"PCAT samples, $f_{{min}}={float(flux_cut):g}$",
        )
        reference = result["grid_reference_quantiles"]
        axis.plot(reference[1], float(flux_cut), marker="|", color="black", ms=13)
    axis.axvline(scene["true_background"], color="#222222", ls="--", label="True background")
    axis.axvline(
        scene["true_background"] + scene["omitted_surface_brightness"],
        color="#7A5195", ls=":", label="True background + omitted faint mean",
    )
    axis.set_yticks((1.0, 10.0), labels=("Include faint sources", "Omit faint sources"))
    axis.set_xlabel("Inferred background [counts pixel$^{-1}$]")
    axis.set_title("Faint-source omission shifts the fitted background")
    axis.legend(frameon=False, loc="best")
    figure_path = save_example_figure(
        figure, Path(example_root), f"faint_source_background_{run_tag}", typefileplot
    )
    return {
        "run_directory": run_directory,
        "figure_path": figure_path,
        "summary": summary,
    }


def run_single_star_15x15(
    output_root: Path | None = None,
    *,
    seed: int = 173,
    numbswep: int = 6000,
    typefileplot: str = "png",
    float_psf: bool = False,
    example_root: Path = OUTPUT_ROOT,
) -> dict[str, object]:
    """Fit one centered source with fixed or sampled PSF width in a synthetic image."""
    import matplotlib.pyplot as plt

    from pcat.demo import configure_example_style, save_example_figure
    from pcat.fixed import sample_fixed

    if numbswep < 500:
        raise ValueError("numbswep must be at least 500 for the single-star fit")
    if typefileplot not in {"png", "pdf"}:
        raise ValueError("typefileplot must be 'png' or 'pdf'")
    if not isinstance(float_psf, bool):
        raise ValueError("float_psf must be a boolean")
    scene = make_single_star_scene(seed)
    output_root = (
        Path(output_root) if output_root is not None
        else Path(example_root) / "data" / "single_star_15x15"
    )
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    run_directory = output_root / run_tag
    run_directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        run_directory / "synthetic_scene.npz",
        source_parameters=scene["source_parameters"],
        unit_source_image=scene["unit_source_image"],
        unit_source_template=scene["unit_source_template"],
        source_counts=scene["source_counts"],
        observed_counts=scene["observed_counts"],
    )

    burn_count = numbswep // 5
    parameter_names = ["source_flux", "background"]
    prior_minima = [0.01, 0.01]
    prior_maxima = [300.0, 3.0]
    initial_values = [scene["true_source_flux_counts"], scene["true_background_counts_per_pixel"]]
    proposal_scales = [2.0, 0.05]
    if float_psf:
        parameter_names.append("psf_sigma_pixels")
        prior_minima.append(0.4)
        prior_maxima.append(1.2)
        initial_values.append(scene["psf_sigma_pixels"])
        proposal_scales.append(0.03)

    posterior = sample_fixed(
        retr_llik=_single_star_log_likelihood,
        observed_counts=scene["observed_counts"],
        unit_source_image=scene["unit_source_image"],
        unit_source_template=scene["unit_source_template"],
        parameter_names=tuple(parameter_names),
        prior_types=("self",) * len(parameter_names),
        prior_minima=tuple(prior_minima),
        prior_maxima=tuple(prior_maxima),
        initial_values=tuple(initial_values),
        proposal_scales=tuple(proposal_scales),
        pathbase=run_directory,
        strgcnfg="single_star_15x15",
        typeseed=seed,
        numbproc=1,
        numbswep=numbswep,
        numbburn=burn_count,
        numbsamp=numbswep - burn_count,
        booladaptstdp=True,
        boolmakeplot=False,
        makeanim=False,
        typeverb=0,
    )
    samples = np.asarray(posterior.listpostparagenrscalbase, dtype=float).reshape(
        -1, len(parameter_names)
    )
    quantiles = np.quantile(samples, (0.16, 0.5, 0.84), axis=0)
    posterior_quantiles = {
        name: quantiles[:, index].tolist()
        for index, name in enumerate(("source_flux_counts", "background_counts_per_pixel"))
    }
    if float_psf:
        posterior_quantiles["psf_sigma_pixels"] = quantiles[:, 2].tolist()
    summary = {
        "seed": seed,
        "numbswep": numbswep,
        "image_shape": list(scene["image_shape"]),
        "source_count": 1,
        "source_position_pixels": list(scene["source_position_pixels"]),
        "true_source_flux_counts": scene["true_source_flux_counts"],
        "true_background_counts_per_pixel": scene["true_background_counts_per_pixel"],
        "true_psf_sigma_pixels": scene["psf_sigma_pixels"],
        "psf_mode": "sampled" if float_psf else "fixed",
        "psf_prior_pixels": [0.4, 1.2] if float_psf else None,
        "posterior_quantiles": posterior_quantiles,
        "retained_draws": len(samples),
        "interpretation": (
            "Synthetic fixed-position one-source fit; PCAT samples source flux and constant background. "
            + ("PSF width is sampled with a uniform 0.4-to-1.2-pixel prior. " if float_psf else
               "PSF width is fixed at the injected value. ")
            + "Source position is known and the source count is fixed at one."
        ),
    }
    (run_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    configure_example_style("white")
    median_flux, median_background = quantiles[1, :2]
    if float_psf:
        from pcat.image import forward_model_image

        model_template = forward_model_image(
            scene["unit_source_image"], quantiles[1, 2]
        )["observed_image"]
    else:
        model_template = scene["unit_source_template"]
    model_counts = model_template * median_flux + median_background
    figure, axes = plt.subplots(1, 4, figsize=(12.0, 3.5), constrained_layout=True)
    maximum_counts = max(float(scene["observed_counts"].max()), float(model_counts.max()))
    image = axes[0].imshow(scene["observed_counts"], origin="lower", cmap="magma",
                           vmin=0.0, vmax=maximum_counts)
    axes[0].scatter(*scene["source_position_pixels"], marker="x", color="cyan", s=55)
    axes[0].set_title("Observed counts")
    axes[1].imshow(model_counts, origin="lower", cmap="magma", vmin=0.0, vmax=maximum_counts)
    axes[1].set_title("Posterior median model")
    for axis in axes[:2]:
        axis.set(xlabel="x [pixel]", ylabel="y [pixel]", xlim=(-0.5, 14.5), ylim=(-0.5, 14.5))
        axis.set_aspect("equal")
    figure.colorbar(image, ax=axes[:2], label="Counts [count]", shrink=0.85)
    axes[2].hist(samples[:, 0], bins=35, color="#007360", alpha=0.85)
    axes[2].axvline(scene["true_source_flux_counts"], color="#A51417", linestyle="--",
                    label="Injected flux")
    axes[2].axvline(median_flux, color="black", label="Posterior median")
    axes[2].set(xlabel="Source flux [count]", ylabel="Posterior draws [count]")
    axes[2].legend(frameon=False)
    if float_psf:
        axes[3].hist(samples[:, 2], bins=30, color="#0072B2", alpha=0.85)
        axes[3].axvline(scene["psf_sigma_pixels"], color="#A51430", linestyle="--",
                        label="Injected width")
        axes[3].axvline(quantiles[1, 2], color="black", label="Posterior median")
        axes[3].set(xlabel="PSF width [pixel]", ylabel="Posterior draws [count]",
                    xlim=(0.4, 1.2))
        axes[3].legend(frameon=False)
        axes[3].set_title("Sampled PSF width")
    else:
        axes[3].text(0.5, 0.5, f"PSF fixed at {scene['psf_sigma_pixels']:.1f} pixel",
                     ha="center", va="center", transform=axes[3].transAxes)
        axes[3].set_axis_off()
    figure.suptitle("One centered point source in a 15-by-15 image")
    figure_path = save_example_figure(
        figure, Path(example_root), f"single_star_15x15_{run_tag}", typefileplot
    )
    return {"run_directory": run_directory, "figure_path": figure_path, "summary": summary}


def main(argv=None) -> None:
    from pcat.demo import run_pipeline_demo

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--faint-source-background', action='store_true')
    parser.add_argument('--single-star-15x15', action='store_true')
    parser.add_argument('--float-psf', action='store_true')
    parser.add_argument('--numbswep', type=int, default=6000)
    parser.add_argument('--typefileplot', choices=('png', 'pdf'), default='png')
    arguments = parser.parse_args(argv)
    if arguments.float_psf and not arguments.single_star_15x15:
        parser.error('--float-psf requires --single-star-15x15')
    if arguments.single_star_15x15:
        output_root = arguments.output_root or OUTPUT_ROOT / 'data' / 'single_star_15x15'
        run_single_star_15x15(
            output_root=output_root,
            seed=173 if arguments.seed is None else arguments.seed,
            numbswep=arguments.numbswep,
            typefileplot=arguments.typefileplot,
            float_psf=arguments.float_psf,
        )
        return
    if arguments.faint_source_background:
        output_root = arguments.output_root or OUTPUT_ROOT / 'data' / 'faint_source_background'
        run_background_cut_comparison(
            output_root=output_root,
            seed=173 if arguments.seed is None else arguments.seed,
            numbswep=arguments.numbswep,
            typefileplot=arguments.typefileplot,
        )
        return
    cfg = {
        "typeexpr": "chan",
        "typeelem": ["lghtpnts"],
        "truenumbelempop0": 2,
        "fittminmnumbelempop0": 1,
        "fittmaxmnumbelempop0": 3,
        "dicttrue": {"typeelemspateval": ["full"]},
        "dictfitt": {"typeelemspateval": ["full"]},
        "probspmr": 0.4,
        "numbswep": 250,
        "numbsamp": 50,
        "numbswepplot": 25,
        "numbsidecart": 8,
        "strgcnfg": RUN_NAME,
    }
    if arguments.seed is not None:
        cfg['typeseed'] = arguments.seed
    run_pipeline_demo(arguments.output_root or OUTPUT_ROOT, **cfg)


if __name__ == "__main__":
    main()
