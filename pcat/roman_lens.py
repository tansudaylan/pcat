"""Compact Roman strong-lens injection and probabilistic-catalog benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chalcedon
import numpy as np
from scipy.ndimage import gaussian_filter

from .diagnostics import binomial_wilson_interval


@dataclass(frozen=True)
class RomanLensConfig:
    """Instrument and population assumptions for the synthetic benchmark."""

    number_side: int = 31
    pixel_scale: float = 0.11  # [arcsec pixel^-1]
    psf_fwhm: float = 0.105  # [arcsec]
    background: float = 30.0  # [electron pixel^-1]
    read_noise: float = 6.0  # [electron pixel^-1]
    source_counts: float = 8.0e4  # [electron]
    subhalo_fraction: float = 0.5
    reference_einstein_radius: float = 0.03  # [arcsec]
    number_candidates: int = 12
    # even sub-pixel sampling integrates each pixel and never evaluates the singular lens center
    subpixel_sampling: int = 4


@dataclass(frozen=True)
class LensImagePipelineResult:
    """Posterior state and visual products from a PCAT lens-image run."""

    posterior: object
    draws: np.ndarray
    median_parameters: np.ndarray
    model_counts: np.ndarray
    image_fit_path: Path
    parameter_path: Path


def coordinate_grid(config: RomanLensConfig, subpixel_sampling: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return a square grid of (sub-)pixel centers centered on zero in arcseconds."""
    number = config.number_side * subpixel_sampling
    coordinate = (np.arange(number) - (number - 1) / 2) * config.pixel_scale / subpixel_sampling
    return np.meshgrid(coordinate, coordinate, indexing="xy")


def _deflect(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    einstein_radius: float,
    subhalo: tuple[float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the deflection of a singular isothermal sphere plus an optional cored point-mass subhalo."""
    deflection = chalcedon.retr_deflsie(x_grid.ravel(), y_grid.ravel(), 0.0, 0.0, einstein_radius).reshape(x_grid.shape + (2,))
    if subhalo is not None:
        x_subhalo, y_subhalo, subhalo_einstein_radius = subhalo
        core_radius = 0.03  # [arcsec]
        deflection = deflection + chalcedon.retr_deflplum(
            x_grid, y_grid, x_subhalo, y_subhalo, subhalo_einstein_radius, core_radius
        )
    return deflection[..., 0], deflection[..., 1]


def render_lens(
    config: RomanLensConfig,
    einstein_radius: float,
    source_x: float,
    source_y: float,
    source_size: float,
    axis_ratio: float,
    source_angle: float,
    subhalo: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Render a PSF-convolved lensed Gaussian source in detector electrons, integrated over each pixel."""
    sampling = config.subpixel_sampling
    if sampling < 1 or (config.number_side % 2 == 1 and sampling % 2 == 1 and sampling > 1):
        raise ValueError("subpixel_sampling must be 1 or even for an odd number of pixels per side")
    x_grid, y_grid = coordinate_grid(config, sampling)
    alpha_x, alpha_y = _deflect(x_grid, y_grid, einstein_radius, subhalo)
    beta_x = x_grid - alpha_x - source_x  # [arcsec]
    beta_y = y_grid - alpha_y - source_y  # [arcsec]
    cosine = np.cos(source_angle)
    sine = np.sin(source_angle)
    major = cosine * beta_x + sine * beta_y  # [arcsec]
    minor = -sine * beta_x + cosine * beta_y  # [arcsec]
    profile = np.exp(-0.5 * ((major / source_size) ** 2 + (minor / (axis_ratio * source_size)) ** 2))
    side = config.number_side
    profile = profile.reshape(side, sampling, side, sampling).mean(axis=(1, 3))
    profile *= config.source_counts / profile.sum()
    psf_sigma = config.psf_fwhm / (2.355 * config.pixel_scale)  # [pixel]
    return gaussian_filter(profile, psf_sigma, mode="constant")


def render_lens_counts(
    config: RomanLensConfig,
    parameters: np.ndarray,
    source_size: float,
    source_axis_ratio: float,
    source_angle: float,
) -> np.ndarray:
    """Return PSF-convolved lens and uniform background counts per pixel."""
    return render_lens(
        config, *parameters, source_size, source_axis_ratio, source_angle
    ) + config.background


def poisson_lens_log_likelihood(state: object, model_name: str, parameters: np.ndarray) -> float:
    """Evaluate a Poisson lens-image likelihood up to data-only terms."""
    model_counts = render_lens_counts(
        state.lens_config,
        parameters,
        state.lens_source_size,
        state.lens_source_axis_ratio,
        state.lens_source_angle,
    )
    observed_counts = state.lens_observed_counts
    return float(np.sum(observed_counts * np.log(model_counts) - model_counts))


def gaussian_lens_log_likelihood(state: object, model_name: str, parameters: np.ndarray) -> float:
    """Evaluate an independent-pixel Gaussian lens-image likelihood."""
    model_image = render_lens_counts(
        state.lens_config,
        parameters,
        state.lens_source_size,
        state.lens_source_axis_ratio,
        state.lens_source_angle,
    )
    residual = state.lens_observed_image - model_image
    variance = state.lens_variance
    valid = np.isfinite(residual) & np.isfinite(variance) & (variance > 0.0)
    return float(-0.5 * np.sum(residual[valid] ** 2 / variance[valid] + np.log(variance[valid])))


def run_lens_image_pipeline(
    config: RomanLensConfig,
    observed_counts: np.ndarray,
    source_size: float,
    source_axis_ratio: float,
    source_angle: float,
    true_parameters: np.ndarray,
    output_root: Path,
    run_name: str,
    visual_stem: str,
    typefileplot: str = "png",
    numbswep: int = 400,
    numbburn: int = 150,
    numbsamp: int = 250,
    initial_parameters: tuple[float, float, float] | None = (),
    numbframanim: int | None = None,
) -> LensImagePipelineResult:
    """Sample a Poisson lens image and write PCAT fit and posterior figures.

    ``initial_parameters`` defaults to the field center with half the maximum
    Einstein radius; pass ``None`` to start from a random draw from the prior.
    ``numbframanim`` records whole-run animation snapshots.
    """
    from .main import sample
    from .plotting import plot_lens_image_fit, plot_lens_parameter_recovery

    output_root = Path(output_root)
    half_width = (config.number_side - 1) * config.pixel_scale / 2.0  # [arcsec]
    if initial_parameters == ():
        initial_parameters = (0.5 * half_width, 0.0, 0.0)  # [arcsec]
    posterior = sample(
        typeexpr="gener",
        retr_llik=poisson_lens_log_likelihood,
        lens_config=config,
        lens_observed_counts=observed_counts,
        lens_source_size=source_size,
        lens_source_axis_ratio=source_axis_ratio,
        lens_source_angle=source_angle,
        parameter_names=("einstein_radius", "source_x", "source_y"),
        prior_types=("self", "self", "self"),
        prior_minima=(0.1, -0.3 * half_width, -0.3 * half_width),  # [arcsec]
        prior_maxima=(0.9 * half_width, 0.3 * half_width, 0.3 * half_width),  # [arcsec]
        initial_values=initial_parameters,  # [arcsec]
        proposal_scales=(0.08, 0.03, 0.03),  # [arcsec]
        propwithsing=True,
        probpropblock=0.0,
        pathbase=str(output_root),
        strgcnfg=run_name,
        numbproc=1,
        numbswep=numbswep,
        numbburn=numbburn,
        numbsamp=numbsamp,
        booladaptstdp=True,
        numbframanim=numbframanim,
        typeseed=42,
        typeverb=-1,
    )
    draws = np.asarray(posterior.listpostparagenrscalbase)
    median_parameters = np.median(draws, axis=0)  # [arcsec]
    model_counts = render_lens_counts(
        config, median_parameters, source_size, source_axis_ratio, source_angle
    )
    visual_root = output_root / "visuals"
    image_fit_path = plot_lens_image_fit(
        visual_root / f"{visual_stem}_image_fit",
        observed_counts,
        model_counts,
        model_counts,
        config.pixel_scale,
        typefileplot,
    )
    parameter_path = plot_lens_parameter_recovery(
        visual_root / f"{visual_stem}_posterior",
        draws,
        np.asarray(true_parameters),
        typefileplot,
    )
    return LensImagePipelineResult(
        posterior=posterior,
        draws=draws,
        median_parameters=median_parameters,
        model_counts=model_counts,
        image_fit_path=image_fit_path,
        parameter_path=parameter_path,
    )


def candidate_positions(einstein_radius: float, number_candidates: int) -> np.ndarray:
    """Place candidate perturbers around the macro Einstein ring."""
    angle = np.linspace(0.0, 2.0 * np.pi, number_candidates, endpoint=False)  # [rad]
    return np.column_stack((einstein_radius * np.cos(angle), einstein_radius * np.sin(angle)))


def infer_catalog_probability(
    data: np.ndarray,
    macro_image: np.ndarray,
    templates: np.ndarray,
    variance: np.ndarray,
) -> tuple[float, int, float]:
    """Approximate the posterior for a zero- versus one-perturber catalog."""
    residual = data - macro_image
    weighted_templates = templates / variance
    information = np.sum(templates * weighted_templates, axis=(1, 2))
    projection = np.sum(residual * weighted_templates, axis=(1, 2))
    # A null template contributes no evidence and should not contaminate the catalog odds.
    amplitudes = np.zeros_like(projection)
    np.divide(projection, information, out=amplitudes, where=information > 0.0)
    amplitudes = np.maximum(amplitudes, 0.0)
    delta_chi_squared = amplitudes**2 * information
    best_index = int(np.argmax(delta_chi_squared))
    parameter_penalty = 3.0 * np.log(data.size)
    look_elsewhere_penalty = 2.0 * np.log(templates.shape[0])
    log_odds = 0.5 * (delta_chi_squared[best_index] - parameter_penalty - look_elsewhere_penalty)
    probability = float(1.0 / (1.0 + np.exp(-np.clip(log_odds, -40.0, 40.0))))
    return probability, best_index, float(amplitudes[best_index])


def simulate_population(
    number_lenses: int = 500,
    seed: int = 814,
    config: RomanLensConfig | None = None,
) -> tuple[list[dict[str, float | int | bool]], dict[str, np.ndarray]]:
    """Simulate and analyze an ensemble with a zero-or-one perturber catalog."""
    config = config or RomanLensConfig()
    number_injected = round(number_lenses * config.subhalo_fraction)
    if not 0 < number_injected < number_lenses:
        raise ValueError("The benchmark requires at least one injected and null lens.")
    random = np.random.default_rng(seed)
    records: list[dict[str, float | int | bool]] = []
    examples: dict[str, np.ndarray] = {}

    for lens_index in range(number_lenses):
        einstein_radius = random.uniform(0.6, 1.2)  # [arcsec]
        source_x, source_y = random.normal(0.0, 0.08, size=2)  # [arcsec]
        source_size = random.uniform(0.06, 0.14)  # [arcsec]
        axis_ratio = random.uniform(0.55, 1.0)
        source_angle = random.uniform(0.0, np.pi)  # [rad]
        has_subhalo = lens_index < number_injected
        positions = candidate_positions(einstein_radius, config.number_candidates)
        injected_index = int(random.integers(config.number_candidates))
        subhalo_einstein_radius = random.uniform(0.015, 0.045) if has_subhalo else 0.0  # [arcsec]
        subhalo = (*positions[injected_index], subhalo_einstein_radius) if has_subhalo else None

        macro_signal = render_lens(
            config, einstein_radius, source_x, source_y, source_size, axis_ratio, source_angle
        )
        true_signal = render_lens(
            config, einstein_radius, source_x, source_y, source_size, axis_ratio, source_angle, subhalo
        )
        expectation = true_signal + config.background  # [electron pixel^-1]
        data = random.poisson(expectation) + random.normal(0.0, config.read_noise, expectation.shape)
        macro_image = macro_signal + config.background  # [electron pixel^-1]
        variance = expectation + config.read_noise**2  # [electron^2 pixel^-2]
        injected_signal_to_noise = float(
            np.sqrt(np.sum((true_signal - macro_signal) ** 2 / variance))
        )
        templates = np.stack(
            [
                render_lens(
                    config,
                    einstein_radius,
                    source_x,
                    source_y,
                    source_size,
                    axis_ratio,
                    source_angle,
                    (*position, config.reference_einstein_radius),
                )
                - macro_signal
                for position in positions
            ]
        )
        probability, recovered_index, recovered_scale = infer_catalog_probability(
            data, macro_image, templates, variance
        )
        records.append(
            {
                "lens_index": lens_index,
                "has_subhalo": has_subhalo,
                "macro_einstein_radius_arcsec": einstein_radius,
                "source_size_arcsec": source_size,
                "source_axis_ratio": axis_ratio,
                "injected_index": injected_index if has_subhalo else -1,
                "subhalo_einstein_radius_arcsec": subhalo_einstein_radius,
                "injected_signal_to_noise": injected_signal_to_noise,
                "posterior_one": probability,
                "recovered_index": recovered_index,
                "recovered_scale": recovered_scale,
            }
        )
        if lens_index in (0, number_injected):
            examples[f"lens_{lens_index}_data"] = data
            examples[f"lens_{lens_index}_macro"] = macro_image
            examples[f"lens_{lens_index}_residual"] = data - macro_image

    return records, examples


def summarize_population(
    records: list[dict[str, float | int | bool]],
    detection_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Return detection and localization metrics at a posterior threshold."""
    if not 0.0 <= detection_threshold <= 1.0:
        raise ValueError("detection_threshold must be between zero and one.")
    truth = np.array([record["has_subhalo"] for record in records], dtype=bool)
    probability = np.array([record["posterior_one"] for record in records], dtype=float)
    detected = probability >= detection_threshold
    injected_index = np.array([record["injected_index"] for record in records], dtype=int)
    recovered_index = np.array([record["recovered_index"] for record in records], dtype=int)
    number_injected = int(truth.sum())
    number_null = len(records) - number_injected
    true_positive_interval = binomial_wilson_interval(int(detected[truth].sum()), number_injected)
    false_positive_interval = binomial_wilson_interval(int(detected[~truth].sum()), number_null)
    return {
        "number_lenses": len(records),
        "number_injected": number_injected,
        "true_positive_rate": float(detected[truth].mean()),
        "true_positive_rate_lower": true_positive_interval[0],
        "true_positive_rate_upper": true_positive_interval[1],
        "false_positive_rate": float(detected[~truth].mean()),
        "false_positive_rate_lower": false_positive_interval[0],
        "false_positive_rate_upper": false_positive_interval[1],
        "localization_rate": float((recovered_index[truth] == injected_index[truth]).mean()),
        "mean_posterior_injected": float(probability[truth].mean()),
        "mean_posterior_null": float(probability[~truth].mean()),
    }


from .plotting import plot_detection_diagnostic