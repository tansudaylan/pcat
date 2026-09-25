"""Compact Roman strong-lens injection and probabilistic-catalog benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


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


def coordinate_grid(config: RomanLensConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return a square image grid centered on zero in arcseconds."""
    coordinate = (np.arange(config.number_side) - (config.number_side - 1) / 2) * config.pixel_scale
    return np.meshgrid(coordinate, coordinate, indexing="xy")


def _deflect(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    einstein_radius: float,
    subhalo: tuple[float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    radius = np.hypot(x_grid, y_grid) + 1.0e-4  # [arcsec]
    alpha_x = einstein_radius * x_grid / radius  # [arcsec]
    alpha_y = einstein_radius * y_grid / radius  # [arcsec]
    if subhalo is not None:
        x_subhalo, y_subhalo, subhalo_einstein_radius = subhalo
        delta_x = x_grid - x_subhalo  # [arcsec]
        delta_y = y_grid - y_subhalo  # [arcsec]
        core_radius = 0.03  # [arcsec]
        radius_squared = delta_x**2 + delta_y**2 + core_radius**2  # [arcsec^2]
        alpha_x += subhalo_einstein_radius**2 * delta_x / radius_squared  # [arcsec]
        alpha_y += subhalo_einstein_radius**2 * delta_y / radius_squared  # [arcsec]
    return alpha_x, alpha_y


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
    """Render a PSF-convolved lensed Gaussian source in detector electrons."""
    x_grid, y_grid = coordinate_grid(config)
    alpha_x, alpha_y = _deflect(x_grid, y_grid, einstein_radius, subhalo)
    beta_x = x_grid - alpha_x - source_x  # [arcsec]
    beta_y = y_grid - alpha_y - source_y  # [arcsec]
    cosine = np.cos(source_angle)
    sine = np.sin(source_angle)
    major = cosine * beta_x + sine * beta_y  # [arcsec]
    minor = -sine * beta_x + cosine * beta_y  # [arcsec]
    profile = np.exp(-0.5 * ((major / source_size) ** 2 + (minor / (axis_ratio * source_size)) ** 2))
    profile *= config.source_counts / profile.sum()
    psf_sigma = config.psf_fwhm / (2.355 * config.pixel_scale)  # [pixel]
    return gaussian_filter(profile, psf_sigma, mode="constant")


def candidate_positions(einstein_radius: float, number_candidates: int) -> np.ndarray:
    """Place candidate perturbers around the macro Einstein ring."""
    angle = np.linspace(0.0, 2.0 * np.pi, number_candidates, endpoint=False)
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
    amplitudes = np.maximum(projection / information, 0.0)
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
    random = np.random.default_rng(seed)
    records: list[dict[str, float | int | bool]] = []
    examples: dict[str, np.ndarray] = {}

    for lens_index in range(number_lenses):
        einstein_radius = random.uniform(0.6, 1.2)  # [arcsec]
        source_x, source_y = random.normal(0.0, 0.08, size=2)  # [arcsec]
        source_size = random.uniform(0.06, 0.14)  # [arcsec]
        axis_ratio = random.uniform(0.55, 1.0)
        source_angle = random.uniform(0.0, np.pi)  # [rad]
        has_subhalo = lens_index < round(number_lenses * config.subhalo_fraction)
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
        variance = expectation + config.read_noise**2  # [electron^2 pixel^-1]
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
        if lens_index in (0, round(number_lenses * config.subhalo_fraction)):
            examples[f"lens_{lens_index}_data"] = data
            examples[f"lens_{lens_index}_macro"] = macro_image
            examples[f"lens_{lens_index}_residual"] = data - macro_image

    return records, examples


def summarize_population(records: list[dict[str, float | int | bool]]) -> dict[str, float | int]:
    """Return thresholded detection and localization calibration metrics."""
    truth = np.array([record["has_subhalo"] for record in records], dtype=bool)
    probability = np.array([record["posterior_one"] for record in records], dtype=float)
    detected = probability >= 0.5
    injected_index = np.array([record["injected_index"] for record in records], dtype=int)
    recovered_index = np.array([record["recovered_index"] for record in records], dtype=int)
    return {
        "number_lenses": len(records),
        "number_injected": int(truth.sum()),
        "true_positive_rate": float(detected[truth].mean()),
        "false_positive_rate": float(detected[~truth].mean()),
        "localization_rate": float((recovered_index[truth] == injected_index[truth]).mean()),
        "mean_posterior_injected": float(probability[truth].mean()),
        "mean_posterior_null": float(probability[~truth].mean()),
    }


def plot_detection_diagnostic(
    records: list[dict[str, float | int | bool]], output_path: Path
) -> Path:
    """Plot catalog probability against injected perturber signal-to-noise."""
    output_path = Path(output_path)
    if output_path.suffix not in (".png", ".pdf"):
        raise ValueError("Output format must be 'png' or 'pdf'.")

    truth = np.array([record["has_subhalo"] for record in records], dtype=bool)
    signal_to_noise = np.array(
        [record["injected_signal_to_noise"] for record in records], dtype=float
    )
    probability = np.array([record["posterior_one"] for record in records], dtype=float)
    summary = summarize_population(records)

    figure, axis = plt.subplots(figsize=(6.5, 4.0), facecolor="white")
    axis.scatter(
        signal_to_noise[~truth],
        probability[~truth],
        color="0.55",
        alpha=0.65,
        s=24,
        label="No perturber",
    )
    axis.scatter(
        signal_to_noise[truth],
        probability[truth],
        color="#A51417",
        alpha=0.75,
        s=28,
        label="Injected perturber",
    )
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1.0, label="Detection threshold")
    axis.set_xlabel(r"Injected perturbation signal-to-noise [$\sigma$]")
    axis.set_ylabel(r"Posterior probability $P(N_{\rm sub}=1\mid d)$")
    axis.set_title("Simulated Roman strong-lens catalog inference")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(False)
    axis.legend(frameon=True, fancybox=True, framealpha=1.0, loc="upper left")
    axis.text(
        0.98,
        0.04,
        "TPR %.0f%%   FPR %.0f%%"
        % (100.0 * summary["true_positive_rate"], 100.0 * summary["false_positive_rate"]),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
    )
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path