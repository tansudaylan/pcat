#!/usr/bin/env python3
"""Run PCAT's seeded Roman strong-lens population benchmark."""

from tdpy.verbosity import print

import argparse
from tdpy.cli import add_plot_arguments
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pcat.plotting import plot_detection_diagnostic
from pcat.roman_lens import (
    RomanLensConfig, render_lens, render_lens_counts,
    run_lens_image_pipeline, simulate_population, summarize_population,
)


def run_example(output_path: Path, number_lenses: int = 100) -> dict[str, float | int]:
    """Simulate the lens population and write its catalog-detection diagnostic."""
    records, examples = simulate_population(number_lenses=number_lenses, seed=814)
    plot_detection_diagnostic(records, output_path, examples)
    return summarize_population(records)


def render_posterior_lens_frames(smoke: bool = False) -> list[Path]:
    """Plot PSF-convolved simulated Roman arcs from actual PCAT posterior draws."""
    output_root = Path(__file__).resolve().parent
    config = RomanLensConfig()
    truth = np.array([0.9, 0.06, -0.04])  # [arcsec]
    source_size = 0.09  # [arcsec]
    axis_ratio = 0.8
    source_angle = 0.35  # [rad]
    expected_counts = render_lens_counts(config, truth, source_size, axis_ratio, source_angle)
    observed = np.random.default_rng(814).poisson(expected_counts)
    result = run_lens_image_pipeline(
        config, observed, source_size, axis_ratio, source_angle, truth,
        output_root, "roman_wfi_lens_posterior", "roman_wfi_lens_posterior",
        numbswep=36 if smoke else 140, numbburn=6 if smoke else 30,
        numbsamp=30 if smoke else 110,
    )
    max_signal = float(np.arcsinh(np.max(observed - config.background)))
    half_width = config.number_side * config.pixel_scale / 2  # [arcsec]
    visual_root = output_root / "visuals"
    paths = []
    for sample_index in np.linspace(0, len(result.draws) - 1, min(12, len(result.draws)), dtype=int):
        signal = render_lens(
            config, *result.draws[sample_index], source_size, axis_ratio, source_angle
        )
        figure, axis = plt.subplots(figsize=(5.2, 5.2), facecolor="white")
        axis.imshow(np.arcsinh(signal), origin="lower", cmap="magma", vmin=0,
                    vmax=max(max_signal, 1.0),
                    extent=(-half_width, half_width, -half_width, half_width))
        axis.set(xlabel="Field offset 1 [arcsec]", ylabel="Field offset 2 [arcsec]",
                 title="Simulated Roman/WFI strong-lens arcs")
        axis.grid(False)
        path = visual_root / f"roman_wfi_lensed_posterior_swep{sample_index:09d}.png"
        print(f"Writing to {path}...")
        figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_arguments(parser)
    parser.add_argument("--posterior-frames", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.posterior_frames:
        render_posterior_lens_frames(smoke=arguments.smoke)
        return 0
    output_path = Path(__file__).with_name("visuals") / (
        f"roman_strong_lens_perturber_catalog.{arguments.typefileplot}"
    )
    summary = run_example(output_path)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())