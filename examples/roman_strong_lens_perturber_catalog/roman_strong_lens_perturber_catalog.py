#!/usr/bin/env python3
"""Run PCAT's seeded Roman strong-lens population benchmark."""

from tdpy.verbosity import print

import argparse
import shutil
from tdpy.cli import add_plot_arguments
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pcat.plotting import animation_phase_label, animation_states, plot_detection_diagnostic
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
    """Plot replicated Roman images, with sky background and noise, from a prior draw to posterior samples."""
    output_root = Path(__file__).resolve().parent
    config = RomanLensConfig()
    truth = np.array([0.9, 0.06, -0.04])  # [arcsec]
    source_size = 0.09  # [arcsec]
    axis_ratio = 0.8
    source_angle = 0.35  # [rad]
    expected_counts = render_lens_counts(config, truth, source_size, axis_ratio, source_angle)
    observed = np.random.default_rng(814).poisson(expected_counts)
    cached_run = output_root / "pcat_runs" / "roman_wfi_lens_posterior"
    if cached_run.exists():
        print(f"Removing cached lens run {cached_run}...")
        shutil.rmtree(cached_run)
    result = run_lens_image_pipeline(
        config, observed, source_size, axis_ratio, source_angle, truth,
        output_root, "roman_wfi_lens_posterior", "roman_wfi_lens_posterior",
        numbswep=600 if smoke else 6000, numbburn=200 if smoke else 2000,
        numbsamp=200 if smoke else 2000, initial_parameters=None, numbframanim=24,
    )
    # one fixed noise realization, so frames differ only through the lens model
    noise_seed = 2027
    stretch = 3.0 * np.sqrt(config.background)  # [electron pixel^-1], arcsinh softening near the sky noise
    upper = float(np.arcsinh(np.max(observed) / stretch))
    half_width = config.number_side * config.pixel_scale / 2  # [arcsec]
    visual_root = output_root / "visuals"
    for old_frame in visual_root.glob("roman_wfi_lensed_posterior_swep*.png"):
        print(f"Removing previous frame {old_frame}...")
        old_frame.unlink()
    paths = []
    for snapshot in animation_states(result.posterior):
        model_counts = render_lens_counts(config, snapshot["paragenrscalfull"][:3], source_size,
                                          axis_ratio, source_angle)
        replicated = np.random.default_rng(noise_seed).poisson(model_counts)
        figure, axis = plt.subplots(figsize=(5.2, 5.2), facecolor="white")
        image = axis.imshow(np.arcsinh(replicated / stretch), origin="lower", cmap="magma", vmin=0.0,
                            vmax=upper, extent=(-half_width, half_width, -half_width, half_width))
        figure.colorbar(image, ax=axis, shrink=0.8,
                        label=r"arcsinh(counts / $3\sigma_{\rm sky}$)")
        axis.set(xlabel="Field offset 1 [arcsec]", ylabel="Field offset 2 [arcsec]",
                 title=animation_phase_label(snapshot))
        axis.grid(False)
        path = visual_root / f"roman_wfi_lensed_posterior_swep{snapshot['cntrswep']:09d}.png"
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