#!/usr/bin/env python3
"""Demonstrate PCAT's cubic subpixel point-spread-function model."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from pcat.psf import psf_poly_fit


def reconstruct_psf(
    oversampling: int = 8,
    number_pixels: int = 8,
    psf_sigma: float = 0.85,  # [pixel]
    number_offsets: int = 41,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit and densely evaluate the piecewise-cubic detector PSF model."""
    sample_index = np.arange(oversampling * number_pixels)
    sample_coordinate = sample_index / oversampling - number_pixels / 2.0  # [pixel]
    sampled_psf = np.exp(-0.5 * (sample_coordinate / psf_sigma) ** 2)
    sampled_psf /= sampled_psf.sum()

    coefficients = psf_poly_fit(SimpleNamespace(), sampled_psf, oversampling)
    offsets = np.linspace(0.0, 1.0, number_offsets)
    design = np.column_stack((np.ones_like(offsets), offsets, offsets**2, offsets**3))
    reconstructed = (design @ coefficients).T.reshape(-1)
    coordinates = (
        np.repeat(np.arange(number_pixels), number_offsets)
        + np.tile(offsets, number_pixels)
        - number_pixels / 2.0
    )  # [pixel]
    analytic = np.exp(-0.5 * (coordinates / psf_sigma) ** 2) / sampled_psf.sum()
    analytic *= sampled_psf.max() / analytic.max()
    return coordinates, analytic, reconstructed


def run_example(output_path: Path) -> dict[str, float | int]:
    """Plot the subpixel PSF reconstruction and its interpolation residuals."""
    coordinates, analytic, reconstructed = reconstruct_psf()
    residual_percent = 100.0 * (reconstructed - analytic) / analytic.max()

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(6.4, 5.0),
        sharex=True,
        facecolor="white",
        gridspec_kw={"height_ratios": (3, 1)},
    )
    axes[0].plot(coordinates, analytic, color="black", linewidth=2.0, label="Analytic Gaussian")
    axes[0].plot(
        coordinates,
        reconstructed,
        color="#A51417",
        linewidth=1.4,
        linestyle="--",
        label="PCAT cubic model",
    )
    axes[0].set_ylabel("Normalized PSF response")
    axes[0].legend(frameon=True, fancybox=True, framealpha=1.0)
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].plot(coordinates, residual_percent, color="#007360", linewidth=1.5)
    axes[1].set_xlabel("Subpixel coordinate [pixel]")
    axes[1].set_ylabel("Residual [% peak]")
    for axis in axes:
        axis.grid(False)
    figure.tight_layout()

    output_path = Path(output_path)
    if output_path.suffix not in (".png", ".pdf"):
        raise ValueError("Output format must be 'png' or 'pdf'.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {
        "number_evaluations": coordinates.size,
        "maximum_residual_percent": float(np.max(np.abs(residual_percent))),
        "rms_residual_percent": float(np.sqrt(np.mean(residual_percent**2))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typefileplot", choices=("png", "pdf"), default="png")
    arguments = parser.parse_args()
    output_path = Path(__file__).with_name("visuals") / f"psf_subpixel.{arguments.typefileplot}"
    print(run_example(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())