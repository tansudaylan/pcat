#!/usr/bin/env python3
"""Measure catalog-association completeness and purity versus match radius."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pcat.associate import associate_catalogs


def simulate_catalogs(
    number_sources: int = 80,
    seed: int = 814,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return seeded reference and measured catalogs with contaminants."""
    random = np.random.default_rng(seed)
    coordinates_source = random.uniform(-2.0, 2.0, size=(number_sources, 2))  # [arcsec]
    magnitude_source = random.uniform(19.0, 24.0, size=number_sources)  # [mag]

    recovered = random.random(number_sources) < 0.8
    coordinates_recovered = coordinates_source[recovered] + random.normal(
        0.0, 0.08, size=(recovered.sum(), 2)
    )  # [arcsec]
    magnitude_recovered = magnitude_source[recovered] + random.normal(
        0.0, 0.12, size=recovered.sum()
    )  # [mag]

    number_contaminants = 24
    coordinates_contaminants = random.uniform(
        -2.0, 2.0, size=(number_contaminants, 2)
    )  # [arcsec]
    magnitude_contaminants = random.uniform(19.0, 24.0, size=number_contaminants)  # [mag]
    coordinates_target = np.vstack((coordinates_recovered, coordinates_contaminants))
    magnitude_target = np.concatenate((magnitude_recovered, magnitude_contaminants))
    return coordinates_source, magnitude_source, coordinates_target, magnitude_target


def association_rates(
    coordinates_source: np.ndarray,
    magnitude_source: np.ndarray,
    coordinates_target: np.ndarray,
    magnitude_target: np.ndarray,
    radii: np.ndarray,
    magnitude_difference_maximum: float = 0.5,  # [mag]
) -> tuple[np.ndarray, np.ndarray]:
    """Return completeness and purity across angular matching radii."""
    completeness = np.empty(radii.size)
    purity = np.empty(radii.size)
    for index, radius in enumerate(radii):
        matched_source = associate_catalogs(
            coordinates_source,
            magnitude_source,
            coordinates_target,
            magnitude_target,
            radius,
            magnitude_difference_maximum,
        )
        matched_target = associate_catalogs(
            coordinates_target,
            magnitude_target,
            coordinates_source,
            magnitude_source,
            radius,
            magnitude_difference_maximum,
        )
        completeness[index] = matched_source.mean()
        purity[index] = matched_target.mean()
    return completeness, purity


def run_example(
    output_path: Path,
    number_sources: int = 80,
    seed: int = 814,
) -> dict[str, float | int]:
    """Simulate two catalogs and plot their association performance."""
    catalogs = simulate_catalogs(number_sources=number_sources, seed=seed)
    radii = np.linspace(0.02, 0.40, 20)  # [arcsec]
    completeness, purity = association_rates(*catalogs, radii)

    figure, axis = plt.subplots(figsize=(6.4, 4.0), facecolor="white")
    axis.plot(radii, completeness, color="#A51417", linewidth=2.0, label="Completeness")
    axis.plot(radii, purity, color="#007360", linewidth=2.0, label="Purity")
    axis.set_xlabel("Association radius [arcsec]")
    axis.set_ylabel("Catalog fraction")
    axis.set_ylim(0.0, 1.03)
    axis.grid(False)
    axis.legend(frameon=True, fancybox=True, framealpha=1.0)
    figure.tight_layout()

    output_path = Path(output_path)
    if output_path.suffix not in (".png", ".pdf"):
        raise ValueError("Output format must be 'png' or 'pdf'.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    reference_index = int(np.argmin(np.abs(radii - 0.20)))
    return {
        "number_sources": number_sources,
        "number_targets": catalogs[2].shape[0],
        "reference_radius_arcsec": float(radii[reference_index]),
        "completeness": float(completeness[reference_index]),
        "purity": float(purity[reference_index]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typefileplot", choices=("png", "pdf"), default="png")
    arguments = parser.parse_args()
    output_path = Path(__file__).with_name("visuals") / (
        f"catalog_association.{arguments.typefileplot}"
    )
    print(run_example(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())