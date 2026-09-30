#!/usr/bin/env python3
"""Run PCAT's seeded Roman strong-lens population benchmark."""

from tdpy.verbosity import print

import argparse
from pathlib import Path

from pcat.plotting import plot_detection_diagnostic
from pcat.roman_lens import simulate_population, summarize_population


def run_example(output_path: Path, number_lenses: int = 100) -> dict[str, float | int]:
    """Simulate the lens population and write its catalog-detection diagnostic."""
    records, examples = simulate_population(number_lenses=number_lenses, seed=814)
    plot_detection_diagnostic(records, output_path, examples)
    return summarize_population(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typefileplot", choices=("png", "pdf"), default="png")
    arguments = parser.parse_args()
    output_path = Path(__file__).with_name("visuals") / (
        f"roman_strong_lens_perturber_catalog.{arguments.typefileplot}"
    )
    summary = run_example(output_path)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())