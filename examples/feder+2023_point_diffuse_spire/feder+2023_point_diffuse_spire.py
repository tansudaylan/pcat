#!/usr/bin/env python3
"""Run a scaled point-plus-diffuse analog of Feder et al. (2023)."""

from pathlib import Path

from pcat.publication_examples import run_spire_component_example

OUTPUT_ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    run_spire_component_example(OUTPUT_ROOT, "feder2023_point_diffuse_spire", 10, 0.35)
