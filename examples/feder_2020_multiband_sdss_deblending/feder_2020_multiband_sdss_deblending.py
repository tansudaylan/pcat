#!/usr/bin/env python3
"""Run a scaled multiband SDSS analog of Feder et al. (2020)."""

from pathlib import Path

from pcat.publication_examples import run_optical_catalog_example

OUTPUT_ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    run_optical_catalog_example(OUTPUT_ROOT, "feder2020_multiband_sdss", 12)
