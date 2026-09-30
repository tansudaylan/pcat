#!/usr/bin/env python3
"""Run a scaled SDSS crowded-field analog of Portillo et al. (2017)."""

from pathlib import Path

from pcat.publication_examples import run_optical_catalog_example

OUTPUT_ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    run_optical_catalog_example(OUTPUT_ROOT, "portillo2017_sdss_m2", 8)
