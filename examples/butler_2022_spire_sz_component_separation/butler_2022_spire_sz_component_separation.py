#!/usr/bin/env python3
"""Run a scaled point-plus-diffuse analog of Butler et al. (2022)."""

from pathlib import Path

from pcat.publication_examples import run_spire_component_example

OUTPUT_ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    run_spire_component_example(OUTPUT_ROOT, "butler2022_spire_sz", 6, 0.20)
