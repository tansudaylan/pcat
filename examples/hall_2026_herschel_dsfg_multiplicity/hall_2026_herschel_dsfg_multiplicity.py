#!/usr/bin/env python3
"""Run a scaled Herschel DSFG multiplicity analog of Hall et al. (2026)."""

from pathlib import Path

from pcat.publication_examples import run_spire_component_example

OUTPUT_ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    run_spire_component_example(OUTPUT_ROOT, "hall2026_dsfg_multiplicity", 14, 0.12)
