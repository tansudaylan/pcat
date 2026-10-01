#!/usr/bin/env python3
"""Draw corner, histogram, and pair plots of two simulated populations.

The samples are simulated: five features drawn from unit Gaussians with random
offsets, 100 samples for the "Positive" population and 10,000 for "Negative".
"""

import argparse
from tdpy.cli import add_plot_arguments
from pathlib import Path

import numpy as np

from pcat import plot_population_grid

EXAMPLE_PATH = Path(__file__).resolve().parent


def simulate_populations(seed=0):
    """Return per-population sample arrays with shape (samples, features)."""
    rng = np.random.default_rng(seed)
    numbfeat = 5
    return [rng.standard_normal((numbsamp, numbfeat)) + 5.0 * rng.standard_normal(numbfeat)[None, :]
            for numbsamp in (100, 10_000)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_arguments(parser)
    arguments = parser.parse_args()

    listpara = simulate_populations()
    listlablpara = [[f"Feature {index + 1}", ""] for index in range(listpara[0].shape[1])]
    plot_population_grid(
        listlablpara,
        listpara=listpara,
        listlablpopl=["Positive", "Negative"],
        pathbase=f"{EXAMPLE_PATH / 'visuals'}/",
        strgextn="simulated_two_populations",
        typefileplot=arguments.typefileplot,
        boolplottria=True,
        typeverb=0,
    )


if __name__ == "__main__":
    main()
