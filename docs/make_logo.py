#!/usr/bin/env python3
"""Draw the PCAT logo: posterior sample clouds of three sources above the name.

The clouds are Gaussian draws around fixed centers, a stylized picture of a
probabilistic catalog in which each source is an ensemble of samples.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

PATH_STATIC = Path(__file__).resolve().parent / "_static"
NAVY = "#0b1f3a"
COLORS = ["#f2b134", "#4fb0c6", "#ef6f6c"]
# source centers and cloud widths in logo coordinates
CENTERS = np.array([[-0.32, 0.18], [0.30, 0.30], [0.05, -0.28]])
WIDTHS = np.array([0.075, 0.06, 0.09])


def draw_logo(path_stem):
    mpl.rcParams["text.usetex"] = False
    rng = np.random.default_rng(2017)
    figure, axis = plt.subplots(figsize=(3.0, 3.6))
    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(-1.45, 1.0)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.add_patch(Circle((0.0, 0.0), 0.95, color=NAVY, zorder=0))
    for center, width, color in zip(CENTERS, WIDTHS, COLORS):
        cloud = center + width * rng.standard_normal((260, 2))
        axis.scatter(*cloud.T, s=4, color=color, alpha=0.55, lw=0, zorder=1)
        axis.scatter(*center, s=90, marker="+", color="white", lw=1.6, zorder=2)
    axis.text(0.0, -1.22, "PCAT", ha="center", va="center", fontsize=40,
              fontweight="bold", color=NAVY, family="DejaVu Sans")
    PATH_STATIC.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        path = PATH_STATIC / f"{path_stem}.{suffix}"
        print(f"Writing to {path}...")
        figure.savefig(path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


if __name__ == "__main__":
    draw_logo("pcat_logo")
