#!/usr/bin/env python3
"""Draw the PCAT logo in Harvard Crimson and black.

Three stacked catalog planes hold one, two, and three sources, each drawn as a small cloud of
posterior samples. Arrows hop between the planes, the birth and death moves that let PCAT change
the dimension of its model. The script writes a stacked logo with the name, a wide banner, and a
square icon without text for favicons and social previews.
"""

from tdpy.verbosity import print

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

PATH_STATIC = Path(__file__).resolve().parent / "_static"
CRIMSON = "#A51C30"  # Harvard Crimson
BLACK = "#000000"
WHITE = "#FFFFFF"
# plane shear and size in icon coordinates
SHEAR = 0.35
HALFWIDTH = 0.62
HALFDEPTH = 0.17
PLANE_HEIGHTS = (-0.52, 0.0, 0.52)
# source positions in plane coordinates (x across, y into the plane)
SOURCES = (
    [(0.05, 0.1)],
    [(-0.3, -0.2), (0.3, 0.25)],
    [(-0.38, 0.15), (0.02, -0.35), (0.38, 0.2)],
)


def to_icon(x, y, height):
    """Project plane coordinates onto the icon with a sheared oblique view."""
    return x + SHEAR * y * HALFDEPTH / 0.5, height + y * HALFDEPTH


def draw_icon(axis, rng):
    """Draw the crimson tile with three catalog planes and the transdimensional hops."""
    axis.add_patch(FancyBboxPatch((-1.0, -1.0), 2.0, 2.0, boxstyle="round,pad=0,rounding_size=0.28",
                                  color=CRIMSON, zorder=0))
    for height, sources in zip(PLANE_HEIGHTS, SOURCES):
        corners = [to_icon(x, y, height) for x, y in [(-HALFWIDTH, -1.), (HALFWIDTH, -1.), (HALFWIDTH, 1.),
                                                        (-HALFWIDTH, 1.)]]
        axis.add_patch(Polygon(corners, closed=True, facecolor=WHITE, edgecolor=BLACK, lw=2.2, zorder=1))
        for x, y in sources:
            # each source is a cloud of posterior samples around its position
            cloud = np.array([x, y]) + np.array([0.07, 0.25]) * rng.standard_normal((70, 2))
            # keep every sample on its plane
            cloud = cloud[(np.abs(cloud[:, 0]) < HALFWIDTH - 0.08) & (np.abs(cloud[:, 1]) < 0.8)]
            axis.scatter(*to_icon(cloud[:, 0], cloud[:, 1], height), s=2.5, color=CRIMSON, alpha=0.6, lw=0, zorder=2)
            axis.scatter(*to_icon(x, y, height), s=38, color=BLACK, lw=0, zorder=3)
    # birth (up) and death (down) hops between catalogs of different dimension
    for start, end, sign in [((0.8, -0.44), (0.8, -0.08), -1), ((0.8, 0.08), (0.8, 0.44), -1),
                             ((-0.8, 0.44), (-0.8, 0.08), -1), ((-0.8, -0.08), (-0.8, -0.44), -1)]:
        axis.add_patch(FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={0.5 * sign}", arrowstyle="-|>",
                                       mutation_scale=11, lw=1.8, color=WHITE, zorder=4))
    axis.text(0.87, 0.26, "+", color=WHITE, fontsize=13, fontweight="bold", ha="center", va="center", zorder=4)
    axis.text(-0.87, -0.26, "\u2212", color=WHITE, fontsize=13, fontweight="bold", ha="center", va="center", zorder=4)


def save(figure, path_stem):
    PATH_STATIC.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        path = PATH_STATIC / f"{path_stem}.{suffix}"
        print(f"Writing to {path}...")
        figure.savefig(path, dpi=400, transparent=True, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)


def draw_logo():
    mpl.rcParams["text.usetex"] = False
    font = dict(fontweight="bold", color=BLACK, family="DejaVu Sans")

    # square icon for favicons and social previews
    figure, axis = plt.subplots(figsize=(3.0, 3.0))
    axis.set(xlim=(-1.02, 1.02), ylim=(-1.02, 1.02), aspect="equal")
    axis.axis("off")
    draw_icon(axis, np.random.default_rng(2017))
    save(figure, "pcat_icon")

    # stacked logo with the name below the icon
    figure, axis = plt.subplots(figsize=(3.0, 3.9))
    axis.set(xlim=(-1.02, 1.02), ylim=(-1.62, 1.02), aspect="equal")
    axis.axis("off")
    draw_icon(axis, np.random.default_rng(2017))
    axis.text(0.0, -1.33, "PCAT", ha="center", va="center", fontsize=42, **font)
    save(figure, "pcat_logo")

    # wide banner with the name and its expansion beside the icon
    figure, axis = plt.subplots(figsize=(7.5, 2.0))
    axis.set(xlim=(-1.02, 4.45), ylim=(-1.02, 1.02), aspect="equal")
    axis.axis("off")
    draw_icon(axis, np.random.default_rng(2017))
    axis.text(1.3, 0.2, "PCAT", ha="left", va="center", fontsize=48, **font)
    axis.text(1.34, -0.62, "Probabilistic Cataloger", ha="left", va="center", fontsize=17, color=CRIMSON,
              family="DejaVu Sans")
    save(figure, "pcat_banner")

    # GitHub social preview at its recommended 1280 x 640 pixels
    figure = plt.figure(figsize=(6.4, 3.2), dpi=200)
    axis = figure.add_axes([0.06, 0.12, 0.88, 0.76])
    axis.set(xlim=(-1.02, 4.45), ylim=(-1.02, 1.02), aspect="equal")
    axis.axis("off")
    draw_icon(axis, np.random.default_rng(2017))
    axis.text(1.3, 0.2, "PCAT", ha="left", va="center", fontsize=40, **font)
    axis.text(1.34, -0.62, "Probabilistic Cataloger", ha="left", va="center", fontsize=14, color=CRIMSON,
              family="DejaVu Sans")
    path = PATH_STATIC / "pcat_social_preview.png"
    print(f"Writing to {path}...")
    figure.savefig(path, dpi=200, facecolor=WHITE)
    plt.close(figure)


if __name__ == "__main__":
    draw_logo()
