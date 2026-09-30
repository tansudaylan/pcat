#!/usr/bin/env python3
"""Draw the PCAT logo in Harvard Crimson and black.

A point, a line segment, a square, and a cube are parameter spaces of dimension zero to three.
Reversible arrows jump between them, the birth and death moves with which PCAT changes the
dimension of its model, and a black dot in each space marks the sampled state. The script writes
a stacked logo with the name, a wide banner, and a square icon without text for favicons and
social previews.
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
# centers of the 0-, 1-, 2-, and 3-dimensional spaces in icon coordinates, read in a Z pattern
CENTERS = np.array([[-0.55, 0.52], [0.4, 0.52], [-0.56, -0.44], [0.5, -0.5]])
SIZE = 0.46  # edge length of the segment, square, and cube
DEPTH = np.array([0.18, 0.15])  # oblique offset of the cube's back face


def draw_icon(axis, rng=None):
    """Draw the crimson tile with spaces of increasing dimension joined by transdimensional jumps."""
    axis.add_patch(FancyBboxPatch((-1.0, -1.0), 2.0, 2.0, boxstyle="round,pad=0,rounding_size=0.28",
                                  color=CRIMSON, zorder=0))
    line = dict(color=WHITE, lw=3.2, solid_capstyle="round", zorder=2)
    half = 0.5 * SIZE

    # 0-D: a point
    axis.scatter(*CENTERS[0], s=260, color=WHITE, zorder=2)
    axis.scatter(*CENTERS[0], s=70, color=BLACK, zorder=3)

    # 1-D: a segment with end caps
    x, y = CENTERS[1]
    axis.plot([x - half, x + half], [y, y], **line)
    for end in (x - half, x + half):
        axis.plot([end, end], [y - 0.07, y + 0.07], **line)
    axis.scatter(x + 0.1, y, s=70, color=BLACK, zorder=3)

    # 2-D: a square
    x, y = CENTERS[2]
    axis.add_patch(Polygon([(x - half, y - half), (x + half, y - half), (x + half, y + half), (x - half, y + half)],
                           closed=True, facecolor="none", edgecolor=WHITE, lw=3.2, joinstyle="round", zorder=2))
    axis.scatter(x - 0.08, y + 0.07, s=70, color=BLACK, zorder=3)

    # 3-D: a wireframe cube in oblique projection
    x, y = CENTERS[3] - 0.5 * DEPTH
    front = np.array([(x - half, y - half), (x + half, y - half), (x + half, y + half), (x - half, y + half)])
    back = front + DEPTH
    for face in (back, front):
        axis.add_patch(Polygon(face, closed=True, facecolor="none", edgecolor=WHITE, lw=3.2 if face is front else 2.,
                               joinstyle="round", zorder=2))
    for corner in range(4):
        axis.plot(*np.array([front[corner], back[corner]]).T, color=WHITE, lw=2., zorder=2)
    axis.scatter(x + 0.12, y + 0.02, s=70, color=BLACK, zorder=3)

    # reversible jumps between neighboring dimensions
    jumps = [((-0.4, 0.52), (0.08, 0.52), 0.), ((0.15, 0.32), (-0.3, -0.12), 0.), ((-0.26, -0.44), (0.16, -0.44), 0.)]
    for start, end, rad in jumps:
        axis.add_patch(FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}", arrowstyle="<|-|>",
                                       mutation_scale=13, lw=2., color=BLACK, zorder=4))


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
