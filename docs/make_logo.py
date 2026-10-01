#!/usr/bin/env python3
"""Generate PCAT's circular logo and site-ready exports."""

from tdpy.verbosity import print

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

PATH_STATIC = Path(__file__).resolve().parent / "_static"
CRIMSON = "#A51C30"  # Harvard Crimson
BLACK = "#000000"
WHITE = "#FFFFFF"
DISC_RADIUS = 0.98
CENTERS = np.array([[-0.62, 0.30], [-0.21, 0.30], [0.21, 0.30], [0.62, 0.30]])
SIZE = 0.28  # relative edge length in logo coordinates
DEPTH = np.array([0.07, 0.07])  # relative oblique offset of cube's back face
CUBE_EDGE_WIDTH = 3.0
LOGO_TEXT_POSITION = (0.0, -0.33)
LOGO_FONT_SIZE = 44
TRANSITIONS = [
    ((-0.54, 0.30), (-0.36, 0.30)),
    ((-0.07, 0.30), (0.07, 0.30)),
    ((0.35, 0.30), (0.45, 0.30)),
]


def draw_disc(axis):
    """Draw the circular Harvard Crimson field and black outline."""
    axis.add_patch(Circle((0.0, 0.0), DISC_RADIUS, facecolor=CRIMSON, edgecolor=BLACK, linewidth=2.5, zorder=0))


def draw_icon(axis):
    """Draw the four-dimensionality glyphs and PCAT wordmark inside the circle."""
    draw_disc(axis)
    line = dict(color=WHITE, lw=3.6, solid_capstyle="round", zorder=2)
    half = 0.5 * SIZE

    # 0-D: a point
    axis.scatter(*CENTERS[0], s=190, color=WHITE, zorder=2)
    axis.scatter(*CENTERS[0], s=54, color=BLACK, zorder=3)

    # 1-D: a segment with end caps
    x, y = CENTERS[1]
    axis.plot([x - half, x + half], [y, y], **line)
    for end in (x - half, x + half):
        axis.plot([end, end], [y - 0.045, y + 0.045], **line)
    axis.scatter(x + 0.04, y, s=54, color=BLACK, zorder=3)

    # 2-D: a square
    x, y = CENTERS[2]
    axis.add_patch(Polygon([(x - half, y - half), (x + half, y - half), (x + half, y + half), (x - half, y + half)],
                           closed=True, facecolor="none", edgecolor=WHITE, lw=3.6, joinstyle="round", zorder=2))
    axis.scatter(x - 0.035, y + 0.025, s=54, color=BLACK, zorder=3)

    # 3-D: a wireframe cube in oblique projection
    x, y = CENTERS[3] - 0.5 * DEPTH
    front = np.array([(x - half, y - half), (x + half, y - half), (x + half, y + half), (x - half, y + half)])
    back = front + DEPTH
    for face in (back, front):
        for edge in range(4):
            points = np.array([face[edge], face[(edge + 1) % 4]])
            line_artist, = axis.plot(*points.T, color=WHITE, lw=CUBE_EDGE_WIDTH, solid_capstyle="round", zorder=2)
            line_artist.set_gid("pcat-cube-edge")
    for corner in range(4):
        line_artist, = axis.plot(*np.array([front[corner], back[corner]]).T, color=WHITE,
                                 lw=CUBE_EDGE_WIDTH, solid_capstyle="round", zorder=2)
        line_artist.set_gid("pcat-cube-edge")
    axis.scatter(x + 0.045, y + 0.02, s=54, color=BLACK, zorder=3)

    # Reversible transitions connect the four parameter-space glyphs.
    for start, end in TRANSITIONS:
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="<|-|>", mutation_scale=3,
                                       lw=1.5, color=BLACK, zorder=4))

    axis.text(*LOGO_TEXT_POSITION, "PCAT", color=WHITE, fontsize=LOGO_FONT_SIZE, fontweight="bold",
              family="DejaVu Sans", ha="center", va="center", zorder=5)


def save(figure, path_stem, bbox_inches="tight"):
    PATH_STATIC.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        path = PATH_STATIC / f"{path_stem}.{suffix}"
        print(f"Writing to {path}...")
        metadata = {"Date": None, "Creator": "PCAT"} if suffix == "svg" else None
        figure.savefig(path, dpi=400, transparent=True, bbox_inches=bbox_inches,
                   pad_inches=0.03 if bbox_inches else 0.0, metadata=metadata)
        if suffix == "svg":
            source = path.read_text()
            path.write_text("\n".join(line.rstrip() for line in source.splitlines()) + "\n")
    plt.close(figure)


def draw_logo():
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["svg.hashsalt"] = "pcat-logo"

    def draw_circular_asset(path_stem, figsize, bbox_inches=None):
        figure, axis = plt.subplots(figsize=figsize)
        axis.set(xlim=(-1.02, 1.02), ylim=(-1.02, 1.02), aspect="equal")
        axis.axis("off")
        draw_icon(axis)
        save(figure, path_stem, bbox_inches=bbox_inches)

    draw_circular_asset("pcat_icon", (1.0, 1.0))
    draw_circular_asset("pcat_logo", (3.0, 3.0))

    # Wide site exports repeat the same circular mark without alternate layouts.
    figure, axis = plt.subplots(figsize=(7.5, 2.0))
    axis.set(xlim=(-3.75, 3.75), ylim=(-1.0, 1.0), aspect="equal")
    axis.axis("off")
    draw_icon(axis)
    save(figure, "pcat_banner", bbox_inches=None)

    figure = plt.figure(figsize=(6.4, 3.2), dpi=200, facecolor=WHITE)
    axis = figure.add_axes([0.25, 0.0, 0.5, 1.0])
    axis.set(xlim=(-1.02, 1.02), ylim=(-1.02, 1.02), aspect="equal")
    axis.axis("off")
    draw_icon(axis)
    path = PATH_STATIC / "pcat_social_preview.png"
    print(f"Writing to {path}...")
    figure.savefig(path, dpi=200, facecolor=WHITE)
    plt.close(figure)


if __name__ == "__main__":
    draw_logo()
