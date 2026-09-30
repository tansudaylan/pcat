#!/usr/bin/env python3
"""Draw the PCAT logo in Harvard Crimson and black.

A point, a line segment, a square, and a cube are parameter spaces of dimension zero to three.
Reversible arrows jump between them, the birth and death moves with which PCAT changes the
dimension of its model, and a black dot in each space marks the sampled state. The script writes
a stacked logo with the name, a wide banner, and a circular icon without text for favicons and
social previews.
"""

from tdpy.verbosity import print

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

PATH_STATIC = Path(__file__).resolve().parent / "_static"
CRIMSON = "#A51C30"  # Harvard Crimson
BLACK = "#000000"
WHITE = "#FFFFFF"
# centers of the 0-, 1-, 2-, and 3-dimensional spaces in icon coordinates, read in a Z pattern
CENTERS = np.array([[-0.55, 0.52], [0.4, 0.52], [-0.56, -0.44], [0.5, -0.5]])
SIZE = 0.46  # edge length of the segment, square, and cube
DEPTH = np.array([0.18, 0.15])  # oblique offset of the cube's back face


def draw_disc(axis):
    """Draw the common circular Harvard Crimson brand field."""
    axis.add_patch(Circle((0.0, 0.0), 0.98, facecolor=CRIMSON, edgecolor=BLACK, linewidth=2.5, zorder=0))


def draw_icon(axis):
    """Draw the circular mark with spaces of increasing dimension and reversible jumps."""
    draw_disc(axis)
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


def draw_nested_spaces(axis):
    """Show nested parameter spaces linked by a reversible outer orbit."""
    draw_disc(axis)
    for radius, linewidth in ((0.22, 3.2), (0.43, 2.8), (0.67, 2.4)):
        axis.add_patch(Circle((0.0, 0.0), radius, fill=False, edgecolor=WHITE, linewidth=linewidth, zorder=2))
    axis.scatter((0.0, 0.22, -0.3, 0.45), (0.0, 0.0, 0.3, -0.35), s=(80, 58, 58, 58), color=BLACK, zorder=3)
    axis.add_patch(FancyArrowPatch((-0.72, 0.42), (0.72, 0.42), connectionstyle="arc3,rad=-0.48",
                                   arrowstyle="<|-|>", mutation_scale=14, lw=2.2, color=BLACK, zorder=4))


def draw_catalog_birth_death(axis):
    """Show reversible movement between catalogs containing two and three objects."""
    draw_disc(axis)
    left = np.array([[-0.58, 0.22], [-0.58, -0.22]])
    right = np.array([[0.58, 0.32], [0.43, -0.18], [0.73, -0.25]])
    axis.scatter(left[:, 0], left[:, 1], s=180, facecolor=WHITE, edgecolor=BLACK, linewidth=2.0, zorder=3)
    axis.scatter(right[:, 0], right[:, 1], s=180, facecolor=WHITE, edgecolor=BLACK, linewidth=2.0, zorder=3)
    axis.add_patch(FancyArrowPatch((-0.3, 0.0), (0.25, 0.0), arrowstyle="<|-|>", mutation_scale=17,
                                   lw=2.8, color=BLACK, zorder=4))
    axis.text(-0.58, -0.52, "2", color=WHITE, fontsize=22, fontweight="bold", ha="center", va="center")
    axis.text(0.58, -0.52, "3", color=WHITE, fontsize=22, fontweight="bold", ha="center", va="center")


def draw_dimension_ladder(axis):
    """Show sampled states ascending through model dimensions zero to three."""
    draw_disc(axis)
    heights = (0.20, 0.38, 0.56, 0.74)
    x_positions = (-0.67, -0.23, 0.21, 0.65)
    for dimension, (x_position, height) in enumerate(zip(x_positions, heights)):
        axis.add_patch(Rectangle((x_position - 0.16, -0.62), 0.32, height, fill=False,
                                 edgecolor=WHITE, linewidth=2.5, zorder=2))
        axis.scatter(x_position, -0.62 + 0.72 * height, s=58, color=BLACK, zorder=3)
        axis.text(x_position, -0.78, str(dimension), color=WHITE, fontsize=15, fontweight="bold",
                  ha="center", va="center")
    axis.add_patch(FancyArrowPatch((-0.75, 0.3), (0.73, 0.56), connectionstyle="arc3,rad=-0.16",
                                   arrowstyle="<|-|>", mutation_scale=14, lw=2.2, color=BLACK, zorder=4))


def draw_model_orbit(axis):
    """Show posterior states moving between four model nodes."""
    draw_disc(axis)
    nodes = np.array([[-0.52, 0.46], [0.52, 0.46], [0.52, -0.46], [-0.52, -0.46]])
    for index, (start, end) in enumerate(zip(nodes, np.roll(nodes, -1, axis=0))):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                       lw=2.4, color=WHITE, zorder=2))
        axis.text(*(0.82 * nodes[index]), str(index), color=BLACK, fontsize=14, fontweight="bold",
                  ha="center", va="center", zorder=4)
    axis.scatter(nodes[:, 0], nodes[:, 1], s=(95, 135, 175, 215), facecolor=CRIMSON,
                 edgecolor=BLACK, linewidth=2.5, zorder=3)
    axis.scatter(0.0, 0.0, s=95, color=BLACK, zorder=3)


def draw_reversible_branching(axis):
    """Show one catalog state branching reversibly into larger catalogs."""
    draw_disc(axis)
    levels = (
        np.array([[0.0, 0.62]]),
        np.array([[-0.34, 0.08], [0.34, 0.08]]),
        np.array([[-0.56, -0.52], [0.0, -0.52], [0.56, -0.52]]),
    )
    for parent, children in zip(levels[:-1], levels[1:]):
        for child_index, child in enumerate(children):
            source = parent[min(child_index * len(parent) // len(children), len(parent) - 1)]
            axis.plot((source[0], child[0]), (source[1], child[1]), color=WHITE, lw=2.2, zorder=1)
    for level in levels:
        axis.scatter(level[:, 0], level[:, 1], s=145, facecolor=WHITE, edgecolor=BLACK, linewidth=2.0, zorder=3)
    axis.add_patch(FancyArrowPatch((-0.78, 0.35), (-0.78, -0.38), arrowstyle="<|-|>", mutation_scale=15,
                                   lw=2.5, color=BLACK, zorder=4))


CONCEPTS = (
    ("Nested spaces", "pcat_logo_concept_nested_spaces", draw_nested_spaces),
    ("Catalog birth/death", "pcat_logo_concept_catalog_birth_death", draw_catalog_birth_death),
    ("Dimension ladder", "pcat_logo_concept_dimension_ladder", draw_dimension_ladder),
    ("Posterior model orbit", "pcat_logo_concept_model_orbit", draw_model_orbit),
    ("Reversible branching", "pcat_logo_concept_reversible_branching", draw_reversible_branching),
)


def save(figure, path_stem):
    PATH_STATIC.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        path = PATH_STATIC / f"{path_stem}.{suffix}"
        print(f"Writing to {path}...")
        metadata = {"Date": None, "Creator": "PCAT"} if suffix == "svg" else None
        figure.savefig(path, dpi=400, transparent=True, bbox_inches="tight", pad_inches=0.03, metadata=metadata)
        if suffix == "svg":
            source = path.read_text()
            path.write_text("\n".join(line.rstrip() for line in source.splitlines()) + "\n")
    plt.close(figure)


def draw_concepts():
    """Write five circular alternatives and one labeled comparison sheet."""
    for _, path_stem, draw_concept in CONCEPTS:
        figure, axis = plt.subplots(figsize=(3.0, 3.0))
        axis.set(xlim=(-1.02, 1.02), ylim=(-1.02, 1.02), aspect="equal")
        axis.axis("off")
        draw_concept(axis)
        save(figure, path_stem)

    figure, axes = plt.subplots(2, 3, figsize=(9.0, 6.4), facecolor=WHITE)
    entries = (("Current circular mark", draw_icon),) + tuple((label, function) for label, _, function in CONCEPTS)
    for axis, (label, draw_concept) in zip(axes.flat, entries):
        axis.set(xlim=(-1.02, 1.02), ylim=(-1.18, 1.02), aspect="equal")
        axis.axis("off")
        draw_concept(axis)
        axis.text(0.0, -1.08, label, color=BLACK, fontsize=11, fontweight="bold", ha="center", va="center")
    figure.tight_layout()
    path = PATH_STATIC / "pcat_logo_concepts.png"
    print(f"Writing to {path}...")
    figure.savefig(path, dpi=300, facecolor=WHITE, bbox_inches="tight")
    plt.close(figure)


def draw_logo():
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["svg.hashsalt"] = "pcat-logo"
    font = dict(fontweight="bold", color=BLACK, family="DejaVu Sans")

    # circular icon for favicons and social previews
    figure, axis = plt.subplots(figsize=(3.0, 3.0))
    axis.set(xlim=(-1.02, 1.02), ylim=(-1.02, 1.02), aspect="equal")
    axis.axis("off")
    draw_icon(axis)
    save(figure, "pcat_icon")

    # stacked logo with the name below the icon
    figure, axis = plt.subplots(figsize=(3.0, 3.9))
    axis.set(xlim=(-1.02, 1.02), ylim=(-1.62, 1.02), aspect="equal")
    axis.axis("off")
    draw_icon(axis)
    axis.text(0.0, -1.33, "PCAT", ha="center", va="center", fontsize=42, **font)
    save(figure, "pcat_logo")

    # wide banner with the name and its expansion beside the icon
    figure, axis = plt.subplots(figsize=(7.5, 2.0))
    axis.set(xlim=(-1.02, 4.45), ylim=(-1.02, 1.02), aspect="equal")
    axis.axis("off")
    draw_icon(axis)
    axis.text(1.3, 0.2, "PCAT", ha="left", va="center", fontsize=48, **font)
    axis.text(1.34, -0.62, "Probabilistic Cataloger", ha="left", va="center", fontsize=17, color=CRIMSON,
              family="DejaVu Sans")
    save(figure, "pcat_banner")

    # GitHub social preview at its recommended 1280 x 640 pixels
    figure = plt.figure(figsize=(6.4, 3.2), dpi=200)
    axis = figure.add_axes([0.06, 0.12, 0.88, 0.76])
    axis.set(xlim=(-1.02, 4.45), ylim=(-1.02, 1.02), aspect="equal")
    axis.axis("off")
    draw_icon(axis)
    axis.text(1.3, 0.2, "PCAT", ha="left", va="center", fontsize=40, **font)
    axis.text(1.34, -0.62, "Probabilistic Cataloger", ha="left", va="center", fontsize=14, color=CRIMSON,
              family="DejaVu Sans")
    path = PATH_STATIC / "pcat_social_preview.png"
    print(f"Writing to {path}...")
    figure.savefig(path, dpi=200, facecolor=WHITE)
    plt.close(figure)
    draw_concepts()


if __name__ == "__main__":
    draw_logo()
