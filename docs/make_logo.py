#!/usr/bin/env python3
"""Generate PCAT's circular logo and site-ready exports.

A point, a segment, a square, and a cube, the parameter spaces of dimension zero
to three, sit on two rows joined by reversible arrows, above the PCAT wordmark.
The artwork is scaled and shifted to the largest size at which every stroke
stays inside the disc, which keeps the unused crimson area small.
"""

from tdpy.verbosity import print

import itertools
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Polygon
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

PATH_STATIC = Path(__file__).resolve().parent / "_static"
CRIMSON = "#A51C30"  # Harvard Crimson
BLACK = "#000000"
WHITE = "#FFFFFF"
DISC_RADIUS = 0.98
OUTLINE_WIDTH = 0.022  # disc outline width in logo coordinates
CONTENT_RADIUS = DISC_RADIUS - OUTLINE_WIDTH - 0.02  # every stroke stays inside this radius

# Lengths below are in units of the glyph edge length and are scaled to fit the disc.
STROKE = 0.085  # glyph line width
DEPTH = np.array([0.28, 0.24])  # oblique offset of the cube's back face
CAP = 0.13  # half height of the segment end caps
POINT_DISC = 0.42  # diameter of the white disc of the 0-D point
SAMPLE_DOT = 0.17  # diameter of the black sample dot in each space
ARROW_WIDTH = 0.055  # arrow shaft width
ARROW_HEAD = 0.13  # arrow head length
GAP = 0.1  # clearance between an arrow tip and a glyph
ARROW_LENGTH = 0.72  # length of the horizontal arrows
ROW_GAP = 0.6  # vertical clearance between the two rows
TEXT_GAP = 0.3  # clearance between the bottom row and the wordmark
LETTER_RATIO = 0.7  # minimum letter height relative to the glyph edge length
WORDMARK = TextPath((0.0, 0.0), "PCAT", size=1.0,
                    prop=FontProperties(family="DejaVu Sans", weight="bold"))


def cube_faces(center):
    """Return the front and back faces of the oblique wireframe cube."""
    x, y = center - 0.5 * DEPTH
    front = np.array([(x - 0.5, y - 0.5), (x + 0.5, y - 0.5), (x + 0.5, y + 0.5), (x - 0.5, y + 0.5)])
    return front, front + DEPTH


def build_layout(text_fraction):
    """Return glyph centers, arrows, the wordmark transform, outline points, and letter height.

    ``text_fraction`` is the wordmark width relative to the width of the glyph block.
    """
    top = 0.5 + 0.5 * DEPTH[1] + ROW_GAP + CAP
    # place the left glyphs so that both horizontal arrows have length ARROW_LENGTH
    right = 0.5 + GAP + 0.5 * ARROW_LENGTH
    centers = {
        "point": np.array([right - 0.5 - 2 * GAP - ARROW_LENGTH - 0.5 * POINT_DISC, top]),
        "segment": np.array([right, top]),
        "square": np.array([right - 1.0 - 2 * GAP - ARROW_LENGTH, 0.0]),
        "cube": np.array([right + 0.5 * DEPTH[0], 0.0]),
    }
    arrows = [
        (centers["point"] + [0.5 * POINT_DISC + GAP, 0.0], centers["segment"] - [0.5 + GAP, 0.0]),
        (centers["segment"] + [-0.35, -CAP - GAP], centers["square"] + [0.35, 0.5 + GAP]),
        (centers["square"] + [0.5 + GAP, 0.0], centers["cube"] - [0.5 + 0.5 * DEPTH[0] + GAP, 0.0]),
    ]
    front, back = cube_faces(centers["cube"])
    glyph_points = np.concatenate([
        centers["point"] + 0.5 * POINT_DISC * np.stack(
            [np.cos(np.linspace(0, 2 * np.pi, 72)), np.sin(np.linspace(0, 2 * np.pi, 72))], 1),
        centers["segment"] + np.array([[-0.5, -CAP], [-0.5, CAP], [0.5, -CAP], [0.5, CAP]]) * (1 + STROKE),
        centers["square"] + np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) * (1 + STROKE),
        front + 0.5 * STROKE * np.sign(front - front.mean(axis=0)),
        back + 0.5 * STROKE * np.sign(back - back.mean(axis=0)),
    ])

    # the wordmark spans the width of the glyph block
    vertices = WORDMARK.vertices[np.isfinite(WORDMARK.vertices).all(axis=1)]
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    left, right = glyph_points[:, 0].min(), glyph_points[:, 0].max()
    text_scale = text_fraction * (right - left) / (high[0] - low[0])
    text_top = glyph_points[:, 1].min() - TEXT_GAP
    text = Affine2D().translate(-0.5 * (low[0] + high[0]), -high[1]).scale(text_scale).translate(
        0.5 * (left + right), text_top)
    outline = np.concatenate([glyph_points, text.transform(vertices)])
    return centers, arrows, text, outline, (high[1] - low[1]) * text_scale


def fit_to_disc(outline):
    """Return the largest scale and the shift that keep the outline inside the content radius."""
    best_scale, best_shift = 0.0, np.zeros(2)
    middle = 0.5 * (outline.min(axis=0) + outline.max(axis=0))
    for dx, dy in itertools.product(np.linspace(-0.3, 0.3, 31), np.linspace(-0.6, 0.6, 121)):
        shift = np.array([dx, dy]) - middle
        scale = CONTENT_RADIUS / np.max(np.hypot(*(outline + shift).T))
        if scale > best_scale:
            best_scale, best_shift = scale, shift
    return best_scale, best_shift


def choose_layout():
    """Pick the wordmark width at which glyphs and letters are jointly as large as possible."""
    best = None
    for fraction in np.linspace(0.55, 1.0, 46):
        centers, arrows, text, outline, letter_height = build_layout(fraction)
        scale, shift = fit_to_disc(outline)
        score = scale * min(1.0, letter_height / LETTER_RATIO)
        if best is None or score > best[0]:
            best = (score, centers, arrows, text, scale, shift)
    return best[1:]


CENTERS, ARROWS, TEXT_TRANSFORM, SCALE, SHIFT = choose_layout()


def draw_disc(axis, points_per_unit):
    """Draw the circular Harvard Crimson field and black outline."""
    axis.add_patch(Circle((0.0, 0.0), DISC_RADIUS - 0.5 * OUTLINE_WIDTH, facecolor=CRIMSON,
                          edgecolor=BLACK, linewidth=OUTLINE_WIDTH * points_per_unit, zorder=0))


def draw_icon(axis):
    """Draw the four-dimensionality glyphs and PCAT wordmark inside the circle."""
    axis.apply_aspect()
    pixels_per_unit = axis.transData.transform((1.0, 0.0))[0] - axis.transData.transform((0.0, 0.0))[0]
    points_per_unit = pixels_per_unit * 72.0 / axis.figure.dpi  # [point] per logo-coordinate unit
    draw_disc(axis, points_per_unit)

    def place(points):
        return SCALE * (np.asarray(points, dtype=float) + SHIFT)

    def width(value):
        return SCALE * value * points_per_unit  # [point]

    def dot(center, diameter, color, zorder):
        axis.scatter(*place(center), s=width(diameter) ** 2, color=color, linewidths=0, zorder=zorder)

    line = dict(color=WHITE, lw=width(STROKE), solid_capstyle="round", solid_joinstyle="round", zorder=2)

    # 0-D: a point
    dot(CENTERS["point"], POINT_DISC, WHITE, 2)
    dot(CENTERS["point"], SAMPLE_DOT, BLACK, 3)

    # 1-D: a segment with end caps
    x, y = CENTERS["segment"]
    axis.plot(*place([[x - 0.5, y], [x + 0.5, y]]).T, **line)
    for end in (x - 0.5, x + 0.5):
        axis.plot(*place([[end, y - CAP], [end, y + CAP]]).T, **line)
    dot((x + 0.15, y), SAMPLE_DOT, BLACK, 3)

    # 2-D: a square
    x, y = CENTERS["square"]
    axis.add_patch(Polygon(place([(x - 0.5, y - 0.5), (x + 0.5, y - 0.5), (x + 0.5, y + 0.5), (x - 0.5, y + 0.5)]),
                           closed=True, facecolor="none", edgecolor=WHITE, lw=width(STROKE),
                           joinstyle="round", zorder=2))
    dot((x - 0.13, y + 0.1), SAMPLE_DOT, BLACK, 3)

    # 3-D: a wireframe cube in oblique projection
    front, back = cube_faces(CENTERS["cube"])
    for face in (back, front):
        for edge in range(4):
            artist, = axis.plot(*place([face[edge], face[(edge + 1) % 4]]).T, **line)
            artist.set_gid("pcat-cube-edge")
    for corner in range(4):
        artist, = axis.plot(*place([front[corner], back[corner]]).T, **line)
        artist.set_gid("pcat-cube-edge")
    dot(front.mean(axis=0) + [0.17, 0.09], SAMPLE_DOT, BLACK, 3)

    # Reversible transitions connect the four parameter-space glyphs.
    head = width(ARROW_HEAD)
    for start, end in ARROWS:
        axis.add_patch(FancyArrowPatch(place(start), place(end),
                                       arrowstyle=f"<|-|>,head_length={head:.3f},head_width={0.32 * head:.3f}",
                                       mutation_scale=1.0, lw=width(ARROW_WIDTH),
                                       shrinkA=0, shrinkB=0, color=BLACK, zorder=4))

    transform = TEXT_TRANSFORM + Affine2D().translate(*SHIFT).scale(SCALE)
    axis.add_patch(PathPatch(transform.transform_path(WORDMARK), facecolor=WHITE, edgecolor="none",
                             zorder=5, gid="pcat-wordmark"))


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
        figure = plt.figure(figsize=figsize)
        axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
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
