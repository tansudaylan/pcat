#!/usr/bin/env python3
"""Draw PCAT schematics: the pipeline, the likelihood families, and the extension points.

Each figure is written as PNG and SVG to docs/_static/diagrams. Box sizes are fixed
in inches, and the script fails if any text overflows its box or two boxes overlap.
"""

from tdpy.verbosity import print

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PATH_DIAGRAMS = Path(__file__).resolve().parent / "_static" / "diagrams"
CRIMSON = "#A51C30"  # Harvard Crimson
INK = "#1A1A1A"
FILLS = {"stage": "#F6E8EA", "step": "#FFFFFF", "data": "#EAF1F7", "core": "#F2F2F2", "hook": "#FFF7E6"}
FONT_SIZE = 10  # [point]
LINE_HEIGHT = 0.2  # [inch]


@dataclass
class Box:
    """A titled box whose center, width, and height are in inches."""

    x: float
    y: float
    width: float
    height: float
    title: str
    lines: tuple[str, ...] = ()
    kind: str = "step"

    def edge(self, side):
        """Return the midpoint of one side: 'left', 'right', 'top', or 'bottom'."""
        return {"left": (self.x - self.width / 2, self.y), "right": (self.x + self.width / 2, self.y),
                "top": (self.x, self.y + self.height / 2), "bottom": (self.x, self.y - self.height / 2)}[side]


def new_figure(width, height):
    """Return a figure whose data units are inches."""
    figure = plt.figure(figsize=(width, height), facecolor="white")
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.set(xlim=(0, width), ylim=(0, height))
    axis.axis("off")
    return figure, axis


def draw_box(axis, box):
    """Draw a rounded box with a bold title and plain body lines, returning its text artists."""
    axis.add_patch(FancyBboxPatch((box.x - box.width / 2, box.y - box.height / 2), box.width, box.height,
                                  boxstyle="round,pad=0,rounding_size=0.08", facecolor=FILLS[box.kind],
                                  edgecolor=CRIMSON if box.kind in ("stage", "core") else INK,
                                  linewidth=1.6 if box.kind in ("stage", "core") else 1.0, zorder=2))
    count = 1 + len(box.lines)
    top = box.y + 0.5 * (count - 1) * LINE_HEIGHT
    artists = [axis.text(box.x, top, box.title, ha="center", va="center", fontsize=FONT_SIZE,
                         fontweight="bold", color=INK, zorder=3)]
    for index, line in enumerate(box.lines, start=1):
        artists.append(axis.text(box.x, top - index * LINE_HEIGHT, line, ha="center", va="center",
                                 fontsize=FONT_SIZE, color=INK, zorder=3))
    return artists


def connect(axis, start, end, label=None, bend=0.0, both=False, label_offset=(0.0, 0.14)):
    """Draw an arrow between two points with an optional label at its midpoint."""
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="<|-|>" if both else "-|>", mutation_scale=12,
                                   lw=1.3, color=INK, connectionstyle=f"arc3,rad={bend}",
                                   shrinkA=2, shrinkB=2, zorder=1))
    if label:
        middle = (0.5 * (start[0] + end[0]) + label_offset[0], 0.5 * (start[1] + end[1]) + label_offset[1])
        axis.text(*middle, label, ha="center", va="center", fontsize=FONT_SIZE, style="italic", color=INK,
                  zorder=3, bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))


def check_layout(figure, boxes, artists):
    """Raise if any text leaves its box or two boxes overlap."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    to_inches = figure.dpi_scale_trans.inverted()
    for box, texts in zip(boxes, artists):
        for text in texts:
            extent = text.get_window_extent(renderer).transformed(to_inches)
            if (extent.x0 < box.x - box.width / 2 + 0.05 or extent.x1 > box.x + box.width / 2 - 0.05
                    or extent.y0 < box.y - box.height / 2 + 0.03 or extent.y1 > box.y + box.height / 2 - 0.03):
                raise ValueError(f"Text {text.get_text()!r} overflows the box {box.title!r}")
    for index, first in enumerate(boxes):
        for second in boxes[index + 1:]:
            if (abs(first.x - second.x) < 0.5 * (first.width + second.width)
                    and abs(first.y - second.y) < 0.5 * (first.height + second.height)):
                raise ValueError(f"Boxes {first.title!r} and {second.title!r} overlap")


def save(figure, name):
    """Write one diagram as PNG (300 dpi) and SVG."""
    PATH_DIAGRAMS.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        path = PATH_DIAGRAMS / f"{name}.{suffix}"
        print(f"Writing to {path}...")
        metadata = {"Date": None} if suffix == "svg" else None
        figure.savefig(path, dpi=300, facecolor="white", metadata=metadata)
    plt.close(figure)


def draw_boxes(axis, boxes):
    return [draw_box(axis, box) for box in boxes]


def draw_pipeline():
    """Five run stages above the cycle of one Metropolis-Hastings sweep."""
    figure, axis = new_figure(14.0, 8.4)
    axis.text(7.0, 8.05, "PCAT pipeline", ha="center", va="center", fontsize=FONT_SIZE, fontweight="bold")
    stages = [
        Box(1.5, 6.6, 2.5, 1.7, "1. Configure", ("sampling.sample(**configuration)", "data, model, element types",
                                                  "priors and proposal settings", "sweeps, burn-in, chains"), "stage"),
        Box(4.25, 6.6, 2.5, 1.7, "2. Set up", ("read or simulate the data", "build the parameter vector",
                                               "unit-cube prior transforms", "write gdatinit"), "stage"),
        Box(7.0, 6.6, 2.5, 1.7, "3. Sample chains", ("numbproc parallel workers", "independent random seeds",
                                                     "adapt and temper in burn-in", "thin and keep samples"), "stage"),
        Box(9.75, 6.6, 2.5, 1.7, "4. Combine", ("merge chains (proc_finl)", "R-hat, autocorrelation, ESS",
                                                "posterior summaries", "write gdatfinlpost"), "stage"),
        Box(12.5, 6.6, 2.5, 1.7, "5. Report", ("frames and animations", "convergence figures",
                                               "sampler operation views", "catalog summaries"), "stage"),
    ]
    steps = [
        Box(2.0, 3.5, 2.9, 1.5, "Propose (prop_stat)", ("choose a move type:", "within, birth, death,",
                                                          "split, merge, or jump"), "step"),
        Box(5.35, 3.5, 2.9, 1.5, "Predict (proc_samp)", ("elements to profiles or images,", "add backgrounds and response,",
                                                           "evaluate log L and log prior"), "step"),
        Box(8.7, 3.5, 2.9, 1.5, "Score (calc_probprop)", (r"$\ln\alpha=\beta\,\Delta\ln L+\Delta\ln\pi$",
                                                            r"$-\ln q_{\mathrm{aux}}+\ln r_{\mathrm{move}}+\ln|J|$"),
            "step"),
        Box(12.0, 3.5, 2.9, 1.5, "Decide", (r"accept with $\min(1,\alpha)$", "update or keep the state",
                                            "adapt step sizes in burn-in"), "step"),
        Box(7.0, 1.1, 3.6, 1.3, "Record", ("store move, acceptance, and timings,", "save every thinned sample after burn-in,",
                                            "optionally stop when converged"), "step"),
    ]
    boxes = stages + steps
    artists = draw_boxes(axis, boxes)
    for first, second in zip(stages[:-1], stages[1:]):
        connect(axis, first.edge("right"), second.edge("left"))
    for first, second in zip(steps[:3], steps[1:4]):
        connect(axis, first.edge("right"), second.edge("left"))
    connect(axis, steps[3].edge("bottom"), steps[4].edge("right"), bend=-0.3)
    connect(axis, steps[4].edge("left"), steps[0].edge("bottom"), "next sweep", bend=-0.3,
            label_offset=(-0.9, -0.45))
    connect(axis, stages[2].edge("bottom"), (7.0, 4.75), "every sweep of every chain", label_offset=(1.25, 0.0))
    axis.add_patch(FancyBboxPatch((0.3, 0.25), 13.4, 4.55, boxstyle="round,pad=0,rounding_size=0.12",
                                  facecolor="none", edgecolor=CRIMSON, linestyle="--", linewidth=1.0, zorder=0))
    axis.text(0.5, 4.55, "One sweep", ha="left", va="center", fontsize=FONT_SIZE, fontweight="bold", color=CRIMSON)
    check_layout(figure, boxes, artists)
    save(figure, "pcat_pipeline")


def draw_likelihoods():
    """Parameters, the forward model of each data family, its likelihood, and the posterior."""
    figure, axis = new_figure(14.0, 8.6)
    axis.text(7.0, 8.25, "From parameters to posterior", ha="center", va="center", fontsize=FONT_SIZE,
              fontweight="bold")
    parameters = Box(1.55, 4.3, 2.7, 3.2, "Parameters", ("fixed parameters, e.g.", "backgrounds, PSF, hyperpriors",
                                                         "", "catalog of N elements,", "each with its own", "position, amplitude, shape",
                                                         "", "sampled in the unit cube"), "core")
    families = [
        ("Images", "chan, ferm, sdss, HST_WFC3",
         ("point sources, extended emission,", "lens deflectors, host light"),
         ("brightness + background templates,", "PSF kernel or convolution,", "× exposure: counts per pixel,",
          "energy, and event class"),
         ("Poisson (liketype='pois')", "or Gaussian ('gaus')")),
        ("Spectra and time series", "fire",
         ("lines: gaus, lore, voig, pvoi, ...", "flares: flargauss, flarfred, ...", "rotating starspots: spotrot"),
         ("profiles + continuum template,", "optional line-spread function,", "× exposure: counts per bin"),
         ("Poisson or Gaussian,", "or retr_llik on the model")),
        ("Radial velocities", "fire, lghtlinekepl",
         ("Keplerian orbits:", "K, P, phase, e, omega"),
         ("sum of Keplerian", "velocity curves [m/s]"),
         ("Gaussian, instrument offsets", "marginalized analytically,", "jitter numerically")),
        ("Fixed-dimensional", "gener",
         ("named parameters", "with chosen priors"),
         ("any user model, e.g.", "Ephesos transit times,", "Roman lens images"),
         ("user retr_llik, e.g.", "log_likelihood_transit_times")),
    ]
    rows = (7.0, 5.25, 3.35, 1.45)
    heights = (1.55, 1.55, 1.3, 1.3)
    boxes, links = [parameters], []
    for (title, typeexpr, elements, model, likelihood), y, height in zip(families, rows, heights):
        element_box = Box(4.75, y, 2.9, height, title, (f"typeexpr: {typeexpr}",) + elements, "data")
        model_box = Box(8.05, y, 2.9, height, "Forward model", model, "step")
        likelihood_box = Box(11.0, y, 2.4, height, "Likelihood", likelihood, "step")
        boxes += [element_box, model_box, likelihood_box]
        links.append((element_box, model_box, likelihood_box))
    posterior = Box(13.15, 4.3, 1.5, 2.2, "Posterior", (r"$\ln P=$", r"$\beta\ln L$", r"$+\ln\pi$",
                                                        r"$\beta<1$ only in", "tempered burn-in"), "core")
    boxes.append(posterior)
    artists = draw_boxes(axis, boxes)
    for element_box, model_box, likelihood_box in links:
        connect(axis, parameters.edge("right"), element_box.edge("left"))
        connect(axis, element_box.edge("right"), model_box.edge("left"))
        connect(axis, model_box.edge("right"), likelihood_box.edge("left"))
        connect(axis, likelihood_box.edge("right"), posterior.edge("left"))
    check_layout(figure, boxes, artists)
    save(figure, "pcat_likelihoods")


def draw_extension_points():
    """The sampler core surrounded by the parts users can replace or tune."""
    figure, axis = new_figure(14.0, 8.4)
    axis.text(7.0, 8.05, "Custom parts and the settings that control them", ha="center", va="center",
              fontsize=FONT_SIZE, fontweight="bold")
    core = Box(7.0, 4.2, 3.4, 2.2, "PCAT sampler core", ("unit-cube parameters and priors",
                                                          "Metropolis-Hastings moves", "parallel chains, persistence",
                                                          "convergence diagnostics"), "core")
    hooks = [
        Box(2.25, 6.75, 3.7, 1.55, "Likelihood", ("retr_llik(gdat, strgmodl, values)", "for typeexpr='gener', or on the",
                                                  "predicted model of a catalog run"), "hook"),
        Box(7.0, 6.75, 3.7, 1.55, "Data and response", ("typedata='inpt' with sbrt and expo FITS,",
                                                         "background templates (sbrtbacknorm),",
                                                         "PSF (typeevalpsfn), LSF (lsftype)"), "hook"),
        Box(11.75, 6.75, 3.7, 1.55, "Element profiles", ("typeelem and spectype select the shape;",
                                                         "new shapes join pcat.spectral or",
                                                         "pcat.time_series and retr_elem_spec"), "hook"),
        Box(2.25, 4.2, 3.7, 1.55, "Priors", ("prior_types, prior_minima, prior_maxima,", "limtparaelem, spatdisttype,",
                                             "typeprioflux, catalog-size limits"), "hook"),
        Box(11.75, 4.2, 3.7, 1.55, "Data-informed proposals", ("retr_drawpropelem, retr_lpdfpropelem,",
                                                                "built by retr_dictpropelemtmpl;",
                                                                "Hastings term keeps the chain exact"), "hook"),
        Box(2.25, 1.65, 3.7, 1.55, "Move mix and step sizes", ("probtran, probspmr, probjump, radispmr,",
                                                               "proposal_scales, proposal_correlation,",
                                                               "proposal_blocks, probdemc"), "hook"),
        Box(7.0, 1.65, 3.7, 1.55, "Burn-in and stopping", ("numbburn, booladaptstdp,", "boolburntmpr, factburntmpr,",
                                                           "boolcheckconv with R-hat and ESS"), "hook"),
        Box(11.75, 1.65, 3.7, 1.55, "Plots", ("plot_func(gdat) during sampling;", "pcat.plotting views of gdatfinlpost,",
                                              "docs/make_plot_gallery.py"), "hook"),
    ]
    boxes = [core] + hooks
    artists = draw_boxes(axis, boxes)
    for hook in hooks:
        if abs(hook.x - core.x) < 1e-6:
            start = hook.edge("bottom") if hook.y > core.y else hook.edge("top")
            end = core.edge("top") if hook.y > core.y else core.edge("bottom")
        elif abs(hook.y - core.y) < 1e-6:
            start = hook.edge("right") if hook.x < core.x else hook.edge("left")
            end = core.edge("left") if hook.x < core.x else core.edge("right")
        else:
            start = hook.edge("bottom") if hook.y > core.y else hook.edge("top")
            end = (core.x + (-1.0 if hook.x < core.x else 1.0) * 1.2,
                   core.y + (1.0 if hook.y > core.y else -1.0) * core.height / 2)
        connect(axis, start, end)
    check_layout(figure, boxes, artists)
    save(figure, "pcat_extension_points")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    mpl.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "pcat-diagrams", "text.usetex": False})
    draw_pipeline()
    draw_likelihoods()
    draw_extension_points()


if __name__ == "__main__":
    main()
