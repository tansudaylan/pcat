"""Native plotting functions for PCAT posterior products."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _output_path(path, name, typefileplot):
    """Return the output path for a named PCAT posterior grid."""
    if typefileplot not in ("png", "pdf"):
        raise ValueError("typefileplot must be 'png' or 'pdf'")
    output_path = Path(f"{path}_{name}")
    if output_path.suffix not in (".png", ".pdf"):
        output_path = output_path.with_suffix(f".{typefileplot}")
    return output_path


def plot_grid(
    path,
    name,
    listpara,
    listlablparatotl,
    scalpara=None,
    truepara=None,
    join=False,
    listvarbdraw=None,
    typefileplot="pdf",
    **kwargs,
):
    """Plot all one- and two-dimensional projections of posterior samples."""
    samples = np.asarray(listpara)
    if samples.size == 0:
        raise ValueError("listpara must contain posterior samples")
    if samples.ndim == 1:
        samples = samples[:, None]
    elif samples.ndim > 2:
        samples = samples.reshape(samples.shape[0], -1)
    parameter_count = samples.shape[1]
    if parameter_count == 0:
        raise ValueError("listpara must contain at least one parameter")
    labels = list(listlablparatotl) if listlablparatotl is not None else []
    labels.extend(
        f"para{index:04d}" for index in range(len(labels), parameter_count)
    )
    truth = None
    if truepara is not None:
        truth = np.asarray(
            [np.nan if value is None else value for value in truepara], dtype=float
        ).reshape(-1)
    draws = [
        np.asarray(draw, dtype=float).reshape(-1) for draw in listvarbdraw or []
    ]

    figure, axes = plt.subplots(
        parameter_count,
        parameter_count,
        figsize=(1.7 * parameter_count, 1.7 * parameter_count),
        facecolor="white",
        squeeze=False,
    )
    for row in range(parameter_count):
        for column in range(parameter_count):
            axis = axes[row, column]
            axis.grid(False)
            if column > row:
                axis.axis("off")
                continue
            if row == column:
                axis.hist(
                    samples[:, column],
                    bins=30,
                    histtype="step",
                    color="black",
                    linewidth=1.2,
                )
                if truth is not None and column < truth.size and np.isfinite(truth[column]):
                    axis.axvline(truth[column], color="#007360", linewidth=1.3)
                for draw in draws:
                    if column < draw.size and np.isfinite(draw[column]):
                        axis.axvline(
                            draw[column],
                            color="#B23A48",
                            linewidth=1.0,
                            linestyle="--",
                        )
            else:
                axis.scatter(
                    samples[:, column],
                    samples[:, row],
                    s=1.0,
                    alpha=0.12,
                    color="black",
                    edgecolors="none",
                    rasterized=True,
                )
                if (
                    truth is not None
                    and row < truth.size
                    and np.all(np.isfinite(truth[[column, row]]))
                ):
                    axis.scatter(
                        truth[column],
                        truth[row],
                        color="#007360",
                        marker="x",
                        s=28,
                        linewidths=1.3,
                        zorder=3,
                    )
                for draw in draws:
                    if row < draw.size and np.all(np.isfinite(draw[[column, row]])):
                        axis.scatter(
                            draw[column],
                            draw[row],
                            color="#B23A48",
                            marker="+",
                            s=24,
                            linewidths=1.0,
                            zorder=3,
                        )
            axis.tick_params(labelsize=8, colors="black")
            axis.spines[["top", "right"]].set_visible(False)
            if row == parameter_count - 1:
                axis.set_xlabel(labels[column], fontsize=8)
            else:
                axis.tick_params(labelbottom=False)
            if column == 0 and row > 0:
                axis.set_ylabel(labels[row], fontsize=8)
            elif column > 0:
                axis.tick_params(labelleft=False)
    figure.subplots_adjust(
        left=0.10,
        right=0.99,
        bottom=0.10,
        top=0.99,
        wspace=0.08,
        hspace=0.08,
    )
    output_path = _output_path(path, name, typefileplot)
    print(f"Writing to {output_path}...")
    figure.savefig(
        output_path,
        dpi=300 if typefileplot == "png" else None,
        bbox_inches="tight",
    )
    plt.close(figure)
    return output_path