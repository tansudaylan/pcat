"""Native plotting functions for PCAT posterior products."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tdpy.util import save_figure


EXAMPLES_ROOT = Path(__file__).resolve().parents[1] / "examples"
DEFAULT_POSTERIOR_COLLAGE = EXAMPLES_ROOT / "pcat_posterior_samples.gif"


def plot_lens_image_fit(
    output_path: Path,
    observed: np.ndarray,
    model: np.ndarray,
    variance: np.ndarray,
    pixel_scale_arcsec: float,
    typefileplot: str = "png",
) -> Path:
    """Write observed, median-model, and standardized-residual lens maps."""
    if typefileplot not in ("png", "pdf"):
        raise ValueError("typefileplot must be 'png' or 'pdf'")
    residual = (observed - model) / np.sqrt(variance)
    half_width_arcsec = observed.shape[0] * pixel_scale_arcsec / 2.0  # [arcsec]
    extent = (-half_width_arcsec, half_width_arcsec, -half_width_arcsec, half_width_arcsec)
    count_limit = np.nanpercentile(observed, 99.5)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, image, title, bounds, colorbar_label in zip(
        axes,
        (observed, model, residual),
        ("Simulated observation", "PCAT median model", "Standardized residual"),
        ((0.0, count_limit), (0.0, count_limit), (-5.0, 5.0)),
        ("Electrons [pixel$^{-1}$]", "Electrons [pixel$^{-1}$]", "Residual [$\\sigma$]"),
    ):
        shown = axis.imshow(
            image,
            origin="lower",
            extent=extent,
            cmap="Greys" if title != "Standardized residual" else "RdBu_r",
            vmin=bounds[0],
            vmax=bounds[1],
            interpolation="nearest",
        )
        axis.set_title(title)
        axis.set_xlabel("Offset [arcsec]")
        axis.set_ylabel("Offset [arcsec]")
        axis.grid(False)
        figure.colorbar(shown, ax=axis, label=colorbar_label, shrink=0.8)
    output_path = Path(output_path).with_suffix(f".{typefileplot}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(output_path, dpi=300 if typefileplot == "png" else None, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_lens_parameter_recovery(
    output_path: Path,
    draws: np.ndarray,
    true_parameters: np.ndarray,
    typefileplot: str = "png",
) -> Path:
    """Write marginal lens-parameter distributions with injected values."""
    if typefileplot not in ("png", "pdf"):
        raise ValueError("typefileplot must be 'png' or 'pdf'")
    labels = ("Einstein radius [arcsec]", "Source x [arcsec]", "Source y [arcsec]")
    medians = np.median(draws, axis=0)
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
    for index, (axis, label) in enumerate(zip(axes, labels)):
        axis.hist(draws[:, index], bins=22, density=True, color="#007360", alpha=0.75,
                  label="PCAT samples")
        axis.axvline(true_parameters[index], color="#A51417", linewidth=2,
                     label="Injected value")
        axis.axvline(medians[index], color="black", linestyle="--", linewidth=1.5,
                     label="Posterior median")
        axis.set_xlabel(label)
        axis.set_ylabel("Density [arcsec$^{-1}$]")
        axis.grid(False)
    axes[0].legend(loc="upper left", frameon=True, fancybox=True, framealpha=1.0, fontsize=8)
    output_path = Path(output_path).with_suffix(f".{typefileplot}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(output_path, dpi=300 if typefileplot == "png" else None, bbox_inches="tight")
    plt.close(figure)
    return output_path


@dataclass(frozen=True)
class PosteriorAnimationPanel:
    """Describe one posterior-frame sequence in an animation collage."""

    label: str
    pattern: str


POSTERIOR_ANIMATION_PANELS = (
    PosteriorAnimationPanel(
        "Gaussian mixture | model intensity",
        "gaussian_mixture_catalog/visuals/post/fram/thiscntpmodl_*.png",
    ),
    PosteriorAnimationPanel(
        "Point sources | model counts",
        "daylan_2017_fermi_point_sources/visuals/post/fram/thiscntpmodlen00evt0_*.png",
    ),
    PosteriorAnimationPanel(
        "Strong lens | model counts",
        "simulated_hst_strong_lens/visuals/post/fram/thiscntpmodl_*.png",
    ),
    PosteriorAnimationPanel(
        "Spectral lines | model and data",
        "voigt_spectral_line_catalog/visuals/post/fram/thisscatcntpevt0_*.png",
    ),
)


def _animation_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Use a widely available font with a portable fallback."""
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _animation_frame_paths(
    panel: PosteriorAnimationPanel, examples_root: Path
) -> list[Path]:
    paths = sorted(examples_root.glob(panel.pattern))
    if len(paths) < 2:
        raise RuntimeError(
            f"{panel.label} requires at least two posterior frames matching "
            f"{examples_root / panel.pattern}. Run examples/run_all_examples.py first."
        )
    return paths


def _animation_sample_index(
    frame_index: int, output_count: int, input_count: int
) -> int:
    """Spread each available sequence over the common animation timeline."""
    if output_count == 1:
        return input_count - 1
    return round(frame_index * (input_count - 1) / (output_count - 1))


def make_posterior_animation_collage(
    output_path: Path = DEFAULT_POSTERIOR_COLLAGE,
    examples_root: Path = EXAMPLES_ROOT,
    panels: tuple[PosteriorAnimationPanel, ...] = POSTERIOR_ANIMATION_PANELS,
    frame_count: int = 10,
    duration_ms: int = 500,
    panel_size: int = 480,
) -> Path:
    """Write a synchronized two-column collage of posterior frame sequences."""
    if frame_count < 2:
        raise ValueError("frame_count must be at least two.")
    if not panels:
        raise ValueError("panels must contain at least one frame sequence.")

    sequences = [_animation_frame_paths(panel, examples_root) for panel in panels]
    column_count = min(2, len(panels))
    row_count = int(np.ceil(len(panels) / column_count))
    margin = 20
    title_height = 66
    label_height = 42
    canvas_size = (
        column_count * panel_size + (column_count + 1) * margin,
        title_height + row_count * (panel_size + label_height) + (row_count + 1) * margin,
    )
    title_font = _animation_font(28, bold=True)
    label_font = _animation_font(20, bold=True)
    counter_font = _animation_font(18)
    frames: list[Image.Image] = []

    for frame_index in range(frame_count):
        canvas = Image.new("RGB", canvas_size, "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, margin), "PCAT posterior samples", fill="black", font=title_font)
        counter = f"draw {frame_index + 1:02d} / {frame_count:02d}"
        counter_box = draw.textbbox((0, 0), counter, font=counter_font)
        draw.text(
            (canvas.width - margin - (counter_box[2] - counter_box[0]), margin + 6),
            counter,
            fill="#3f4b4b",
            font=counter_font,
        )

        for panel_index, (panel, paths) in enumerate(zip(panels, sequences)):
            row, column = divmod(panel_index, column_count)
            x = margin + column * (panel_size + margin)
            y = title_height + margin + row * (panel_size + label_height + margin)
            source_index = _animation_sample_index(frame_index, frame_count, len(paths))
            source_path = paths[source_index]
            print(f"Reading from {source_path}...")
            with Image.open(source_path) as source:
                panel_image = source.convert("RGB").resize(
                    (panel_size, panel_size), Image.Resampling.LANCZOS
                )
            canvas.paste(panel_image, (x, y))
            draw.rectangle(
                (x, y, x + panel_size - 1, y + panel_size - 1),
                outline="#c4cccc",
                width=2,
            )
            draw.text(
                (x, y + panel_size + 8),
                panel.label,
                fill="black",
                font=label_font,
            )
        frames.append(canvas)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=True,
    )
    return output_path


def plot_gelman_rubin(path, statistics, typefileplot='pdf', typeplotback='norm'):
    """Save the convergence-statistic distribution for a PCAT posterior."""
    values = np.asarray(statistics)
    bins = np.linspace(1., np.max(values), 41)
    figure, axis = plt.subplots()
    axis.hist(values, bins=bins)
    axis.set_title('Gelman-Rubin Convergence Test')
    axis.set_xlabel('PSRF')
    axis.set_ylabel('$N_p$')
    output = save_figure(figure, path + 'gmrb', typefileplot, typeplotback)
    plt.close(figure)
    return output


def plot_autocorrelation(path, autocorrelation, correlation_time, strgextn='',
                         typefileplot='pdf', typeplotback='norm'):
    """Save a sampled parameter's autocorrelation sequence."""
    values = np.asarray(autocorrelation).reshape(-1)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(np.arange(values.size), values)
    axis.set_xlabel(r'$\tau$')
    axis.set_ylabel(r'$\xi(\tau)$')
    axis.text(0.8, 0.8, r'$\tau_{exp} = %.3g$' % correlation_time,
              ha='center', va='center', transform=axis.transAxes)
    axis.axhline(0., ls='--', alpha=0.5)
    plt.tight_layout()
    output = save_figure(figure, path + 'atcr%s' % strgextn, typefileplot, typeplotback)
    plt.close(figure)
    return output


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