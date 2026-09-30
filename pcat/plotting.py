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


def plot_detection_diagnostic(
    records: list[dict[str, float | int | bool]],
    output_path: Path,
    examples: dict[str, np.ndarray] | None = None,
) -> Path:
    """Plot representative lens images and population detection results."""
    from .roman_lens import summarize_population

    output_path = Path(output_path)
    if output_path.suffix not in (".png", ".pdf"):
        raise ValueError("Output format must be 'png' or 'pdf'.")
    truth = np.array([record["has_subhalo"] for record in records], dtype=bool)
    signal_to_noise = np.array(
        [record["injected_signal_to_noise"] for record in records], dtype=float
    )
    probability = np.array([record["posterior_one"] for record in records], dtype=float)
    summary = summarize_population(records)

    if examples:
        example_prefix = sorted(
            key.removesuffix("_data") for key in examples if key.endswith("_data")
        )[0]
        data = examples[f"{example_prefix}_data"]
        macro = examples[f"{example_prefix}_macro"]
        residual = examples[f"{example_prefix}_residual"]
        figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), facecolor="white")
        image_axes = axes.flat[:3]
        image_minimum = min(np.percentile(data, 1.0), np.percentile(macro, 1.0))
        image_maximum = max(np.percentile(data, 99.5), np.percentile(macro, 99.5))
        residual_limit = np.max(np.abs(residual))
        panels = (
            (data, "Simulated detector input", "viridis", image_minimum, image_maximum),
            (macro, "Macro-lens model", "viridis", image_minimum, image_maximum),
            (residual, "Data - macro model", "RdBu_r", -residual_limit, residual_limit),
        )
        for image_axis, (image, title, color_map, minimum, maximum) in zip(image_axes, panels):
            image_artist = image_axis.imshow(
                image, origin="lower", cmap=color_map, vmin=minimum, vmax=maximum
            )
            image_axis.set_title(title)
            image_axis.set_xlabel("Detector x [pixel]")
            image_axis.set_ylabel("Detector y [pixel]")
            figure.colorbar(image_artist, ax=image_axis, label="Signal [electron pixel$^{-1}$]")
        axis = axes[1, 1]
    else:
        figure, axis = plt.subplots(figsize=(6.5, 4.0), facecolor="white")
    axis.scatter(
        signal_to_noise[~truth], probability[~truth], color="0.55", alpha=0.65, s=24,
        label="No perturber",
    )
    axis.scatter(
        signal_to_noise[truth], probability[truth], color="#A51417", alpha=0.75, s=28,
        label="Injected perturber",
    )
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1.0, label="Detection threshold")
    axis.set_xlabel(r"Injected perturbation signal-to-noise [$\sigma$]")
    axis.set_ylabel(r"Posterior probability $P(N_{\rm sub}=1\mid d)$")
    axis.set_title("Catalog inference")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(False)
    axis.legend(frameon=True, fancybox=True, framealpha=1.0, loc="upper left")
    axis.text(
        0.98,
        0.04,
        "TPR %.0f%% [%.0f, %.0f]\nFPR %.0f%% [%.0f, %.0f]\nWilson intervals ($z=1$)"
        % (
            100.0 * summary["true_positive_rate"],
            100.0 * summary["true_positive_rate_lower"],
            100.0 * summary["true_positive_rate_upper"],
            100.0 * summary["false_positive_rate"],
            100.0 * summary["false_positive_rate_lower"],
            100.0 * summary["false_positive_rate_upper"],
        ),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
    )
    figure.suptitle("Simulated Roman strong-lens catalog inference")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
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


def make_image_sequence_animation(
    images: tuple[np.ndarray, ...] | list[np.ndarray],
    labels: tuple[str, ...] | list[str],
    output_path: Path,
    duration_ms: int = 800,
    image_size: int = 640,
) -> Path:
    """Write one labeled GIF frame for every image using shared intensity limits."""
    if len(images) == 0 or len(images) != len(labels):
        raise ValueError("images and labels must have the same nonzero length.")
    if duration_ms < 1 or image_size < 1:
        raise ValueError("duration_ms and image_size must be positive.")

    valid_values = [np.asarray(image, dtype=float)[np.isfinite(image)] for image in images]
    valid_values = [values for values in valid_values if values.size]
    if not valid_values:
        raise ValueError("images must contain at least one finite pixel value.")
    shared_values = np.concatenate(valid_values)
    lower = float(np.percentile(shared_values, 1.0))
    upper = float(np.percentile(shared_values, 99.5))
    if upper <= lower:
        upper = lower + 1.0

    frames: list[Image.Image] = []
    header_height = 58
    footer_height = 44
    for index, (image, label) in enumerate(zip(images, labels), start=1):
        values = np.asarray(image, dtype=float)
        finite_values = values[np.isfinite(values)]
        if values.ndim != 2 or finite_values.size == 0:
            raise ValueError(f"image {index} must be a 2D array with finite pixels.")
        scaled = np.zeros(values.shape, dtype=np.uint8)
        scaled[np.isfinite(values)] = np.clip(
            (values[np.isfinite(values)] - lower) / (upper - lower) * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)
        rendered = Image.fromarray(scaled, mode="L")
        rendered.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (image_size, image_size + header_height + footer_height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 16), "Rubin DP1 lens cutouts", fill="black", font=_animation_font(22, bold=True))
        caption = f"{index:03d}/{len(images):03d}  {label}"
        label_size = max(12, min(22, int((image_size - 32) * 1.6 / max(len(caption), 1))))
        draw.text(
            (16, header_height + (image_size - rendered.height) // 2 + rendered.height + 10),
            caption,
            fill="black",
            font=_animation_font(label_size, bold=True),
        )
        canvas.paste(
            rendered.convert("RGB"),
            ((image_size - rendered.width) // 2, header_height + (image_size - rendered.height) // 2),
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