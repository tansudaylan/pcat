"""Native plotting functions for PCAT posterior products."""

from tdpy.verbosity import print

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tdpy.util import save_figure

from .diagnostics import autocorrelation_time, catalog_count_transitions, gelman_rubin


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
    pattern: str | None = None
    static_image: str | None = None


def histogram_frame_limits(reference_count: float, maximum_model_count: float) -> tuple[float, float]:
    """Keep all posterior histogram bars within one stable logarithmic y-range."""

    maximum = max(1.0, float(np.max(np.atleast_1d(reference_count))),
                  float(np.max(np.atleast_1d(maximum_model_count))))
    return 0.5, 1.1 * maximum


def plot_posterior_convergence(state, output_directory: Path, typefileplot: str = "png") -> dict[str, Path]:
    """Plot fixed-parameter mixing and transdimensional catalog-size diagnostics."""

    if typefileplot not in ("png", "pdf"):
        raise ValueError("typefileplot must be 'png' or 'pdf'")
    fixed = np.asarray(state.listpostparagenrscalbase, dtype=float)
    if fixed.ndim == 1:
        fixed = fixed[:, None]
    if fixed.ndim != 2 or fixed.shape[0] < 2:
        raise ValueError("Fixed-parameter draws must be a sample-by-parameter array")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    chain_count = int(getattr(state, "numbproc", 1))
    sample_count = int(getattr(state, "numbsamp", fixed.shape[0]))
    if chain_count > 1 and chain_count * sample_count != fixed.shape[0]:
        raise ValueError("The number of posterior draws must match the independent chains")

    def split_chains(values):
        if chain_count == 1:
            return values[:, None, :]
        return values.reshape(sample_count, chain_count, values.shape[1])

    def save(figure, name):
        path = output_directory / f"{name}.{typefileplot}"
        print(f"Writing to {path}...")
        figure.savefig(path, dpi=300 if typefileplot == "png" else None,
                       bbox_inches="tight", facecolor="white")
        plt.close(figure)
        paths[name] = path

    def series_and_mixing(values, prefix, names):
        if values.shape[1] == 0:
            return None
        labels = list(names) if names is not None else [f"Parameter {index + 1}" for index in range(values.shape[1])]
        if len(labels) != values.shape[1]:
            raise ValueError("Convergence labels must match the number of series")
        chains = split_chains(values)
        figure, axes = plt.subplots(len(labels), 1, figsize=(9, max(3.0, 1.8 * len(labels))),
                                   sharex=True, squeeze=False, facecolor="white")
        for index, axis in enumerate(axes[:, 0]):
            for chain_index in range(chain_count):
                axis.plot(chains[:, chain_index, index], lw=0.65, alpha=0.7,
                          label=f"Chain {chain_index + 1}" if index == 0 else None)
            axis.set_ylabel(labels[index], fontsize=10)
            axis.grid(False)
        if chain_count > 1:
            axes[0, 0].legend(framealpha=1, facecolor="white", edgecolor="black")
        axes[-1, 0].set_xlabel("Posterior sample index", fontsize=10)
        save(figure, f"{prefix}_trace")

        window = chains[-min(2000, sample_count):]
        correlation, times = autocorrelation_time(window)
        figure, axis = plt.subplots(figsize=(8, 4), facecolor="white")
        for index, label in enumerate(labels):
            for chain_index in range(chain_count):
                axis.plot(np.arange(correlation.shape[-1]), correlation[chain_index, index],
                          label=label if chain_index == 0 else None, alpha=0.75)
        axis.axhline(0, color="black", lw=0.7)
        axis.set(xlabel="Lag [posterior samples]", ylabel="Autocorrelation")
        axis.set_xlim(0, max(1, correlation.shape[-1] - 1))
        axis.grid(False)
        axis.legend(loc="upper right", framealpha=1, facecolor="white", edgecolor="black")
        save(figure, f"{prefix}_autocorrelation")
        time_per_parameter = np.max(np.where(np.isfinite(times), times, np.inf), axis=0)
        effective = values.shape[0] / np.maximum(time_per_parameter, 1)
        rhat = np.array([gelman_rubin(chains[:, :, index]) for index in range(values.shape[1])]) \
            if chain_count > 1 else np.full(values.shape[1], np.nan)
        return labels, effective, rhat

    if fixed.shape[1]:
        parameter_names = getattr(state, "convergence_parameter_names", None)
        if parameter_names is None:
            parameter_names = getattr(getattr(getattr(state, "fitt", None), "namepara", None), "genrbase", None)
        names, effective, rhat = series_and_mixing(fixed, "fixed_parameter", parameter_names)
        figure, axes = plt.subplots(2, 1, figsize=(8, max(4.4, len(names) * 0.48 + 2)),
                                   sharex=True, facecolor="white")
        axes[0].bar(np.arange(len(names)), effective, color="#19796D")
        axes[0].set_ylabel("Approximate ESS [samples]")
        axes[1].plot(np.arange(len(names)), rhat, marker="o", color="#A64135")
        axes[1].axhline(1.05, color="black", ls="--", lw=0.8)
        axes[1].set_ylabel(r"Multi-chain $\hat R$")
        if not np.isfinite(rhat).any():
            axes[1].text(0.5, 0.5, "Multiple chains required", ha="center", va="center",
                         transform=axes[1].transAxes)
        axes[1].set_xticks(np.arange(len(names)), names, rotation=35, ha="right")
        for axis in axes:
            axis.grid(False)
        save(figure, "fixed_parameter_mixing")

    count = np.asarray(getattr(state, "listpostnumbelem", np.empty((fixed.shape[0], 0))))
    if count.ndim == 1:
        count = count[:, None]
    if count.ndim != 2 or count.shape[0] != fixed.shape[0]:
        raise ValueError("Element counts must be a matching sample-by-population array")
    if count.shape[1]:
        if not np.isfinite(count).all() or np.any(count < 0) or np.any(count != np.floor(count)):
            raise ValueError("Element counts must be finite nonnegative integers")
        count = count.astype(int)
        names, effective, rhat = series_and_mixing(count, "element_count", [f"Population {index + 1}"
                                    for index in range(count.shape[1])])
        figure, axes = plt.subplots(2, 1, figsize=(8, max(4.4, count.shape[1] * 0.48 + 2)),
                                   sharex=True, facecolor="white")
        axes[0].bar(np.arange(len(names)), effective, color="#19796D")
        axes[0].set_ylabel("Approximate ESS [samples]")
        axes[1].plot(np.arange(len(names)), rhat, marker="o", color="#A64135")
        axes[1].axhline(1.05, color="black", ls="--", lw=0.8)
        if chain_count == 1:
            axes[1].text(0.5, 0.5, "Multiple chains required", transform=axes[1].transAxes,
                         ha="center", va="center")
        axes[1].set_ylabel(r"Multi-chain $\hat R$")
        axes[1].set_xticks(np.arange(len(names)), names)
        for axis in axes:
            axis.grid(False)
        save(figure, "element_count_mixing")
        figure, axis = plt.subplots(figsize=(8, 4), facecolor="white")
        for index, label in enumerate(names):
            bins = np.arange(count[:, index].max() + 2)
            axis.step(bins[:-1], np.bincount(count[:, index], minlength=len(bins) - 1) / len(count),
                      where="mid", label=label)
        axis.set(xlabel="Number of elements", ylabel="Posterior occupancy", ylim=(0, 1))
        axis.grid(False)
        axis.legend(loc="upper right", framealpha=1, facecolor="white", edgecolor="black")
        save(figure, "element_count_occupancy")

        figure, axes = plt.subplots(1, count.shape[1], figsize=(5 * count.shape[1], 4),
                                   squeeze=False, facecolor="white")
        count_chains = split_chains(count)
        for index, axis in enumerate(axes[0]):
            transitions = catalog_count_transitions(count_chains[:, :, index])
            axis.imshow(transitions, origin="lower", cmap="Greens", interpolation="nearest")
            axis.set(xlabel="Next catalog size", ylabel="Current catalog size", title=names[index])
            axis.grid(False)
        save(figure, "element_count_transitions")

        catalogs = getattr(state, "listpostdictelem", None)
        if catalogs is not None and len(catalogs) == len(count):
            for population in range(count.shape[1]):
                features = sorted({feature for sample in catalogs if len(sample) > population
                                   for feature in sample[population]})
                for feature in features:
                    halves = []
                    for subset in (catalogs[:len(catalogs) // 2], catalogs[len(catalogs) // 2:]):
                        series = []
                        for sample in subset:
                            if len(sample) <= population or feature not in sample[population]:
                                continue
                            try:
                                values = np.asarray(sample[population][feature], dtype=float)
                            except (TypeError, ValueError):
                                continue
                            if values.ndim == 1:
                                series.extend(values[np.isfinite(values)])
                        halves.append(np.asarray(series))
                    if not all(values.size for values in halves):
                        continue
                    combined = np.concatenate(halves)
                    if not np.all(np.isfinite(combined)):
                        continue
                    figure, axis = plt.subplots(figsize=(7, 4), facecolor="white")
                    bins = np.histogram_bin_edges(combined, bins=30)
                    for values, label, color in zip(halves, ("First half", "Second half"),
                                                     ("#19796D", "#A64135")):
                        axis.hist(values, bins=bins, histtype="step", density=True,
                                  linewidth=1.8, color=color, label=label)
                    axis.set(xlabel=f"{feature} [model units]", ylabel="Density",
                             title=f"Population {population + 1} element distribution")
                    axis.grid(False)
                    axis.legend(framealpha=1, facecolor="white", edgecolor="black")
                    save(figure, f"element_parameter_pop{population}_{feature}_stability")

    return paths


POSTERIOR_ANIMATION_PANELS = (
    PosteriorAnimationPanel(
        "Unbinned Gaussian mixture | event catalog",
        "unbinned_gaussian_mixture/visuals/gmix_events_swep*.png",
    ),
    PosteriorAnimationPanel(
        "Daylan+2017 NG cap mock | Fermi-LAT point sources",
        "daylan+2017_fermi_point_sources/visuals/fermi_ngpc_sources_swep*.png",
    ),
    PosteriorAnimationPanel(
        "Voigt lines | spectral model",
        "voigt_spectral_line_catalog/pcat_runs/voigt_nomi/visuals/post/fram/thisscatcntpevt0_swep*.png",
    ),
    PosteriorAnimationPanel(
        "Simulated photometry | rotating starspots",
        "variable_number_stellar_spots/visuals/stellar_spot_photometry_swep*.png",
    ),
    PosteriorAnimationPanel(
        "PCAT | transdimensional inference",
        static_image="docs/_static/pcat_logo.png",
    ),
    PosteriorAnimationPanel(
        "Roman/WFI | simulated strong-lens arcs",
        "roman_strong_lens_perturber_catalog/visuals/roman_wfi_lensed_posterior_swep*.png",
    ),
    PosteriorAnimationPanel(
        "Simulated photometry | PCAT flare catalog",
        "variable_number_stellar_flares/visuals/flare_photometry_swep*.png",
    ),
    PosteriorAnimationPanel(
        "Simulated RV | variable-number exoplanets",
        "variable_number_exoplanets_radial_velocity/visuals/rv_posterior_swep*.png",
    ),
    PosteriorAnimationPanel(
        "Simulated TTV | transit-time series",
        "transit_timing_variations/visuals/ttv_posterior_swep*.png",
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
    if panel.static_image is not None:
        path = examples_root.parent / panel.static_image
        if not path.is_file():
            raise RuntimeError(f"{panel.label} requires a static image at {path}.")
        return [path]
    if panel.pattern is None:
        raise ValueError(f"{panel.label} needs a frame pattern or static image.")
    paths = sorted(examples_root.glob(panel.pattern))
    if len(paths) < 2:
        raise RuntimeError(
            f"{panel.label} requires at least two posterior frames matching "
            f"{examples_root / panel.pattern}. Run examples/run_all_examples.py first."
        )
    unique_frames = set()
    for path in paths:
        with Image.open(path) as image:
            unique_frames.add(image.convert("RGB").tobytes())
    if len(unique_frames) < 2:
        raise RuntimeError(
            f"{panel.label} has no visual evolution across {len(paths)} frames. "
            "Increase the example depth or select a changing posterior product."
        )
    return paths


def _quantize_shared_palette(frames: list[Image.Image]) -> list[Image.Image]:
    """Quantize RGB animation frames against one shared palette."""
    if not frames:
        raise ValueError("frames must contain at least one image")
    sample_size = 128
    samples = []
    for frame in frames:
        sample = frame.copy()
        sample.thumbnail((sample_size, sample_size), Image.Resampling.LANCZOS)
        samples.append(sample.convert("RGB"))
    palette_source = Image.new("RGB", (sample_size, sample_size * len(samples)), "white")
    for index, sample in enumerate(samples):
        palette_source.paste(sample, (0, index * sample_size))
    adaptive_palette = palette_source.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    palette = Image.new("P", (1, 1))
    palette.putpalette(adaptive_palette.getpalette())
    return [
        frame.convert("RGB").quantize(palette=palette, dither=Image.Dither.NONE)
        for frame in frames
    ]


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
    frames = _quantize_shared_palette(frames)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )
    return output_path


def make_posterior_animation_collage(
    output_path: Path = DEFAULT_POSTERIOR_COLLAGE,
    examples_root: Path = EXAMPLES_ROOT,
    panels: tuple[PosteriorAnimationPanel, ...] = POSTERIOR_ANIMATION_PANELS,
    frame_count: int = 16,
    duration_ms: int = 120,
    panel_size: int = 560,
) -> Path:
    """Write a synchronized, high-resolution collage of posterior frame sequences."""
    if frame_count < 2:
        raise ValueError("frame_count must be at least two.")
    if not panels:
        raise ValueError("panels must contain at least one frame sequence.")

    sequences = [_animation_frame_paths(panel, examples_root) for panel in panels]
    column_count = min(3, len(panels))
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
        counter = f"frame {frame_index + 1:02d} / {frame_count:02d}"
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
                rgba = source.convert("RGBA")
                white_background = Image.new("RGBA", rgba.size, "white")
                white_background.alpha_composite(rgba)
                panel_image = ImageOps.contain(
                    white_background.convert("RGB"),
                    (panel_size, panel_size),
                    Image.Resampling.LANCZOS,
                )
            canvas.paste(
                panel_image,
                (x + (panel_size - panel_image.width) // 2,
                 y + (panel_size - panel_image.height) // 2),
            )
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
    frames = _quantize_shared_palette(frames)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
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