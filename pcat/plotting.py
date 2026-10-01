"""Native plotting functions for PCAT posterior products."""

from tdpy.verbosity import print

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tdpy.util import save_figure

from .diagnostics import (
    autocorrelation_time, binomial_wilson_interval, catalog_count_transitions, gelman_rubin,
)


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
    labels: tuple[str, ...] = ("Einstein radius [arcsec]", "Source x [arcsec]", "Source y [arcsec]"),
) -> Path:
    """Write marginal lens-parameter distributions with injected values, three panels per row."""
    if typefileplot not in ("png", "pdf"):
        raise ValueError("typefileplot must be 'png' or 'pdf'")
    medians = np.median(draws, axis=0)
    number_rows = int(np.ceil(len(labels) / 3))
    figure, axes = plt.subplots(number_rows, 3, figsize=(11, 3.3 * number_rows), constrained_layout=True,
                                squeeze=False)
    for index, (axis, label) in enumerate(zip(axes.flat, labels)):
        axis.hist(draws[:, index], bins=22, density=True, color="#007360", alpha=0.75,
                  label="PCAT samples")
        axis.axvline(true_parameters[index], color="#A51417", linewidth=2,
                     label="Injected value")
        axis.axvline(medians[index], color="black", linestyle="--", linewidth=1.5,
                     label="Posterior median")
        axis.set_xlabel(label)
        axis.set_ylabel("Posterior density")
        axis.grid(False)
    for axis in axes.flat[len(labels):]:
        axis.set_visible(False)
    axes[0, 0].legend(loc="upper left", frameon=True, fancybox=True, framealpha=1.0, fontsize=8)
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
    title: str = "Rubin DP1 lens cutouts",
) -> Path:
    """Write one labeled GIF frame for every image using shared intensity limits.

    ``title`` is drawn above every frame and each label below its image. The
    stretch spans the 1st to 99.5th percentile of all finite pixels.
    """
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
        # fit the longer side to image_size; small cutouts are enlarged with sharp detector pixels
        factor = image_size / max(rendered.size)
        resample = Image.Resampling.NEAREST if factor > 1.0 else Image.Resampling.LANCZOS
        rendered = rendered.resize(
            (max(1, round(rendered.width * factor)), max(1, round(rendered.height * factor))), resample
        )
        canvas = Image.new("RGB", (image_size, image_size + header_height + footer_height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 16), title, fill="black", font=_animation_font(22, bold=True))
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
        draw.text((margin, margin), "PCAT: from a prior draw to posterior samples", fill="black",
                  font=title_font)
        # panel frames follow PCAT's animation schedule, which spends the first third in burn-in
        phase = "burn-in" if frame_index < frame_count / 3 else "posterior samples"
        counter = f"{phase} | frame {frame_index + 1:02d} / {frame_count:02d}"
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
            if panel.static_image is None:
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


def animation_states(state):
    """Return the chain-0 snapshots PCAT recorded for ``numbframanim``, ordered by sweep."""
    snapshots = list(getattr(state, "listanimstate", []) or [])
    if len(snapshots) < 2:
        raise ValueError("Run PCAT with numbframanim >= 2 to record animation snapshots")
    return sorted(snapshots, key=lambda snapshot: snapshot["cntrswep"])


def animation_phase_label(snapshot):
    """Describe a snapshot as the initial prior draw, a burn-in state, or a posterior sample."""
    if snapshot["cntrswep"] == 0:
        return "Initial random draw from the prior"
    phase = "Burn-in" if snapshot["boolburn"] else "Posterior sample"
    return f"{phase}, sweep {snapshot['cntrswep']:,}"


MOVE_LABELS = {"with": "Within-model", "brth": "Birth", "deth": "Death", "splt": "Split",
               "merg": "Merge", "jump": "Jump"}
MOVE_COLORS = {"with": "#4D4D4D", "brth": "#1B7837", "deth": "#A50026", "splt": "#2166AC",
               "merg": "#B35806", "jump": "#762A83"}
LEGEND_STYLE = dict(frameon=True, fancybox=True, framealpha=1.0, facecolor="white", edgecolor="black")


def _save_view(figure, output_path, typefileplot):
    """Write one sampler view at the workspace resolution and close it."""
    if typefileplot not in ("png", "pdf"):
        raise ValueError("typefileplot must be 'png' or 'pdf'")
    output_path = Path(output_path).with_suffix(f".{typefileplot}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(output_path, dpi=300 if typefileplot == "png" else None,
                   bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def _proposal_record(state):
    """Return per-proposal move names, flags, and sweep indices pooled over chains."""
    move_index = np.asarray(state.listpostindxproptype, dtype=int).ravel()
    accepted = np.asarray(state.listpostboolpropaccp, dtype=bool).ravel()
    evaluated = np.asarray(getattr(state, "listpostboolpropfilt", np.ones_like(accepted)), dtype=bool).ravel()
    if not move_index.size == accepted.size == evaluated.size:
        raise ValueError("Proposal arrays must have one entry per proposal")
    names = np.asarray(getattr(state, "nameproptype", ["with"]))
    if move_index.max(initial=0) >= names.size:
        raise ValueError("A proposal type index exceeds the configured move names")
    chain_count = max(int(getattr(state, "numbproc", 1)), 1)
    # proposals are stored sweep-major across chains
    sweep = np.arange(move_index.size) // chain_count
    return names, move_index, accepted, evaluated, sweep, chain_count


def has_proposal_record(state):
    """Return whether a state stores the per-proposal arrays the operation views need."""
    return all(np.size(getattr(state, name, [])) > 0
               for name in ("listpostindxproptype", "listpostboolpropaccp"))


def plot_proposal_ledger(state, output_path, typefileplot="png", number_windows=120):
    """Plot acceptance by move type through burn-in and sampling, with per-move totals.

    The left panel shows the accepted fraction of each move type in sweep windows
    pooled over chains, the burn-in interval, and the likelihood inverse
    temperature when tempering is active. The right panel shows each move's
    share of all proposals, its acceptance with 1-sigma Wilson intervals after
    burn-in, and the fraction rejected before the likelihood is evaluated.
    """
    names, move_index, accepted, evaluated, sweep, _ = _proposal_record(state)
    burn_count = int(getattr(state, "numbburn", 0) or 0)
    edges = np.linspace(0, sweep[-1] + 1, number_windows + 1)
    window = np.clip(np.searchsorted(edges, sweep, side="right") - 1, 0, number_windows - 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    present = [index for index in range(names.size) if np.any(move_index == index)]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white",
                                gridspec_kw={"width_ratios": (2.2, 1.0)})
    floor = 1e-4
    axis = axes[0]
    if burn_count > 0:
        axis.axvspan(0, burn_count, color="0.92", label="Burn-in")
    for index in present:
        chosen = move_index == index
        attempts = np.bincount(window[chosen], minlength=number_windows)
        successes = np.bincount(window[chosen], weights=accepted[chosen], minlength=number_windows)
        rate = np.where(attempts >= 5, np.maximum(successes / np.maximum(attempts, 1), floor), np.nan)
        name = str(names[index])
        axis.plot(centers, rate, color=MOVE_COLORS.get(name, "black"), lw=1.4,
                  label=MOVE_LABELS.get(name, name))
    temperature = np.asarray(getattr(state, "listpostfacttmpr", []), dtype=float).ravel()
    if temperature.size == sweep.size and np.ptp(temperature) > 0:
        inverse = np.bincount(window, weights=temperature, minlength=number_windows) / \
            np.maximum(np.bincount(window, minlength=number_windows), 1)
        axis.plot(centers, np.maximum(inverse, floor), color="black", ls="--", lw=1.2,
                  label=r"Likelihood inverse temperature $\beta$")
    axis.set_yscale("log")
    axis.set(xlabel="Sweep", ylabel=f"Accepted fraction (floor {floor:g})", ylim=(0.7 * floor, 1.5),
             xlim=(0, edges[-1]), title="Proposal acceptance through the run")
    axis.grid(False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, **LEGEND_STYLE)

    axis = axes[1]
    sampled = sweep >= burn_count
    positions = np.arange(len(present))
    labeled = set()
    for row, index in enumerate(present):
        chosen = (move_index == index) & sampled
        name = str(names[index])
        share = chosen.sum() / max(sampled.sum(), 1)
        axis.barh(row + 0.2, share, height=0.35, color="0.75",
                  label="Share of proposals" if row == 0 else None)
        if chosen.sum() == 0:
            continue
        rate = max(accepted[chosen].mean(), floor)
        lower, upper = binomial_wilson_interval(int(accepted[chosen].sum()), int(chosen.sum()))
        axis.barh(row - 0.2, rate, height=0.35, color=MOVE_COLORS.get(name, "black"),
                  xerr=[[rate - max(lower, floor)], [max(upper - rate, 0.0)]], capsize=3,
                  label="Accepted fraction" if row == 0 else None)
        filtered = 1.0 - evaluated[chosen].mean()
        if filtered > 0:
            axis.plot(filtered, row - 0.2, marker="D", markersize=5, color="black", linestyle="none",
                      label=None if "filtered" in labeled else "Rejected before likelihood")
            labeled.add("filtered")
    axis.set_yticks(positions, [MOVE_LABELS.get(str(names[index]), str(names[index])) for index in present])
    axis.set_xscale("log")
    axis.set(xlabel="Fraction after burn-in", xlim=(floor, 1.5), title="Move totals")
    axis.invert_yaxis()
    axis.grid(False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, **LEGEND_STYLE)
    figure.tight_layout()
    return _save_view(figure, output_path, typefileplot)


def _signed_log(values):
    """Compress values spanning many decades while keeping their sign."""
    return np.sign(values) * np.log10(1.0 + np.abs(values))


def plot_acceptance_decomposition(state, output_path, typefileplot="png"):
    """Show which terms of the Metropolis-Hastings log ratio drive each move's decision.

    For every evaluated proposal after burn-in, the log acceptance ratio is split
    into the posterior and auxiliary-density change, the move-selection ratio,
    and the split/merge Jacobian. Each is drawn per move type on a signed
    logarithmic axis, so a reader can see whether, for example, deaths fail
    because of the likelihood or because of the proposal bookkeeping.
    """
    names, move_index, _, evaluated, sweep, _ = _proposal_record(state)
    total = np.asarray(state.listpostaccplprb, dtype=float).ravel()
    selection = np.asarray(getattr(state, "listpostltrp", np.zeros_like(total)), dtype=float).ravel()
    jacobian = np.asarray(getattr(state, "listpostljcb", np.zeros_like(total)), dtype=float).ravel()
    keep = evaluated & (sweep >= int(getattr(state, "numbburn", 0) or 0)) & np.isfinite(total)
    terms = (
        ("Posterior and auxiliary\ndensity change [nat]", total - selection - jacobian),
        ("Move-selection\nlog ratio [nat]", selection),
        ("Split/merge log\nJacobian [nat]", jacobian),
        ("Log acceptance\nratio [nat]", total),
    )
    present = [index for index in range(names.size) if np.sum(keep & (move_index == index)) >= 5]
    if not present:
        raise ValueError("No move type has at least five evaluated proposals after burn-in")
    figure, axes = plt.subplots(1, len(terms), figsize=(12, 0.55 * len(present) + 2.4),
                                sharey=True, facecolor="white")
    decades = np.arange(0, 6)
    for axis, (label, values) in zip(axes, terms):
        data = [_signed_log(values[keep & (move_index == index)]) for index in present]
        limit = max(1.0, np.ceil(max(np.max(np.abs(values)) for values in data)))
        jitter = [values + 1e-9 * np.arange(values.size) for values in data]
        violins = axis.violinplot(jitter, positions=np.arange(len(present)), orientation="horizontal",
                                  showmedians=True, showextrema=False, widths=0.8)
        violins["cmedians"].set_color("black")
        for body, index in zip(violins["bodies"], present):
            body.set_facecolor(MOVE_COLORS.get(str(names[index]), "black"))
            body.set_alpha(0.65)
        axis.axvline(0.0, color="black", lw=0.8, ls="--")
        shown = decades[decades <= limit]
        if shown.size > 4:
            shown = shown[::2]
        ticks = np.concatenate((-shown[::-1][:-1], shown))
        axis.set_xticks(_signed_log(np.sign(ticks) * (10.0 ** np.abs(ticks) - (ticks == 0))),
                        ["0" if tick == 0 else ("-" if tick < 0 else "") + f"$10^{{{abs(tick)}}}$"
                         for tick in ticks])
        axis.set(xlabel=label, xlim=(-limit - 0.3, limit + 0.3))
        axis.grid(False)
    axes[0].set_yticks(np.arange(len(present)),
                       [MOVE_LABELS.get(str(names[index]), str(names[index])) for index in present])
    axes[0].invert_yaxis()
    figure.suptitle("Metropolis-Hastings terms per move type after burn-in")
    figure.tight_layout()
    return _save_view(figure, output_path, typefileplot)


STAGE_LABELS = {
    "prop": "Proposal", "diag": "Diagnostics", "save": "Saving", "plot": "Plotting",
    "proc": "State processing", "elem": "Element evaluation", "modl": "Model assembly",
    "llik": "Likelihood", "sbrtmodl": "Surface brightness", "spec": "Element spectra",
    "elemsbrtdfnc": "Element brightness", "psfnconv": "PSF convolution", "expo": "Exposure",
    "lpri": "Prior", "tert": "Tertiary products", "deflzero": "Zero deflection",
    "deflhost": "Host deflection", "deflextr": "External shear", "sbrtlens": "Lensed source",
    "sbrthost": "Host light", "elemdeflsubh": "Subhalo deflection",
    "elemsbrtextsbgrd": "Extended background",
}


def plot_compute_budget(state, output_path, typefileplot="png"):
    """Map the mean wall time of each sampler stage for every move type.

    PCAT times each stage of every proposal (proposal construction, element
    evaluation, model and likelihood evaluation, prior, saving, plotting). The
    heat map shows the mean time [ms] per stage and move on a logarithmic color
    scale. Stages nest inside one another, so their times do not add up to the
    total shown beside each move; stages timed once per sweep are omitted.
    """
    names, move_index, _, _, _, _ = _proposal_record(state)
    stages = [name for name in getattr(state, "listnamechro", []) if name != "totl"
              and np.size(getattr(state, "listpostchro" + name, [])) == move_index.size]
    if not stages:
        raise ValueError("The state stores no per-proposal stage timings")
    present = [index for index in range(names.size) if np.any(move_index == index)]
    total = np.asarray(getattr(state, "listpostchrototl"), dtype=float).ravel()
    budget = np.zeros((len(present), len(stages)))  # [ms]
    for row, index in enumerate(present):
        chosen = move_index == index
        for column, stage in enumerate(stages):
            budget[row, column] = 1e3 * np.nanmean(np.asarray(getattr(state, "listpostchro" + stage)).ravel()[chosen])
    # stages timed across a whole sweep, such as frame plotting, exceed every proposal's total
    active = (budget.max(axis=0) > 0) & (budget.max(axis=0) <= 1e3 * np.nanmax(
        [np.nanmean(total[move_index == index]) for index in present]))
    budget = budget[:, active]
    stages = [stage for stage, keep in zip(stages, active) if keep]
    figure, axis = plt.subplots(figsize=(1.0 * len(stages) + 3.0, 0.6 * len(present) + 2.0),
                                facecolor="white")
    image = axis.imshow(budget, cmap="magma_r", aspect="auto", norm=plt.matplotlib.colors.LogNorm(
        max(budget[budget > 0].min(), 1e-4), budget.max()))
    threshold = np.sqrt(budget[budget > 0].min() * budget.max())
    for row in range(len(present)):
        for column in range(len(stages)):
            axis.text(column, row, f"{budget[row, column]:.2g}", ha="center", va="center",
                      color="white" if budget[row, column] > threshold else "black")
    axis.set_xticks(np.arange(len(stages)), [STAGE_LABELS.get(stage, stage) for stage in stages],
                    rotation=35, ha="right")
    axis.set_yticks(np.arange(len(present)),
                    [f"{MOVE_LABELS.get(str(names[index]), str(names[index]))} "
                     f"({1e3 * np.nanmean(total[move_index == index]):.2f} ms)" for index in present])
    axis.set(xlabel="Sampler stage (nested)", title="Mean wall time per proposal stage [ms]")
    axis.grid(False)
    figure.colorbar(image, ax=axis, label="Mean time [ms]")
    figure.tight_layout()
    return _save_view(figure, output_path, typefileplot)


def plot_catalog_trace(state, output_path, position="elin", amplitude="flux", population=0,
                       chain=0, position_label=None, amplitude_label=None, typefileplot="png"):
    """Draw a transdimensional trace: every element of every retained catalog of one chain.

    Each column is one retained catalog, each marker is one element at its
    ``position`` and colored by ``amplitude``, so births, deaths, splits, and
    merges appear as tracks that start, stop, fork, or join. The lower strip
    shows the catalog size, and the right panel shows the expected number of
    elements per position bin pooled over all chains.
    """
    catalogs = getattr(state, "listpostdictelem", None)
    if not catalogs:
        raise ValueError("The state stores no posterior element catalogs")
    chain_count = max(int(getattr(state, "numbproc", 1)), 1)
    if not 0 <= chain < chain_count:
        raise ValueError("chain must index one of the sampled chains")

    def element_values(sample, name):
        if len(sample) <= population or name not in sample[population]:
            return np.empty(0)
        return np.asarray(sample[population][name], dtype=float).ravel()

    chain_catalogs = catalogs[chain::chain_count]
    rows = [(index, element_values(sample, position), element_values(sample, amplitude))
            for index, sample in enumerate(chain_catalogs)]
    sample_index = np.concatenate([np.full(values.size, index) for index, values, _ in rows])
    positions = np.concatenate([values for _, values, _ in rows])
    amplitudes = np.concatenate([values for _, _, values in rows])
    sizes = np.array([values.size for _, values, _ in rows])
    pooled = np.concatenate([element_values(sample, position) for sample in catalogs])
    if pooled.size == 0:
        raise ValueError(f"No retained catalog contains the element parameter {position!r}")

    figure = plt.figure(figsize=(10, 5.6), facecolor="white")
    grid = figure.add_gridspec(2, 2, width_ratios=(4.0, 1.0), height_ratios=(3.2, 1.0),
                               hspace=0.08, wspace=0.05)
    axis = figure.add_subplot(grid[0, 0])
    positive = amplitudes[amplitudes > 0]
    norm = plt.matplotlib.colors.LogNorm(positive.min(), positive.max()) \
        if positive.size and positive.max() > positive.min() else None
    shown = axis.scatter(sample_index, positions, c=amplitudes, s=6, cmap="viridis", norm=norm,
                         linewidths=0, rasterized=True)
    axis.set(ylabel=position_label or position, title=f"Catalog trace of chain {chain + 1}")
    axis.tick_params(labelbottom=False)
    axis.grid(False)
    colorbar_axis = axis.inset_axes((1.30, 0.0, 0.03, 1.0))
    figure.colorbar(shown, cax=colorbar_axis, label=amplitude_label or amplitude)

    side = figure.add_subplot(grid[0, 1], sharey=axis)
    bins = np.histogram_bin_edges(pooled, bins=60)
    counts, _ = np.histogram(pooled, bins=bins)
    side.barh(0.5 * (bins[1:] + bins[:-1]), counts / len(catalogs), height=np.diff(bins),
              color="#2166AC")
    side.set(xlabel="Expected elements\nper bin")
    side.tick_params(labelleft=False)
    side.grid(False)

    strip = figure.add_subplot(grid[1, 0], sharex=axis)
    strip.step(np.arange(sizes.size), sizes, where="mid", color="black", lw=1.0)
    strip.set(xlabel="Retained sample index", ylabel="Elements",
              xlim=(-0.5, sizes.size - 0.5))
    strip.yaxis.get_major_locator().set_params(integer=True)
    strip.grid(False)
    return _save_view(figure, output_path, typefileplot)


def has_one_dimensional_prediction(state):
    """Return whether a state stores data and model counts along a single axis."""
    data = np.asarray(getattr(state, "cntpdata", []))
    models = np.asarray(getattr(state, "listpostcntpmodl", []))
    return data.size > 1 and models.ndim >= 2 and models.shape[1:] == data.shape \
        and sum(length > 1 for length in data.shape) == 1


def plot_posterior_predictive(state, output_path, axis_values=None, axis_label="Data axis",
                              data_label="Counts per bin", typefileplot="png", seed=0):
    """Compare one-dimensional data to replicated data drawn from the posterior.

    For every retained sample, PCAT's model is turned into a replicated data set
    with the run's likelihood noise (Poisson counts, or Gaussian with the data
    variance). The upper panel shows the data, the posterior median model, and
    the 68% and 95% bands of the replicated data. The lower panels show the
    residual in units of the replicated scatter and the posterior predictive
    tail probability P(replicated > data), which should be spread over the unit
    interval when the model describes the data.
    """
    if not has_one_dimensional_prediction(state):
        raise ValueError("Posterior predictive view requires data and models along one axis")
    data = np.asarray(state.cntpdata, dtype=float).ravel()
    models = np.asarray(state.listpostcntpmodl, dtype=float).reshape(-1, data.size)
    if axis_values is None:
        energy = np.asarray(getattr(getattr(state, "bctrpara", None), "ener", []), dtype=float)
        axis_values = energy if energy.size == data.size else np.arange(data.size)
    axis_values = np.asarray(axis_values, dtype=float)
    random = np.random.default_rng(seed)
    if getattr(state, "liketype", "pois") == "gaus":
        scatter = np.sqrt(np.asarray(state.varidata, dtype=float).ravel())
        replicated = models + random.normal(size=models.shape) * scatter[None, :]
    else:
        replicated = random.poisson(np.maximum(models, 0.0)).astype(float)
    model_median = np.median(models, axis=0)
    quantiles = np.percentile(replicated, [2.5, 16.0, 84.0, 97.5], axis=0)
    residual = (data - model_median) / np.maximum(np.std(replicated, axis=0), 1e-12)
    exceedance = np.mean(replicated > data[None, :], axis=0) + 0.5 * np.mean(replicated == data[None, :], axis=0)

    figure, axes = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True, facecolor="white",
                                gridspec_kw={"height_ratios": (3.0, 1.2, 1.2)})
    axes[0].fill_between(axis_values, quantiles[0], quantiles[3], color="#C6DBEF", lw=0,
                         label="95% replicated-data band")
    axes[0].fill_between(axis_values, quantiles[1], quantiles[2], color="#6BAED6", lw=0,
                         label="68% replicated-data band")
    axes[0].plot(axis_values, model_median, color="#08306B", lw=1.2, label="Posterior median model")
    axes[0].plot(axis_values, data, ".", color="black", markersize=3, label="Data")
    axes[0].set(ylabel=data_label, title="Posterior predictive check")
    axes[0].legend(loc="lower left", ncol=2, **LEGEND_STYLE)
    axes[1].axhspan(-2, 2, color="0.9", label=r"$\pm 2\sigma$")
    axes[1].plot(axis_values, residual, ".", color="black", markersize=3)
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set(ylabel=r"Residual [$\sigma$]")
    axes[1].legend(loc="upper right", **LEGEND_STYLE)
    axes[2].axhspan(0.025, 0.975, color="0.9", label="Central 95%")
    axes[2].plot(axis_values, exceedance, ".", color="#A50026", markersize=3)
    axes[2].set(xlabel=axis_label, ylabel="P(replicated > data)", ylim=(-0.03, 1.03))
    axes[2].legend(loc="upper right", **LEGEND_STYLE)
    for axis in axes:
        axis.grid(False)
    figure.tight_layout()
    return _save_view(figure, output_path, typefileplot)


def plot_sampler_overview(state, output_directory, typefileplot="png", **catalog_options):
    """Write every operation, catalog, and predictive view that applies to a state.

    Returns a dictionary from view name to the written path. ``catalog_options``
    are passed to :func:`plot_catalog_trace`.
    """
    output_directory = Path(output_directory)
    paths = {}
    if has_proposal_record(state):
        paths["proposal_ledger"] = plot_proposal_ledger(
            state, output_directory / "proposal_ledger", typefileplot)
        if np.size(getattr(state, "listpostaccplprb", [])) == np.size(state.listpostindxproptype):
            paths["acceptance_decomposition"] = plot_acceptance_decomposition(
                state, output_directory / "acceptance_decomposition", typefileplot)
        if np.size(getattr(state, "listpostchrototl", [])) == np.size(state.listpostindxproptype):
            paths["compute_budget"] = plot_compute_budget(
                state, output_directory / "compute_budget", typefileplot)
    catalogs = getattr(state, "listpostdictelem", None)
    position = catalog_options.get("position", "elin")
    if catalogs and any(len(sample) > catalog_options.get("population", 0)
                        and position in sample[catalog_options.get("population", 0)]
                        for sample in catalogs[:50]):
        paths["catalog_trace"] = plot_catalog_trace(
            state, output_directory / "catalog_trace", typefileplot=typefileplot, **catalog_options)
    if has_one_dimensional_prediction(state):
        paths["posterior_predictive"] = plot_posterior_predictive(
            state, output_directory / "posterior_predictive", typefileplot=typefileplot)
    return paths