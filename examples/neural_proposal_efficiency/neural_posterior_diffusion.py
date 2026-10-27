"""Compare neural source-flux uncertainties for two blended point sources."""

from tdpy.verbosity import print

import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit, logsumexp, logit

from pcat import NeuralElementProposal
from examples.neural_proposal_efficiency.point_source_detection import (
    BACKGROUND_BOUNDS, IMAGE_SHAPE, MAXIMUM_FLUX, PIXEL_NOISE,
    PSF_SIGMA, SOURCE_PIXELS, point_source_templates,
)

EXAMPLE_PATH = Path(__file__).resolve().parent
RESULTS_PATH = EXAMPLE_PATH / "data"
FIGURES_PATH = EXAMPLE_PATH / "visuals"
NOISE = PIXEL_NOISE  # [count pixel^-1]
BOUND = MAXIMUM_FLUX / 2.0  # [count]
BACKGROUND = sum(BACKGROUND_BOUNDS) / 2.0  # [count pixel^-1]
GRID_SIZE = 201
HISTOGRAM_BINS = 24
FIGURE_FONT_SIZE = 12.0  # [pt]


def _design_matrix():
    return point_source_templates()


def _center_image(observations, design):
    return np.asarray(observations).reshape(-1) - BACKGROUND - BOUND * design.sum(axis=1)


def _condition(observations, design):
    return _center_image(observations, design) @ design / NOISE**2


def _simulate_pairs(design, sample_count, seed):
    random = np.random.default_rng(seed)
    coefficients = random.uniform(-BOUND, BOUND, size=(sample_count, 2))
    observations = coefficients @ design.T + random.normal(
        0.0, NOISE, size=(sample_count, design.shape[0])
    )
    conditions = observations @ design / NOISE**2
    unit_coefficients = (coefficients + BOUND) / (2.0 * BOUND)
    return conditions, unit_coefficients


class ConditionalDiffusionPosterior:
    """Conditional DDPM sampler for the bounded two-coefficient posterior."""

    def __init__(self, network, condition_mean, condition_scale, steps):
        import torch

        self.network = network.eval()
        self.condition_mean = np.asarray(condition_mean, dtype=np.float32)
        self.condition_scale = np.asarray(condition_scale, dtype=np.float32)
        self.steps = int(steps)
        self.betas = torch.linspace(1e-4, 0.02, self.steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    @classmethod
    def fit(cls, conditions, unit_coefficients, *, steps=64, epochs=80,
            batch_size=256, seed=0):
        import torch
        from torch import nn

        conditions = np.asarray(conditions, dtype=np.float32)
        unit_coefficients = np.asarray(unit_coefficients, dtype=np.float32)
        if conditions.ndim != 2 or conditions.shape[1] != 2:
            raise ValueError("conditions must have shape (n, 2).")
        if unit_coefficients.shape != conditions.shape:
            raise ValueError("unit_coefficients must align with conditions.")
        if not np.isfinite(conditions).all() or not np.isfinite(unit_coefficients).all():
            raise ValueError("training arrays must be finite.")
        if np.any((unit_coefficients <= 0.0) | (unit_coefficients >= 1.0)):
            raise ValueError("unit_coefficients must lie strictly inside (0, 1).")
        if steps < 2 or epochs < 1 or batch_size < 1:
            raise ValueError("steps must be at least two; epochs and batch_size must be positive.")

        condition_mean = conditions.mean(axis=0)
        condition_scale = conditions.std(axis=0)
        condition_scale[condition_scale == 0.0] = 1.0
        normalized_conditions = (conditions - condition_mean) / condition_scale
        latent = logit(unit_coefficients)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            network = nn.Sequential(
                nn.Linear(20, 128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU(),
                nn.Linear(128, 2),
            )

        posterior = cls(network, condition_mean, condition_scale, steps)
        tensor_latent = torch.as_tensor(latent)
        tensor_conditions = torch.as_tensor(normalized_conditions)
        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-5)
        random = torch.Generator().manual_seed(int(seed) + 1)
        frequencies = torch.exp(torch.linspace(0.0, np.log(1000.0), 8))

        for _ in range(int(epochs)):
            permutation = torch.randperm(conditions.shape[0], generator=random)
            network.train()
            for indices in permutation.split(int(batch_size)):
                clean = tensor_latent[indices]
                condition = tensor_conditions[indices]
                timestep = torch.randint(0, steps, (len(indices),), generator=random)
                noise = torch.randn(clean.shape, generator=random)
                alpha_bar = posterior.alpha_bars[timestep, None]
                noisy = alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise
                phase = (timestep.float() / max(steps - 1, 1))[:, None] * frequencies[None, :] * 2.0 * np.pi
                time_features = torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)
                prediction = network(torch.cat((noisy, condition, time_features), dim=1))
                loss = torch.mean((prediction - noise) ** 2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=10.0)
                optimizer.step()

        return posterior

    def sample(self, condition, draws, seed):
        import torch

        condition = np.asarray(condition, dtype=np.float32)
        if condition.shape != (2,) or not np.isfinite(condition).all():
            raise ValueError("condition must be a finite vector with two entries.")
        if isinstance(draws, bool) or int(draws) != draws or draws < 1:
            raise ValueError("draws must be a positive integer.")
        normalized = (condition - self.condition_mean) / self.condition_scale
        condition_tensor = torch.as_tensor(normalized).expand(int(draws), -1)
        random = torch.Generator().manual_seed(int(seed))
        latent = torch.randn((int(draws), 2), generator=random)
        frequencies = torch.exp(torch.linspace(0.0, np.log(1000.0), 8))
        with torch.no_grad():
            for index in range(self.steps - 1, -1, -1):
                timestep = torch.full((int(draws),), index / max(self.steps - 1, 1))
                phase = timestep[:, None] * frequencies[None, :] * 2.0 * np.pi
                features = torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)
                noise_prediction = self.network(torch.cat((latent, condition_tensor, features), dim=1))
                beta = self.betas[index]
                alpha = self.alphas[index]
                alpha_bar = self.alpha_bars[index]
                mean = (latent - beta / torch.sqrt(1.0 - alpha_bar) * noise_prediction) / torch.sqrt(alpha)
                if index:
                    latent = mean + torch.sqrt(beta) * torch.randn(
                        latent.shape, generator=random
                    )
                else:
                    latent = mean
        unit = torch.sigmoid(latent).numpy()
        return (2.0 * unit - 1.0) * BOUND


def _exact_posterior(observations, design):
    axis = np.linspace(-BOUND, BOUND, GRID_SIZE)
    first, second = np.meshgrid(axis, axis, indexing="ij")
    coefficients = np.column_stack((first.ravel(), second.ravel()))
    residual = _center_image(observations, design)[None, :] - coefficients @ design.T
    log_probability = -0.5 * np.sum((residual / NOISE) ** 2, axis=1)
    probability = np.exp(log_probability - logsumexp(log_probability)).reshape(
        GRID_SIZE, GRID_SIZE
    )
    return axis, probability


def _sample_exact(axis, probability, draws, seed):
    random = np.random.default_rng(seed)
    indices = random.choice(probability.size, size=draws, p=probability.ravel())
    first, second = np.unravel_index(indices, probability.shape)
    return np.column_stack((axis[first], axis[second]))


def _total_variation(samples, axis, probability):
    edges = np.linspace(-BOUND, BOUND, HISTOGRAM_BINS + 1)
    sampled = np.histogram2d(samples[:, 0], samples[:, 1], bins=(edges, edges))[0]
    sampled /= sampled.sum()
    grid_first, grid_second = np.meshgrid(axis, axis, indexing="ij")
    exact = np.histogram2d(
        grid_first.ravel(), grid_second.ravel(), bins=(edges, edges),
        weights=probability.ravel(),
    )[0]
    exact /= exact.sum()
    return 0.5 * float(np.abs(sampled - exact).sum())


def _weighted_interval(axis, probability, dimension, mass=0.9):
    marginal = probability.sum(axis=1 - dimension)
    cumulative = np.cumsum(marginal)
    tail = 0.5 * (1.0 - mass)
    return np.interp([tail, 1.0 - tail], cumulative, axis)


def _wilson_interval(successes, trials, z=1.96):
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("successes and trials must define a valid binomial sample.")
    fraction = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (fraction + z**2 / (2.0 * trials)) / denominator
    radius = z * np.sqrt(
        fraction * (1.0 - fraction) / trials + z**2 / (4.0 * trials**2)
    ) / denominator
    return [float(center - radius), float(center + radius)]


def _bootstrap_median_interval(values, seed, draws=10000):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a nonempty finite vector.")
    random = np.random.default_rng(seed)
    resampled = random.choice(values, size=(draws, values.size), replace=True)
    return [float(value) for value in np.quantile(np.median(resampled, axis=1), [0.025, 0.975])]


def _run(arguments):
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_PATH / f"{arguments.run_id}_posterior.json"
    figure_path = FIGURES_PATH / f"{arguments.run_id}_posterior.{arguments.typefileplot}"
    arrays_path = RESULTS_PATH / f"{arguments.run_id}_posterior.npz"
    for path in (result_path, figure_path, arrays_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}.")

    design = _design_matrix()
    conditions, unit_coefficients = _simulate_pairs(
        design, arguments.training_samples, arguments.seed + 1
    )
    previous_threads = None
    import torch

    if torch.get_num_threads() > 1:
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
    try:
        training_start = time.perf_counter()
        diffusion = ConditionalDiffusionPosterior.fit(
            conditions, unit_coefficients,
            steps=arguments.diffusion_steps,
            epochs=arguments.epochs,
            seed=arguments.seed + 2,
        )
        mixture = NeuralElementProposal.fit(
            conditions, unit_coefficients,
            number_components=arguments.mixture_components,
            hidden_sizes=(64, 64),
            epochs=arguments.epochs,
            batch_size=256,
            uniform_fraction=0.01,
            seed=arguments.seed + 3,
        )
        training_seconds = time.perf_counter() - training_start

        scenes = []
        observations_saved, truth_saved, samples_saved, references_saved = [], [], [], []
        example = None
        for scene_index in range(arguments.validation_scenes):
            scene_seed = arguments.seed + 100 + scene_index
            scene_random = np.random.default_rng(scene_seed)
            truth = scene_random.uniform(-BOUND, BOUND, size=2)
            observations = BACKGROUND + design @ (truth + BOUND) + scene_random.normal(
                0.0, NOISE, design.shape[0]
            )
            condition = _condition(observations, design)
            axis, exact = _exact_posterior(observations, design)
            reference = _sample_exact(axis, exact, arguments.posterior_draws, scene_seed + 1)
            observations_saved.append(observations.reshape(IMAGE_SHAPE))
            truth_saved.append(truth + BOUND)
            references_saved.append(exact)
            model_samples = {}
            for method in ("mixture", "diffusion"):
                start = time.perf_counter()
                if method == "mixture":
                    state = np.random.get_state()
                    np.random.seed(scene_seed + 2)
                    try:
                        unit = np.asarray([
                            mixture.sample(condition) for _ in range(arguments.posterior_draws)
                        ])
                    finally:
                        np.random.set_state(state)
                    samples = (2.0 * unit - 1.0) * BOUND
                else:
                    samples = diffusion.sample(
                        condition, arguments.posterior_draws, scene_seed + 3
                    )
                sample_seconds = time.perf_counter() - start
                intervals = np.asarray([
                    np.quantile(samples[:, dimension], [0.05, 0.95])
                    for dimension in range(2)
                ])
                model_samples[method] = samples
                samples_saved.append(samples + BOUND)
                scenes.append({
                    "scene_seed": scene_seed,
                    "method": method,
                    "total_variation": _total_variation(samples, axis, exact),
                    "coverage_90": [
                        bool(intervals[dimension, 0] <= truth[dimension] <= intervals[dimension, 1])
                        for dimension in range(2)
                    ],
                    "sampling_seconds": float(sample_seconds),
                })
            if example is None:
                example = (axis, exact, model_samples)
            reference_intervals = np.asarray([
                _weighted_interval(axis, exact, dimension) for dimension in range(2)
            ])
            scenes.append({
                "scene_seed": scene_seed,
                "method": "exact-grid",
                "total_variation": _total_variation(reference, axis, exact),
                "coverage_90": [
                    bool(reference_intervals[dimension, 0] <= truth[dimension] <= reference_intervals[dimension, 1])
                    for dimension in range(2)
                ],
                "sampling_seconds": None,
            })
    finally:
        if previous_threads is not None:
            torch.set_num_threads(previous_threads)

    summary = {}
    for method_index, method in enumerate(("exact-grid", "mixture", "diffusion")):
        selected = [scene for scene in scenes if scene["method"] == method]
        coverage = np.asarray([scene["coverage_90"] for scene in selected], dtype=bool)
        coverage_successes = coverage.sum(axis=0).astype(int).tolist()
        coverage_fraction = np.asarray(coverage_successes, dtype=float) / coverage.shape[0]
        total_variations = [scene["total_variation"] for scene in selected]
        summary[method] = {
            "median_total_variation": float(np.median(total_variations)),
            "median_total_variation_95pct_scene_bootstrap_interval": _bootstrap_median_interval(
                total_variations, arguments.seed + 5000 + method_index
            ),
            "marginal_90pct_coverage_successes": coverage_successes,
            "marginal_90pct_coverage": coverage_fraction.tolist(),
            "marginal_90pct_coverage_95pct_wilson_interval": [
                _wilson_interval(successes, coverage.shape[0])
                for successes in coverage_successes
            ],
            "median_sampling_seconds": (
                None if method == "exact-grid" else float(np.median([
                    scene["sampling_seconds"] for scene in selected
                ]))
            ),
        }
    result = {
        "run_id": arguments.run_id,
        "seed": arguments.seed,
        "training_samples": arguments.training_samples,
        "validation_scenes": arguments.validation_scenes,
        "posterior_draws_per_scene": arguments.posterior_draws,
        "diffusion_steps": arguments.diffusion_steps,
        "epochs": arguments.epochs,
        "mixture_components": arguments.mixture_components,
        "typefileplot": arguments.typefileplot,
        "training_seconds": training_seconds,
        "design": "two blended point sources at known positions with independent Gaussian pixel noise",
        "image_shape": list(IMAGE_SHAPE),
        "source_pixels": [list(position) for position in SOURCE_PIXELS],
        "psf_sigma_pixels": PSF_SIGMA,
        "background_counts_per_pixel": BACKGROUND,
        "noise_counts_per_pixel": NOISE,
        "prior": f"independent uniform source fluxes on [0, {2.0 * BOUND}] count",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": version("torch"),
            "torch_threads_during_training": 1,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "limitations": [
            "fixed-dimensional toy posterior, not a transdimensional PCAT benchmark",
            "diffusion samples do not provide the normalized proposal density required by PCAT",
        ],
        "summary": summary,
        "scenes": scenes,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    print(f"Writing to {arrays_path}...")
    np.savez_compressed(
        arrays_path, observations=np.asarray(observations_saved),
        source_fluxes=np.asarray(truth_saved), samples=np.asarray(samples_saved),
        reference_probabilities=np.asarray(references_saved), flux_axis=axis + BOUND,
        templates=design,
    )
    print(f"Writing to {result_path}...")
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    _plot_example(example, figure_path)
    return result


def _plot_example(example, path):
    axis, probability, samples = example
    grid_first, grid_second = np.meshgrid(axis + BOUND, axis + BOUND, indexing="ij")
    ranked = np.sort(probability.ravel())[::-1]
    cumulative = np.cumsum(ranked)
    levels = sorted([
        ranked[np.searchsorted(cumulative, 0.5)],
        ranked[np.searchsorted(cumulative, 0.9)],
    ])
    figure, axes = plt.subplots(1, 2, figsize=(5.4, 4.0), constrained_layout=True)
    for plot_axis, method, color, title in (
        (axes[0], "mixture", "#D55E00", "Mixture"),
        (axes[1], "diffusion", "#0072B2", "Diffusion"),
    ):
        plot_axis.contour(grid_first, grid_second, probability, levels=levels, colors="black")
        plot_axis.scatter(
            samples[method][:500, 0] + BOUND, samples[method][:500, 1] + BOUND,
            s=8, alpha=0.35, color=color, label="Posterior draws",
        )
        plot_axis.set_title(title, fontsize=FIGURE_FONT_SIZE)
        plot_axis.set_xlabel(r"Flux $F_1$ [count]", fontsize=FIGURE_FONT_SIZE)
        plot_axis.set_ylabel(r"Flux $F_2$ [count]", fontsize=FIGURE_FONT_SIZE)
        plot_axis.tick_params(labelsize=FIGURE_FONT_SIZE)
        plot_axis.set_xlim(0.0, 2.0 * BOUND)
        plot_axis.set_ylim(0.0, 2.0 * BOUND)
        plot_axis.set_aspect("equal")
        plot_axis.grid(False)
    figure.legend(
        [
            plt.Line2D([], [], color="black"),
            plt.Line2D([], [], marker="o", color="#D55E00", linestyle="none"),
            plt.Line2D([], [], marker="o", color="#0072B2", linestyle="none"),
        ],
        ["Reference 50% and 90% regions", "Mixture samples", "Diffusion samples"],
        loc="outside lower center", ncol=1, fontsize=FIGURE_FONT_SIZE,
        frameon=True, fancybox=True, framealpha=1.0, edgecolor="black",
    )
    figure.patch.set_facecolor("white")
    print(f"Writing to {path}...")
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="point_source_posterior_diffusion")
    parser.add_argument("--typefileplot", choices=("png", "pdf"), default="png")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--training-samples", type=int, default=12000)
    parser.add_argument("--validation-scenes", type=int, default=20)
    parser.add_argument("--posterior-draws", type=int, default=2000)
    parser.add_argument("--diffusion-steps", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--mixture-components", type=int, default=8)
    arguments = parser.parse_args()
    if min(arguments.training_samples, arguments.validation_scenes,
           arguments.posterior_draws, arguments.epochs) < 1:
        parser.error("sample counts and epochs must be positive.")
    if arguments.diffusion_steps < 2 or arguments.mixture_components < 1:
        parser.error("diffusion_steps must be at least two and mixture_components positive.")
    _run(arguments)


if __name__ == "__main__":
    main()