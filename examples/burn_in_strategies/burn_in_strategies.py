#!/usr/bin/env python3
"""Compare PCAT burn-in strategies on two seeded simulated posterior targets.

The correlated Gaussian tests proposal-scale tuning. The equal-weight bimodal
target tests whether burn-in explores both modes when initialized in one mode.
No observational data or fabricated scientific measurements are used.
"""

from __future__ import annotations

from tdpy.verbosity import print

import argparse
from tdpy.cli import add_plot_arguments
from pathlib import Path
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np

from pcat import sampling
from pcat.diagnostics import autocorrelation_time
from pcat.main import retr_pathrun


EXAMPLE_PATH = Path(__file__).resolve().parent
VISUAL_PATH = EXAMPLE_PATH / "visuals"
PROBLEMS = ("correlated_gaussian", "bimodal")
STRATEGIES = {
    "fixed": {"label": "Fixed scale", "booladaptstdp": False, "boolburntmpr": False},
    "adaptive": {"label": "Adaptive", "booladaptstdp": True, "boolburntmpr": False},
    "tempered_adaptive": {
        "label": "Tempered + adaptive",
        "booladaptstdp": True,
        "boolburntmpr": True,
    },
}
COLORS = {"fixed": "#555555", "adaptive": "#007360", "tempered_adaptive": "#A51C30"}


def benchmark_log_likelihood(gdat, model_name, values):
    """Return one of the two normalized-shape benchmark log likelihoods."""
    if gdat.burn_problem == "correlated_gaussian":
        covariance = np.array(((1.0, 0.95), (0.95, 1.0)))
        return float(-0.5 * values @ np.linalg.solve(covariance, values))
    if gdat.burn_problem == "bimodal":
        standard_deviation = 0.35
        log_components = -0.5 * ((values[0] - np.array((-2.0, 2.0))) / standard_deviation) ** 2
        maximum = np.max(log_components)
        return float(maximum + np.log(np.mean(np.exp(log_components - maximum))))
    raise ValueError(f"Unknown benchmark problem {gdat.burn_problem!r}")


def _problem_configuration(problem):
    if problem == "correlated_gaussian":
        return {
            "parameter_names": ("x", "y"),
            "prior_minima": (-6.0, -6.0),
            "prior_maxima": (6.0, 6.0),
            "initial_values": (4.5, -4.5),
            "proposal_scales": (0.005, 0.005),
        }
    return {
        "parameter_names": ("x",),
        "prior_minima": (-6.0,),
        "prior_maxima": (6.0,),
        "initial_values": (-2.0,),
        "proposal_scales": (0.05,),
    }


def _effective_sample_size(samples):
    _, correlation_time = autocorrelation_time(samples)
    maximum_time = np.nanmax(correlation_time)
    return float(samples.shape[0] / maximum_time) if np.isfinite(maximum_time) else 0.0


def _posterior_error(problem, samples):
    if problem == "correlated_gaussian":
        return float(np.linalg.norm(np.mean(samples, axis=0)))
    return float(abs(np.mean(samples[:, 0] > 0.0) - 0.5))


def run_benchmark(number_sweeps=1200, burn_fraction=0.25):
    """Run all problem/strategy pairs and return samples with performance summaries."""
    burn_count = int(number_sweeps * burn_fraction)
    sample_count = number_sweeps - burn_count
    results = {}
    for problem in PROBLEMS:
        configuration = _problem_configuration(problem)
        for strategy, options in STRATEGIES.items():
            run_name = f"{problem}_{strategy}"
            run_root = Path(retr_pathrun(EXAMPLE_PATH, run_name))
            if run_root.exists():
                print(f"Removing cached benchmark output {run_root}...")
                shutil.rmtree(run_root)
            start = time.perf_counter()
            posterior = sampling.sample_fixed(
                retr_llik=benchmark_log_likelihood,
                burn_problem=problem,
                prior_types=("self",) * len(configuration["parameter_names"]),
                pathbase=EXAMPLE_PATH,
                strgcnfg=run_name,
                numbswep=number_sweeps,
                numbburn=burn_count,
                numbsamp=sample_count,
                factburntmpr=0.75,
                typeseed=19,
                numbproc=1,
                boolmakeplot=False,
                makeanim=False,
                typeverb=0,
                **configuration,
                **{name: value for name, value in options.items() if name.startswith("bool")},
            )
            elapsed_seconds = time.perf_counter() - start  # [s]
            samples = np.asarray(posterior.listpostparagenrscalbase)
            accepted = np.asarray(posterior.listpostboolpropaccp).reshape(-1).astype(bool)
            temperature = np.asarray(posterior.listpostfacttmpr).reshape(-1)
            results[(problem, strategy)] = {
                "samples": samples,
                "elapsed_seconds": elapsed_seconds,
                "acceptance_fraction": float(np.mean(accepted[burn_count:])),
                "effective_sample_size": _effective_sample_size(samples),
                "posterior_error": _posterior_error(problem, samples),
                "inverse_temperature": temperature,
            }
    return results


def _save_figure(figure, name, typefileplot):
    VISUAL_PATH.mkdir(parents=True, exist_ok=True)
    path = VISUAL_PATH / f"{name}.{typefileplot}"
    print(f"Writing to {path}...")
    figure.savefig(path, dpi=300 if typefileplot == "png" else None, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_posterior_comparison(results, typefileplot="png"):
    """Plot retained posterior samples for both simulated benchmark problems."""
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), facecolor="white")
    axis = axes[0]
    for strategy, options in STRATEGIES.items():
        samples = results[("correlated_gaussian", strategy)]["samples"]
        axis.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.18, color=COLORS[strategy], label=options["label"])
    axis.scatter(0.0, 0.0, marker="x", s=65, color="black", linewidth=2.0, label="Target mean")
    axis.set(xlabel="x", ylabel="y", title="Correlated Gaussian")
    axis.legend(loc="upper left", frameon=True, fancybox=True, framealpha=1.0, fontsize=8)

    axis = axes[1]
    bins = np.linspace(-4.0, 4.0, 60)
    for strategy, options in STRATEGIES.items():
        samples = results[("bimodal", strategy)]["samples"][:, 0]
        axis.hist(samples, bins=bins, density=True, histtype="step", lw=1.6, color=COLORS[strategy], label=options["label"])
    axis.axvline(-2.0, color="black", ls="--", lw=1.0)
    axis.axvline(2.0, color="black", ls="--", lw=1.0, label="Target modes")
    axis.set(xlabel="x", ylabel="Posterior density", title="Equal-weight bimodal target")
    axis.legend(loc="upper center", frameon=True, fancybox=True, framealpha=1.0, fontsize=8)
    for axis in axes:
        axis.grid(False)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return _save_figure(figure, "burn_in_posterior_comparison", typefileplot)


def plot_performance_comparison(results, typefileplot="png"):
    """Plot acceptance, effective sample rate, error, and runtime by strategy."""
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), facecolor="white")
    metrics = (
        ("acceptance_fraction", "Post-burn acceptance fraction", False),
        ("effective_samples_per_second", "Effective samples [s$^{-1}$]", True),
        ("posterior_error", "Target-summary error", True),
        ("elapsed_seconds", "Runtime [s]", False),
    )
    x_positions = np.arange(len(STRATEGIES))
    width = 0.36
    for axis, (metric, label, logarithmic) in zip(axes.flat, metrics):
        for problem_index, problem in enumerate(PROBLEMS):
            values = []
            for strategy in STRATEGIES:
                result = results[(problem, strategy)]
                value = result[metric] if metric in result else result["effective_sample_size"] / result["elapsed_seconds"]
                values.append(value)
            offset = (problem_index - 0.5) * width
            axis.bar(x_positions + offset, values, width=width, label=problem.replace("_", " ").title(),
                     color=("#007360", "#A51C30")[problem_index], alpha=0.82)
        axis.set_xticks(x_positions)
        axis.set_xticklabels([options["label"] for options in STRATEGIES.values()], rotation=14)
        axis.set_ylabel(label)
        axis.grid(False)
        axis.spines[["top", "right"]].set_visible(False)
        if logarithmic:
            axis.set_yscale("log")
    axes[0, 0].legend(loc="upper right", frameon=True, fancybox=True, framealpha=1.0, fontsize=8)
    figure.tight_layout()
    return _save_figure(figure, "burn_in_performance_comparison", typefileplot)


def plot_temperature_schedule(results, typefileplot="png"):
    """Plot the recorded inverse-likelihood-temperature history."""
    figure, axis = plt.subplots(figsize=(7.2, 3.8), facecolor="white")
    for strategy, options in STRATEGIES.items():
        schedule = results[("correlated_gaussian", strategy)]["inverse_temperature"]
        axis.plot(np.arange(schedule.size), schedule, color=COLORS[strategy], lw=1.8, label=options["label"])
    axis.set(xlabel="Sweep", ylabel=r"Inverse likelihood temperature $\beta$", ylim=(-0.03, 1.05))
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower right", frameon=True, fancybox=True, framealpha=1.0)
    figure.tight_layout()
    return _save_figure(figure, "burn_in_temperature_schedule", typefileplot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbswep", type=int, default=1200)
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    results = run_benchmark(arguments.numbswep)
    plot_posterior_comparison(results, arguments.typefileplot)
    plot_performance_comparison(results, arguments.typefileplot)
    plot_temperature_schedule(results, arguments.typefileplot)
    for (problem, strategy), result in results.items():
        print(
            f"{problem} | {strategy}: acceptance={result['acceptance_fraction']:.3f}, "
            f"ESS={result['effective_sample_size']:.1f}, error={result['posterior_error']:.3g}, "
            f"runtime={result['elapsed_seconds']:.2f} s"
        )


if __name__ == "__main__":
    main()