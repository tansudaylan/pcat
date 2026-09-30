#!/usr/bin/env python3
"""Compare PCAT with emcee and dynesty on the same simulated sinusoid posterior.

The data are simulated and labeled as such: 40 epochs of a sinusoidal signal
(for example a radial-velocity curve) with Gaussian noise. All three samplers
use the same likelihood and the same uniform priors. The script records wall-
clock time, number of likelihood evaluations, and effective sample size (ESS),
then plots the marginal posteriors and the sampling efficiency.
"""

from tdpy.verbosity import print

import argparse
import shutil
import time
from pathlib import Path

import dynesty
import emcee
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

EXAMPLE_PATH = Path(__file__).resolve().parent
RUN_NAME = EXAMPLE_PATH.name

# simulated-data truth and priors
NAMES = ("ampl", "peri", "phas")
LABELS = ("Amplitude [m s$^{-1}$]", "Period [d]", "Phase [rad]")
TRUTH = np.array([5.0, 3.7, 1.2])  # [m s^-1], [d], [rad]
MINIMA = np.array([0.0, 2.0, 0.0])  # [m s^-1], [d], [rad]
MAXIMA = np.array([15.0, 6.0, 2.0 * np.pi])  # [m s^-1], [d], [rad]
NOISE = 2.0  # [m s^-1]
COLORS = {"PCAT": "C0", "emcee": "C1", "dynesty": "C2"}


def simulate_data(seed=7):
    """Return simulated epochs [d] and velocities [m s^-1]."""
    rng = np.random.default_rng(seed)
    time_obs = np.sort(rng.uniform(0.0, 30.0, 40))  # [d]
    velocity = TRUTH[0] * np.sin(2.0 * np.pi * time_obs / TRUTH[1] + TRUTH[2])
    return time_obs, velocity + rng.normal(0.0, NOISE, time_obs.size)


class Likelihood:
    """Gaussian log-likelihood that counts its own evaluations."""

    def __init__(self, time_obs, velocity):
        self.time_obs = time_obs
        self.velocity = velocity
        self.count = 0

    def __call__(self, values):
        self.count += 1
        model = values[0] * np.sin(2.0 * np.pi * self.time_obs / values[1] + values[2])
        return -0.5 * np.sum(((self.velocity - model) / NOISE) ** 2)


def effective_sample_size(chain):
    """Return the minimum ESS over parameters for a (steps, walkers, dims) chain."""
    tau = emcee.autocorr.integrated_time(chain, quiet=True)
    return float(np.min(chain.shape[0] * chain.shape[1] / tau))


# PCAT pickles its state, so its likelihood must be a module-level function
PCAT_LIKELIHOOD = None


def pcat_log_likelihood(gdat, strgmodl, values):
    return PCAT_LIKELIHOOD(values)


def run_pcat(likelihood, numbswep):
    global PCAT_LIKELIHOOD
    from pcat import sampling

    PCAT_LIKELIHOOD = likelihood
    # PCAT resumes from cached state, so remove it to time a fresh run
    path_cache = EXAMPLE_PATH / "data" / "outp" / RUN_NAME
    if path_cache.exists():
        print(f"Removing cached PCAT state {path_cache}...")
        shutil.rmtree(path_cache)
    numbburn = numbswep // 5
    start = time.perf_counter()
    result = sampling.sample_fixed(
        retr_llik=pcat_log_likelihood,
        parameter_names=NAMES,
        prior_types=("self",) * 3,
        prior_minima=MINIMA,
        prior_maxima=MAXIMA,
        initial_values=0.5 * (MINIMA + MAXIMA),
        booladaptstdp=True,
        pathbase=str(EXAMPLE_PATH),
        strgcnfg=RUN_NAME,
        numbproc=1,
        numbswep=numbswep,
        numbburn=numbburn,
        numbsamp=numbswep - numbburn,
        boolmakeplot=False,
        typeverb=0,
    )
    elapsed = time.perf_counter() - start
    samples = np.asarray(result.listpostparagenrscalbase)
    return samples, elapsed, effective_sample_size(samples[:, None, :])


def run_emcee(likelihood, numbstep, seed=3):
    def log_probability(values):
        if np.any(values < MINIMA) or np.any(values > MAXIMA):
            return -np.inf
        return likelihood(values)

    numbwalk = 16
    rng = np.random.default_rng(seed)
    initial = rng.uniform(MINIMA, MAXIMA, (numbwalk, 3))
    sampler = emcee.EnsembleSampler(numbwalk, 3, log_probability)
    start = time.perf_counter()
    sampler.run_mcmc(initial, numbstep)
    elapsed = time.perf_counter() - start
    chain = sampler.get_chain(discard=numbstep // 5)
    return chain.reshape(-1, 3), elapsed, effective_sample_size(chain)


def run_dynesty(likelihood, seed=5):
    sampler = dynesty.NestedSampler(likelihood, lambda unit: MINIMA + unit * (MAXIMA - MINIMA), 3,
                                    nlive=500, rstate=np.random.default_rng(seed))
    start = time.perf_counter()
    sampler.run_nested(print_progress=False)
    elapsed = time.perf_counter() - start
    results = sampler.results
    samples = results.samples_equal()
    # Kish effective size of the importance weights
    weights = np.exp(results.logwt - results.logz[-1])
    ess = float(np.sum(weights) ** 2 / np.sum(weights ** 2))
    return samples, elapsed, ess, (results.logz[-1], results.logzerr[-1])


def configure_style(typeplotback):
    colrfore = "white" if typeplotback == "dark" else "black"
    colrback = "black" if typeplotback == "dark" else "white"
    mpl.rcParams.update({
        "font.size": 10, "text.usetex": False, "axes.grid": False,
        "figure.facecolor": colrback, "axes.facecolor": colrback, "savefig.facecolor": colrback,
        "axes.edgecolor": colrfore, "axes.labelcolor": colrfore, "xtick.color": colrfore,
        "ytick.color": colrfore, "text.color": colrfore,
        "legend.fancybox": True, "legend.framealpha": 1.0,
    })


def save(figure, name, typefileplot):
    path = EXAMPLE_PATH / "visuals" / f"{name}.{typefileplot}"
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {path}...")
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_marginals(results, typefileplot):
    """Overlay the marginal posteriors from the three samplers with the simulated truth."""
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    for k, axis in enumerate(axes):
        values = np.concatenate([result["samples"][:, k] for result in results.values()])
        bins = np.linspace(*np.percentile(values, [0.5, 99.5]), 40)
        for name, result in results.items():
            axis.hist(result["samples"][:, k], bins=bins, density=True, histtype="step",
                      lw=1.5, color=COLORS[name], label=name)
        axis.axvline(TRUTH[k], color="0.4", ls="--", label="Simulated truth")
        axis.set_xlabel(LABELS[k])
        axis.set_yticks([])
    axes[0].set_ylabel("Posterior density")
    axes[0].legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    save(figure, "sampler_comparison_marginal_posteriors", typefileplot)


def plot_efficiency(results, typefileplot):
    """Compare global-mode recovery, effective samples per second, and per likelihood evaluation."""
    names = list(results)
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    for axis, key, label in [(axes[0], "frac_mode", "Fraction in true period mode"),
                             (axes[1], "ess_per_second", "ESS per second [s$^{-1}$]"),
                             (axes[2], "ess_per_call", "ESS per likelihood call")]:
        values = [results[name][key] for name in names]
        axis.bar(names, values, color=[COLORS[name] for name in names])
        for position, value in enumerate(values):
            axis.text(position, value * 1.05, f"{value:.2g}", ha="center")
        if key == "frac_mode":
            axis.set_ylim(0.0, 1.15)
        else:
            axis.set_yscale("log")
            axis.set_ylim(min(values) / 3.0, max(values) * 4.0)
        axis.set_ylabel(label)
    figure.tight_layout()
    save(figure, "sampler_comparison_efficiency", typefileplot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbswep", type=int, default=60000, help="PCAT sweeps.")
    parser.add_argument("--numbstep", type=int, default=4000, help="emcee steps per walker.")
    parser.add_argument("--typefileplot", choices=("png", "pdf"), default="png")
    parser.add_argument("--typeplotback", choices=("white", "dark"), default="white")
    arguments = parser.parse_args()

    time_obs, velocity = simulate_data()
    results = {}
    runners = {
        "PCAT": lambda like: run_pcat(like, arguments.numbswep),
        "emcee": lambda like: run_emcee(like, arguments.numbstep),
        "dynesty": run_dynesty,
    }
    for name, runner in runners.items():
        likelihood = Likelihood(time_obs, velocity)
        output = runner(likelihood)
        samples, elapsed, ess = output[:3]
        results[name] = {
            "samples": samples, "elapsed": elapsed, "calls": likelihood.count, "ess": ess,
            "ess_per_second": ess / elapsed, "ess_per_call": ess / likelihood.count,
            # samples within 0.2 d of the simulated period; ESS is meaningful only when this is near 1
            "frac_mode": float(np.mean(np.abs(samples[:, 1] - TRUTH[1]) < 0.2)),
        }
        summary = ", ".join(f"{n} {np.median(samples[:, k]):.3f}+-{np.std(samples[:, k]):.3f}"
                            for k, n in enumerate(NAMES))
        print(f"{name}: {elapsed:.1f} s, {likelihood.count} calls, ESS {ess:.0f}, "
              f"fraction in true period mode {results[name]['frac_mode']:.2f}, {summary}")
        if name == "dynesty":
            print(f"dynesty log-evidence {output[3][0]:.2f} +- {output[3][1]:.2f}")

    configure_style(arguments.typeplotback)
    plot_marginals(results, arguments.typefileplot)
    plot_efficiency(results, arguments.typefileplot)


if __name__ == "__main__":
    main()
