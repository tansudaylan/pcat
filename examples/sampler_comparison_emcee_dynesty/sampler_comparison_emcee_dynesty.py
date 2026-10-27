#!/usr/bin/env python3
"""Compare PCAT with emcee and dynesty on simulated radial-velocity (RV) data.

All data are simulated and labeled as such. The comparison has two parts.

Fixed dimension: 40 epochs of one sinusoidal signal with Gaussian noise. All three samplers use the
same likelihood and uniform priors. The script records wall-clock time, likelihood evaluations,
effective sample size (ESS), and the fraction of samples in the true period mode.

Variable dimension: 60 epochs containing two Keplerian planets, with the number of planets unknown
(0 to 3). PCAT samples the number of planets and their orbits in one run. emcee cannot compare models
of different dimension, and dynesty needs one nested-sampling run per planet count, whose evidences
give the posterior on the count. Both use the same marginalized likelihood (offset analytically,
jitter numerically) and the same priors, including a uniform prior on the count, so their posteriors
on the number of planets target the same result; finite runs can disagree through incomplete exploration.
"""

from tdpy.verbosity import print

import argparse
from datetime import datetime, timezone
import hashlib
import json
from tdpy.cli import add_plot_arguments
import time
from pathlib import Path

import dynesty
import emcee
import matplotlib.pyplot as plt
import numpy as np
from pcat.demo import configure_example_style, save_example_figure

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
    """Return the minimum ESS, or NaN when emcee's chain-length check fails."""
    try:
        tau = emcee.autocorr.integrated_time(chain, quiet=False)
    except emcee.autocorr.AutocorrError:
        return np.nan
    if not np.isfinite(tau).all() or np.any(tau <= 0.0):
        return np.nan
    return float(np.min(chain.shape[0] * chain.shape[1] / tau))


# PCAT pickles its state, so its likelihood must be a module-level function
PCAT_LIKELIHOOD = None


def pcat_log_likelihood(gdat, strgmodl, values):
    return PCAT_LIKELIHOOD(values)


def fresh_comparison_root(run_name):
    from pcat.main import retr_pathrun

    root = Path(retr_pathrun(EXAMPLE_PATH, run_name))
    if ((root / "data/outp").exists() or (root / "visuals").exists()
            or (EXAMPLE_PATH / "data/outp" / run_name).exists()):
        raise FileExistsError(f"Refusing to overwrite comparison run {run_name}")
    return root


def run_pcat(likelihood, numbswep):
    global PCAT_LIKELIHOOD
    from pcat import sampling

    PCAT_LIKELIHOOD = likelihood
    fresh_comparison_root(RUN_NAME)
    numbburn = numbswep // 5
    start = time.perf_counter()
    sampling.sample_fixed(
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
        makeanim=False,
        typeverb=0,
    )
    elapsed = time.perf_counter() - start
    from pcat.main import readfile, retr_pathrun

    result = readfile(str(Path(retr_pathrun(EXAMPLE_PATH, RUN_NAME)) / "data/outp" / RUN_NAME / "gdatfinlpost"))
    samples = np.asarray(result.listpostparagenrscalbase)
    return samples, elapsed, effective_sample_size(samples[:, None, :])


def run_emcee(likelihood, numbstep, seed=3):
    """Seed walker positions and transitions without changing NumPy's global state."""
    def log_probability(values):
        if np.any(values < MINIMA) or np.any(values > MAXIMA):
            return -np.inf
        return likelihood(values)

    numbwalk = 16
    rng = np.random.default_rng(seed)
    initial = rng.uniform(MINIMA, MAXIMA, (numbwalk, 3))
    sampler = emcee.EnsembleSampler(numbwalk, 3, log_probability)
    sampler.random_state = np.random.RandomState(rng.integers(0, 2**32)).get_state()
    start = time.perf_counter()
    sampler.run_mcmc(initial, numbstep)
    elapsed = time.perf_counter() - start
    chain = sampler.get_chain(discard=numbstep // 5)
    return chain.reshape(-1, 3), elapsed, effective_sample_size(chain)


def run_dynesty(likelihood, seed=5, *, nlive=500, maxcall=None):
    sampler = dynesty.NestedSampler(likelihood, lambda unit: MINIMA + unit * (MAXIMA - MINIMA), 3,
                                    nlive=nlive, rstate=np.random.default_rng(seed))
    start = time.perf_counter()
    sampler.run_nested(print_progress=False, maxcall=maxcall)
    elapsed = time.perf_counter() - start
    results = sampler.results
    samples = results.samples_equal()
    # Kish effective size of the importance weights
    weights = np.exp(results.logwt - results.logz[-1])
    ess = float(np.sum(weights) ** 2 / np.sum(weights ** 2))
    return samples, elapsed, ess, (results.logz[-1], results.logzerr[-1])


def plot_marginals(results, typefileplot):
    """Overlay the marginal posteriors from the three samplers with the simulated truth."""
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), layout="constrained")
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
    save_comparison_figure(figure, "sampler_comparison_marginal_posteriors", typefileplot)


def plot_efficiency(results, typefileplot):
    """Compare global-mode recovery, effective samples per second, and per likelihood evaluation."""
    names = list(results)
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), layout="constrained")
    for axis, key, label in [(axes[0], "frac_mode", "Fraction within 0.2 d of injected period"),
                             (axes[1], "ess_per_second", "Reported ESS per second [s$^{-1}$]"),
                             (axes[2], "ess_per_call", "Reported ESS per likelihood call")]:
        values = [results[name][key] for name in names]
        valid = [np.isfinite(value) and value >= 0.0 for value in values]
        axis.bar(names, [value if good else 0.0 for value, good in zip(values, valid)],
                 color=[COLORS[name] for name in names])
        for position, value in enumerate(values):
            if np.isfinite(value):
                axis.text(position, value * 1.05, f"{value:.2g}", ha="center")
            else:
                axis.text(position, 0.05, "unresolved", transform=axis.get_xaxis_transform(), ha="center")
        if key == "frac_mode":
            axis.set_ylim(0.0, 1.15)
        else:
            positive = [value for value in values if np.isfinite(value) and value > 0.0]
            if positive:
                axis.set_yscale("log")
                axis.set_ylim(min(positive) / 3.0, max(positive) * 4.0)
        axis.set_ylabel(label)
        if key != "frac_mode":
            axis.set_xticks(range(len(names)), ["PCAT\nMCMC", "emcee\nMCMC", "dynesty\nKish"])
    save_comparison_figure(figure, "sampler_comparison_efficiency", typefileplot)


def main():
    global RUN_NAME, DISPLAY_TAG

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbswep", type=int, default=60000, help="PCAT sweeps.")
    parser.add_argument("--numbstep", type=int, default=4000, help="emcee steps per walker.")
    parser.add_argument("--numbswepplan", type=int, default=200000, help="PCAT sweeps, variable planet count.")
    parser.add_argument("--nliveplan", type=int, default=1000, help="dynesty live points, variable planet count.")
    parser.add_argument("--part", choices=("fixed", "variable", "both"), default="both")
    parser.add_argument("--time-tag", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--nlive", type=int, default=500, help="dynesty live points, fixed dimension.")
    parser.add_argument("--maxcall", type=int, help="Optional likelihood-call cap per dynesty run; capped runs are integration checks.")
    add_plot_arguments(parser)
    parser.add_argument("--typeplotback", choices=("white", "dark"), default="white")
    arguments = parser.parse_args()
    if datetime.strptime(arguments.time_tag, "%Y%m%d_%H%M%S").strftime("%Y%m%d_%H%M%S") != arguments.time_tag:
        raise ValueError("time-tag must use YYYYMMDD_HHMMSS")
    if min(arguments.numbswep, arguments.numbstep, arguments.numbswepplan) < 40:
        raise ValueError("At least 40 MCMC steps are required")
    if min(arguments.nlive, arguments.nliveplan) < 20 or (arguments.maxcall is not None and arguments.maxcall < 1):
        raise ValueError("At least 20 live points and a positive call budget are required")
    DISPLAY_TAG = arguments.time_tag
    RUN_NAME = f"{EXAMPLE_PATH.name}_{DISPLAY_TAG}"
    if list((EXAMPLE_PATH / "visuals").glob(f"sampler_comparison_*_{DISPLAY_TAG}.*")):
        raise FileExistsError("Refusing to overwrite an earlier comparison output set")
    configure_example_style(arguments.typeplotback)
    if arguments.part in ("fixed", "both"):
        compare_fixed_dimension(arguments)
    if arguments.part in ("variable", "both"):
        compare_variable_dimension(arguments)


# simulated two-planet system: period [d], semi-amplitude [m s^-1], eccentricity, argument of periastron [rad], mean anomaly [rad]
PLANETS = np.array([[9.1, 6.0, 0.1, 1.0, 0.5], [61.0, 3.0, 0.0, 0.0, 2.0]])
MAXM_NUMB_PLAN = 3
NOISE_PLAN = 1.5  # [m s^-1], reported uncertainty
JITTER_PLAN = 1.0  # [m s^-1], extra white noise the samplers must marginalize


def simulate_planets(seed=11):
    """Return simulated epochs [d], velocities [m s^-1], uncertainties [m s^-1], and the reference time [d]."""
    from tdpy.exoplanet import keplerian_radial_velocity

    rng = np.random.default_rng(seed)
    time_obs = np.sort(rng.uniform(0.0, 800.0, 60)) + 8000.0  # [d]
    timerefr = 0.5 * (time_obs[0] + time_obs[-1])
    velocity = np.sum(keplerian_radial_velocity(time_obs[:, None], *PLANETS.T[[0, 1, 2, 3, 4]], timerefr), 1)
    velocity += 4.0 + rng.normal(0.0, np.hypot(NOISE_PLAN, JITTER_PLAN), time_obs.size)
    return time_obs, velocity, np.full(time_obs.size, NOISE_PLAN), timerefr


class PlanetLikelihood:
    """PCAT's marginalized RV likelihood for a fixed number of planets, counting its evaluations."""

    def __init__(self, time_obs, velocity, stdv, timerefr, numbplan):
        from pcat.radial_velocity import retr_matrdesi

        self.time_obs, self.velocity, self.stdv, self.timerefr, self.numbplan = time_obs, velocity, stdv, timerefr, numbplan
        self.matrdesi = retr_matrdesi(time_obs, np.zeros(time_obs.size, int), False, timerefr)
        self.listjitt = np.geomspace(0.1, 30.0, 24)  # [m s^-1], same grid as pcat.radial_velocity
        self.maxmperi = 2.0 * (time_obs[-1] - time_obs[0])  # [d]
        self.count = 0

    def prior_transform(self, unit):
        """Map the unit cube to (P, K, e, omega, M) per planet with PCAT's element priors."""
        unit = unit.reshape(self.numbplan, 5)
        return np.column_stack([1.2 * (self.maxmperi / 1.2) ** unit[:, 1], 0.3 * 1000.0 ** unit[:, 0],
                                0.8 * unit[:, 3], 2.0 * np.pi * unit[:, 4], 2.0 * np.pi * unit[:, 2]]).ravel()

    def __call__(self, values):
        from pcat.radial_velocity import retr_llik_rvelmarg
        from tdpy.exoplanet import keplerian_radial_velocity

        self.count += 1
        model = np.zeros(self.time_obs.size)
        if self.numbplan > 0:
            planets = values.reshape(self.numbplan, 5)
            model = np.sum(keplerian_radial_velocity(self.time_obs[:, None], *planets.T, self.timerefr), 1)
        return retr_llik_rvelmarg(self.velocity - model, self.stdv, self.matrdesi, self.listjitt)


PCAT_PLANET_CALLS = [0]


def pcat_planet_log_likelihood(gdat, strgmodl, cntpmodl):
    """PCAT's RV likelihood callback, counting evaluations."""
    from pcat.radial_velocity import retr_llik_rvel

    PCAT_PLANET_CALLS[0] += 1
    return retr_llik_rvel(gdat, strgmodl, cntpmodl)


def run_pcat_planets(time_obs, velocity, stdv, numbswep):
    """One transdimensional PCAT run over 0 to MAXM_NUMB_PLAN planets with a uniform count prior."""
    from pcat import sampling
    from pcat.main import readfile, retr_pathrun
    from pcat.radial_velocity import retr_dictpcatrvel

    strgcnfg = RUN_NAME + "_variable_planet_count"
    fresh_comparison_root(strgcnfg)
    dictpcat = retr_dictpcatrvel(time_obs, velocity, stdv, np.zeros(time_obs.size, int), str(EXAMPLE_PATH), strgcnfg,
                                 maxmnumbplan=MAXM_NUMB_PLAN, factpriodoff=0.0, probtran=0.7, probspmr=0.4,
                                 probjump=0.2, numbswep=numbswep, numbsamp=numbswep // 50,
                                 numbswepplot=max(numbswep // 40, 1), inittype="rand", typeseed=0,
                                 stdvpropelemfire=[1e-2, 1e-4, 3e-2, 3e-2, 3e-2], boolmakeplot=False,
                                 boolmakeplotinit=False, makeanim=False, typeverb=0,
                                 numbproc=1,
                                 retr_llik=pcat_planet_log_likelihood)
    PCAT_PLANET_CALLS[0] = 0
    start = time.perf_counter()
    sampling.sample(**dictpcat)
    elapsed = time.perf_counter() - start
    pathrun = Path(retr_pathrun(str(EXAMPLE_PATH), strgcnfg))
    posterior = readfile(str(pathrun / "data" / "outp" / strgcnfg / "gdatfinlpost"))
    numbelem = np.asarray(posterior.listpostnumbelem).astype(int).ravel()
    listperi = [np.asarray(sample[0]["elin"]) for sample, n in zip(posterior.listpostdictelem, numbelem) if n == 2]
    return {"prob": np.bincount(numbelem, minlength=MAXM_NUMB_PLAN + 1) / numbelem.size, "elapsed": elapsed,
            "calls": PCAT_PLANET_CALLS[0], "peri": np.sort(np.asarray(listperi).reshape(-1, 2), 1)}


def run_dynesty_planets(time_obs, velocity, stdv, timerefr, nlive, seed=5, *, maxcall=None):
    """One nested-sampling run per planet count; returns log-evidences, costs, and two-planet periods."""
    output = {"logz": [], "logzerr": [], "elapsed": [], "calls": []}
    for numbplan in range(MAXM_NUMB_PLAN + 1):
        likelihood = PlanetLikelihood(time_obs, velocity, stdv, timerefr, numbplan)
        start = time.perf_counter()
        if numbplan == 0:
            logz, logzerr = likelihood(np.zeros(0)), 0.0
        else:
            sampler = dynesty.NestedSampler(likelihood, likelihood.prior_transform, 5 * numbplan, nlive=nlive,
                                            sample="rslice", rstate=np.random.default_rng(seed + numbplan))
            sampler.run_nested(print_progress=False, dlogz=0.1, maxcall=maxcall)
            results = sampler.results
            logz, logzerr = results.logz[-1], results.logzerr[-1]
            if numbplan == 2:
                output["peri"] = np.sort(results.samples_equal().reshape(-1, 2, 5)[:, :, 0], 1)
        output["elapsed"].append(time.perf_counter() - start)
        output["calls"].append(likelihood.count)
        output["logz"].append(logz)
        output["logzerr"].append(logzerr)
    logz = np.array(output["logz"])
    # uniform prior on the count, so its posterior is proportional to the evidence
    output["prob"] = np.exp(logz - np.logaddexp.reduce(logz))
    # propagate evidence uncertainties by resampling the log-evidences
    rng = np.random.default_rng(seed)
    draws = rng.normal(logz, output["logzerr"], (2000, logz.size))
    draws = np.exp(draws - np.logaddexp.reduce(draws, axis=1)[:, None])
    output["proberr"] = np.std(draws, 0)
    return output


def plot_planet_count(pcat_result, dynesty_result, typefileplot):
    """Posterior on the number of planets from one PCAT run and from dynesty evidences."""
    numbplan = np.arange(MAXM_NUMB_PLAN + 1)
    figure, axis = plt.subplots(figsize=(6.5, 3.6), layout="constrained")
    axis.bar(numbplan - 0.2, pcat_result["prob"], 0.4, color=COLORS["PCAT"], label="PCAT, 1 run")
    axis.bar(numbplan + 0.2, dynesty_result["prob"], 0.4, yerr=dynesty_result["proberr"], color=COLORS["dynesty"],
             label=f"dynesty, {MAXM_NUMB_PLAN} runs")
    axis.axvline(PLANETS.shape[0], color="0.4", ls="--", label="Simulated count")
    axis.set_yscale("log")
    axis.set_ylim(1e-4, 2.0)
    axis.set_xticks(numbplan)
    axis.set_xlabel("Number of planets")
    axis.set_ylabel("Posterior probability")
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=9,
                  fancybox=True, framealpha=1.0)
    save_comparison_figure(figure, "sampler_comparison_planet_count_posterior", typefileplot)


def plot_planet_cost(pcat_result, dynesty_result, typefileplot):
    """Likelihood evaluations and wall-clock time to obtain the posterior on the number of planets."""
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.3), layout="constrained")
    for axis, key, label in [(axes[0], "calls", "Likelihood evaluations"), (axes[1], "elapsed", "Wall-clock time [s]")]:
        axis.bar("PCAT", pcat_result[key], color=COLORS["PCAT"], label="PCAT, all counts at once")
        bottom = 0.0
        for numbplan in range(1, MAXM_NUMB_PLAN + 1):
            value = dynesty_result[key][numbplan]
            axis.bar("dynesty", value, bottom=bottom, color=COLORS["dynesty"], alpha=0.35 + 0.2 * numbplan,
                     label=f"dynesty, {numbplan} planet{'s' if numbplan > 1 else ''}")
            bottom += value
        axis.set_ylabel(label)
    axes[1].legend(loc="upper left")
    save_comparison_figure(figure, "sampler_comparison_planet_count_cost", typefileplot)


def plot_planet_periods(pcat_result, dynesty_result, typefileplot):
    """Period posteriors of the two-planet solution from PCAT and dynesty."""
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.3), layout="constrained")
    for k, axis in enumerate(axes):
        samples = {name: np.asarray(result.get("peri", []), dtype=float).reshape(-1, 2)
                   for name, result in (("PCAT", pcat_result), ("dynesty", dynesty_result))}
        available = [values[:, k] for values in samples.values() if len(values)]
        values = np.concatenate(available) if available else PLANETS[:, 0]
        lower, upper = float(values.min()), float(values.max())
        if lower == upper:
            lower, upper = lower * 0.9, upper * 1.1
        bins = np.linspace(lower, upper, 40)
        for name, posterior in samples.items():
            if len(posterior):
                axis.hist(posterior[:, k], bins=bins, density=True, histtype="step", lw=1.5,
                          color=COLORS[name], label=name)
            else:
                axis.plot([], [], color=COLORS[name], label=f"{name}: zero retained two-planet draws")
        axis.axvline(PLANETS[k, 0], color="0.4", ls="--", label="Simulated truth")
        axis.set_xlabel(f"Period of planet {'bc'[k]} [d]")
        axis.set_yticks([])
    axes[0].set_ylabel("Posterior density")
    axes[0].legend(loc="upper left")
    save_comparison_figure(figure, "sampler_comparison_two_planet_periods", typefileplot)


def compare_variable_dimension(arguments):
    time_obs, velocity, stdv, timerefr = simulate_planets()
    pcat_result = run_pcat_planets(time_obs, velocity, stdv, arguments.numbswepplan)
    dynesty_result = run_dynesty_planets(time_obs, velocity, stdv, timerefr, arguments.nliveplan,
                                       maxcall=arguments.maxcall)
    for name, result in [("PCAT", pcat_result), ("dynesty", dynesty_result)]:
        print(f"{name}: P(N) {np.round(result['prob'], 4)}, {np.sum(result['calls'])} calls, "
              f"{np.sum(result['elapsed']):.0f} s")
    plot_planet_count(pcat_result, dynesty_result, arguments.typefileplot)
    plot_planet_cost(pcat_result, dynesty_result, arguments.typefileplot)
    plot_planet_periods(pcat_result, dynesty_result, arguments.typefileplot)
    write_comparison_metrics("variable", {"PCAT": pcat_result, "dynesty": dynesty_result}, arguments)


def compare_fixed_dimension(arguments):
    time_obs, velocity = simulate_data()
    results = {}
    runners = {
        "PCAT": lambda like: run_pcat(like, arguments.numbswep),
        "emcee": lambda like: run_emcee(like, arguments.numbstep),
        "dynesty": lambda like: run_dynesty(like, nlive=arguments.nlive, maxcall=arguments.maxcall),
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
            "ess_estimator": ("Kish importance-weight ESS" if name == "dynesty"
                              else "minimum MCMC autocorrelation ESS"),
        }
        summary = ", ".join(f"{n} {np.median(samples[:, k]):.3f}+-{np.std(samples[:, k]):.3f}"
                            for k, n in enumerate(NAMES))
        print(f"{name}: {elapsed:.1f} s, {likelihood.count} calls, ESS {ess:.0f}, "
              f"fraction in true period mode {results[name]['frac_mode']:.2f}, {summary}")
        if name == "dynesty":
            print(f"dynesty log-evidence {output[3][0]:.2f} +- {output[3][1]:.2f}")

    plot_marginals(results, arguments.typefileplot)
    plot_efficiency(results, arguments.typefileplot)
    write_comparison_metrics("fixed", results, arguments)


DISPLAY_TAG = None


def save_comparison_figure(figure, name, typefileplot):
    suffix = f"_{DISPLAY_TAG}" if DISPLAY_TAG else ""
    target = EXAMPLE_PATH / "visuals" / f"{name}{suffix}.{typefileplot}"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite comparison figure {target}")
    save_example_figure(figure, EXAMPLE_PATH, name + suffix, typefileplot)


def write_comparison_metrics(part, results, arguments):
    def portable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return portable(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [portable(item) for item in value]
        return value

    record = {"part": part, "time_tag": DISPLAY_TAG, "settings": vars(arguments),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "assumptions": "Seeded simulated radial velocities; bounded short runs demonstrate integration rather than sampler superiority.",
              "cost_scope": "PCAT includes native setup and persistence; other timers cover sampler execution. MCMC and Kish ESS are different estimators.",
              "nested_sampling_call_cap_used": arguments.maxcall is not None,
              "results": portable(results)}
    path = EXAMPLE_PATH / "visuals" / f"sampler_comparison_{part}_{DISPLAY_TAG}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {path}...")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
