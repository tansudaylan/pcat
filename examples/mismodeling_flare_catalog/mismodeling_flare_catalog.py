#!/usr/bin/env python3
"""Show what mismodeling does to a PCAT catalog, using simulated stellar-flare photometry.

Data: simulated (not real observations). A star is observed at a 2 minute cadence for one day,
and four flares with a fast rise and exponential decay (FRED) are injected. Light curve A has a
constant quiescent level. Light curve B adds a 2% rotational modulation of that level.

Each light curve is fit twice with identical priors and sampler settings, once with the model
that generated it and once with a misspecified forward model.

- Profile mismatch: light curve A is fit with symmetric Gaussian flares instead of FRED flares.
- Baseline mismatch: light curve B is fit with a constant quiescent level instead of the modulated one.

Compare the recovered catalogs and residuals rather than assuming that a small chi-squared
identifies the correct flare population. Short runs exercise the pipeline and do not establish
convergence or a particular catalog-size bias.
"""

from __future__ import annotations

from tdpy.verbosity import print

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tdpy.cli import add_plot_arguments, add_smoke_argument

from pcat.demo import write_time_series_fits
from pcat.main import readfile, retr_pathrun
from pcat.time_series import evaluate_flare_profile

EXAMPLE_PATH = Path(__file__).resolve().parent
VISUAL_PATH = EXAMPLE_PATH / "visuals"
TIME_OFFSET = 1.0  # [day], keeps the time axis away from zero for PCAT's geometric binning
DURATION = 1.0  # [day]
CADENCE = 2.0 / 1440.0  # [day]
BASELINE = 5000.0  # [counts per bin], mean quiescent level
MODULATION_AMPLITUDE = 0.02  # fractional rotational modulation in light curve B
MODULATION_PERIOD = 0.45  # [day]
INJECTED = {
    "elin": TIME_OFFSET + np.array([0.15, 0.38, 0.62, 0.84]),  # [day], peak times
    "flux": np.array([2200.0, 1500.0, 2600.0, 1800.0]),  # [counts per bin], peak excess
    "scalrise": np.array([5.0, 4.0, 7.0, 6.0]) / 1440.0,  # [day]
    "scalfall": np.array([40.0, 25.0, 55.0, 30.0]) / 1440.0,  # [day]
}
# identical priors for every fit, wide enough that extra or merged components are allowed
LIMITS = {
    "flux": (200.0, 1.0e4),  # [counts per bin]
    "scalrise": (2.0 / 1440.0, 15.0 / 1440.0),  # [day]
    "scalfall": (6.0 / 1440.0, 90.0 / 1440.0),  # [day]
    "fwhm": (2.0 / 1440.0, 120.0 / 1440.0),  # [day]
}
# (run name, light curve, fitted flare profile, fitted baseline, legend label)
FITS = (
    ("fit_profile_correct", "A", "flarfred", "constant", "FRED flares (correct)"),
    ("fit_profile_wrong", "A", "flargauss", "constant", "Gaussian flares (misspecified)"),
    ("fit_baseline_correct", "B", "flarfred", "modulated", "Modulated baseline (correct)"),
    ("fit_baseline_wrong", "B", "flarfred", "constant", "Constant baseline (misspecified)"),
)
NUMBER_CHAINS = 4
NUMBER_PREDICTIVE_REPLICATES = 1000
# prior upper limit on the number of flares, high enough that misspecified fits do not pile up against it
MAXIMUM_NUMBER = 24


def simulate_light_curves(seed: int = 3) -> dict:
    """Return time bin edges and centers [day], the two quiescent levels, and Poisson counts."""
    edges = TIME_OFFSET + np.arange(int(round(DURATION / CADENCE)) + 1) * CADENCE  # [day]
    time = 0.5 * (edges[1:] + edges[:-1])  # [day]
    flares = evaluate_flare_profile(time, "flarfred", INJECTED["flux"], INJECTED["elin"],
                                    rise_time=INJECTED["scalrise"], decay_time=INJECTED["scalfall"]).sum(axis=1)
    baselines = {
        "constant": np.full(time.size, BASELINE),
        "modulated": BASELINE * (1.0 + MODULATION_AMPLITUDE * np.sin(2.0 * np.pi * (time - TIME_OFFSET) / MODULATION_PERIOD)),
    }
    random = np.random.default_rng(seed)
    counts = {"A": random.poisson(baselines["constant"] + flares).astype(float),
              "B": random.poisson(baselines["modulated"] + flares).astype(float)}  # [counts per bin]
    return {"edges": edges, "time": time, "baselines": baselines, "counts": counts}


def run_fit(light_curves: dict, run_name: str, light_curve: str, profile: str, baseline: str,
            numbswep: int, *, number_chains=None) -> object:
    """Fit one light curve with one forward model and return PCAT's final posterior state."""
    from pcat import sampling

    run_root = Path(retr_pathrun(EXAMPLE_PATH, run_name))
    cached_state = run_root / "data" / "outp" / run_name
    legacy_state = EXAMPLE_PATH / "data" / "outp" / run_name
    if cached_state.parent.exists() or legacy_state.exists():
        raise FileExistsError(f"Refusing to overwrite cached PCAT run {run_name}")
    input_path = run_root / "data" / "inpt"
    baseline_template = write_time_series_fits(
        input_path, light_curves["edges"], light_curves["counts"][light_curve],
        light_curves["baselines"][baseline],
    )
    shape_names = ("scalrise", "scalfall") if profile == "flarfred" else ("fwhm",)
    sampling.sample(
        typeexpr="fire", typedata="inpt", strgexprsbrt="sbrt.fits", typeexpo="file", strgexpo="expo.fits",
        binsenerfull=light_curves["edges"], spectype=[profile], spatdisttype=["line"], typeelem=["lghtlinevoig"],
        # the quiescent level is a fixed template, so a wrong template cannot be absorbed by a normalization
        dictfitt={"typeelem": ["lghtlinevoig"], "spectype": [profile], "listnamediff": ["back0000"],
                  "sbrtbacknorm": [baseline_template]},
        limtparaelem={name: LIMITS[name] for name in ("flux", *shape_names)},
        maxmgangdata=100.0 / (3600.0 * 180.0 / np.pi),  # [rad], unused for non-spatial data
        anlytype="spec", fittminmnumbelempop0=0, fittmaxmnumbelempop0=MAXIMUM_NUMBER,
        inittype="rand", typeseed=0, numbproc=NUMBER_CHAINS if number_chains is None else number_chains,
        probtran=0.5, probspmr=0.3, radispmr=0.01,  # [day]
        stdvpropelemfire=[0.01, 1.0e-3] + [0.02] * len(shape_names),
        booladaptstdp=True, boolburntmpr=True, factburntmpr=0.9,
        numbswep=numbswep, numbburn=numbswep // 2, numbsamp=min(numbswep // 20, 1000),
        boolmakeplot=False, boolmakeplotinit=False, makeanim=False, booldiag=False, typeverb=0,
        pathbase=str(EXAMPLE_PATH), strgcnfg=run_name,
    )
    return readfile(str(cached_state / "gdatfinlpost"))


def calculate_residual_diagnostics(counts, posterior_models, rng, number_replicates):
    """Compare Pearson residual and total chi-squared shapes with Gaussian references."""
    if np.ma.is_masked(counts) or np.ma.is_masked(posterior_models):
        raise ValueError("counts and posterior models must contain measured values")
    counts = np.asarray(counts, dtype=float)
    posterior_models = np.asarray(posterior_models, dtype=float)
    if counts.ndim != 1 or counts.size == 0 or posterior_models.ndim != 2 or posterior_models.shape[1] != counts.size:
        raise ValueError("counts and posterior models must have matching one-dimensional bins")
    if (posterior_models.shape[0] == 0 or not np.isfinite(counts).all()
            or np.any(counts < 0.0) or not np.isfinite(posterior_models).all() or np.any(posterior_models < 0.0)):
        raise ValueError("counts and posterior model draws must be finite and nonnegative")
    if number_replicates < 2:
        raise ValueError("at least two predictive replicates are required")

    model = np.median(posterior_models, axis=0)
    variance = np.maximum(model, 1.0)
    residual = (counts - model) / np.sqrt(variance)
    chi2_total = float(np.sum(residual**2))
    index = rng.integers(posterior_models.shape[0], size=number_replicates)
    predictive_model = posterior_models[index]
    predictive_counts = rng.poisson(predictive_model)
    chi2_predictive = np.sum(
        (predictive_counts - model[None, :])**2 / variance[None, :], axis=1
    )
    return {
        "model": model,
        "residual": residual,
        "chi2_total": chi2_total,
        "chi2_per_bin": chi2_total / counts.size,
        "chi2_predictive": chi2_predictive,
        "chi2_upper_tail_fraction": float(
            (1 + np.count_nonzero(chi2_predictive >= chi2_total))
            / (number_replicates + 1)
        ),
    }


def summarize(light_curves: dict, posteriors: dict) -> dict:
    """Return catalog-size and residual-shape statistics for every fit."""
    summary = {}
    rng = np.random.default_rng(0)
    for run_name, light_curve, *_ in FITS:
        posterior = posteriors[run_name]
        counts = light_curves["counts"][light_curve]
        models = np.asarray(posterior.listpostcntpmodl)[:, :, 0, 0]  # [counts per bin]
        diagnostics = calculate_residual_diagnostics(
            counts, models, rng, NUMBER_PREDICTIVE_REPLICATES
        )
        number = np.asarray(posterior.listpostnumbelem).ravel()
        summary[run_name] = {
            "mean_number": float(number.mean()),
            "probability_injected_number": float(np.mean(number == INJECTED["elin"].size)),
            **diagnostics,
        }
    return summary


def gaussian_density(values, mean, standard_deviation):
    """Evaluate a Gaussian probability density on ``values``."""
    standard_deviation = max(float(standard_deviation), 1e-12)
    standardized = (values - mean) / standard_deviation
    return np.exp(-0.5 * standardized**2) / (standard_deviation * np.sqrt(2.0 * np.pi))


def plot_residual_chi2_shapes(summary: dict, typefileplot: str, time_tag=None) -> Path:
    """Compare residual histograms and predictive total chi-squared with Gaussians."""
    from matplotlib.ticker import MaxNLocator

    figure, axes = plt.subplots(len(FITS), 2, figsize=(8.2, 9.0))
    for row, (run_name, _, _, _, label) in enumerate(FITS):
        result = summary[run_name]
        residual = result["residual"]
        resid_mean = float(np.mean(residual))
        resid_stdv = float(np.std(residual, ddof=1))
        resid_grid = np.linspace(float(residual.min()), float(residual.max()), 300)
        axis = axes[row, 0]
        axis.hist(residual, bins=30, density=True, color="0.25", alpha=0.55,
                  label="Observed Pearson residuals")
        axis.plot(resid_grid, gaussian_density(resid_grid, 0.0, 1.0), color="black",
                  ls="--", label="Standard Gaussian")
        axis.plot(resid_grid, gaussian_density(resid_grid, resid_mean, resid_stdv),
                  color="#A51C30", label="Gaussian fit")
        axis.set_title(f"{label}\nmean={resid_mean:.2f}, sd={resid_stdv:.2f}", loc="left", fontsize=9)
        axis.set_xlabel("Pearson residual")
        axis.set_ylabel("Density")

        chi2_predictive = result["chi2_predictive"]
        chi2_mean = float(np.mean(chi2_predictive))
        chi2_stdv = float(np.std(chi2_predictive, ddof=1))
        chi2_total = result["chi2_total"]
        chi2_grid = np.linspace(0.0, max(float(chi2_predictive.max()), chi2_total) * 1.05, 300)
        axis = axes[row, 1]
        axis.hist(chi2_predictive, bins=30, density=True, color="0.25", alpha=0.55,
                  label="Posterior-predictive $\\chi^2$")
        axis.plot(chi2_grid, gaussian_density(chi2_grid, chi2_mean, chi2_stdv),
                  color="#A51C30", ls="--", label="Gaussian approximation")
        axis.axvline(chi2_total, color="black", lw=1.2,
                     label="Observed total $\\chi^2$")
        axis.set_xlabel(r"Total Pearson $\chi^2$")
        axis.set_ylabel("Density")
        axis.set_title(f"Upper-tail fraction={result['chi2_upper_tail_fraction']:.3f}", loc="left", fontsize=9)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4), useMathText=True)

    handles = []
    labels = []
    for axis in axes[0]:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        handles.extend(axis_handles)
        labels.extend(axis_labels)
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
                  ncol=3, fontsize=9, fancybox=True, framealpha=1.0)
    for axis in axes.flat:
        axis.grid(False)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return save(figure, "mismodeling_residual_chi2_shapes", typefileplot, time_tag)


def plot_light_curve_fits(light_curves: dict, summary: dict, light_curve: str, typefileplot: str, time_tag=None) -> Path:
    """Data, correct and misspecified median models, and both standardized residuals for one light curve."""
    hours = (light_curves["time"] - TIME_OFFSET) * 24.0  # [hour]
    counts = light_curves["counts"][light_curve]
    fits = [row for row in FITS if row[1] == light_curve]
    figure, axes = plt.subplots(3, 1, figsize=(7.0, 5.6), sharex=True, gridspec_kw={"height_ratios": [2.4, 1, 1]})
    axes[0].plot(hours, counts, color="black", marker=".", markersize=2, linestyle="none", label="Simulated counts")
    if light_curve == "B":
        axes[0].plot(hours, light_curves["baselines"]["modulated"], color="black", ls=":", lw=1.0,
                     label="Injected quiescent level")
    for (run_name, _, _, _, label), color, axis in zip(fits, ("#007360", "#A51C30"), axes[1:]):
        model = summary[run_name]["model"]
        axes[0].plot(hours, model, color=color, lw=1.2,
                     label=f"{label}, {summary[run_name]['mean_number']:.1f} flares on average")
        residual = (counts - model) / np.sqrt(np.maximum(model, 1.0))
        axis.plot(hours, residual, color="black",
              marker=".", markersize=2, linestyle="none")
        axis.axhline(0.0, color="black", lw=0.6)
        residual_limit = max(5.0, 1.1 * float(np.max(np.abs(residual))))
        axis.set_ylim(-residual_limit, residual_limit)
        axis.set_ylabel(r"Residual [$\sigma$]")
        axis.set_title(f"{label}, $\\chi^2$ per bin {summary[run_name]['chi2_per_bin']:.2f}",
                   loc="left", fontsize=9)
    for elin in INJECTED["elin"]:
        axes[0].axvline((elin - TIME_OFFSET) * 24.0, color="black", ls="--", lw=0.5)
    # headroom above the brightest flare keeps the legend clear of the data
    span = float(np.ptp(counts))  # [counts per bin]
    axes[0].set_ylim(counts.min() - 0.05 * span, counts.max() + 0.45 * span)
    axes[0].set_ylabel("Counts per 2 min bin")
    axes[0].legend(loc="upper right", fontsize=8, fancybox=True, framealpha=1.0)
    axes[-1].set_xlabel("Time [hour]")
    axes[-1].set_xlim(0.0, 24.0)
    for axis in axes:
        axis.grid(False)
    figure.subplots_adjust(hspace=0.3)
    name = "profile" if light_curve == "A" else "baseline"
    return save(figure, f"mismodeling_{name}_light_curve", typefileplot, time_tag)


def plot_catalogs(light_curves: dict, posteriors: dict, typefileplot: str, time_tag=None) -> Path:
    """Posterior flare peak times and excesses of all four fits against the injected catalog."""
    figure, axes = plt.subplots(len(FITS), 1, figsize=(7.0, 6.4), sharex=True, sharey=True)
    for axis, (run_name, _, _, _, label) in zip(axes, FITS):
        posterior = posteriors[run_name]
        elin = np.concatenate([np.asarray(sample[0]["elin"]) for sample in posterior.listpostdictelem])  # [day]
        flux = np.concatenate([np.asarray(sample[0]["flux"]) for sample in posterior.listpostdictelem])
        color = "#A51C30" if "misspecified" in label else "#007360"
        axis.scatter((elin - TIME_OFFSET) * 24.0, flux, s=4, lw=0, alpha=0.3, color=color)
        axis.scatter((INJECTED["elin"] - TIME_OFFSET) * 24.0, INJECTED["flux"], marker="*", s=110,
                     facecolor="none", edgecolor="black", lw=1.0, label="Injected flares")
        # the label sits above the brightest injected flare, between flares in time
        axis.text(0.5, 0.94, f"{label}, {len(posterior.listpostdictelem)} posterior catalogs",
                  transform=axis.transAxes, ha="center", va="top", fontsize=8, color=color,
                  bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": color})
        axis.set_yscale("log")
        axis.set_ylim(150.0, 1.5e4)  # [counts per bin]
        axis.set_ylabel("Peak excess\n[counts per bin]")
        axis.grid(False)
    axes[0].legend(loc="upper left", fontsize=8, fancybox=True, framealpha=1.0)
    axes[-1].set_xlabel("Time [hour]")
    axes[-1].set_xlim(0.0, 24.0)
    figure.subplots_adjust(hspace=0.08)
    return save(figure, "mismodeling_catalog_samples", typefileplot, time_tag)


def plot_catalog_size(posteriors: dict, typefileplot: str, time_tag=None) -> Path:
    """Posterior probability of the number of flares for all four fits."""
    figure, axis = plt.subplots(figsize=(7.0, 3.0))
    width = 0.2
    numbers = np.arange(MAXIMUM_NUMBER + 1)
    for offset, (run_name, _, _, _, label), color, hatch in zip(
        (-1.5, -0.5, 0.5, 1.5), FITS, ("#007360", "#A51C30", "#4D9A8A", "#D47B87"), ("", "", "//", "//")
    ):
        number = np.asarray(posteriors[run_name].listpostnumbelem).ravel().astype(int)
        probability = np.bincount(number, minlength=numbers.size)[:numbers.size] / number.size
        axis.bar(numbers + offset * width, probability, width=width, color=color, hatch=hatch,
                 edgecolor="white", label=label)
    axis.axvline(INJECTED["elin"].size, color="black", ls="--", lw=1.0, label="Injected number")
    axis.set_xticks(numbers[::2])
    axis.set_xlabel("Number of flares")
    axis.set_ylabel("Posterior probability")
    axis.legend(loc="upper right", fontsize=8, fancybox=True, framealpha=1.0)
    axis.grid(False)
    return save(figure, "mismodeling_catalog_size", typefileplot, time_tag)


def save(figure, name: str, typefileplot: str, time_tag=None) -> Path:
    suffix = f"_{time_tag}" if time_tag else ""
    path = VISUAL_PATH / f"{name}{suffix}.{typefileplot}"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite misspecification figure {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {path}...")
    figure.savefig(path, dpi=300 if typefileplot == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def run_example(numbswep: int = 40000, typefileplot: str = "png", *, time_tag=None, number_chains=None) -> dict:
    """Run all four fits and write the comparison figures."""
    time_tag = time_tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if datetime.strptime(time_tag, "%Y%m%d_%H%M%S").strftime("%Y%m%d_%H%M%S") != time_tag:
        raise ValueError("time_tag must use YYYYMMDD_HHMMSS")
    if numbswep < 40 or typefileplot not in {"png", "pdf"}:
        raise ValueError("At least 40 sweeps and a PNG or PDF output are required")
    number_chains = NUMBER_CHAINS if number_chains is None else number_chains
    if number_chains < 1:
        raise ValueError("number_chains must be positive")
    for run_name, *_ in FITS:
        run_root = Path(retr_pathrun(EXAMPLE_PATH, f"{run_name}_{time_tag}"))
        if (run_root / "data" / "outp").exists():
            raise FileExistsError(f"Refusing to overwrite cached PCAT run {run_root}")
    if list(VISUAL_PATH.glob(f"mismodeling_*_{time_tag}.*")):
        raise FileExistsError("Refusing to overwrite a previous misspecification output set")
    plt.rcParams.update({"font.size": 9, "text.usetex": False, "axes.grid": False})
    light_curves = simulate_light_curves()
    posteriors = {run_name: run_fit(light_curves, f"{run_name}_{time_tag}", light_curve, profile, baseline, numbswep,
                                  number_chains=number_chains)
                  for run_name, light_curve, profile, baseline, _ in FITS}
    summary = summarize(light_curves, posteriors)
    summary["figure_paths"] = [
        plot_light_curve_fits(light_curves, summary, "A", typefileplot, time_tag),
        plot_light_curve_fits(light_curves, summary, "B", typefileplot, time_tag),
        plot_catalogs(light_curves, posteriors, typefileplot, time_tag),
        plot_catalog_size(posteriors, typefileplot, time_tag),
        plot_residual_chi2_shapes(summary, typefileplot, time_tag),
    ]
    record = {"time_tag": time_tag, "sweeps_per_chain": numbswep, "chains": number_chains,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "assumptions": "Seeded Poisson counts with four injected FRED flares; short runs verify integration rather than convergence.",
              "fits": {name: {key: value.tolist() if isinstance(value, np.ndarray) else value
                              for key, value in summary[name].items()} for name, *_ in FITS}}
    record_path = VISUAL_PATH / f"mismodeling_summary_{time_tag}.json"
    print(f"Writing to {record_path}...")
    with record_path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbswep", type=int, default=40000)
    add_smoke_argument(parser, help="Run a short pipeline check.")
    parser.add_argument("--time-tag")
    parser.add_argument("--chains", type=int, default=NUMBER_CHAINS)
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    summary = run_example(2000 if arguments.smoke else arguments.numbswep, arguments.typefileplot,
                          time_tag=arguments.time_tag, number_chains=arguments.chains)
    for run_name, *_ , label in FITS:
        result = summary[run_name]
        print(f"{label}: mean number of flares {result['mean_number']:.2f}, "
              f"P(N = {INJECTED['elin'].size}) = {result['probability_injected_number']:.2f}, "
              f"chi-squared per bin {result['chi2_per_bin']:.2f}, "
              f"residual mean {result['residual'].mean():.2f}, "
              f"residual standard deviation {result['residual'].std(ddof=1):.2f}, "
              f"predictive chi-squared upper-tail fraction "
              f"{result['chi2_upper_tail_fraction']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
