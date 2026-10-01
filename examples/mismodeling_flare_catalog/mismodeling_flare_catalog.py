#!/usr/bin/env python3
"""Show what mismodeling does to a PCAT catalog, using simulated stellar-flare photometry.

Data: simulated (not real observations). A star is observed at a 2 minute cadence for one day,
and four flares with a fast rise and exponential decay (FRED) are injected. Light curve A has a
constant quiescent level. Light curve B adds a 2% rotational modulation of that level.

Each light curve is fit twice with identical priors and sampler settings, once with the model
that generated it and once with a misspecified forward model.

- Profile mismatch: light curve A is fit with symmetric Gaussian flares instead of FRED flares.
- Baseline mismatch: light curve B is fit with a constant quiescent level instead of the modulated one.

The correct models recover the four injected flares. The misspecified models converge to confident
and wrong catalogs. Gaussian components tile each asymmetric decay, so about 15 flares fit the
data with a residual indistinguishable from noise. Faint flares absorb the unmodeled variability
of light curve B, which leaves a coherent residual. A good fit therefore does not validate a catalog.
"""

from __future__ import annotations

from tdpy.verbosity import print

import argparse
import shutil
from pathlib import Path

import astropy.io.fits
import matplotlib.pyplot as plt
import numpy as np
from tdpy.cli import add_plot_arguments

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
            numbswep: int) -> object:
    """Fit one light curve with one forward model and return PCAT's final posterior state."""
    from pcat import sampling

    run_root = Path(retr_pathrun(EXAMPLE_PATH, run_name))
    cached_state = run_root / "data" / "outp" / run_name
    if cached_state.exists():
        print(f"Removing cached PCAT state {cached_state}...")
        shutil.rmtree(cached_state)
    # counts = sbrt * expo * width, so unit-per-width exposure makes PCAT read the counts unchanged
    input_path = run_root / "data" / "inpt"
    input_path.mkdir(parents=True, exist_ok=True)
    for name, values in (("sbrt.fits", light_curves["counts"][light_curve][:, None, None, None]),
                         ("expo.fits", (1.0 / np.diff(light_curves["edges"]))[:, None, None])):
        print(f"Writing to {input_path / name}...")
        astropy.io.fits.writeto(input_path / name, values, overwrite=True)
    shape_names = ("scalrise", "scalfall") if profile == "flarfred" else ("fwhm",)
    sampling.sample(
        typeexpr="fire", typedata="inpt", strgexprsbrt="sbrt.fits", typeexpo="file", strgexpo="expo.fits",
        binsenerfull=light_curves["edges"], spectype=[profile], spatdisttype=["line"], typeelem=["lghtlinevoig"],
        # the quiescent level is a fixed template, so a wrong template cannot be absorbed by a normalization
        dictfitt={"typeelem": ["lghtlinevoig"], "spectype": [profile], "listnamediff": ["back0000"],
                  "sbrtbacknorm": [light_curves["baselines"][baseline][:, None, None]]},
        limtparaelem={name: LIMITS[name] for name in ("flux", *shape_names)},
        maxmgangdata=100.0 / (3600.0 * 180.0 / np.pi),  # [rad], unused for non-spatial data
        anlytype="spec", fittminmnumbelempop0=0, fittmaxmnumbelempop0=MAXIMUM_NUMBER,
        inittype="rand", typeseed=0, numbproc=NUMBER_CHAINS,
        probtran=0.5, probspmr=0.3, radispmr=0.01,  # [day]
        stdvpropelemfire=[0.01, 1.0e-3] + [0.02] * len(shape_names),
        booladaptstdp=True, boolburntmpr=True, factburntmpr=0.9,
        numbswep=numbswep, numbburn=numbswep // 2, numbsamp=min(numbswep // 20, 1000),
        boolmakeplot=False, boolmakeplotinit=False, makeanim=False, booldiag=False, typeverb=0,
        pathbase=str(EXAMPLE_PATH), strgcnfg=run_name,
    )
    return readfile(str(cached_state / "gdatfinlpost"))


def summarize(light_curves: dict, posteriors: dict) -> dict:
    """Return the catalog size and residual statistics of every fit."""
    summary = {}
    for run_name, light_curve, *_ in FITS:
        posterior = posteriors[run_name]
        counts = light_curves["counts"][light_curve]
        model = np.median(np.asarray(posterior.listpostcntpmodl)[:, :, 0, 0], axis=0)  # [counts per bin]
        number = np.asarray(posterior.listpostnumbelem).ravel()
        summary[run_name] = {
            "mean_number": float(number.mean()),
            "probability_injected_number": float(np.mean(number == INJECTED["elin"].size)),
            # Poisson chi-squared per bin of the posterior-median model
            "chi2_per_bin": float(np.mean((counts - model) ** 2 / np.maximum(model, 1.0))),
            "model": model,
        }
    return summary


def plot_light_curve_fits(light_curves: dict, summary: dict, light_curve: str, typefileplot: str) -> Path:
    """Data, correct and misspecified median models, and both standardized residuals for one light curve."""
    hours = (light_curves["time"] - TIME_OFFSET) * 24.0  # [hour]
    counts = light_curves["counts"][light_curve]
    fits = [row for row in FITS if row[1] == light_curve]
    figure, axes = plt.subplots(3, 1, figsize=(7.0, 5.6), sharex=True, gridspec_kw={"height_ratios": [2.4, 1, 1]})
    axes[0].plot(hours, counts, color="0.55", lw=0.6, label="Simulated counts")
    if light_curve == "B":
        axes[0].plot(hours, light_curves["baselines"]["modulated"], color="black", ls=":", lw=1.0,
                     label="Injected quiescent level")
    for (run_name, _, _, _, label), color, axis in zip(fits, ("#007360", "#A51C30"), axes[1:]):
        model = summary[run_name]["model"]
        axes[0].plot(hours, model, color=color, lw=1.2,
                     label=f"{label}, {summary[run_name]['mean_number']:.1f} flares on average")
        axis.plot(hours, (counts - model) / np.sqrt(np.maximum(model, 1.0)), color=color, lw=0.6)
        axis.axhline(0.0, color="black", lw=0.6)
        axis.set_ylim(-5.0, 5.0)
        axis.set_ylabel(r"Residual [$\sigma$]")
        axis.text(0.01, 0.92, f"{label}, $\\chi^2$ per bin {summary[run_name]['chi2_per_bin']:.2f}",
                  transform=axis.transAxes, va="top", fontsize=8,
                  bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "none"})
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
    figure.subplots_adjust(hspace=0.06)
    name = "profile" if light_curve == "A" else "baseline"
    return save(figure, f"mismodeling_{name}_light_curve", typefileplot)


def plot_catalogs(light_curves: dict, posteriors: dict, typefileplot: str) -> Path:
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
    return save(figure, "mismodeling_catalog_samples", typefileplot)


def plot_catalog_size(posteriors: dict, typefileplot: str) -> Path:
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
    return save(figure, "mismodeling_catalog_size", typefileplot)


def save(figure, name: str, typefileplot: str) -> Path:
    path = VISUAL_PATH / f"{name}.{typefileplot}"
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {path}...")
    figure.savefig(path, dpi=300 if typefileplot == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def run_example(numbswep: int = 40000, typefileplot: str = "png") -> dict:
    """Run all four fits and write the comparison figures."""
    plt.rcParams.update({"font.size": 9, "text.usetex": False, "axes.grid": False})
    light_curves = simulate_light_curves()
    posteriors = {run_name: run_fit(light_curves, run_name, light_curve, profile, baseline, numbswep)
                  for run_name, light_curve, profile, baseline, _ in FITS}
    summary = summarize(light_curves, posteriors)
    summary["figure_paths"] = [
        plot_light_curve_fits(light_curves, summary, "A", typefileplot),
        plot_light_curve_fits(light_curves, summary, "B", typefileplot),
        plot_catalogs(light_curves, posteriors, typefileplot),
        plot_catalog_size(posteriors, typefileplot),
    ]
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbswep", type=int, default=40000)
    parser.add_argument("--smoke", action="store_true", help="Run a short pipeline check.")
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    summary = run_example(2000 if arguments.smoke else arguments.numbswep, arguments.typefileplot)
    for run_name, *_ , label in FITS:
        result = summary[run_name]
        print(f"{label}: mean number of flares {result['mean_number']:.2f}, "
              f"P(N = {INJECTED['elin'].size}) = {result['probability_injected_number']:.2f}, "
              f"chi-squared per bin {result['chi2_per_bin']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
