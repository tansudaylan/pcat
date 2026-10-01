#!/usr/bin/env python3
"""Catalog a variable number of stellar flares in a simulated photometric time series with PCAT.

Data: simulated (not real observations). A quiescent star is observed at a TESS-like 2 minute
cadence for 1 day. A random number of flares (3 to 6, drawn uniformly) is injected using the
fast-rise, exponential-decay (FRED) template from nicomedia.retr_lcurmodl_flarsing, with peak
times, amplitudes, and rise/decay time scales drawn from simple illustrative distributions (not
fit to any specific stellar sample). Photon counts are then drawn from a Poisson distribution
around the expected count rate, matching PCAT's own likelihood.

PCAT fits a transdimensional catalog of FRED-profile bursts on top of a fixed, flat quiescent
baseline by reusing its 1D element machinery (typeexpr='fire'). Time plays the role of the energy
axis, and each flare has a peak amplitude plus independent rise and decay time scales.
"""

from tdpy.verbosity import print

import argparse
from tdpy.cli import add_plot_arguments
import shutil
from pathlib import Path

import astropy.io.fits
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tdpy
from nicomedia import retr_lcurmodl_flarsing
from pcat.time_series import evaluate_flare_profile, retr_dictpropelemtmpl

EXAMPLE_PATH = Path(__file__).resolve().parent
RUN_NAME = EXAMPLE_PATH.name
PATH_INPT = EXAMPLE_PATH / "data" / "inpt"

DURATION_DAYS = 1.0  # [day]
CADENCE_MINUTES = 2.0  # [minute], TESS-like short cadence
TIME_OFFSET_DAYS = 1.0  # [day], keeps the time axis away from zero for PCAT's geometric plot binning
BASELINE_COUNT_RATE = 5000.0  # [counts per bin], arbitrary but plausible quiescent level
MINM_NUMB_FLAR, MAXM_NUMB_FLAR = 3, 6
MINM_AMPL_FLAR, MAXM_AMPL_FLAR = 0.30, 0.38  # [relative flux above baseline]
SLOP_AMPL_FLAR = 2.0  # power-law index of the illustrative flare amplitude distribution
MINM_SCAL_RISE, MAXM_SCAL_RISE = 3.0, 10.0  # [minute]
MINM_RATI_FALL_RISE, MAXM_RATI_FALL_RISE = 3.0, 8.0  # fall/rise time-scale ratio
LIMITS = {
    # a floor above half the brightest flare keeps one flare from being fit as two stacked halves
    "flux": (1300.0, 1.0e4),  # [counts per bin], peak excess
    # the simulated rise and fall time-scale ranges, so births propose plausible flare shapes
    "scalrise": (MINM_SCAL_RISE / 1440.0, MAXM_SCAL_RISE / 1440.0),  # [day]
    "scalfall": (MINM_RATI_FALL_RISE * MINM_SCAL_RISE / 1440.0,
                 MAXM_RATI_FALL_RISE * MAXM_SCAL_RISE / 1440.0),  # [day]
}
NUMBER_CHAINS = 12


def simulate_flare_catalog(rng):
    """Draw a simulated ground-truth catalog of flares; return times, amplitudes, and time scales [day]."""
    numbflar = rng.integers(MINM_NUMB_FLAR, MAXM_NUMB_FLAR + 1)
    timeflar = TIME_OFFSET_DAYS + np.sort(rng.uniform(0.05 * DURATION_DAYS, 0.95 * DURATION_DAYS, numbflar))
    amplflar = tdpy.icdf_powr(rng.random(numbflar), MINM_AMPL_FLAR, MAXM_AMPL_FLAR, SLOP_AMPL_FLAR)
    scalrise = rng.uniform(MINM_SCAL_RISE, MAXM_SCAL_RISE, numbflar) / (24.0 * 60.0)  # [day]
    scalfall = scalrise * rng.uniform(MINM_RATI_FALL_RISE, MAXM_RATI_FALL_RISE, numbflar)  # [day]
    return {"timeflar": timeflar, "amplflar": amplflar, "scalrise": scalrise, "scalfall": scalfall}


def simulate_light_curve(catalog, rng):
    """Return time bin edges [day], observed counts, and the expected (noise-free) count rate."""
    numbtime = int(round(DURATION_DAYS * 24.0 * 60.0 / CADENCE_MINUTES))
    edges = TIME_OFFSET_DAYS + np.linspace(0.0, DURATION_DAYS, numbtime + 1)
    meantime = 0.5 * (edges[1:] + edges[:-1])

    relflux = np.ones_like(meantime)
    for timeflar, amplflar, scalrise, scalfall in zip(
        catalog["timeflar"], catalog["amplflar"], catalog["scalrise"], catalog["scalfall"]
    ):
        relflux += retr_lcurmodl_flarsing(meantime, timeflar, amplflar, scalrise, scalfall)

    expcnts = relflux * BASELINE_COUNT_RATE
    obsvcnts = rng.poisson(expcnts).astype(float)
    return edges, meantime, obsvcnts, expcnts


def write_pcat_inputs(edges, obsvcnts):
    """Write the observed counts and the constant exposure PCAT needs to recover them exactly."""
    width = np.diff(edges)  # [day]
    exposure = 1.0 / width  # so that counts = sbrt * expo * width recovers obsvcnts exactly

    PATH_INPT.mkdir(parents=True, exist_ok=True)
    for name, array in [("sbrt.fits", obsvcnts[:, None, None, None]), ("expo.fits", exposure[:, None, None])]:
        path = PATH_INPT / name
        print(f"Writing to {path}...")
        astropy.io.fits.writeto(path, array, overwrite=True)
    template = np.full((edges.size - 1, 1, 1), BASELINE_COUNT_RATE)
    return template


def build_birth_proposal(edges, observed_counts):
    """Return a matched-filter flare birth density on PCAT's log-uniform peak-time grid."""
    meantime = 0.5 * (edges[1:] + edges[:-1])  # [day]
    minimum, maximum = edges[0], edges[-1]  # [day]
    peak_times = minimum * (maximum / minimum) ** ((np.arange(1000) + 0.5) / 1000)  # [day]
    ones = np.ones(peak_times.size)
    templates = evaluate_flare_profile(
        meantime, "flarfred", ones, peak_times, rise_time=6.0 / 1440.0 * ones,
        decay_time=30.0 / 1440.0 * ones,
    ).T
    return retr_dictpropelemtmpl(
        observed_counts - BASELINE_COUNT_RATE, np.maximum(observed_counts, 1.0), templates,
        *LIMITS["flux"], index_position=1, index_flux=0, number_parameters=4,
    )


def run_pcat(edges, template, numbswep, proposal):
    from pcat import sampling

    path_cache = EXAMPLE_PATH / "data" / "outp" / RUN_NAME
    if path_cache.exists():
        print(f"Removing cached PCAT state {path_cache}...")
        shutil.rmtree(path_cache)
    return sampling.sample(
        typeexpr="fire",
        typedata="inpt",
        strgexprsbrt="sbrt.fits",
        typeexpo="file",
        strgexpo="expo.fits",
        binsenerfull=edges,
        spectype=["flarfred"],
        spatdisttype=["line"],
        typeelem=["lghtlinevoig"],
        dictfitt={"typeelem": ["lghtlinevoig"], "spectype": ["flarfred"], "sbrtbacknorm": [template],
                  "listnamediff": ["back0000"]},
        # peak excess counts [counts per bin] and rise/fall time scales [day]
        limtparaelem=LIMITS,
        maxmgangdata=100.0 / (3600.0 * 180.0 / np.pi),  # [rad], unused for non-spatial data
        anlytype="spec",
        fittminmnumbelempop0=0,
        fittmaxmnumbelempop0=10,
        inittype="rand",
        typeseed=0,
        numbproc=NUMBER_CHAINS,
        probtran=0.5,
        probspmr=0.3,
        probjump=0.3,
        radispmr=0.01,  # [day]
        stdvpropelemfire=[0.01, 1.0e-3, 0.02, 0.02],
        booladaptstdp=True,
        boolburntmpr=True,
        factburntmpr=0.9,
        numbswep=numbswep,
        numbburn=3 * numbswep // 4,
        numbsamp=min(max(numbswep // 40, 16), 2000),
        numbswepplot=max(numbswep // 20, 1),
        **proposal,
        makeanim=True,
        boolmakeplotinit=False,
        booldiag=False,
        typeverb=0,
        pathbase=str(EXAMPLE_PATH),
        strgcnfg=RUN_NAME,
    )


def read_posterior():
    """Return PCAT's final posterior state."""
    from pcat.main import readfile

    return readfile(str(EXAMPLE_PATH / "data" / "outp" / RUN_NAME / "gdatfinlpost"))


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
    return colrfore


def save(figure, name, typefileplot):
    path = EXAMPLE_PATH / "visuals" / f"{name}.{typefileplot}"
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {path}...")
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_light_curve_fit(meantime, obsvcnts, catalog, posterior, typefileplot, colrfore):
    """Simulated data, injected ground-truth flares, and PCAT's posterior model versus time."""
    modl = posterior.listpostcntpmodl[:, :, 0, 0]
    quantiles = np.percentile(modl, [16.0, 50.0, 84.0], axis=0)
    figure, axes = plt.subplots(2, 1, figsize=(7.0, 4.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].step((meantime - TIME_OFFSET_DAYS) * 24.0, obsvcnts, where="mid", color=colrfore, lw=0.6, label="Simulated counts")
    for timeflar in catalog["timeflar"]:
        axes[0].axvline((timeflar - TIME_OFFSET_DAYS) * 24.0, color="C3", ls=":", lw=0.8)
    axes[0].fill_between((meantime - TIME_OFFSET_DAYS) * 24.0, quantiles[0], quantiles[2], color="C0", alpha=0.4, lw=0)
    axes[0].plot((meantime - TIME_OFFSET_DAYS) * 24.0, quantiles[1], color="C0", lw=1.0, label="PCAT posterior median model")
    axes[0].axhline(BASELINE_COUNT_RATE, color="C2", ls="--", lw=1.0, label="Quiescent baseline")
    axes[0].set_ylabel("Counts per 2 min bin")
    axes[0].legend(loc="upper right", fontsize=8)
    sigma = np.sqrt(np.clip(obsvcnts, 1.0, None))
    axes[1].step((meantime - TIME_OFFSET_DAYS) * 24.0, (obsvcnts - quantiles[1]) / sigma, where="mid", color=colrfore, lw=0.6)
    axes[1].axhline(0.0, color="0.5", lw=0.8)
    axes[1].set_ylabel(r"Residual [$\sigma$]")
    axes[1].set_xlabel("Time [hour]")
    figure.subplots_adjust(hspace=0.05)
    save(figure, "variable_number_stellar_flares_light_curve_fit", typefileplot)


def render_flare_posterior_frames(meantime, observed_counts, posterior):
    """Render simulated photometry and changing PCAT flare models on fixed axes."""
    models = np.asarray(posterior.listpostcntpmodl, dtype=float)[:, :, 0, 0]
    indices = np.linspace(0, len(models) - 1, min(12, len(models)), dtype=int)
    lower = 0.85 * min(BASELINE_COUNT_RATE, observed_counts.min(), models[indices].min())  # [counts per bin]
    upper = 1.08 * max(observed_counts.max(), models[indices].max())  # [counts per bin]
    visual_root = EXAMPLE_PATH / "visuals"
    visual_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in indices:
        figure, axis = plt.subplots(figsize=(6.0, 4.3), facecolor="white")
        hours = (meantime - TIME_OFFSET_DAYS) * 24.0  # [hour]
        axis.plot(hours, observed_counts, color="#555D61", lw=0.7, alpha=0.75,
                  label="Simulated 2-min photometry")
        axis.plot(hours, models[index], color="#B04435", lw=1.6,
                  label="PCAT flare-catalog model")
        axis.set(xlim=(0, 24), ylim=(lower, upper), xlabel="Time [hour]",
                 ylabel="Counts per 2 min bin", title="Simulated stellar-flare time series")
        axis.grid(False)
        axis.legend(loc="upper right", framealpha=1, facecolor="white")
        path = visual_root / f"flare_photometry_swep{index:09d}.png"
        print(f"Writing to {path}...")
        figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        paths.append(path)
    return paths


def plot_flare_catalog_samples(catalog, posterior, typefileplot, colrfore):
    """Posterior samples of flare time and peak excess counts, with the injected flares marked."""
    listelin = np.concatenate([np.asarray(sample[0]["elin"]) for sample in posterior.listpostdictelem])  # [day]
    listflux = np.concatenate([np.asarray(sample[0]["flux"]) for sample in posterior.listpostdictelem])  # [counts day]
    figure, axis = plt.subplots(figsize=(7.0, 3.4))
    axis.scatter((listelin - TIME_OFFSET_DAYS) * 24.0, listflux, s=4, alpha=0.25, color="C0", lw=0,
                label=f"Posterior flare samples ({len(posterior.listpostdictelem)} catalogs)")
    trueflux = catalog["amplflar"] * BASELINE_COUNT_RATE
    axis.scatter((catalog["timeflar"] - TIME_OFFSET_DAYS) * 24.0, trueflux, marker="*", s=120, color="C3", zorder=5,
                label="Injected flares (approximate)")
    axis.set_yscale("log")
    axis.set_xlabel("Time [hour]")
    axis.set_ylabel("Peak excess [counts per bin]")
    axis.legend(loc="upper left", fontsize=8)
    save(figure, "variable_number_stellar_flares_catalog_samples", typefileplot)


def plot_flare_count_posterior(catalog, posterior, typefileplot, colrfore):
    """Posterior probability of the number of flares, compared to the injected count."""
    numbelem = np.asarray(posterior.listpostnumbelem).ravel().astype(int)
    values, counts = np.unique(numbelem, return_counts=True)
    figure, axis = plt.subplots(figsize=(3.4, 2.6))
    axis.bar(values, counts / counts.sum(), color="C0")
    axis.axvline(catalog["timeflar"].size, color="C3", ls="--", lw=1.2, label="Injected count")
    axis.set_xlabel("Number of flares")
    axis.set_ylabel("Posterior probability")
    axis.legend(loc="upper right", fontsize=8)
    save(figure, "variable_number_stellar_flares_count_posterior", typefileplot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbswep", type=int, default=100_000)
    parser.add_argument("--smoke", action="store_true", help="Run a short pipeline check.")
    add_plot_arguments(parser)
    parser.add_argument("--typeplotback", choices=("white", "dark"), default="white")
    parser.add_argument("--skip-sampling", action="store_true", help="Replot an existing chain.")
    arguments = parser.parse_args()
    numbswep = 4000 if arguments.smoke else arguments.numbswep

    rng = np.random.default_rng(0)
    catalog = simulate_flare_catalog(rng)
    edges, meantime, obsvcnts, expcnts = simulate_light_curve(catalog, rng)
    template = write_pcat_inputs(edges, obsvcnts)
    if not arguments.skip_sampling:
        run_pcat(edges, template, numbswep, build_birth_proposal(edges, obsvcnts))
    posterior = read_posterior()
    colrfore = configure_style(arguments.typeplotback)
    render_flare_posterior_frames(meantime, obsvcnts, posterior)
    plot_light_curve_fit(meantime, obsvcnts, catalog, posterior, arguments.typefileplot, colrfore)
    plot_flare_catalog_samples(catalog, posterior, arguments.typefileplot, colrfore)
    plot_flare_count_posterior(catalog, posterior, arguments.typefileplot, colrfore)


if __name__ == "__main__":
    main()
