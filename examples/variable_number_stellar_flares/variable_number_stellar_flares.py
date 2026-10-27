#!/usr/bin/env python3
"""Catalog a variable number of stellar flares in a simulated photometric time series with PCAT.

Data: simulated (not real observations). A quiescent star is observed at a TESS-like 2 minute
cadence for 1 day. A random number of flares (3 to 6, drawn uniformly) is injected using the
fast-rise, exponential-decay (FRED) template from tdpy.retr_lcurmodl_flarsing, with peak
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
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tdpy
from tdpy import retr_lcurmodl_flarsing
from pcat.demo import (
    configure_example_style,
    load_example_posterior,
    save_example_figure,
    write_time_series_fits,
)
from pcat.time_series import (
    FLARE_PROFILE_PARAMETERS,
    evaluate_flare_profile,
    flare_profile_parameters,
    retr_dictpropelemtmpl,
)
from pcat.plotting import animation_phase_labels, animation_states
from pcat.main import retr_pathrun

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
FLARE_WIDTH_BOUNDS = {
    "flargauss": {"fwhm": (4.0 / 1440.0, 180.0 / 1440.0)},
    "flarexpd": {"scalfall": (5.0 / 1440.0, 240.0 / 1440.0)},
    "flarfred": {"scalrise": LIMITS["scalrise"], "scalfall": LIMITS["scalfall"]},
    "flardav": {"fwhm": (5.0 / 1440.0, 180.0 / 1440.0)},
}
FLARE_PARAMETER_SCALES = {"flux": 0.01, "elin": 1.0e-3, "fwhm": 0.02,
                          "scalrise": 0.02, "scalfall": 0.02}
NUMBER_CHAINS = 12


def flare_profile_configuration(flare_profile):
    """Return bounds and proposal scales for one supported flare profile."""
    parameters = flare_profile_parameters(flare_profile)
    limits = {"flux": LIMITS["flux"], **FLARE_WIDTH_BOUNDS[flare_profile]}
    return limits, [FLARE_PARAMETER_SCALES[name] for name in parameters]


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


def build_birth_proposal(edges, observed_counts, flare_profile="flarfred"):
    """Return a matched-filter flare birth density on PCAT's log-uniform peak-time grid."""
    meantime = 0.5 * (edges[1:] + edges[:-1])  # [day]
    minimum, maximum = edges[0], edges[-1]  # [day]
    peak_times = minimum * (maximum / minimum) ** ((np.arange(1000) + 0.5) / 1000)  # [day]
    ones = np.ones(peak_times.size)
    if flare_profile == "flargauss":
        shape_parameters = {"fwhm": 30.0 / 1440.0 * ones}
    elif flare_profile == "flarexpd":
        shape_parameters = {"decay_time": 30.0 / 1440.0 * ones}
    elif flare_profile == "flarfred":
        shape_parameters = {"rise_time": 6.0 / 1440.0 * ones, "decay_time": 30.0 / 1440.0 * ones}
    elif flare_profile == "flardav":
        shape_parameters = {"fwhm": 30.0 / 1440.0 * ones}
    else:
        flare_profile_parameters(flare_profile)
        raise AssertionError("unreachable")
    templates = evaluate_flare_profile(meantime, flare_profile, ones, peak_times, **shape_parameters).T
    limits, _ = flare_profile_configuration(flare_profile)
    return retr_dictpropelemtmpl(
        observed_counts - BASELINE_COUNT_RATE, np.maximum(observed_counts, 1.0), templates,
        *limits["flux"], index_position=1, index_flux=0,
        number_parameters=len(flare_profile_parameters(flare_profile)),
    )


def _fresh_run_root(run_name):
    root = Path(retr_pathrun(EXAMPLE_PATH, run_name))
    legacy = EXAMPLE_PATH / "data" / "outp" / run_name
    if legacy.exists() or (root / "data" / "outp").exists() or (root / "visuals").exists():
        raise FileExistsError(f"Refusing to overwrite cached flare run {run_name}")
    return root


def run_pcat(edges, template, numbswep, proposal, flare_profile="flarfred", *,
             run_name=None, number_chains=NUMBER_CHAINS):
    from pcat import sampling

    run_name = run_name or (RUN_NAME if flare_profile == "flarfred" else f"{RUN_NAME}_{flare_profile}")
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("run_name must be a single directory name")
    if not isinstance(number_chains, int) or number_chains < 1:
        raise ValueError("number_chains must be a positive integer")
    _fresh_run_root(run_name)
    limits, proposal_scales = flare_profile_configuration(flare_profile)
    return sampling.sample(
        typeexpr="fire",
        typedata="inpt",
        strgexprsbrt="sbrt.fits",
        typeexpo="file",
        strgexpo="expo.fits",
        binsenerfull=edges,
        spectype=[flare_profile],
        spatdisttype=["line"],
        typeelem=["lghtlinevoig"],
        dictfitt={"typeelem": ["lghtlinevoig"], "spectype": [flare_profile], "sbrtbacknorm": [template],
                  "listnamediff": ["back0000"]},
        # peak excess counts [counts per bin] and rise/fall time scales [day]
        limtparaelem=limits,
        maxmgangdata=100.0 / (3600.0 * 180.0 / np.pi),  # [rad], unused for non-spatial data
        anlytype="spec",
        fittminmnumbelempop0=0,
        fittmaxmnumbelempop0=10,
        inittype="rand",
        typeseed=0,
        numbframanim=24,
        numbproc=number_chains,
        probtran=0.5,
        probspmr=0.3,
        probjump=0.3,
        radispmr=0.01,  # [day]
        stdvpropelemfire=proposal_scales,
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
        strgcnfg=run_name,
    )


def plot_light_curve_fit(meantime, obsvcnts, catalog, posterior, typefileplot, colrfore, time_tag=None):
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
    return save_flare_figure(figure, "variable_number_stellar_flares_light_curve_fit", typefileplot, time_tag)


def render_flare_posterior_frames(meantime, observed_counts, posterior):
    """Render simulated photometry and PCAT flare models from the prior draw to posterior samples."""
    snapshots = animation_states(posterior)
    models = np.array([np.asarray(snapshot["cntpmodl"], dtype=float)[:, 0, 0] for snapshot in snapshots])
    if not np.isfinite(observed_counts).all() or not np.isfinite(models).all():
        raise ValueError("Flare animation requires finite observations and model draws")
    minimum = min(float(np.min(observed_counts)), float(np.min(models)))
    maximum = max(float(np.max(observed_counts)), float(np.max(models)))
    span = max(1.0, maximum - minimum)
    lower = minimum - 0.05 * span  # [counts per bin]
    upper = maximum + 0.25 * span  # [counts per bin]
    visual_root = EXAMPLE_PATH / "visuals"
    visual_root.mkdir(parents=True, exist_ok=True)
    tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    paths = []
    phase_labels = animation_phase_labels(snapshots)
    count_width = len(str(max(int(np.ravel(snapshot["numbelem"])[0]) for snapshot in snapshots)))
    for snapshot, model, phase_label in zip(snapshots, models, phase_labels):
        figure, axis = plt.subplots(figsize=(6.0, 4.3), facecolor="white")
        hours = (meantime - TIME_OFFSET_DAYS) * 24.0  # [hour]
        axis.plot(hours, observed_counts, color="black", marker=".", markersize=2,
                  linestyle="none", alpha=0.75,
                  label="Simulated 2-min photometry")
        axis.plot(hours, model, color="#B04435", lw=1.6,
                  label=f"PCAT model, {int(np.ravel(snapshot['numbelem'])[0]):{count_width}d} flares")
        axis.set(xlim=(0, 24), ylim=(lower, upper), xlabel="Time [hour]",
                 ylabel="Counts per 2 min bin")
        axis.set_title(phase_label, loc="left", fontfamily="monospace")
        axis.grid(False)
        axis.legend(loc="upper right", framealpha=1, facecolor="white", prop={"family": "monospace"})
        path = visual_root / f"flare_photometry_{tag}_swep{snapshot['cntrswep']:09d}.png"
        print(f"Writing to {path}...")
        figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        paths.append(path)
    return paths


def plot_flare_catalog_samples(catalog, posterior, typefileplot, colrfore, time_tag=None):
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
    return save_flare_figure(figure, "variable_number_stellar_flares_catalog_samples", typefileplot, time_tag)


def plot_flare_count_posterior(catalog, posterior, typefileplot, colrfore, time_tag=None):
    """Posterior probability of the number of flares, compared to the injected count."""
    numbelem = np.asarray(posterior.listpostnumbelem).ravel().astype(int)
    values, counts = np.unique(numbelem, return_counts=True)
    figure, axis = plt.subplots(figsize=(3.4, 2.6))
    axis.bar(values, counts / counts.sum(), color="C0")
    axis.axvline(catalog["timeflar"].size, color="C3", ls="--", lw=1.2, label="Injected count")
    axis.set_xlabel("Number of flares")
    axis.set_ylabel("Posterior probability")
    axis.legend(loc="upper right", fontsize=8)
    return save_flare_figure(figure, "variable_number_stellar_flares_count_posterior", typefileplot, time_tag)


def save_flare_figure(figure, name, typefileplot, time_tag=None):
    name = f"{name}_{time_tag}" if time_tag else name
    target = EXAMPLE_PATH / "visuals" / f"{name}.{typefileplot}"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite flare figure {target}")
    save_example_figure(figure, EXAMPLE_PATH, name, typefileplot)
    return target


def main():
    global EXAMPLE_PATH, PATH_INPT
    import hashlib
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbswep", type=int, default=100_000)
    parser.add_argument("--output-root", type=Path, default=EXAMPLE_PATH)
    parser.add_argument("--run-name", help="Existing run to replot or a new flare run name.")
    parser.add_argument("--chains", type=int, default=NUMBER_CHAINS)
    parser.add_argument("--profile", choices=tuple(FLARE_PROFILE_PARAMETERS), default="flarfred")
    parser.add_argument("--smoke", action="store_true", help="Run a short pipeline check.")
    add_plot_arguments(parser)
    parser.add_argument("--typeplotback", choices=("white", "dark"), default="white")
    parser.add_argument("--skip-sampling", action="store_true", help="Replot an existing chain.")
    parser.add_argument("--time-tag", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    arguments = parser.parse_args()
    if datetime.strptime(arguments.time_tag, "%Y%m%d_%H%M%S").strftime("%Y%m%d_%H%M%S") != arguments.time_tag:
        raise ValueError("time-tag must use YYYYMMDD_HHMMSS")
    EXAMPLE_PATH = arguments.output_root.resolve()
    if list((EXAMPLE_PATH / "visuals").glob(f"*_{arguments.time_tag}.*")):
        raise FileExistsError("Refusing to overwrite an earlier flare output set")
    base_name = RUN_NAME if arguments.profile == "flarfred" else f"{RUN_NAME}_{arguments.profile}"
    run_name = arguments.run_name or (
        base_name if arguments.skip_sampling
        else f"{base_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    run_root = Path(retr_pathrun(EXAMPLE_PATH, run_name))
    if not arguments.skip_sampling:
        _fresh_run_root(run_name)
    PATH_INPT = run_root / "data" / "inpt"
    numbswep = 4000 if arguments.smoke else arguments.numbswep

    rng = np.random.default_rng(0)
    catalog = simulate_flare_catalog(rng)
    edges, meantime, obsvcnts, expcnts = simulate_light_curve(catalog, rng)
    template = (write_time_series_fits(PATH_INPT, edges, obsvcnts, BASELINE_COUNT_RATE)
                if not arguments.skip_sampling else None)
    if not arguments.skip_sampling:
        run_pcat(edges, template, numbswep, build_birth_proposal(edges, obsvcnts, arguments.profile),
                 arguments.profile, run_name=run_name, number_chains=arguments.chains)
    posterior = load_example_posterior(
        f"{EXAMPLE_PATH.name}/data/outp/{run_name}/gdatfinlpost", output_root=EXAMPLE_PATH,
    )
    colrfore = configure_example_style(arguments.typeplotback)
    frames = render_flare_posterior_frames(meantime, obsvcnts, posterior)
    figures = [
        plot_light_curve_fit(meantime, obsvcnts, catalog, posterior, arguments.typefileplot, colrfore, arguments.time_tag),
        plot_flare_catalog_samples(catalog, posterior, arguments.typefileplot, colrfore, arguments.time_tag),
        plot_flare_count_posterior(catalog, posterior, arguments.typefileplot, colrfore, arguments.time_tag),
    ]
    record = {"run_name": run_name, "time_tag": arguments.time_tag,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "native_frame_paths": [path.name for path in frames],
              "native_frame_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in frames},
              "observation_bins": len(obsvcnts), "observed_minimum": float(np.min(obsvcnts)),
              "observed_maximum": float(np.max(obsvcnts)), "input_kind": "Seeded simulated stellar photometry",
              "figure_paths": [path.name for path in figures]}
    target = EXAMPLE_PATH / "visuals" / f"flare_collage_run_{arguments.time_tag}.json"
    print(f"Writing to {target}...")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
