#!/usr/bin/env python3
"""Fit an unknown number of rotating starspots to simulated photometry."""

from tdpy.verbosity import print

import argparse
from pathlib import Path
import shutil

import astropy.io.fits
import matplotlib.pyplot as plt
import numpy as np

from pcat import sampling
from pcat.time_series import evaluate_rotating_spot_profile


EXAMPLE_PATH = Path(__file__).resolve().parent
RUN_NAME = EXAMPLE_PATH.name
TIME_OFFSET_DAYS = 1.0  # [day]
DURATION_DAYS = 25.6  # [day]
CADENCE_DAYS = 0.1  # [day]
ROTATION_PERIOD_DAYS = 3.2  # [day], fixed for this illustration
BASELINE_COUNTS = 4000.0  # [counts per cadence bin]
INJECTED_SPOTS = {
    "depth_counts": np.array([420.0, 330.0, 280.0]),
    "phase_epoch_days": np.array([0.28, 1.35, 2.42]),
    "fwhm_days": np.array([0.16, 0.22, 0.18]),
}


def simulate_light_curve(seed=12):
    """Return time edges [day], centers [day], counts, and injected mean counts."""

    random = np.random.default_rng(seed)
    numb_bins = int(round(DURATION_DAYS / CADENCE_DAYS))
    edges = TIME_OFFSET_DAYS + np.arange(numb_bins + 1) * CADENCE_DAYS
    time_days = 0.5 * (edges[1:] + edges[:-1])
    spot_model = evaluate_rotating_spot_profile(
        time_days,
        INJECTED_SPOTS["depth_counts"],
        INJECTED_SPOTS["phase_epoch_days"],
        INJECTED_SPOTS["fwhm_days"],
        period_days=ROTATION_PERIOD_DAYS,
        reference_time_days=TIME_OFFSET_DAYS,
    )
    expected_counts = BASELINE_COUNTS + spot_model.sum(axis=1)
    observed_counts = random.poisson(expected_counts).astype(float)
    return edges, time_days, observed_counts, expected_counts


def write_pcat_inputs(edges, observed_counts):
    """Write simulated Poisson counts and exposure for PCAT's input-data path."""

    input_path = EXAMPLE_PATH / "data" / "inpt"
    input_path.mkdir(parents=True, exist_ok=True)
    exposure = np.full(observed_counts.size, 1.0 / CADENCE_DAYS)
    for name, values in (
        ("sbrt.fits", observed_counts[:, None, None, None]),
        ("expo.fits", exposure[:, None, None]),
    ):
        path = input_path / name
        print(f"Writing to {path}...")
        astropy.io.fits.writeto(path, values, overwrite=True)
    return np.full((edges.size - 1, 1, 1), BASELINE_COUNTS)


def run_pcat(edges, baseline_template, number_sweeps):
    """Fit a zero-to-five-spot catalog with a shared known rotation period."""

    cached_output = EXAMPLE_PATH / "data" / "outp" / RUN_NAME
    if cached_output.exists():
        print(f"Removing cached spot-chain output {cached_output}...")
        shutil.rmtree(cached_output)
    return sampling.sample(
        typeexpr="fire",
        typedata="inpt",
        strgexprsbrt="sbrt.fits",
        typeexpo="file",
        strgexpo="expo.fits",
        binsenerfull=edges,
        spectype=["spotrot"],
        spatdisttype=["line"],
        typeelem=["lghtlinevoig"],
        dictfitt={
            "typeelem": ["lghtlinevoig"],
            "spectype": ["spotrot"],
            "sbrtbacknorm": [baseline_template],
            "listnamediff": ["back0000"],
        },
        spot_period_days=ROTATION_PERIOD_DAYS,
        spot_reference_time_days=TIME_OFFSET_DAYS,
        fittminmnumbelempop0=0,
        fittmaxmnumbelempop0=5,
        limtparaelem={
            "flux": (100.0, 650.0),  # [counts per cadence bin]
            "elin": (0.05, ROTATION_PERIOD_DAYS - 0.05),  # [day]
            "fwhm": (0.08, 0.45),  # [day]
        },
        stdvpropelemfire=[0.04, 0.04, 0.04],
        anlytype="spec",
        maxmgangdata=1e-4,
        inittype="rand",
        typeseed=17,
        probtran=0.8,
        probspmr=0.4,
        numbswep=number_sweeps,
        numbburn=number_sweeps // 3,
        numbsamp=max(number_sweeps // 20, 16),
        numbswepplot=max(number_sweeps // 20, 1),
        boolmakeplot=True,
        boolmakeplotinit=False,
        boolmakeplotfram=True,
        makeanim=True,
        booldiag=False,
        typeverb=0,
        pathbase=str(EXAMPLE_PATH),
        strgcnfg=RUN_NAME,
    )


def read_posterior():
    """Load the final PCAT chain state from this example's run directory."""

    from pcat.main import readfile

    path = (
        EXAMPLE_PATH / "data" / "outp" / RUN_NAME / "gdatfinlpost"
    )
    return readfile(str(path))


def render_spot_posterior_frames(time_days, observed_counts, posterior):
    """Render changing transdimensional spot fits for the posterior collage."""

    models = np.asarray(posterior.listpostcntpmodl, dtype=float)[:, :, 0, 0]
    number_spots = np.asarray(posterior.listpostnumbelem, dtype=int).reshape(-1)
    frame_indices = np.unique(
        np.linspace(0, models.shape[0] - 1, min(18, models.shape[0]), dtype=int)
    )
    hours = (time_days - TIME_OFFSET_DAYS) * 24.0  # [hour]
    upper = 1.05 * max(float(observed_counts.max()), float(models.max()))
    residual_limit = max(1.0, 1.1 * float(np.max(np.abs(observed_counts - models))))
    output_directory = EXAMPLE_PATH / "visuals"
    output_directory.mkdir(parents=True, exist_ok=True)
    for old_frame in output_directory.glob("stellar_spot_photometry_swep*.png"):
        print(f"Removing previous spot frame {old_frame}...")
        old_frame.unlink()

    output_paths = []
    for frame_index in frame_indices:
        model = models[frame_index]
        spot_count = int(number_spots[min(frame_index, number_spots.size - 1)])
        figure, axes = plt.subplots(
            2, 1, figsize=(7.0, 4.5), sharex=True,
            gridspec_kw={"height_ratios": (3.0, 1.0)},
        )
        axes[0].scatter(hours, observed_counts, s=7, color="#60686B", alpha=0.65,
                        linewidths=0, label="Simulated photometry")
        axes[0].plot(hours, model, color="#A51C30", lw=1.5,
                     label="PCAT starspot model")
        axes[0].set(xlim=(hours[0], hours[-1]), ylim=(0.0, upper),
                    ylabel="Counts per 0.1 day bin",
                    title=f"Variable starspot catalog | N = {spot_count}")
        axes[0].legend(loc="upper right", fontsize=8, frameon=True,
                       fancybox=True, framealpha=1.0)
        axes[1].axhline(0.0, color="black", lw=0.8, linestyle="dashed")
        axes[1].scatter(hours, observed_counts - model, s=6, color="#0072B2",
                        alpha=0.7, linewidths=0)
        axes[1].set(xlim=(hours[0], hours[-1]), ylim=(-residual_limit, residual_limit),
                    xlabel="Time [hour]", ylabel="Residual [counts]")
        for axis in axes:
            axis.grid(False)
        figure.subplots_adjust(hspace=0.06)
        path = output_directory / f"stellar_spot_photometry_swep{frame_index:09d}.png"
        print(f"Writing to {path}...")
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(path)
    return output_paths


def run_example(number_sweeps=3000, seed=12):
    """Simulate rotating dark spots and fit a variable-size spot catalog."""

    edges, time_days, observed_counts, _ = simulate_light_curve(seed)
    template = write_pcat_inputs(edges, observed_counts)
    run_pcat(edges, template, number_sweeps)
    posterior = read_posterior()
    frames = render_spot_posterior_frames(time_days, observed_counts, posterior)
    return {"frames": frames, "posterior": posterior}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbswep", type=int, default=3000)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    run_example(600 if arguments.smoke else arguments.numbswep)


if __name__ == "__main__":
    main()