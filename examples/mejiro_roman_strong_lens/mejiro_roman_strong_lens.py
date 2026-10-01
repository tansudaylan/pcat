#!/usr/bin/env python3
"""Infer a mejiro-simulated Roman strong lens with PCAT.

mejiro (Wedig et al., https://github.com/AstroMusers/mejiro) renders the SLSim
galaxy-galaxy lens ``SampleGG`` in the Roman Wide Field Instrument (WFI) F129 band
with its Roman point-spread function (PSF) width, zero point, and sky background.
PCAT samples the singular isothermal ellipsoid (SIE) lens, external shear, and
source position from that Poisson exposure, starting from a random prior draw.
A second fit uses a PSF 30% wider than the one mejiro used, to show how a
misspecified PSF biases the lens parameters without widening their posteriors.
Requires ``pip install -e <mejiro>`` and ``pip install roman-technical-information``.
"""

from __future__ import annotations

from tdpy.verbosity import print

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tdpy.cli import add_plot_arguments

from pcat.diagnostics import gelman_rubin
from pcat.main import retr_pathrun
from pcat.mejiro_lens import LABELS, render_mejiro_counts, run_mejiro_lens_inference, simulate_mejiro_exposure
from pcat.plotting import (
    animation_phase_label, animation_states, make_image_sequence_animation,
    plot_lens_image_fit, plot_lens_parameter_recovery,
)

EXAMPLE_PATH = Path(__file__).resolve().parent
VISUAL_PATH = EXAMPLE_PATH / "visuals"
# differs from the folder name, so the run lives under pcat_runs/ instead of in the example folder
RUN_NAME = "mejiro_samplegg_f129"
NUMBER_CHAINS = 4
# the misspecified fit assumes a PSF this many times wider than the one that generated the data
PSF_SCALE_MISMODELED = 1.3


def sample(specification: dict, run_name: str, smoke: bool) -> object:
    """Sample one forward model of the mejiro exposure with PCAT."""
    # PCAT reuses a finished run if its state exists, so only that state is cleared
    cached_state = Path(retr_pathrun(EXAMPLE_PATH, run_name)) / "data" / "outp" / run_name
    if cached_state.exists():
        print(f"Removing cached PCAT state {cached_state}...")
        shutil.rmtree(cached_state)
    return run_mejiro_lens_inference(
        specification, EXAMPLE_PATH, run_name, numbproc=NUMBER_CHAINS, numbframanim=24,
        numbswep=6000 if smoke else 60000, numbburn=4000 if smoke else 40000,
        numbsamp=500 if smoke else 2000,
    )


def run_example(smoke: bool = False, typefileplot: str = "png") -> dict:
    """Simulate the mejiro exposure, sample it with correct and mismodeled PSFs, and write figures."""
    specification = simulate_mejiro_exposure()
    posterior = sample(specification, RUN_NAME, smoke)
    mismodeled = dict(specification, psf_fwhm=PSF_SCALE_MISMODELED * specification["psf_fwhm"])  # [arcsec]
    posterior_mismodeled = sample(mismodeled, f"{RUN_NAME}_wide_psf", smoke)
    # retained samples are ordered sample-major across chains
    draws = np.asarray(posterior.listpostparagenrscalbase)
    chains = draws.reshape(-1, NUMBER_CHAINS, draws.shape[-1])  # [sample, chain, parameter]
    median_counts = render_mejiro_counts(specification, np.median(draws, axis=0))  # [counts pixel^-1]
    figure_paths = [
        plot_lens_image_fit(VISUAL_PATH / "mejiro_roman_lens_image_fit", specification["observed_counts"],
                            median_counts, median_counts, specification["pixel_scale"], typefileplot),
        plot_lens_parameter_recovery(VISUAL_PATH / "mejiro_roman_lens_posterior", draws,
                                     specification["true_parameters"], typefileplot, labels=LABELS),
        write_replicated_image_animation(posterior, specification),
        plot_psf_mismodeling(specification, posterior, posterior_mismodeled, typefileplot),
    ]
    draws_mismodeled = np.asarray(posterior_mismodeled.listpostparagenrscalbase)
    return {
        "maximum_rhat": max(gelman_rubin(chains[:, :, index]) for index in range(chains.shape[-1])),
        "acceptance_fraction": float(np.mean(posterior.listpostboolpropaccp)),
        "true_parameters": specification["true_parameters"],
        "posterior_medians": np.median(draws, axis=0),
        "posterior_stdvs": draws.std(axis=0),
        "pulls_mismodeled": (np.median(draws_mismodeled, axis=0) - specification["true_parameters"])
        / draws_mismodeled.std(axis=0),
        # mean log-likelihood loss of the mismodeled fit relative to the correct fit
        "delta_log_likelihood": float(np.mean(posterior_mismodeled.listpostlliktotl) - np.mean(posterior.listpostlliktotl)),
        "figure_paths": figure_paths,
    }


def plot_psf_mismodeling(specification: dict, posterior: object, posterior_mismodeled: object,
                         typefileplot: str) -> Path:
    """Posterior offsets from the injected values, in posterior standard deviations, for both PSFs."""
    truth = specification["true_parameters"]
    figure, axis = plt.subplots(figsize=(7.0, 3.4), constrained_layout=True)
    axis.axhspan(-1.0, 1.0, color="0.9", label=r"$\pm 1\sigma$")
    positions = np.arange(truth.size)
    for state, offset, color, label in (
        (posterior, -0.12, "#007360", f"Correct PSF, FWHM {specification['psf_fwhm']:.3f} arcsec"),
        (posterior_mismodeled, 0.12, "#A51C30",
         f"PSF {PSF_SCALE_MISMODELED:.1f} times wider (misspecified)"),
    ):
        draws = np.asarray(state.listpostparagenrscalbase)
        axis.scatter(positions + offset, (np.median(draws, axis=0) - truth) / draws.std(axis=0), s=36,
                     color=color, label=label, zorder=3)
    axis.axhline(0.0, color="black", lw=0.8)
    axis.set_xticks(positions, [label.split(" [")[0] for label in LABELS], rotation=30, ha="right")
    axis.set_ylabel("(Posterior median - injected)\n/ posterior std")
    # above the axes, so no offset can fall behind it
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=8, fancybox=True, framealpha=1.0)
    axis.grid(False)
    path = VISUAL_PATH / f"mejiro_roman_lens_psf_mismodeling.{typefileplot}"
    print(f"Writing to {path}...")
    figure.savefig(path, dpi=300 if typefileplot == "png" else None, bbox_inches="tight")
    plt.close(figure)
    return path


def write_replicated_image_animation(posterior: object, specification: dict) -> Path:
    """Animate replicated mejiro exposures from the initial prior draw to posterior samples."""
    # one fixed noise realization, so frames differ only through the lens model
    noise_seed = 2027
    images, labels = [], []
    for snapshot in animation_states(posterior):
        counts = render_mejiro_counts(specification, snapshot["paragenrscalfull"][:len(LABELS)])
        # counts in units of the sky level, with an arcsinh stretch to show faint arcs beside the bright lens
        replicated = np.arcsinh(np.random.default_rng(noise_seed).poisson(counts) / specification["sky_counts"])
        # GIF rows run top-down, so flip to match the origin='lower' figures
        images.append(np.flipud(replicated))
        labels.append(animation_phase_label(snapshot))
    return make_image_sequence_animation(
        images, labels, VISUAL_PATH / "mejiro_roman_lens_replicated.gif", duration_ms=300,
        title="mejiro Roman F129 lens, PCAT replicated exposure",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run a short pipeline check.")
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    summary = run_example(arguments.smoke, arguments.typefileplot)
    for label, truth, median, stdv in zip(LABELS, summary["true_parameters"], summary["posterior_medians"],
                                          summary["posterior_stdvs"]):
        print(f"{label}: injected {truth:+.4f}, posterior {median:+.4f} +/- {stdv:.4f}")
    print(f"Maximum Gelman-Rubin R-hat {summary['maximum_rhat']:.3f}, "
          f"acceptance fraction {summary['acceptance_fraction']:.2f}")
    print(f"Wide PSF: largest offset {np.max(np.abs(summary['pulls_mismodeled'])):.1f} posterior std, "
          f"log-likelihood change {summary['delta_log_likelihood']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
