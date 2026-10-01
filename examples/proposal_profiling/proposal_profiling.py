#!/usr/bin/env python3
"""Profile the execution time and acceptance of each PCAT proposal type.

The run fits the simulated Voigt spectral-line data set of the
``voigt_spectral_line_catalog`` example with birth, death, split, merge, and
within-model proposals enabled. PCAT records the wall-clock time of every sweep
and the proposal type used, as well as a per-sweep breakdown into pipeline
phases (proposal, likelihood, model evaluation, prior, etc.); this script
summarizes both.
"""

from tdpy.verbosity import print

import argparse
from tdpy.cli import add_plot_arguments
import shutil
import sys
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

EXAMPLE_PATH = Path(__file__).resolve().parent
RUN_NAME = EXAMPLE_PATH.name
sys.path.insert(0, str(EXAMPLE_PATH.parent / "voigt_spectral_line_catalog"))

# proposal-type order used by PCAT (gdat.nameproptype)
LABELS = ["Within-model", "Birth", "Death", "Split", "Merge"]

# descriptive labels for PCAT's per-sweep pipeline phase timers (gdat.listnamechro/listlablchro)
PHASE_LABELS = {
    "prop": "Proposal", "diag": "Diagnostics", "save": "Save", "plot": "Plot",
    "proc": "Process", "elem": "Parse", "modl": "Model", "llik": "Likelihood",
    "sbrtmodl": "Total emission", "spec": "Spectrum calculation",
    "elemsbrtdfnc": "Dfnc S Brght", "elemdeflsubh": "Subh Defl",
    "elemsbrtextsbgrd": "Bkg Exts S Brght", "psfnconv": "Img for PSF Conv.",
    "expo": "Exposure", "lpri": "Prior", "tert": "Tertiary",
}


def run_sampler(numbswep):
    """Sample the simulated Voigt spectrum with all proposal types enabled."""
    from pcat import sampling
    from voigt_spectral_line_catalog import build_configurations

    configuration, _, _ = build_configurations()
    # PCAT resumes from cached state, so remove it to profile a fresh run
    path_cache = EXAMPLE_PATH / "data" / "outp" / RUN_NAME
    if path_cache.exists():
        print(f"Removing cached PCAT state {path_cache}...")
        shutil.rmtree(path_cache)
    configuration.update(
        strgcnfg=RUN_NAME,
        # the run root equals the example folder, so PCAT writes visuals/ here
        pathbase=str(EXAMPLE_PATH),
        truenumbelempop0=2,
        fittminmnumbelempop0=1,
        fittmaxmnumbelempop0=3,
        dicttrue={"typeelem": ["lghtlinevoig"], "spectype": ["voig"]},
        dictfitt={"typeelem": ["lghtlinevoig"], "spectype": ["voig"]},
        # fraction of transdimensional proposals that are split or merge
        probspmr=0.3,
        probtran=0.7,
        numbswep=numbswep,
        numbsamp=numbswep // 10,
        numbswepplot=max(numbswep // 40, 1),
        boolcheckconv=False,
        boolmakeplot=False,
        boolmakeplotinit=False,
        makeanim=True,
        booldiag=False,
        numbproc=1,
        typeverb=0,
    )
    sampling.sample(**configuration)


def read_chain():
    """Return per-sweep proposal type, acceptance, prior filter, time, and pipeline-phase times."""
    path = EXAMPLE_PATH / "data" / "outp" / RUN_NAME / "gdatmodi0000post.h5"
    print(f"Reading from {path}...")
    with h5py.File(path, "r") as file:
        phase = {
            key[len("listpostchro"):]: file[key][()].ravel() * 1e3  # [ms]
            for key in file.keys()
            if key.startswith("listpostchro") and key != "listpostchrototl"
        }
        return {
            "type": file["listpostindxproptype"][()].ravel().astype(int),
            "accp": file["listpostboolpropaccp"][()].ravel().astype(bool),
            "filt": file["listpostboolpropfilt"][()].ravel().astype(bool),
            "time": file["listpostchrototl"][()].ravel() * 1e3,  # [ms]
            "phase": phase,
        }


def configure_style(typeplotback):
    """Apply the white (default) or dark figure theme without gridlines."""
    colrfore = "white" if typeplotback == "dark" else "black"
    colrback = "black" if typeplotback == "dark" else "white"
    mpl.rcParams.update({
        "font.size": 10,
        "text.usetex": False,
        "axes.grid": False,
        "figure.facecolor": colrback,
        "axes.facecolor": colrback,
        "savefig.facecolor": colrback,
        "axes.edgecolor": colrfore,
        "axes.labelcolor": colrfore,
        "xtick.color": colrfore,
        "ytick.color": colrfore,
        "text.color": colrfore,
        "legend.fancybox": True,
        "legend.framealpha": 1.0,
    })


def save(figure, name, typefileplot):
    path = EXAMPLE_PATH / "visuals" / f"{name}.{typefileplot}"
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {path}...")
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_time_per_proposal(chain, typefileplot):
    """Show the distribution of the wall-clock time per sweep by proposal type."""
    indxtype = [k for k in range(len(LABELS)) if np.any(chain["type"] == k)]
    listtime = [chain["time"][chain["type"] == k] for k in indxtype]
    figure, axis = plt.subplots(figsize=(6.5, 3.2))
    boxes = axis.boxplot(listtime, vert=False, showfliers=False, patch_artist=True, widths=0.6)
    for patch, k in zip(boxes["boxes"], indxtype):
        patch.set_facecolor(f"C{k}")
    for position, times in enumerate(listtime, start=1):
        axis.text(np.percentile(times, 75) * 1.03, position + 0.32,
                  f"median {np.median(times):.2f} ms, N={times.size}", va="center")
    axis.set_yticks(range(1, len(indxtype) + 1), [LABELS[k] for k in indxtype])
    axis.set_xlabel("Wall-clock time per sweep [ms]")
    axis.set_xlim(0.0, max(np.percentile(times, 75) for times in listtime) * 1.9)
    save(figure, "proposal_time_per_sweep", typefileplot)


def plot_acceptance_and_cost(chain, typefileplot):
    """Compare acceptance fraction with the share of total compute per proposal type."""
    indxtype = [k for k in range(len(LABELS)) if np.any(chain["type"] == k)]
    fracaccp = [chain["accp"][chain["type"] == k].mean() for k in indxtype]
    fracfilt = [chain["filt"][chain["type"] == k].mean() for k in indxtype]
    fraccost = [chain["time"][chain["type"] == k].sum() / chain["time"].sum() for k in indxtype]
    position = np.arange(len(indxtype))
    width = 0.27
    figure, axis = plt.subplots(figsize=(6.5, 3.2))
    axis.bar(position - width, fracfilt, width, color="0.6", label="Inside prior support")
    axis.bar(position, fracaccp, width, color="C0", label="Accepted")
    axis.bar(position + width, fraccost, width, color="C3", label="Share of total run time")
    axis.set_xticks(position, [LABELS[k] for k in indxtype])
    axis.set_ylabel("Fraction")
    axis.set_ylim(0.0, 1.25)
    axis.legend(loc="upper right", ncol=3)
    save(figure, "proposal_acceptance_and_cost", typefileplot)


def plot_time_breakdown_by_phase(chain, typefileplot):
    """Show the mean per-sweep time spent in each internal pipeline phase."""
    meantime = {name: times.mean() for name, times in chain["phase"].items() if times.mean() > 1e-6}
    # 'tert' is measured after, not inside, the sweep loop, so it is not part of chrototl
    names = sorted((name for name in meantime if name != "tert"), key=meantime.get)
    labels = [PHASE_LABELS.get(name, name.title()) for name in names]
    values = [meantime[name] for name in names]
    figure, axis = plt.subplots(figsize=(6.5, 3.4))
    axis.barh(range(len(names)), values, color="C0")
    axis.set_yticks(range(len(names)), labels)
    axis.axvline(chain["time"].mean(), color="C3", ls="--", lw=1.2,
                label=f"Total sweep, mean {chain['time'].mean():.3f} ms")
    if "tert" in meantime:
        axis.axvline(meantime["tert"], color="0.4", ls=":", lw=1.2,
                    label=f"Tertiary bookkeeping (outside the sweep), mean {meantime['tert']:.3f} ms")
    axis.set_xlabel("Mean wall-clock time per sweep [ms]")
    axis.legend(loc="lower right", fontsize=8)
    save(figure, "proposal_time_breakdown_by_phase", typefileplot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbswep", type=int, default=20000)
    add_plot_arguments(parser)
    parser.add_argument("--typeplotback", choices=("white", "dark"), default="white")
    parser.add_argument("--skip-sampling", action="store_true",
                        help="Replot from an existing chain without sampling again.")
    arguments = parser.parse_args()
    if not arguments.skip_sampling:
        run_sampler(arguments.numbswep)
    chain = read_chain()
    configure_style(arguments.typeplotback)
    plot_time_per_proposal(chain, arguments.typefileplot)
    plot_acceptance_and_cost(chain, arguments.typefileplot)
    plot_time_breakdown_by_phase(chain, arguments.typefileplot)
    for k, label in enumerate(LABELS):
        indx = chain["type"] == k
        if indx.any():
            print(f"{label}: N={indx.sum()}, acceptance {chain['accp'][indx].mean():.3f}, "
                  f"median time {np.median(chain['time'][indx]):.3f} ms")


if __name__ == "__main__":
    main()
