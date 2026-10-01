#!/usr/bin/env python3
"""Generate retained-state and every-proposal animations from one PCAT run.

The data are a seeded simulated spectrum containing two Voigt emission lines.
PCAT fits one to three lines with within-model, birth, death, split, and merge
proposals. The short run demonstrates output semantics rather than convergence.
"""

from tdpy.verbosity import print

import argparse
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np

from pcat import sampling
from pcat.demo import load_example_namespace


EXAMPLE_PATH = Path(__file__).resolve().parent
RUN_NAME = EXAMPLE_PATH.name


def summarize_jacobian_acceptance(worker):
    """Return exact split/merge counterfactuals for uncensored proposals."""
    proposal = np.asarray(worker.listpostindxproptype).reshape(-1).astype(int)
    log_acceptance = np.asarray(worker.listpostaccplprb).reshape(-1)
    jacobian = np.asarray(worker.listpostljcb).reshape(-1)
    accepted = np.asarray(worker.listpostboolpropaccp).reshape(-1).astype(bool)
    supported = np.asarray(worker.listpostboolpropfilt).reshape(-1).astype(bool)
    if not all(array.size == proposal.size for array in (log_acceptance, jacobian, accepted, supported)):
        raise ValueError("proposal arrays must have matching lengths")

    result = {}
    for move, name in ((3, "split"), (4, "merge")):
        selected = (
            (proposal == move) & supported & np.isfinite(log_acceptance)
            & np.isfinite(jacobian)
        )
        if not np.any(selected):
            raise ValueError(f"the run needs a finite, prior-valid {name} proposal")
        actual = np.exp(np.minimum(log_acceptance[selected], 0.0))
        result[name] = {
            "actual": actual,
            "without_jacobian": np.exp(np.minimum(log_acceptance[selected] - jacobian[selected], 0.0)),
            "log_acceptance": log_acceptance[selected],
            "log_acceptance_without_jacobian": log_acceptance[selected] - jacobian[selected],
            "jacobian": jacobian[selected],
            "accepted": accepted[selected],
        }
    return result


def plot_jacobian_acceptance(summary, output_path):
    """Plot each split/merge probability against its no-Jacobian counterfactual."""
    colors = {"split": "#A51C30", "merge": "#007C78"}
    markers = {"split": "o", "merge": "^"}
    figure, axes = plt.subplots(
        1, 2, figsize=(9.0, 4.0), constrained_layout=True,
        gridspec_kw={"width_ratios": (1.0, 1.7)},
    )
    shift_axis, probability_axis = axes
    positions = np.arange(2, dtype=float)
    for index, name in enumerate(("split", "merge")):
        values = summary[name]
        log_shift = (values["log_acceptance_without_jacobian"] - values["log_acceptance"]) / np.log(10.0)
        shift_axis.scatter(
            np.full(log_shift.size, positions[index]), log_shift,
            color=colors[name], marker=markers[name], s=30,
            alpha=0.8, linewidths=0, label=f"{name} (n={log_shift.size})",
        )

        baseline_logratio = np.linspace(-5.0, 1.0, 301)
        median_jacobian = float(np.median(values["jacobian"]))
        probability_axis.plot(
            baseline_logratio,
            np.exp(np.minimum(0.0, baseline_logratio + median_jacobian)),
            color=colors[name], lw=1.5,
            label=f"{name}, with J",
        )
        probability_axis.plot(
            baseline_logratio, np.exp(np.minimum(0.0, baseline_logratio)),
            color=colors[name], lw=1.0, ls="--", alpha=0.7,
            label=f"{name}, without J",
        )

    shift_axis.axhline(0.0, color="0.5", lw=0.8, ls="--")
    shift_axis.set(
        xticks=positions, xticklabels=("Split", "Merge"),
        ylabel="Change in log10 acceptance ratio without J [dex]",
    )
    shift_axis.legend(loc="best", frameon=True, fancybox=True, framealpha=1.0, fontsize=8)
    probability_axis.set(
        xlabel="Log acceptance ratio before Jacobian",
        ylabel="Acceptance probability",
        ylim=(0.0, 1.03),
        title="Median log|J|: split %.2f, merge %.2f" % (
            np.median(summary["split"]["jacobian"]),
            np.median(summary["merge"]["jacobian"]),
        ),
    )
    probability_axis.legend(loc="upper left", frameon=True, fancybox=True, framealpha=1.0, fontsize=8)
    for axis in axes:
        axis.grid(False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def run_example(number_sweeps=120):
    """Run the simulated Voigt problem and return animations and Jacobian plot."""
    namespace = load_example_namespace(
        "voigt_spectral_line_catalog/voigt_spectral_line_catalog.py"
    )
    configuration, variations, _ = namespace["build_configurations"]()
    configuration.update(variations["nomi"])
    configuration.update(
        strgcnfg=RUN_NAME,
        pathbase=str(EXAMPLE_PATH),
        truenumbelempop0=2,
        fittminmnumbelempop0=1,
        fittmaxmnumbelempop0=3,
        dicttrue={"typeelem": ["lghtlinevoig"], "spectype": ["voig"]},
        dictfitt={"typeelem": ["lghtlinevoig"], "spectype": ["voig"]},
        probtran=0.8,
        probspmr=0.5,
        numbswep=number_sweeps,
        numbburn=max(2, number_sweeps // 6),
        numbsamp=max(4, number_sweeps // 3),
        numbswepplot=max(1, number_sweeps // 6),
        boolcheckconv=False,
        boolmakeplot=True,
        boolmakeplotinit=False,
        boolmakeplotfram=True,
        boolmakeplotfinlpost=False,
        makeanim=True,
        boolmakeanimprop=True,
        booldiag=False,
        numbproc=1,
        typeverb=0,
    )
    for path in (EXAMPLE_PATH / "data" / "outp" / RUN_NAME, EXAMPLE_PATH / "visuals"):
        if path.exists():
            print(f"Removing cached example output {path}...")
            shutil.rmtree(path)
    sampling.sample(**configuration)
    from pcat.main import readfile

    worker = readfile(str(EXAMPLE_PATH / "data" / "outp" / RUN_NAME / "gdatmodi0000post"))
    jacobian_summary = summarize_jacobian_acceptance(worker)
    jacobian_path = plot_jacobian_acceptance(
        jacobian_summary, EXAMPLE_PATH / "visuals" / "jacobian_acceptance_effect.png"
    )

    animation_root = EXAMPLE_PATH / "visuals" / "post" / "anim"
    posterior_path = animation_root / "proposal_sequence.gif"
    proposal_path = animation_root / "proposal_candidates.gif"
    if not posterior_path.is_file() or not proposal_path.is_file():
        raise RuntimeError("PCAT did not produce both requested animations.")
    return {
        "posterior_animation": posterior_path,
        "proposal_animation": proposal_path,
        "jacobian_figure": jacobian_path,
        "jacobian_summary": {
            name: {
                "count": values["actual"].size,
                "mean_actual_probability": float(np.mean(values["actual"])),
                "mean_without_jacobian": float(np.mean(values["without_jacobian"])),
                "observed_acceptance_fraction": float(np.mean(values["accepted"])),
                "median_log_jacobian": float(np.median(values["jacobian"])),
            }
            for name, values in jacobian_summary.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbswep", type=int, default=120)
    arguments = parser.parse_args()
    products = run_example(arguments.numbswep)
    for label in ("posterior_animation", "proposal_animation", "jacobian_figure"):
        print(f"{label}: {products[label]}")
    for name, result in products["jacobian_summary"].items():
        print(f"{name}: {result}")


if __name__ == "__main__":
    main()