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
            "accepted": accepted[selected],
        }
    return result


def plot_jacobian_acceptance(summary, output_path):
    """Plot each split/merge probability against its no-Jacobian counterfactual."""
    colors = {"split": "#A51C30", "merge": "#007C78"}
    markers = {"split": "o", "merge": "^"}
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), constrained_layout=True)
    probability_axis, rate_axis = axes
    probability_axis.plot([0.0, 1.0], [0.0, 1.0], color="0.5", lw=0.8, ls="--")
    positions = np.arange(2, dtype=float)
    width = 0.24
    for index, name in enumerate(("split", "merge")):
        values = summary[name]
        actual = values["actual"]
        without = values["without_jacobian"]
        accepted = values["accepted"]
        probability_axis.plot(actual, without, color=colors[name], alpha=0.25, lw=0.8)
        probability_axis.scatter(
            actual, without, color=colors[name], marker=markers[name], s=24,
            alpha=0.8, linewidths=0, label=f"{name} (n={actual.size})",
        )
        rate_axis.bar(positions[index] - width, np.mean(actual), width,
                      color=colors[name], alpha=0.55, label="With Jacobian" if index == 0 else None)
        rate_axis.bar(positions[index], np.mean(without), width,
                      color=colors[name], label="Without Jacobian" if index == 0 else None)
        rate_axis.bar(positions[index] + width, np.mean(accepted), width,
                      color=colors[name], alpha=0.25, hatch="//",
                      label="Observed acceptance" if index == 0 else None)

    probability_axis.set(
        xlabel="PCAT acceptance probability",
        ylabel="Counterfactual probability without Jacobian",
        xlim=(-0.03, 1.03), ylim=(-0.03, 1.03),
    )
    probability_axis.legend(loc="best", frameon=True, fancybox=True, framealpha=1.0)
    rate_axis.set(
        xticks=positions, xticklabels=("Split", "Merge"),
        ylabel="Mean probability or observed fraction", ylim=(0.0, 1.0),
    )
    rate_axis.legend(loc="best", frameon=True, fancybox=True, framealpha=1.0, fontsize=8)
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