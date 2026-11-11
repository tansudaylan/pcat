#!/usr/bin/env python
"""Run a genuine PCAT Gaussian-mixture pipeline example and let it write its own figures."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_NAME = "gaussian_mixture_catalog"
COMPARISON_RUN_NAME = f"{RUN_NAME}_comparison"
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / RUN_NAME


def plot_most_likely_vs_condensed(state, output_path: Path) -> dict[str, object]:
    """Plot the highest-log-posterior retained catalog beside its condensed summary.

    Association frequencies must be unmasked, real, finite fractions in [0, 1].
    Masked and nonfinite log scores are excluded from selection. Complex scores
    are rejected, and at least one available finite score is required.
    Plotted positions must be unmasked, real, and finite. Condensed medians are
    checked independently of uncertainty bounds that this figure does not show.
    Completed empty catalogs are displayed without adding points.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if np.iscomplexobj(state.listpostlpostotl):
        raise ValueError("Posterior log probabilities must be real")
    log_posterior = np.ma.asarray(state.listpostlpostotl, dtype=float).reshape(-1)
    catalog_samples = state.listpostdictelem
    if not catalog_samples or log_posterior.size != len(catalog_samples):
        raise ValueError("Posterior log probabilities must match the retained catalogs")
    finite_samples = np.flatnonzero(
        ~np.ma.getmaskarray(log_posterior) & np.isfinite(np.ma.getdata(log_posterior))
    )
    if not finite_samples.size:
        raise ValueError("At least one retained catalog must have a finite log posterior")
    best_index = int(finite_samples[np.argmax(log_posterior[finite_samples])])
    best_catalog = catalog_samples[best_index][0]

    if "poststkscond" not in state.dictglob:
        raise ValueError("The posterior state must contain a condensed catalog")
    summaries = state.dictglob["poststkscond"]
    condensed_coordinates = []
    for entry in summaries:
        coordinates = []
        for name in ("xpos", "ypos"):
            values = np.ma.asarray(entry[name]).reshape(-1)
            if values.size == 0:
                raise ValueError("Condensed positions must contain median coordinates")
            coordinates.append(values[0])
        condensed_coordinates.append(coordinates)
    plotted_coordinates = [best_catalog["xpos"], best_catalog["ypos"]]
    plotted_coordinates.extend(value for pair in condensed_coordinates for value in pair)
    if any(np.ma.is_masked(value) or np.iscomplexobj(value) for value in plotted_coordinates):
        raise ValueError("Catalog positions must contain unmasked real coordinates")
    best_positions = np.column_stack((best_catalog["xpos"], best_catalog["ypos"])).astype(float)
    condensed_positions = np.asarray(condensed_coordinates, dtype=float).reshape(-1, 2)
    if not np.isfinite(best_positions).all() or not np.isfinite(condensed_positions).all():
        raise ValueError("Catalog positions must be finite")
    if np.ma.is_masked(state.prvl) or np.iscomplexobj(state.prvl):
        raise ValueError("Condensed-catalog frequencies must contain unmasked real values")
    frequencies = np.asarray(state.prvl, dtype=float).reshape(-1)
    if (frequencies.shape != (len(summaries),) or not np.isfinite(frequencies).all()
            or np.any(frequencies < 0.0) or np.any(frequencies > 1.0)):
        raise ValueError("Condensed-catalog frequencies must match the summaries and lie between zero and one")

    all_positions = np.vstack((best_positions, condensed_positions))
    if all_positions.size:
        lower = np.min(all_positions, axis=0)
        upper = np.max(all_positions, axis=0)
        padding = 0.1 * np.maximum(upper - lower, 1.0)
    else:
        lower, upper, padding = np.zeros(2), np.ones(2), np.zeros(2)
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), facecolor="white")
    axes[0].scatter(best_positions[:, 0], best_positions[:, 1], s=70,
                    color="#A51417", edgecolors="white", linewidths=0.7)
    axes[0].set_title(f"Most-likely catalog (sample {best_index + 1})")
    points = axes[1].scatter(
        condensed_positions[:, 0], condensed_positions[:, 1],
        s=40 + 120 * frequencies, c=frequencies, cmap="viridis", vmin=0.0, vmax=1.0,
        edgecolors="black", linewidths=0.5,
    )
    axes[1].set_title("Condensed catalog")
    if frequencies.size:
        figure.colorbar(points, ax=axes[1], label="Fraction of posterior catalogs")
    for axis, positions in zip(axes, (best_positions, condensed_positions)):
        axis.set(xlabel="x position", ylabel="y position",
                 xlim=(lower[0] - padding[0], upper[0] + padding[0]),
                 ylim=(lower[1] - padding[1], upper[1] + padding[1]))
        axis.set_aspect("equal", adjustable="box")
        axis.grid(False)
        if positions.size == 0:
            axis.text(0.5, 0.5, "No catalog elements", transform=axis.transAxes,
                      ha="center", va="center")
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {
        "best_sample_index": best_index,
        "best_positions": best_positions,
        "condensed_positions": condensed_positions,
        "association_frequencies": frequencies,
        "output_path": output_path,
    }


def make_catalog_comparison(output_root: Path, run_name: str) -> dict[str, object]:
    """Load a completed PCAT run and plot its sampled and condensed catalogs."""
    from pcat.main import readfile, retr_condcatl

    output_root = Path(output_root)
    posterior_path = output_root / "data" / "outp" / run_name / "gdatfinlpost"
    state = readfile(str(posterior_path))
    if "poststkscond" not in getattr(state, "dictglob", {}) or not hasattr(state, "prvl"):
        retr_condcatl(state)
    output_path = output_root / "visuals" / "most_likely_vs_condensed_catalog.png"
    return plot_most_likely_vs_condensed(state, output_path)


def main() -> None:
    from pcat.demo import run_pipeline_demo

    cfg = {
        "typeexpr": "gmix",
        "typeelem": ["clusvari"],
        "truenumbelempop0": 2,
        "fittminmnumbelempop0": 1,
        "fittmaxmnumbelempop0": 3,
        "dicttrue": {"typeelem": ["clusvari"], "typeelemspateval": ["full"]},
        "dictfitt": {"typeelem": ["clusvari"], "typeelemspateval": ["full"]},
        "numbsidecart": 8,
        "numbspatdims": 2,
        "strgexpo": 50.0,  # [arbitrary exposure units]
        "typeseedelem": 2_293,
        "inittype": "refr",
        "probspmr": 0.4,
        "numbswep": 1_000,
        "numbsamp": 100,
        "numbswepplot": 100,
        "boolcondcatl": True,
        "strgcnfg": COMPARISON_RUN_NAME,
    }
    run_pipeline_demo(OUTPUT_ROOT, **cfg)
    print(make_catalog_comparison(OUTPUT_ROOT, COMPARISON_RUN_NAME))


if __name__ == "__main__":
    main()
