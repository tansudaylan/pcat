"""Sample an unbinned two-component Gaussian mixture of simulated event positions."""

from tdpy.verbosity import print

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.special import logsumexp

from pcat.fixed import sample_fixed_chains


ROOT = Path(__file__).resolve().parent
WIDTH = 0.24


def simulate_events():
    """Return individual, unbinned 2D events from two known Gaussian components."""

    rng = np.random.default_rng(19)
    centers = np.array([[-0.9, -0.35], [0.8, 0.65]])
    component = rng.integers(0, 2, size=140)
    return centers[component] + rng.normal(0.0, WIDTH, (component.size, 2))


def event_log_likelihood(centers_flat, events):
    """Sum event log-densities directly, without spatial bins or count maps."""

    centers = np.asarray(centers_flat).reshape(2, 2)
    distances = np.sum(((events[:, None, :] - centers[None, :, :]) / WIDTH) ** 2, axis=2)
    return float(np.sum(logsumexp(-0.5 * distances, axis=1) - np.log(4 * np.pi * WIDTH**2)))


def render_posterior_frames(events, chain, output_directory: Path):
    """Plot the actual events and selected PCAT posterior centers on fixed axes."""

    output_directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for sweep in np.linspace(0, len(chain) - 1, min(12, len(chain)), dtype=int):
        centers = np.asarray(chain[sweep]).reshape(2, 2)
        figure, axis = plt.subplots(figsize=(5.2, 5.2), facecolor="white")
        axis.scatter(events[:, 0], events[:, 1], s=10, color="#424E54", alpha=0.55,
                     label="Simulated individual events")
        for index, center in enumerate(centers):
            color = ("#B24A37", "#167D72")[index]
            axis.add_patch(Circle(center, 2 * WIDTH, fill=False, linewidth=2.0,
                                  edgecolor=color, label=f"Posterior component {index + 1}"))
            axis.plot(*center, marker="x", markersize=9, color=color)
        axis.set(xlim=(-2, 2), ylim=(-2, 2), xlabel="Event coordinate x", ylabel="Event coordinate y",
                 title="Unbinned Gaussian-mixture posterior")
        axis.set_aspect("equal")
        axis.grid(False)
        axis.legend(loc="upper left", framealpha=1.0, facecolor="white", edgecolor="black", fontsize=8)
        path = output_directory / f"gmix_events_swep{sweep:09d}.png"
        print(f"Writing to {path}...")
        figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    options = parser.parse_args()
    events = simulate_events()
    chain, _ = sample_fixed_chains(
        events, event_log_likelihood, None,
        ("center_x_1", "center_y_1", "center_x_2", "center_y_2"), ("self",) * 4,
        np.full(4, -1.6), np.full(4, 1.6), None, None,
        np.array([[-0.8, -0.3, 0.7, 0.6]]), 1,
        20 if options.smoke else 80, 5 if options.smoke else 20,
        pathbase=str(ROOT), typeverb=0, boolmakeplot=False,
        boolmakeplotinit=False, boolmakeplotfram=False, makeanim=False,
    )
    render_posterior_frames(events, chain.reshape(-1, 4), ROOT / "visuals")


if __name__ == "__main__":
    main()