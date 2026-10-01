"""Fit a simulated transit-timing-variation time series with PCAT."""

from tdpy.verbosity import print

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ephesos import evaluate_sinusoidal_ttv
from pcat.fixed import sample_fixed_chains
from pcat.time_series import log_likelihood_transit_times


ROOT = Path(__file__).resolve().parent
MINUTES_PER_DAY = 1440.0  # [minute day^-1]
TT_VARIATION_PERIOD = 12.0  # [transit epochs]
TRUE_EPOCH = 0.2  # [day]
TRUE_PERIOD = 3.0  # [day]


def predict_transit_times(parameters, transit_epoch):
    """Return linear transit times plus a periodic timing residual [day]."""
    epoch_time, period_days, amplitude_days, phase = parameters
    return epoch_time + period_days * transit_epoch + evaluate_sinusoidal_ttv(
        transit_epoch, 0.0, phase, amplitude_days, TT_VARIATION_PERIOD
    )


def simulate_transits():
    """Return individual observed epochs and noisy, simulated mid-transit times."""
    transit_epoch = np.arange(24)
    truth = (TRUE_EPOCH, TRUE_PERIOD, 4.0 / MINUTES_PER_DAY, 0.3)
    stdv_days = np.full(transit_epoch.size, 1.0 / MINUTES_PER_DAY)  # [day]
    times = predict_transit_times(truth, transit_epoch)
    times += np.random.default_rng(73).normal(0.0, stdv_days)
    return transit_epoch, times, stdv_days


def timing_likelihood(parameters, data):
    """Evaluate PCAT's Gaussian likelihood on the original, unbinned transit times."""
    transit_epoch, observed_times, timing_errors = data
    return log_likelihood_transit_times(
        observed_times, predict_transit_times(parameters, transit_epoch), timing_errors
    )


def render_ttv_posterior_frames(data, chain):
    """Plot the measured-minus-linear timing series and PCAT models on shared limits."""
    transit_epoch, observed_times, errors = data
    linear = TRUE_EPOCH + TRUE_PERIOD * transit_epoch  # [day]
    measured_minutes = (observed_times - linear) * MINUTES_PER_DAY  # [minute]
    indices = np.linspace(0, len(chain) - 1, min(12, len(chain)), dtype=int)
    curves = [(predict_transit_times(chain[index], transit_epoch) - linear) * MINUTES_PER_DAY
              for index in indices]
    maximum_minutes = 1.15 * max(np.max(np.abs(measured_minutes) + errors * MINUTES_PER_DAY),
                                 *(np.max(np.abs(curve)) for curve in curves))  # [minute]
    output = ROOT / "visuals"
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, curve in zip(indices, curves):
        figure, axis = plt.subplots(figsize=(5.8, 4.3), facecolor="white")
        axis.errorbar(transit_epoch, measured_minutes, yerr=errors * MINUTES_PER_DAY,
                      fmt="o", markersize=3.5, color="#465561", label="Simulated transit times")
        axis.plot(transit_epoch, curve, color="#B34735", lw=2, label="PCAT timing model")
        axis.set(xlim=(-1, transit_epoch[-1] + 1), ylim=(-maximum_minutes, maximum_minutes),
                 xlabel="Transit epoch", ylabel="Observed - linear transit time [minute]",
                 title="Simulated transit timing variations")
        axis.grid(False)
        axis.legend(loc="upper right", framealpha=1, facecolor="white")
        path = output / f"ttv_posterior_swep{index:09d}.png"
        print(f"Writing to {path}...")
        figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        paths.append(path)
    return paths


def run_example(smoke: bool = False):
    """Fit simulated transit times with PCAT and return evolving model figures."""
    data = simulate_transits()
    chain, _ = sample_fixed_chains(
        data, timing_likelihood, None,
        ("epoch", "orbital_period", "ttv_amplitude", "ttv_phase"), ("self",) * 4,
        np.array([0.19, 2.9997, 0.0, -np.pi]),
        np.array([0.21, 3.0003, 10.0 / MINUTES_PER_DAY, np.pi]),
        None, None, np.array([[TRUE_EPOCH, TRUE_PERIOD, 3.0 / MINUTES_PER_DAY, 0.2]]),
        1, 20 if smoke else 80, 5 if smoke else 20,
        pathbase=str(ROOT), typeverb=0, boolmakeplot=False,
        boolmakeplotinit=False, boolmakeplotfram=False, makeanim=False,
    )
    posterior_path = ROOT / "data" / "ttv_posterior.npz"
    posterior_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {posterior_path}...")
    np.savez_compressed(posterior_path, transit_epoch=data[0], observed_times=data[1],
                        timing_errors=data[2], chain=chain)
    return data, chain, render_ttv_posterior_frames(data, chain.reshape(-1, 4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    options = parser.parse_args()
    run_example(smoke=options.smoke)


if __name__ == "__main__":
    main()