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
from pcat.plotting import animation_phase_label, animation_states
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


def render_ttv_posterior_frames(data, snapshots):
    """Plot the measured-minus-linear timing series and PCAT models from the prior draw onward."""
    transit_epoch, observed_times, errors = data
    linear = TRUE_EPOCH + TRUE_PERIOD * transit_epoch  # [day]
    measured_minutes = (observed_times - linear) * MINUTES_PER_DAY  # [minute]
    curves = [(predict_transit_times(snapshot["paragenrscalfull"][:4], transit_epoch) - linear) * MINUTES_PER_DAY
              for snapshot in snapshots]
    # data-based limits keep the axis fixed even when early burn-in models lie far off
    maximum_minutes = 1.6 * np.max(np.abs(measured_minutes) + errors * MINUTES_PER_DAY)  # [minute]
    output = ROOT / "visuals"
    output.mkdir(parents=True, exist_ok=True)
    for old_frame in output.glob("ttv_posterior_swep*.png"):
        print(f"Removing previous frame {old_frame}...")
        old_frame.unlink()
    paths = []
    for snapshot, curve in zip(snapshots, curves):
        figure, axis = plt.subplots(figsize=(5.8, 4.3), facecolor="white")
        axis.errorbar(transit_epoch, measured_minutes, yerr=errors * MINUTES_PER_DAY,
                      fmt="o", markersize=3.5, color="#465561", label="Simulated transit times")
        axis.plot(transit_epoch, curve, color="#B34735", lw=2, label="PCAT timing model")
        axis.set(xlim=(-1, transit_epoch[-1] + 1), ylim=(-maximum_minutes, maximum_minutes),
                 xlabel="Transit epoch", ylabel="Observed - linear transit time [minute]",
                 title=animation_phase_label(snapshot))
        axis.grid(False)
        axis.legend(loc="upper right", framealpha=1, facecolor="white")
        path = output / f"ttv_posterior_swep{snapshot['cntrswep']:09d}.png"
        print(f"Writing to {path}...")
        figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        paths.append(path)
    return paths


def run_example(smoke: bool = False):
    """Fit simulated transit times with PCAT and return evolving model figures."""
    data = simulate_transits()
    sample_count, burn_count = (200, 100) if smoke else (2000, 1000)
    chain, _, state = sample_fixed_chains(
        data, timing_likelihood, None,
        ("epoch", "orbital_period", "ttv_amplitude", "ttv_phase"), ("self",) * 4,
        np.array([0.19, 2.9997, 0.0, -np.pi]),
        np.array([0.21, 3.0003, 10.0 / MINUTES_PER_DAY, np.pi]),
        None, None, None,
        4, sample_count, burn_count,
        pathbase=str(ROOT), typeverb=0, boolmakeplot=False,
        boolmakeplotinit=False, boolmakeplotfram=False, makeanim=False,
        numbframanim=24, return_state=True,
    )
    posterior_path = ROOT / "data" / "ttv_posterior.npz"
    posterior_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {posterior_path}...")
    np.savez_compressed(posterior_path, transit_epoch=data[0], observed_times=data[1],
                        timing_errors=data[2], chain=chain)
    return data, chain, render_ttv_posterior_frames(data, animation_states(state))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    options = parser.parse_args()
    run_example(smoke=options.smoke)


if __name__ == "__main__":
    main()