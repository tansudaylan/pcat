"""Time-series transient profiles for PCAT element populations."""

from __future__ import annotations

import numpy as np


def log_likelihood_transit_times(observed_times, predicted_times, timing_errors):
    """Return the Gaussian transit-time log likelihood up to a constant."""

    observed = np.asarray(observed_times, dtype=float)
    predicted = np.asarray(predicted_times, dtype=float)
    errors = np.asarray(timing_errors, dtype=float)
    if observed.ndim != 1 or observed.shape != predicted.shape or observed.shape != errors.shape:
        raise ValueError("observed, predicted, and timing errors must be matching one-dimensional arrays")
    if not np.isfinite(observed).all() or not np.isfinite(errors).all() or np.any(errors <= 0):
        raise ValueError("observed times must be finite and timing errors must be finite and positive")
    if not np.isfinite(predicted).all():
        return -np.inf
    return float(-0.5 * np.sum(((observed - predicted) / errors) ** 2))


FLARE_PROFILE_PARAMETERS = {
    "flargauss": ("flux", "elin", "fwhm"),
    "flarexpd": ("flux", "elin", "scalfall"),
    "flarfred": ("flux", "elin", "scalrise", "scalfall"),
    "flardav": ("flux", "elin", "fwhm"),
}
ROTATING_SPOT_PARAMETERS = {"spotrot": ("flux", "elin", "fwhm")}


def flare_profile_parameters(profile: str) -> tuple[str, ...]:
    """Return sampled element parameters for one supported flare profile."""
    try:
        return FLARE_PROFILE_PARAMETERS[profile]
    except KeyError as exception:
        raise ValueError(
            f"Unknown flare profile {profile!r}; choose from {tuple(FLARE_PROFILE_PARAMETERS)}."
        ) from exception

def rotating_spot_parameters(profile: str) -> tuple[str, ...]:
    """Return sampled parameters for a rotating stellar-spot profile."""

    try:
        return ROTATING_SPOT_PARAMETERS[profile]
    except KeyError as exception:
        raise ValueError(
            f"Unknown rotating spot profile {profile!r}; "
            f"choose from {tuple(ROTATING_SPOT_PARAMETERS)}."
        ) from exception

def evaluate_rotating_spot_profile(
    time_days,
    depth_counts,
    phase_epoch_days,
    fwhm_days,
    period_days,
    reference_time_days=0.0,  # [day]
):
    """Return periodic Gaussian dimmings, one negative component per spot."""

    time_days = np.asarray(time_days, dtype=float).reshape(-1, 1)
    depth_counts = np.atleast_1d(np.asarray(depth_counts, dtype=float))
    phase_epoch_days = np.atleast_1d(np.asarray(phase_epoch_days, dtype=float))
    fwhm_days = np.atleast_1d(np.asarray(fwhm_days, dtype=float))
    period_days = float(period_days)
    reference_time_days = float(reference_time_days)
    if (time_days.size == 0 or not np.isfinite(time_days).all() or depth_counts.size == 0
            or depth_counts.size != phase_epoch_days.size or depth_counts.size != fwhm_days.size):
        raise ValueError("time and spot parameter arrays must be finite and have matching nonzero sizes")
    if (not np.isfinite(depth_counts).all() or np.any(depth_counts <= 0.0)
            or not np.isfinite(phase_epoch_days).all() or not np.isfinite(fwhm_days).all()
            or np.any(fwhm_days <= 0.0)):
        raise ValueError("spot depths, epochs, and widths must be finite and positive")
    if not np.isfinite(period_days) or period_days <= 0.0 or not np.isfinite(reference_time_days):
        raise ValueError("period must be finite and positive and reference time finite")
    if np.any(fwhm_days >= period_days):
        raise ValueError("spot widths must be shorter than the rotation period")

    phase_offset_days = np.remainder(
        time_days - reference_time_days - phase_epoch_days[None, :] + 0.5 * period_days,
        period_days,
    ) - 0.5 * period_days
    sigma_days = fwhm_days / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # [day]
    profile = np.exp(-0.5 * (phase_offset_days / sigma_days[None, :]) ** 2)
    return -depth_counts[None, :] * profile


def evaluate_flare_profile(
    time,
    profile: str,
    amplitude,
    peak_time,
    rise_time=None,
    decay_time=None,
    fwhm=None,
):
    """Return peak-amplitude-normalized flare profiles on a time axis."""
    time = np.asarray(time, dtype=float).reshape(-1, 1)
    amplitude = np.atleast_1d(np.asarray(amplitude, dtype=float))
    peak_time = np.atleast_1d(np.asarray(peak_time, dtype=float))
    if time.size == 0 or amplitude.size != peak_time.size or not np.isfinite(time).all():
        raise ValueError("time must be finite and amplitude/peak_time arrays must have equal nonzero length")
    if not np.isfinite(amplitude).all() or not np.isfinite(peak_time).all():
        raise ValueError("flare amplitudes and peak times must be finite")
    offset = time - peak_time[None, :]

    def positive(values, name):
        values = np.atleast_1d(np.asarray(values, dtype=float))
        if values.size != amplitude.size or not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError(f"{name} must contain one finite positive value per flare")
        return values

    if profile == "flargauss":
        fwhm = positive(fwhm, "fwhm")
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        shape = np.exp(-0.5 * (offset / sigma) ** 2)
    elif profile == "flarexpd":
        decay_time = positive(decay_time, "decay_time")
        shape = np.where(offset >= 0.0, np.exp(-offset / decay_time), 0.0)
    elif profile == "flarfred":
        rise_time = positive(rise_time, "rise_time")
        decay_time = positive(decay_time, "decay_time")
        shape = np.where(offset < 0.0, np.exp(offset / rise_time), np.exp(-offset / decay_time))
    elif profile == "flardav":
        fwhm = positive(fwhm, "fwhm")
        phase = offset / fwhm
        rise = 1.0 + 1.941 * phase - 0.175 * phase**2 - 2.246 * phase**3 - 1.125 * phase**4
        decay = 0.689 * np.exp(-1.600 * phase) + 0.303 * np.exp(-0.2783 * phase)
        shape = np.where(phase < -1.0, 0.0, np.where(phase <= 0.0, rise, decay))
        shape = np.clip(shape, 0.0, None)
    else:
        flare_profile_parameters(profile)
        raise AssertionError("unreachable")
    return amplitude[None, :] * shape


__all__ = [
    "FLARE_PROFILE_PARAMETERS",
    "ROTATING_SPOT_PARAMETERS",
    "evaluate_flare_profile",
    "evaluate_rotating_spot_profile",
    "flare_profile_parameters",
    "rotating_spot_parameters",
    "log_likelihood_transit_times",
]