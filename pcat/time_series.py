"""Time-series transient profiles for PCAT element populations."""

from __future__ import annotations

import numpy as np


FLARE_PROFILE_PARAMETERS = {
    "flargauss": ("flux", "elin", "fwhm"),
    "flarexpd": ("flux", "elin", "scalfall"),
    "flarfred": ("flux", "elin", "scalrise", "scalfall"),
    "flardav": ("flux", "elin", "fwhm"),
}


def flare_profile_parameters(profile: str) -> tuple[str, ...]:
    """Return sampled element parameters for one supported flare profile."""
    try:
        return FLARE_PROFILE_PARAMETERS[profile]
    except KeyError as exception:
        raise ValueError(
            f"Unknown flare profile {profile!r}; choose from {tuple(FLARE_PROFILE_PARAMETERS)}."
        ) from exception


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
    "evaluate_flare_profile",
    "flare_profile_parameters",
]