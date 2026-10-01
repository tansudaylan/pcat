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
    if (time_days.size == 0 or not np.isfinite(time_days).all()
            or depth_counts.size != phase_epoch_days.size or depth_counts.size != fwhm_days.size):
        raise ValueError("time and spot parameter arrays must be finite and have matching sizes")
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


def retr_dictpropelemtmpl(residual, variance, templates, flux_minimum, flux_maximum,
                          index_position, index_flux, number_parameters, fraction_uniform=0.1,
                          fraction_gaussian=0.5, stdv_log_flux=0.5, exponent=1.0):
    """Return PCAT keywords for a data-informed birth, death, and jump density.

    ``templates`` holds one unit-amplitude element profile per cell of a grid
    uniform in the position unit interval. Positions are drawn with weight
    proportional to the matched-filter chi-squared reduction mixed with a
    uniform fraction; log fluxes [same unit as ``residual``] are drawn around
    the matched-filter amplitude, mixed with the prior. Other element
    parameters are drawn from the prior. The density does not depend on the
    sampler state, so PCAT's Hastings term keeps the chain exact.
    """
    residual = np.asarray(residual, dtype=float).ravel()
    weight = 1.0 / np.asarray(variance, dtype=float).ravel()
    templates = np.asarray(templates, dtype=float)
    if templates.ndim != 2 or templates.shape[1] != residual.size or weight.size != residual.size:
        raise ValueError("templates must have shape (grid, data) matching residual and variance")
    if not 0.0 < fraction_uniform <= 1.0 or not 0.0 <= fraction_gaussian <= 1.0:
        raise ValueError("proposal mixture fractions must lie in the unit interval")
    norm = templates**2 @ weight
    amplitude = (templates @ (weight * residual)) / norm  # [same unit as residual]
    delta_chi2 = np.maximum(amplitude, 0.0) ** 2 * norm
    score = delta_chi2**exponent
    density = fraction_uniform + (1.0 - fraction_uniform) * score / max(np.mean(score), 1e-300)
    log_range = np.log(flux_maximum / flux_minimum)
    unit_flux = np.log(np.clip(amplitude, flux_minimum, flux_maximum) / flux_minimum) / log_range
    return dict(
        pdfnposipropelem=density,
        cdfnposipropelem=np.cumsum(density) / np.sum(density),
        numbposipropelem=density.size,
        unitfluxpropelem=unit_flux,
        stdvunitfluxpropelem=stdv_log_flux / log_range,
        fracgauspropelem=fraction_gaussian,
        indxposipropelem=int(index_position),
        indxfluxpropelem=int(index_flux),
        numbparapropelem=int(number_parameters),
        retr_drawpropelem=retr_drawpropelemtmpl,
        retr_lpdfpropelem=retr_lpdfpropelemtmpl,
    )


def _retr_pdfntruncunit(unit, mean, stdv):
    """Normal density truncated to the unit interval."""
    from scipy.stats import norm

    mass = norm.cdf((1.0 - mean) / stdv) - norm.cdf(-mean / stdv)
    return np.exp(-0.5 * ((unit - mean) / stdv) ** 2) / (np.sqrt(2.0 * np.pi) * stdv * mass)


def retr_drawpropelemtmpl(gdat):
    """Draw unit-cube element parameters from the template-matched density."""
    from scipy.stats import truncnorm

    index = min(int(np.searchsorted(gdat.cdfnposipropelem, np.random.rand())),
                gdat.numbposipropelem - 1)
    unit = np.random.rand(gdat.numbparapropelem)
    unit[gdat.indxposipropelem] = (index + np.random.rand()) / gdat.numbposipropelem
    if np.random.rand() < gdat.fracgauspropelem:
        mean = gdat.unitfluxpropelem[index]
        stdv = gdat.stdvunitfluxpropelem
        unit[gdat.indxfluxpropelem] = truncnorm.rvs(-mean / stdv, (1.0 - mean) / stdv, loc=mean, scale=stdv)
    return unit


def retr_lpdfpropelemtmpl(gdat, unit):
    """Return the log density [unit cube] of the template-matched element proposal."""
    index = min(int(unit[gdat.indxposipropelem] * gdat.numbposipropelem), gdat.numbposipropelem - 1)
    density_position = gdat.pdfnposipropelem[index] / np.mean(gdat.pdfnposipropelem)
    fraction = gdat.fracgauspropelem
    density_flux = 1.0 - fraction + fraction * _retr_pdfntruncunit(
        unit[gdat.indxfluxpropelem], gdat.unitfluxpropelem[index], gdat.stdvunitfluxpropelem
    )
    return float(np.log(density_position) + np.log(density_flux))


__all__ = [
    "FLARE_PROFILE_PARAMETERS",
    "ROTATING_SPOT_PARAMETERS",
    "evaluate_flare_profile",
    "evaluate_rotating_spot_profile",
    "flare_profile_parameters",
    "rotating_spot_parameters",
    "log_likelihood_transit_times",
    "retr_dictpropelemtmpl",
    "retr_drawpropelemtmpl",
    "retr_lpdfpropelemtmpl",
]