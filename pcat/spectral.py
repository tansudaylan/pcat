"""Spectral-line profiles and line-spread-function convolution for PCAT."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, wofz


LINE_PROFILE_PARAMETERS = {
    "gaus": ("flux", "elin", "sigm"),
    "lore": ("flux", "elin", "gamm"),
    "voig": ("flux", "elin", "sigm", "gamm"),
    "pvoi": ("flux", "elin", "sigm", "gamm", "frac"),
    "sinc": ("flux", "elin", "sigm"),
    "skew": ("flux", "elin", "sigm", "skew"),
    "toph": ("flux", "elin", "sigm"),
    "edis": ("flux", "elin"),
}


def spectral_profile_parameters(profile: str) -> tuple[str, ...]:
    """Return sampled element parameters for one supported line profile."""
    try:
        return LINE_PROFILE_PARAMETERS[profile]
    except KeyError as exception:
        raise ValueError(
            f"Unknown line profile {profile!r}; choose from {tuple(LINE_PROFILE_PARAMETERS)}."
        ) from exception


def evaluate_line_profile(
    axis,
    profile: str,
    flux,
    center,
    gaussian_width=None,
    lorentz_width=None,
    mixing_fraction=None,
    skewness=None,
):
    """Return integrated-flux-normalized line profiles on a one-dimensional axis."""
    axis = np.asarray(axis, dtype=float).reshape(-1, 1)
    flux = np.atleast_1d(np.asarray(flux, dtype=float))
    center = np.atleast_1d(np.asarray(center, dtype=float))
    if axis.size == 0 or flux.size != center.size or not np.isfinite(axis).all():
        raise ValueError("axis must be finite and flux/center arrays must have equal nonzero length")
    if not np.isfinite(flux).all() or not np.isfinite(center).all():
        raise ValueError("line fluxes and centers must be finite")
    offset = axis - center[None, :]

    def positive(values, name):
        values = np.atleast_1d(np.asarray(values, dtype=float))
        if values.size != flux.size or not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError(f"{name} must contain one finite positive value per line")
        return values

    if profile in ("gaus", "voig", "pvoi", "sinc", "skew", "toph", "edis"):
        gaussian_width = positive(gaussian_width, "gaussian_width")
    if profile in ("lore", "voig", "pvoi"):
        lorentz_width = positive(lorentz_width, "lorentz_width")

    if profile in ("gaus", "edis"):
        shape = np.exp(-0.5 * (offset / gaussian_width) ** 2) / (gaussian_width * np.sqrt(2.0 * np.pi))
    elif profile == "lore":
        shape = lorentz_width / (np.pi * (offset**2 + lorentz_width**2))
    elif profile == "voig":
        argument = (offset + 1j * lorentz_width) / (np.sqrt(2.0) * gaussian_width)
        shape = np.real(wofz(argument)) / (gaussian_width * np.sqrt(2.0 * np.pi))
    elif profile == "pvoi":
        fraction = np.atleast_1d(np.asarray(mixing_fraction, dtype=float))
        if fraction.size != flux.size or np.any((fraction < 0.0) | (fraction > 1.0)):
            raise ValueError("mixing_fraction must lie between zero and one for every line")
        gaussian = np.exp(-0.5 * (offset / gaussian_width) ** 2) / (gaussian_width * np.sqrt(2.0 * np.pi))
        lorentzian = lorentz_width / (np.pi * (offset**2 + lorentz_width**2))
        shape = (1.0 - fraction) * gaussian + fraction * lorentzian
    elif profile == "sinc":
        shape = np.sinc(offset / gaussian_width) ** 2 / gaussian_width
    elif profile == "skew":
        skewness = np.atleast_1d(np.asarray(skewness, dtype=float))
        if skewness.size != flux.size or not np.isfinite(skewness).all():
            raise ValueError("skewness must contain one finite value per line")
        standardized = offset / gaussian_width
        gaussian = np.exp(-0.5 * standardized**2) / (gaussian_width * np.sqrt(2.0 * np.pi))
        shape = 2.0 * gaussian * ndtr(skewness * standardized)
    elif profile == "toph":
        shape = (np.abs(offset) <= 0.5 * gaussian_width) / gaussian_width
    else:
        spectral_profile_parameters(profile)
        raise AssertionError("unreachable")
    return flux[None, :] * shape


def gaussian_lsf_kernel(sigma_samples: float, truncate: float = 4.0) -> np.ndarray:
    """Return a normalized odd-length Gaussian line-spread kernel in sample units."""
    if not np.isfinite(sigma_samples) or sigma_samples <= 0.0:
        raise ValueError("sigma_samples must be finite and positive")
    if not np.isfinite(truncate) or truncate <= 0.0:
        raise ValueError("truncate must be finite and positive")
    radius = max(1, int(np.ceil(truncate * sigma_samples)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_samples) ** 2)
    return kernel / np.sum(kernel)


def apply_line_spread_function(spectrum, kernel) -> np.ndarray:
    """Convolve spectra along axis zero while preserving each column's total flux."""
    spectrum = np.asarray(spectrum, dtype=float)
    kernel = np.asarray(kernel, dtype=float).reshape(-1)
    if spectrum.ndim not in (1, 2) or not np.isfinite(spectrum).all():
        raise ValueError("spectrum must be a finite one- or two-dimensional array")
    if kernel.size < 1 or kernel.size % 2 == 0 or not np.isfinite(kernel).all() or np.any(kernel < 0.0) or np.sum(kernel) <= 0.0:
        raise ValueError("kernel must have finite nonnegative normalization and odd length")
    kernel = kernel / np.sum(kernel)
    values = spectrum[:, None] if spectrum.ndim == 1 else spectrum
    convolved = np.empty_like(values)
    for index in range(values.shape[1]):
        full = np.convolve(values[:, index], kernel, mode="full")
        start = (kernel.size - 1) // 2
        convolved[:, index] = full[start:start + values.shape[0]]
    before = np.sum(values, axis=0)
    after = np.sum(convolved, axis=0)
    valid = after != 0.0
    convolved[:, valid] *= before[valid] / after[valid]
    return convolved[:, 0] if spectrum.ndim == 1 else convolved


def apply_gaussian_resolving_power(axis, spectrum, centers, resolving_power: float) -> np.ndarray:
    """Convolve each spectral-line column at constant resolving power."""
    axis = np.asarray(axis, dtype=float).reshape(-1)
    values = np.asarray(spectrum, dtype=float)
    centers = np.atleast_1d(np.asarray(centers, dtype=float))
    if axis.size < 2 or values.ndim != 2 or values.shape != (axis.size, centers.size):
        raise ValueError("spectrum must have one row per axis sample and one column per center")
    spacing = np.median(np.diff(axis))
    if not np.allclose(np.diff(axis), spacing, rtol=1e-3, atol=0.0) or spacing <= 0.0:
        raise ValueError("constant-resolving-power convolution requires a uniformly increasing axis")
    if not np.isfinite(resolving_power) or resolving_power <= 0.0:
        raise ValueError("resolving_power must be finite and positive")
    output = np.empty_like(values)
    for index, center in enumerate(centers):
        sigma_samples = center / (2.0 * np.sqrt(2.0 * np.log(2.0)) * resolving_power * spacing)
        output[:, index] = apply_line_spread_function(values[:, index], gaussian_lsf_kernel(sigma_samples))
    return output


__all__ = [
    "LINE_PROFILE_PARAMETERS",
    "apply_gaussian_resolving_power",
    "apply_line_spread_function",
    "evaluate_line_profile",
    "gaussian_lsf_kernel",
    "spectral_profile_parameters",
]