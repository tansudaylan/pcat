"""Reusable forward models for astronomical image formation."""

from __future__ import annotations

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve_fft


def forward_model_image(
    source_image: np.ndarray,
    psf_sigma_pixels: float,
    background: float = 0.0,
) -> dict[str, np.ndarray]:
    """Convolve a two-dimensional source scene with a normalized Gaussian PSF."""

    source_image = np.asarray(source_image, dtype=float)
    if source_image.ndim != 2:
        raise ValueError("source_image must be two-dimensional.")
    if not np.isfinite(source_image).all():
        raise ValueError("source_image must contain only finite values.")
    if not np.isfinite(psf_sigma_pixels) or psf_sigma_pixels <= 0.0:
        raise ValueError("psf_sigma_pixels must be finite and positive.")
    if not np.isfinite(background):
        raise ValueError("background must be finite.")

    psf_kernel = Gaussian2DKernel(x_stddev=psf_sigma_pixels)
    observed_image = convolve_fft(
        source_image,
        psf_kernel,
        boundary="fill",
        fill_value=0.0,
        normalize_kernel=True,
    )
    observed_image += background
    return {
        "source_image": source_image,
        "psf_kernel": np.asarray(psf_kernel.array),
        "observed_image": observed_image,
    }