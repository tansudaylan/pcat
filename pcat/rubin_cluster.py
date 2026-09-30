"""Poisson image likelihood for a simulated cluster-scale lens."""

from __future__ import annotations

import numpy as np

from .roman_lens import RomanLensConfig, render_lens


def predicted_counts(
    config: RomanLensConfig,
    parameters: np.ndarray,
    source_size: float,
    source_axis_ratio: float,
    source_angle: float,
) -> np.ndarray:
    """Return PSF-convolved source and uniform sky counts per pixel."""
    return render_lens(
        config, *parameters, source_size, source_axis_ratio, source_angle
    ) + config.background


def log_likelihood(state: object, model_name: str, parameters: np.ndarray) -> float:
    """Evaluate the Poisson pixel likelihood, dropping data-only terms."""
    model_counts = predicted_counts(
        state.rubin_config,
        parameters,
        state.rubin_source_size,
        state.rubin_source_axis_ratio,
        state.rubin_source_angle,
    )
    observed_counts = state.rubin_observed_counts
    return float(np.sum(observed_counts * np.log(model_counts) - model_counts))