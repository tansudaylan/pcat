"""Convergence summaries for completed PCAT posterior states."""

import numpy as np


def posterior_convergence(state, max_rhat=1.05, min_effective_sample_size=200.0):
    """Return convergence metrics and whether every parameter passes."""
    rhat = np.asarray(state.gmrbparagenrscalbase, dtype=float)
    autocorrelation_time = np.asarray(state.timeatcrpara, dtype=float)
    sample_count = int(state.numbproc) * int(state.numbsamp)
    effective_sample_size = sample_count / np.nanmax(
        autocorrelation_time,
        axis=0,
    )
    finite = np.all(np.isfinite(rhat)) and np.all(
        np.isfinite(effective_sample_size)
    )
    converged = bool(
        finite
        and np.all(rhat <= max_rhat)
        and np.all(effective_sample_size >= min_effective_sample_size)
    )
    return {
        "converged": converged,
        "rhat": rhat,
        "effective_sample_size": effective_sample_size,
        "max_rhat": float(np.max(rhat)),
        "min_effective_sample_size": float(np.min(effective_sample_size)),
    }