"""Convergence summaries for completed PCAT posterior states."""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm


def gelman_rubin(chains):
    """Return the potential scale reduction factor across parallel chains."""
    values = np.asarray(chains, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2:
        return np.nan
    sample_count = values.shape[0]
    within = np.mean(np.var(values, axis=0, ddof=1))
    between = sample_count * np.var(np.mean(values, axis=0), ddof=1)
    if within == 0.:
        return np.inf if between > 0. else np.nan
    variance = (sample_count - 1.) / sample_count * within + between / sample_count
    return float(np.sqrt(variance / within))


def autocorrelation_time(samples, typeverb=0, atcrtype='maxm', verbtype=None):
    """Return correlation sequences and integrated times for a sampled series."""
    values = np.asarray(samples, dtype=float)
    if values.ndim < 1 or values.shape[0] < 2:
        return np.full(values.shape[1:] + (1,), np.nan), np.full(values.shape[1:], np.nan)
    parameter_shape = values.shape[1:]
    flat = values.reshape(values.shape[0], -1)
    lag_count = max(1, values.shape[0] // 2)
    correlations = np.empty((flat.shape[1], lag_count))
    times = np.empty(flat.shape[1])
    for parameter_index, series in enumerate(flat.T):
        centered = series - np.mean(series)
        variance = np.dot(centered, centered)
        if variance == 0.:
            correlations[parameter_index] = np.nan
            times[parameter_index] = np.nan
            continue
        correlation = np.correlate(centered, centered, mode='full')[series.size - 1:series.size - 1 + lag_count]
        correlation /= variance
        correlations[parameter_index] = correlation
        stop_indices = np.where(correlation[1:] <= 0.)[0]
        stop = stop_indices[0] + 1 if stop_indices.size > 0 else correlation.size
        times[parameter_index] = 1. + 2. * np.sum(correlation[1:stop])
    return correlations.reshape(parameter_shape + (lag_count,)), times.reshape(parameter_shape)


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


def estimate_evidence(posterior, log_likelihood, prior_types, prior_minima,
                      prior_maxima, prior_means=None, prior_stdvs=None,
                      sample_count=4000, seed=None):
    """Estimate normalized evidence with independent defensive importance draws.

    The PCAT posterior only fits the proposal. The separate importance draws
    keep the estimate valid conditional on that fit; the prior component gives
    the proposal support everywhere the prior has support.
    """
    posterior = np.asarray(posterior, dtype=float)
    if posterior.ndim != 2 or posterior.shape[0] < 2 or not np.isfinite(posterior).all():
        raise ValueError('posterior must contain at least two finite parameter vectors.')
    dimension = posterior.shape[1]
    prior_types = np.asarray(prior_types)
    prior_minima = np.asarray(prior_minima, dtype=float)
    prior_maxima = np.asarray(prior_maxima, dtype=float)
    prior_means = np.zeros(dimension) if prior_means is None else np.asarray(prior_means, dtype=float)
    prior_stdvs = np.ones(dimension) if prior_stdvs is None else np.asarray(prior_stdvs, dtype=float)
    if any(values.shape != (dimension,) for values in
           (prior_types, prior_minima, prior_maxima, prior_means, prior_stdvs)):
        raise ValueError('Prior arrays must have one entry per parameter.')
    bounded = prior_types == 'self'
    gaussian = prior_types == 'gaus'
    if not np.all(bounded | gaussian) or np.any(prior_maxima[bounded] <= prior_minima[bounded]) or np.any(prior_stdvs[gaussian] <= 0):
        raise ValueError('Evidence requires normalized bounded or Gaussian priors.')
    if sample_count < 2:
        raise ValueError('sample_count must be at least two.')

    rng = np.random.default_rng(seed)
    scale = np.maximum(np.std(posterior, axis=0), 1e-8)
    covariance = np.atleast_2d(np.cov(posterior, rowvar=False))
    covariance += np.diag((0.1 * scale) ** 2)
    proposal = multivariate_normal(mean=np.mean(posterior, axis=0), cov=covariance)
    draws = np.empty((sample_count, dimension))
    draws[:, bounded] = rng.uniform(prior_minima[bounded], prior_maxima[bounded],
                                    size=(sample_count, np.sum(bounded)))
    draws[:, gaussian] = rng.normal(prior_means[gaussian], prior_stdvs[gaussian],
                                    size=(sample_count, np.sum(gaussian)))
    use_gaussian = rng.random(sample_count) < 0.5
    draws[use_gaussian] = rng.multivariate_normal(proposal.mean, covariance, size=np.sum(use_gaussian))

    log_prior = np.zeros(sample_count)
    if np.any(bounded):
        inside = np.all((draws[:, bounded] >= prior_minima[bounded]) &
                        (draws[:, bounded] <= prior_maxima[bounded]), axis=1)
        log_prior[~inside] = -np.inf
        log_prior[inside] -= np.sum(np.log(prior_maxima[bounded] - prior_minima[bounded]))
    if np.any(gaussian):
        log_prior += np.sum(norm.logpdf(draws[:, gaussian], prior_means[gaussian],
                                        prior_stdvs[gaussian]), axis=1)
    log_proposal = np.logaddexp(np.log(0.5) + log_prior,
                                np.log(0.5) + proposal.logpdf(draws))
    log_weights = np.full(sample_count, -np.inf)
    for index in np.flatnonzero(np.isfinite(log_prior)):
        log_weights[index] = float(log_likelihood(draws[index])) + log_prior[index] - log_proposal[index]
    if np.any(np.isnan(log_weights)) or np.any(np.isposinf(log_weights)):
        raise ValueError('Likelihood produced an invalid importance weight.')
    if not np.any(np.isfinite(log_weights)):
        raise ValueError('All evidence importance weights are zero.')
    log_evidence = logsumexp(log_weights) - np.log(sample_count)
    effective_sample_size = np.exp(2 * logsumexp(log_weights) - logsumexp(2 * log_weights))
    relative_error = np.sqrt(max(sample_count / effective_sample_size - 1, 0) / (sample_count - 1))
    return {'log_evidence': float(log_evidence), 'relative_error': float(relative_error),
            'effective_sample_size': float(effective_sample_size)}