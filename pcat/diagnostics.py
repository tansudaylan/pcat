"""Convergence summaries for completed PCAT posterior states."""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm
from tdpy.numerics import logmeanexp


def binomial_wilson_interval(successes, trials, z_score=1.0):
    """Return the Wilson interval for finite scalar binomial counts.

    Counts must be unmasked integer-valued scalars. The confidence score must
    be an unmasked positive finite real scalar.
    Zero and full counts use closed-form limits to avoid cancellation.
    Scores whose squares exceed binary64 range use reciprocal-score scaling.
    Large trial counts use count scaling to avoid squaring the trial count.
    """
    scalars = []
    for value in (successes, trials, z_score):
        if np.ma.is_masked(value):
            raise ValueError("Require valid binomial counts and a positive z_score.")
        scalar = np.asarray(value)
        if scalar.ndim != 0 or scalar.dtype.kind not in 'biuf' or not np.isfinite(scalar):
            raise ValueError("Require valid binomial counts and a positive z_score.")
        scalars.append(scalar.item())
    successes, trials, z_score = scalars
    if any(isinstance(count, (float, np.floating)) and count != np.floor(count)
           for count in (successes, trials)):
        raise ValueError("Require valid binomial counts and a positive z_score.")
    successes, trials = int(successes), int(trials)
    if trials <= 0 or not 0 <= successes <= trials or z_score <= 0.0:
        raise ValueError("Require valid binomial counts and a positive z_score.")
    fraction = successes / trials
    if z_score > np.sqrt(np.finfo(float).max):
        scaled_trials = (trials / z_score) / z_score
        complement = 1.0 - fraction
        root = float(np.sqrt(1.0 + 4.0 * fraction * complement * scaled_trials))
        lower = fraction**2 * scaled_trials / (0.5 + fraction * scaled_trials + 0.5 * root)
        upper = 1.0 - complement**2 * scaled_trials / (0.5 + complement * scaled_trials + 0.5 * root)
        return float(lower), float(upper)
    if trials > np.sqrt(np.finfo(float).max):
        scaled_score = z_score / trials**0.5
        ratio = scaled_score**2
        denominator = 1.0 + ratio
        if successes == 0:
            return 0.0, float(ratio / denominator)
        if successes == trials:
            return float(1.0 / denominator), 1.0
        complement = (trials - successes) / trials
        radius = float(np.hypot(np.sqrt(fraction) * np.sqrt(complement), scaled_score / 2.0))
        spread = scaled_score * radius
        upper_numerator = fraction + ratio / 2.0 + spread
        lower = fraction * (fraction / upper_numerator)
        return float(lower), float(upper_numerator / denominator)
    z_squared = z_score**2
    if successes == 0:
        return 0.0, float(z_squared / (trials + z_squared))
    if successes == trials:
        return float(trials / (trials + z_squared)), 1.0
    denominator = 1.0 + z_squared / trials
    center = (fraction + z_squared / (2.0 * trials)) / denominator
    half_width = z_score * np.sqrt(
        fraction * (1.0 - fraction) / trials + z_squared / (4.0 * trials**2)
    ) / denominator
    return float(center - half_width), float(center + half_width)


def _normalize_trace_scale(values):
    """Rescale only extreme finite traces by an exact power of two."""
    amplitude = np.max(np.abs(values))
    if amplitude != 0.0 and (amplitude < np.sqrt(np.finfo(float).tiny)
                            or amplitude > np.sqrt(np.finfo(float).max / values.size) / 2.0):
        exponent = int(np.frexp(amplitude)[1])
        with np.errstate(under='ignore'):
            return np.ldexp(values, -exponent)
    return values


def gelman_rubin(chains):
    """Return the potential scale reduction factor across parallel chains.

    Masked, nonfinite, nonreal, or undersized traces have an undefined result
    (NaN). Missing samples are not filled or dropped.
    Extreme magnitudes use the same binary normalization as autocorrelation.
    """
    if np.ma.is_masked(chains) or np.iscomplexobj(chains):
        return np.nan
    values = np.asarray(chains, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        return np.nan
    if np.all(values == values[0]):
        return np.inf if np.any(values[0] != values[0, 0]) else np.nan
    values = _normalize_trace_scale(values)
    sample_count = values.shape[0]
    within = np.mean(np.var(values, axis=0, ddof=1))
    between = sample_count * np.var(np.mean(values, axis=0), ddof=1)
    if within == 0.:
        return np.inf if between > 0. else np.nan
    variance = (sample_count - 1.) / sample_count * within + between / sample_count
    return float(np.sqrt(variance / within))


def autocorrelation_time(samples, typeverb=0, atcrtype='maxm', verbtype=None):
    """Return correlation sequences and integrated times for sampled series.

    A masked, nonfinite, nonreal, or constant series has NaN outputs without
    discarding valid neighboring series or filling or dropping samples.
    Extreme scales use power-of-two normalization before forming products.
    """
    values = np.ma.asarray(samples)
    if values.ndim < 1 or values.shape[0] < 2:
        return np.full(values.shape[1:] + (1,), np.nan), np.full(values.shape[1:], np.nan)
    parameter_shape = values.shape[1:]
    flat = values.reshape(values.shape[0], -1)
    lag_count = max(1, values.shape[0] // 2)
    correlations = np.full((flat.shape[1], lag_count), np.nan)
    times = np.full(flat.shape[1], np.nan)
    for parameter_index, series in enumerate(flat.T):
        if np.ma.is_masked(series) or np.iscomplexobj(series):
            continue
        series = np.asarray(series, dtype=float)
        if not np.isfinite(series).all():
            continue
        if np.all(series == series[0]):
            continue
        series = _normalize_trace_scale(series)
        centered = series - np.mean(series)
        variance = np.dot(centered, centered)
        if variance == 0.:
            continue
        correlation = np.correlate(centered, centered, mode='full')[series.size - 1:series.size - 1 + lag_count]
        correlation /= variance
        correlations[parameter_index] = correlation
        stop_indices = np.where(correlation[1:] <= 0.)[0]
        stop = stop_indices[0] + 1 if stop_indices.size > 0 else correlation.size
        times[parameter_index] = 1. + 2. * np.sum(correlation[1:stop])
    return correlations.reshape(parameter_shape + (lag_count,)), times.reshape(parameter_shape)


def _evaluate_log_density(callback, samples):
    values = callback(samples)
    if np.ma.is_masked(values) or np.iscomplexobj(values):
        raise ValueError("log-density callbacks must return unmasked real values")
    values = np.asarray(values, dtype=float)
    if values.shape != (samples.shape[0],):
        raise ValueError("log-density callbacks must return one value per draw")
    return values


def estimate_kl_divergence(samples, log_density_p, log_density_q):
    """Estimate forward ``D_KL(P || Q)`` from normalized log densities and draws from P.

    ``samples`` has shape ``(draws, parameters)`` or
    ``(chains, draws, parameters)``. Both callbacks accept a matrix of draws and
    return one normalized log density per row. The estimate is the sample mean
    of ``log p - log q``. Its Monte Carlo standard error accounts for within-chain
    autocorrelation and variation between independent chains. Finite-sample
    estimates are not clipped at zero. Exact power-of-two scaling protects
    representable means and uncertainties from intermediate overflow or underflow.
    """
    if np.ma.is_masked(samples) or np.iscomplexobj(samples):
        raise ValueError("samples must contain unmasked real values")
    values = np.asarray(samples, dtype=float)
    if values.ndim == 2:
        chains = values[None, :, :]
    elif values.ndim == 3:
        chains = values
    else:
        raise ValueError("samples must be draws-by-parameters or chains-by-draws-by-parameters")
    if min(chains.shape) < 1 or chains.shape[1] < 2 or not np.isfinite(chains).all():
        raise ValueError("samples must contain at least two finite draws per chain")

    chain_count, draw_count, parameter_count = chains.shape
    flat_samples = chains.reshape(-1, parameter_count)
    logprobp = _evaluate_log_density(log_density_p, flat_samples)
    logprobq = _evaluate_log_density(log_density_q, flat_samples)
    if (not np.isfinite(logprobp).all() or np.isnan(logprobq).any()
            or np.isposinf(logprobq).any()):
        raise ValueError("log p must be finite and log q must not be NaN or positive infinity")
    if np.isneginf(logprobq).any():
        return {
            "kl_divergence": float("inf"),
            "standard_error": float("inf"),
            "effective_sample_size": 0.0,
        }

    with np.errstate(over='ignore'):
        logratio = logprobp - logprobq
    scale_exponent = 0
    if not np.isfinite(logratio).all():
        scale_exponent = int(np.frexp(max(np.max(np.abs(logprobp)), np.max(np.abs(logprobq))))[1])
        with np.errstate(under='ignore'):
            logratio = np.ldexp(logprobp, -scale_exponent) - np.ldexp(logprobq, -scale_exponent)
    logratio = logratio.reshape(chain_count, draw_count)
    ratio_exponent = int(np.frexp(np.max(np.abs(logratio)))[1])
    with np.errstate(under='ignore'):
        logratio = np.ldexp(logratio, -ratio_exponent)
    scale_exponent += ratio_exponent
    divergence = float(np.ldexp(np.mean(logratio), scale_exponent))
    means = np.mean(logratio, axis=1)
    variance_mean = 0.0
    effective_sample_size = 0.0
    for chain in logratio:
        variance = float(np.var(chain, ddof=1))
        if variance == 0.0:
            timecorr = 1.0
        else:
            _, times = autocorrelation_time(chain[:, None])
            timecorr = float(np.ravel(times)[0])
            if not np.isfinite(timecorr):
                timecorr = 1.0
            timecorr = max(timecorr, 1.0)
        variance_mean += variance * timecorr / (chain_count**2 * draw_count)
        effective_sample_size += draw_count / timecorr

    if chain_count > 1:
        between_chain_variance = float(np.var(means, ddof=1)) / chain_count
        variance_mean = max(variance_mean, between_chain_variance)
    return {
        "kl_divergence": divergence,
        "standard_error": float(np.ldexp(np.sqrt(variance_mean), scale_exponent)),
        "effective_sample_size": float(effective_sample_size),
    }


def estimate_mutual_information(
    samples, log_density_joint, log_density_x, log_density_y, split_index=1,
):
    """Estimate mutual information in nats from joint draws and known densities.

    Columns before ``split_index`` form X and the remaining columns form Y.
    Samples have the same draw/chain layout as ``estimate_kl_divergence``.
    Each normalized log-density callback receives a matrix and returns one
    value per row. Marginal callbacks receive only their respective columns.
    The supplied densities must describe the same joint distribution. No
    marginal density fitting, independence assumption, or zero clipping is
    performed. The KL estimator supplies chain-aware Monte Carlo uncertainty.
    """
    if np.ma.is_masked(samples) or np.iscomplexobj(samples):
        raise ValueError("samples must contain unmasked real values")
    values = np.asarray(samples, dtype=float)
    if values.ndim not in (2, 3):
        raise ValueError("samples must be draws-by-parameters or chains-by-draws-by-parameters")
    if (isinstance(split_index, (bool, np.bool_))
            or not isinstance(split_index, (int, np.integer))
            or not 0 < split_index < values.shape[-1]):
        raise ValueError("split_index must divide the columns into two nonempty groups")

    def log_density_product(draws):
        log_x = _evaluate_log_density(log_density_x, draws[:, :split_index])
        log_y = _evaluate_log_density(log_density_y, draws[:, split_index:])
        if any(np.isnan(density).any() or np.isposinf(density).any()
               for density in (log_x, log_y)):
            raise ValueError("marginal log densities must not be NaN or positive infinity")
        return log_x + log_y

    result = estimate_kl_divergence(values, log_density_joint, log_density_product)
    return {
        "mutual_information": result["kl_divergence"],
        "standard_error": result["standard_error"],
        "effective_sample_size": result["effective_sample_size"],
    }


def catalog_count_transitions(chains):
    """Count within-chain catalog-size transitions; input is samples by chains."""

    counts = np.asarray(chains)
    if counts.ndim != 2 or counts.shape[0] < 2 or counts.shape[1] < 1:
        raise ValueError("Catalog counts must have sample and chain axes")
    if not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("Catalog counts must be finite nonnegative integers")
    transitions = np.zeros((int(counts.max()) + 1,) * 2, dtype=int)
    np.add.at(transitions, (counts[:-1].astype(int), counts[1:].astype(int)), 1)
    return transitions


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
    log_evidence = logmeanexp(log_weights)
    effective_sample_size = np.exp(2 * logsumexp(log_weights) - logsumexp(2 * log_weights))
    relative_error = np.sqrt(max(sample_count / effective_sample_size - 1, 0) / (sample_count - 1))
    return {'log_evidence': float(log_evidence), 'relative_error': float(relative_error),
            'effective_sample_size': float(effective_sample_size)}
