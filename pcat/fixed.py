"""Fixed-dimensional posterior sampling through PCAT's native pipeline."""

from tdpy.verbosity import print

from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import cloudpickle
import numpy as np

from .diagnostics import estimate_evidence


_callback_cache = {}


def _legacy_likelihood(gdat, strgmodl, values):
    payload = gdat.legacy_payload
    key = id(payload)
    if key not in _callback_cache or _callback_cache[key][0] is not payload:
        if len(_callback_cache) >= 8:
            _callback_cache.pop(next(iter(_callback_cache)))
        _callback_cache[key] = (payload, cloudpickle.loads(payload))
    state, log_likelihood, log_prior = _callback_cache[key][1]
    result = log_likelihood(values, state)
    if log_prior is not None:
        result += log_prior(values, state)
        gaussian = np.asarray(gdat.legacy_scalpara) == 'gaus'
        result += 0.5 * np.sum(((values[gaussian] - gdat.legacy_mean[gaussian]) /
                                gdat.legacy_stdv[gaussian]) ** 2)
    return result


def sample_fixed(**kwargs):
    """Run a generic fixed-dimensional model with PCAT's standard sampler."""
    from .main import sample

    return sample(typeexpr='gener', **kwargs)


def sample_fixed_chains(state, log_likelihood, log_prior, names, scales, minima, maxima,
                        means, stdvs, initial, chain_count, sample_count, burn_count,
                        pathbase=None, typeverb=0, estimate_log_evidence=False,
                        evidence_samples=4000, seed=None):
    """Return PCAT chains in legacy walker-first order, with optional evidence."""
    if estimate_log_evidence and log_prior is not None:
        raise ValueError('Evidence needs a normalized prior; custom priors require a density and sampler.')
    means = np.zeros(len(names)) if means is None else np.asarray(means)
    stdvs = np.ones(len(names)) if stdvs is None else np.asarray(stdvs)
    prior_types = tuple('self' if scale == 'logt' else scale for scale in scales)
    output = nullcontext(pathbase) if pathbase is not None else TemporaryDirectory(prefix='pcat-fixed-')
    with output as root:
        result = sample_fixed(
            retr_llik=_legacy_likelihood, parameter_names=tuple(names),
            prior_types=prior_types, prior_minima=minima, prior_maxima=maxima,
            prior_means=means, prior_stdvs=stdvs,
            initial_values=np.mean(initial, axis=0),
            legacy_payload=cloudpickle.dumps((state, log_likelihood, log_prior)),
            legacy_scalpara=np.asarray(scales), legacy_mean=means, legacy_stdv=stdvs,
            numbproc=chain_count, numbswep=sample_count + burn_count,
            numbburn=burn_count, numbsamp=sample_count, pathbase=root,
            strgcnfg='fixed_' + uuid4().hex[:16], typeverb=typeverb,
        )
        chain = np.asarray(result.listpostparagenrscalbase).reshape(chain_count, sample_count, -1)
        logprob = np.asarray(result.listpostlpostotl).reshape(chain_count, sample_count)
        if estimate_log_evidence:
            evidence = estimate_evidence(
                chain.reshape(-1, len(names)), lambda values: log_likelihood(values, state),
                prior_types, minima, maxima, means, stdvs,
                sample_count=evidence_samples, seed=seed,
            )
            return chain, logprob, evidence
    return chain, logprob


def _allesfitter_likelihood(values, datadir):
    from allesfitter import config
    from allesfitter.mcmc import mcmc_lnlike

    if not hasattr(config, 'BASEMENT') or config.BASEMENT.datadir != datadir:
        config.init(datadir)
    try:
        value = float(mcmc_lnlike(values))
    except Exception:
        return -np.inf
    return value if np.isfinite(value) else -np.inf


def sample_allesfitter_pcat(datadir):
    """Fit an allesfitter model with PCAT and persist its existing HDF contract."""
    import h5py
    from allesfitter import config

    config.init(datadir)
    basement = config.BASEMENT
    bounds = basement.bounds
    prior_types = []
    minima = []
    maxima = []
    means = []
    stdvs = []
    for bound in bounds:
        if bound[0] == 'uniform':
            prior_types.append('self')
            minima.append(bound[1])
            maxima.append(bound[2])
            means.append(0.)
            stdvs.append(1.)
        elif bound[0] == 'normal':
            prior_types.append('gaus')
            means.append(bound[1])
            stdvs.append(bound[2])
            minima.append(bound[1] - 10 * bound[2])
            maxima.append(bound[1] + 10 * bound[2])
        else:
            raise ValueError('PCAT requires normalized uniform or normal allesfitter priors.')
    settings = basement.settings
    walker_count = int(settings['mcmc_nwalkers'])
    step_count = int(settings['mcmc_total_steps']) // int(settings['mcmc_thin_by'])
    chain, logprob = sample_fixed_chains(
        datadir, _allesfitter_likelihood, None,
        ['parameter_%d' % index for index in range(len(bounds))], prior_types,
        np.asarray(minima), np.asarray(maxima), np.asarray(means), np.asarray(stdvs),
        np.asarray(basement.theta_0)[None, :], walker_count, step_count,
        0, datadir, 0,
    )
    path = Path(basement.outdir) / 'mcmc_save.h5'
    path.parent.mkdir(parents=True, exist_ok=True)
    print('Writing to %s...' % path)
    with h5py.File(path, 'w') as output:
        group = output.create_group('mcmc')
        group.attrs.update(version='pcat-compat', nwalkers=walker_count,
                           ndim=len(bounds), has_blobs=False, iteration=step_count)
        group.create_dataset('chain', data=chain.transpose(1, 0, 2), maxshape=(None, walker_count, len(bounds)))
        group.create_dataset('log_prob', data=logprob.T, maxshape=(None, walker_count))
        group.create_dataset('accepted', data=np.zeros(walker_count))
    return path