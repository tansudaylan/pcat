"""Fixed-dimensional posterior sampling through PCAT's native pipeline."""

from tdpy.verbosity import print

from contextlib import nullcontext
import os
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from uuid import uuid4

import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from tdpy.util import cdfn_gaus, cdfn_logt, cdfn_self, icdf_gaus, icdf_logt, icdf_self, retr_axis
from tdpy.verbosity import tqdm

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
                        evidence_samples=4000, seed=None, **dictpcat):
    """Return PCAT chains in legacy walker-first order, with optional evidence.

    Extra keyword arguments, e.g. booladaptstdp, are passed to PCAT.
    """
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
            strgcnfg='fixed_' + uuid4().hex[:16], typeverb=typeverb, **dictpcat,
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



# Posterior sampling with a dictionary interface; PCAT is the only sampler in the ecosystem.
def retr_limtpara(scalpara, minmpara, maxmpara, meanpara, stdvpara):
    
    numbpara = len(scalpara)
    limtpara = np.empty((2, numbpara))
    indxpara = np.arange(numbpara)
    for n in indxpara:
        if scalpara[n] == 'self' or scalpara[n] == 'logt':
            limtpara[0, n] = minmpara[n]
            limtpara[1, n] = maxmpara[n]
        if scalpara[n] == 'gaus':
            limtpara[0, n] = meanpara[n] - 10 * stdvpara[n]
            limtpara[1, n] = meanpara[n] + 10 * stdvpara[n]
    
    return limtpara


def retr_lpos(para, *dictlpos):
     
    gdat, indxpara, scalpara, minmpara, maxmpara, meangauspara, stdvgauspara, retr_llik, retr_lpri = dictlpos
    
    boolreje = False
    for k in indxpara:
        if scalpara[k] != 'gaus':
            if para[k] < minmpara[k] or para[k] > maxmpara[k]:
                lpos = -np.inf
                boolreje = True
    
    if not boolreje:
        llik = retr_llik(para, gdat)
        lpri = 0.
        if retr_lpri is None:
            for k in indxpara:
                if scalpara[k] == 'gaus':
                    lpri += -0.5 * ((para[k] - meangauspara[k]) / stdvgauspara[k])**2
        else:
            lpri = retr_lpri(para, gdat)
        lpos = llik + lpri
    
    #print('lpos')
    #print(lpos)
    #print('')
    
    return lpos


def sample_posterior( \
         gdat, \

         numbsampwalk, \

         retr_llik, \
              
         # model parameters
         ## list of names of parameters
         listnamepara, \
         ## list of labels of parameters
         listlablpara, \
         ## list of scalings of parameters
         scalpara, \
         ## list of minima of parameters
         minmpara, \
         ## list of maxima of parameters
         maxmpara, \
         
         # base path for placing plot and data files
         pathbase=None, \
         
         # Boolean flag to enforce a reprocess and overwrite
         boolforcrepr=False, \

         # Boolean flag to make plots
         boolplot=True, \
         
         # callable drawing joint posteriors, e.g. pcat.plot_population_grid; no joint plots if None
         plot_posterior=None, \
         
         numbsamppostwalk=None, \

         meangauspara=None, \
         
         stdvgauspara=None, \
         
         retr_lpri=None, \
         
         # Boolean flag to turn on multiprocessing
         boolmult=False, \
         
         # burn-in
         ## number of samples in a precursor run whose final state will be used as the initial state of the actual sampler
         numbsampburnwalkinit=0, \
         
         ## number of initial samples to be burned
         numbsampburnwalk=0, \
         
         # function to return derived variables from the parameter vector
         retr_dictderi=None, \
         
         # dictionary of labels and scalings for derived parameters
         dictlablscalparaderi=None, \
         
         # Boolean flag to use tqdm to report the percentage of completion
         booltqdm=True, \

         # Boolean flag to diagnose the code
         booldiag=True, \
         
         # a string used to save and retrieve results
         strgextn='', \
         
         typesamp='mcmc', \

         # type of the file for plots
         typefileplot='png', \
         
         # type of verbosity
         ## -1: absolutely no text
         ##  0: no text output except critical warnings
         ##  1: minimal description of the execution
         ##  2: detailed description of the execution
         typeverb=1, \
        ):
    '''
    Sample a fixed-dimensional posterior with PCAT from a likelihood callable and simple priors.

    Chains start from a small ball around the prior center, or around a saved posterior median, and
    the thinned post-burn-in samples are returned as a dictionary with derived variables, if requested.
    '''
    
    numbpara = len(listlablpara)
   
    if numbsampwalk <= numbsampburnwalk:
        raise Exception('Burn-in samples cannot outnumber samples.')
    
    if isinstance(minmpara, list):
        minmpara = np.array(minmpara)

    if isinstance(maxmpara, list):
        maxmpara = np.array(maxmpara)

    if numbpara != minmpara.size:
        raise Exception('')
    if numbpara != maxmpara.size:
        raise Exception('')
    
    if dictlablscalparaderi is not None and retr_dictderi is None: 
        raise Exception('')

    indxpara = np.arange(numbpara)
    
    if typesamp == 'mcmc':
        numbwalk = max(20, 2 * numbpara)
        indxwalk = np.arange(numbwalk)
        numbsamptotl = numbsampwalk * numbwalk
        
        if typeverb > 0:
            print('numbsampwalk')
            print(numbsampwalk)
            print('numbwalk')
            print(numbwalk)
            print('numbsamptotl')
            print(numbsamptotl)

    pathvisu = None
    if pathbase is not None:
        pathbasesamp = pathbase + '%s/' % typesamp
        pathvisu = pathbasesamp + 'visuals/'
        pathdata = pathbasesamp + 'data/'
        Path(pathvisu).mkdir(parents=True, exist_ok=True)
        Path(pathdata).mkdir(parents=True, exist_ok=True)

    # plotting
    ## plot limits 
    limtpara = retr_limtpara(scalpara, minmpara, maxmpara, meangauspara, stdvgauspara)

    ## plot bins
    numbbins = 20
    indxbins = np.arange(numbbins)
    binsgrid = np.empty((numbbins + 1, numbpara)) 
    midpgrid = np.empty((numbbins, numbpara)) 
    numbpntsgrid = np.empty(numbpara, dtype=int)
    for k in indxpara:
        if limtpara[0, k] == limtpara[1, k]:
            raise Exception('')
        
        binsgrid[:, k], midpgrid[:, k], deltgrid, numbpntsgrid[k], indx = retr_axis(limt=limtpara[:, k], numbpntsgrid=numbbins)
    
    for k in indxpara:
        if minmpara[k] >= maxmpara[k]:
            print('')
            print('')
            print('')
            print('manmpara')
            print(maxxpara)
            print('minmpara')
            print(minmpara)
            raise Exception('minmpara > maxmpara')
    
    if numbsamppostwalk is None:
        numbsamppostwalk = 100

    if typeverb > 0:
        print('listnamepara')
        print(listnamepara)
        print('listlablpara')
        print(listlablpara)
        print('scalpara')
        print(scalpara)
        if minmpara is not None:
            print('minmpara')
            print(minmpara)
            print('maxmpara')
            print(maxmpara)
        if meangauspara is not None:
            print('meangauspara')
            print(meangauspara)
            print('stdvgauspara')
            print(stdvgauspara)
    
    if strgextn != '':
        strgextn = '_%s' % strgextn
    
    # path of the posterior
    if pathbase is not None:
        pathpost = pathdata + 'postpara%s.csv' % strgextn
        pathdict = pathdata + 'samppostpara%s.csv' % strgextn
        pathpickderi = pathdata + 'postderi%s.pickle' % strgextn
    
    if boolforcrepr or not boolforcrepr and (pathbase is None or not os.path.exists(pathdict)):
        
        # initialize
        if pathbase is not None and os.path.exists(pathpost) and np.loadtxt(pathpost, delimiter=',').shape[0] == limtpara.shape[1]:
            print('Reading the initial state from %s...' % pathpost)
            parainitcent = np.loadtxt(pathpost, delimiter=',')[:, 0]
            paraunitinitcent = np.empty_like(parainitcent)
            for m in indxpara:
                if scalpara[m] == 'self':
                    paraunitinitcent[m] = cdfn_self(parainitcent[m], limtpara[0, m], limtpara[1, m])
                if scalpara[m] == 'logt':
                    paraunitinitcent[m] = cdfn_logt(parainitcent[m], limtpara[0, m], limtpara[1, m])
                if scalpara[m] == 'gaus':
                    paraunitinitcent[m] = cdfn_gaus(parainitcent[m], meangauspara[m], stdvgauspara[m])
        else:
            paraunitinitcent = np.full(numbpara, 0.5)
        
        parainit = [np.empty(numbpara) for k in indxwalk]
        for k in indxwalk:
            for m in indxpara:
                paraunit = paraunitinitcent[m] + 0.05 * scipy.stats.norm.rvs()
                paraunit = paraunit % 1.
                if scalpara[m] == 'self':
                    parainit[k][m] = icdf_self(paraunit, limtpara[0, m], limtpara[1, m])
                if scalpara[m] == 'logt':
                    if booldiag:
                        if limtpara[0, m] <= 0.:
                            raise Exception('')
                        if limtpara[1, m] <= 0.:
                            raise Exception('')
                    parainit[k][m] = icdf_logt(paraunit, limtpara[0, m], limtpara[1, m])
                if scalpara[m] == 'gaus':
                    parainit[k][m] = icdf_gaus(paraunit, meangauspara[m], stdvgauspara[m])

        if typesamp == 'mcmc':
            if booltqdm:
                progress = True
            else:
                progress = False
        
            numbsamp = numbwalk * numbsampwalk
            indxsampwalk = np.arange(numbsampwalk)
            indxsamp = np.arange(numbsamp)
            if booldiag:
                if numbsampwalk == 0:
                    raise Exception('')
            
            if typeverb >= 1:
                print('Running PCAT...')
            listparafittwalk, listlposwalk = sample_fixed_chains(
                gdat, retr_llik, retr_lpri, listnamepara, scalpara, minmpara,
                maxmpara, meangauspara, stdvgauspara, parainit, numbwalk,
                numbsampwalk, numbsampburnwalkinit, pathbase, typeverb,
            )
            
            # get rid of burn-in and thin
            numbavail = numbsampwalk - numbsampburnwalk
            if numbavail <= 0:
                raise ValueError('No post-burn-in samples remain to retain.')
            if numbsamppostwalk > numbavail:
                if typeverb > 0:
                    print('Requested post-burn-in samples (%d) exceed the available window (%d); clipping to the available window.' % (numbsamppostwalk, numbavail))
                numbsamppostwalk = numbavail
            indxsampwalkkeep = np.linspace(numbsampburnwalk, numbsampwalk - 1, numbsamppostwalk, dtype=int)
            listparafitt = listparafittwalk[:, indxsampwalkkeep, :].reshape((-1, numbpara))
            
            numbsampkeep = listparafitt.shape[0]
            indxsampkeep = np.arange(numbsampkeep)
            
            listparaderi = None
            dictvarbderi = dict()
            if retr_dictderi is not None:
                numbsampderi = numbsampkeep
                indxsampderi = indxsampkeep

                print('Evaluating derived variables...')
                listdictvarbderi = [[] for n in indxsampderi]
                for n in tqdm(range(numbsampderi)):
                    listdictvarbderi[n] = retr_dictderi(listparafitt[indxsampderi[n], :], gdat)

                print('Placing the evaluated derived variables in the output dictionary...')
                for strg, valu in listdictvarbderi[0].items():
                    if booldiag and isinstance(valu, list):
                        print('strg')
                        print(strg)
                        print('valu')
                        print(valu)
                        raise Exception('')

                    if valu is None:
                        print('Encountered a None derived variable (%s). Skipping...' % strg)
                        continue
                    if np.isscalar(valu):
                        dictvarbderi[strg] = np.empty(numbsampderi)
                    else:
                        dictvarbderi[strg] = np.empty([numbsampderi] + list(valu.shape))
                    
                    for n in range(numbsampderi):
                        dictvarbderi[strg][n, ...] = listdictvarbderi[n][strg]
                
                listnameparaderi = listdictvarbderi[0].keys()
                listnameparaderi = []
                for name, valu in listdictvarbderi[0].items():
                    if np.isscalar(valu) and valu is not None:
                        listnameparaderi.append(name)
                numbparaderi = len(listnameparaderi)
                print('Placing the evaluated derived parameters in the output dictionary...')
                listparaderi = np.empty((numbsampkeep, numbparaderi)) 
                for k, strg in enumerate(listnameparaderi):
                    listparaderi[:, k] = dictvarbderi[strg]
                
            indxsampwalk = np.arange(numbsampwalk)
            
            if boolplot and pathvisu is not None:
                # plot the posterior
                ### trace
                figr, axis = plt.subplots(numbpara + 1, 1, figsize=(12, (numbpara + 1) * 4))
                for i in indxwalk:
                    axis[0].plot(indxsampwalk, listlposwalk[i, :])
                axis[0].axvline(numbsampburnwalk, color='black')
                axis[0].set_ylabel('log P')
                for k in indxpara:
                    for i in indxwalk:
                        axis[k+1].plot(indxsampwalk, listparafittwalk[i, :, k])
                    labl = listlablpara[k][0]
                    if listlablpara[k][1] != '':
                        labl += ' [%s]' % listlablpara[k][1]
                    axis[k+1].axvline(numbsampburnwalk, color='black')
                    axis[k+1].set_ylabel(labl)
                path = pathvisu + 'trac%s.%s' % (strgextn, typefileplot)
                if typeverb == 1:
                    print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
                
                # plot the posterior
                ### trace
                if numbsampburnwalk > 0:
                    figr, axis = plt.subplots(numbpara + 1, 1, figsize=(12, (numbpara + 1) * 4))
                    for i in indxwalk:
                        axis[0].plot(indxsampwalk[numbsampburnwalk:], listlposwalk[i, numbsampburnwalk:])
                    axis[0].set_ylabel('log P')
                    for k in indxpara:
                        for i in indxwalk:
                            axis[k+1].plot(indxsampwalk[numbsampburnwalk:], listparafittwalk[i, numbsampburnwalk:, k])
                        labl = listlablpara[k][0]
                        if listlablpara[k][1] != '':
                            labl += ' [%s]' % listlablpara[k][1]
                        axis[k+1].set_ylabel(labl)
                    path = pathvisu + 'tracgood%s.%s' % (strgextn, typefileplot)
                    if typeverb == 1:
                        print('Writing to %s...' % path)
                    plt.savefig(path)
                    plt.close()
        
        # derived
        if dictlablscalparaderi is not None:
            listnameparaderi = list(dictlablscalparaderi)
            listlablparaderi = [dictlablscalparaderi[name][0] for name in listnameparaderi]
            listlablparatotl = listlablpara + listlablparaderi
            listnameparatotl = listnamepara + listnameparaderi
            listparatotl = np.concatenate([listparafitt, listparaderi], 1)
        else:
            listparatotl = listparafitt
            
        if boolplot and pathvisu is not None and plot_posterior is not None:
            ## joint PDF
            strgextn = 'postparafitt' + strgextn
            plot_posterior(listlablpara, pathbase=pathvisu, listnamepara=listnamepara, strgextn=strgextn, listpara=listparafitt, numbpntsgrid=numbbins+1)
            
            if dictlablscalparaderi is not None:
                strgextn = 'postparaderi' + strgextn
                plot_posterior(listlablparaderi, pathbase=pathvisu, strgextn=strgextn, listnamepara=listnameparaderi, listpara=listparaderi, numbpntsgrid=numbbins+1)
                strgextn = 'postparatotl' + strgextn
                plot_posterior(listlablparatotl, pathbase=pathvisu, strgextn=strgextn, listpara=listparatotl, numbpntsgrid=numbbins+1)
        
        if pathbase is not None:
            if dictlablscalparaderi is not None:
                numbparapost = listparatotl.shape[1]
            else:
                numbparapost = numbpara
            print('Writing the posterior to %s...' % pathpost)
            arry = np.empty((numbparapost, 3))
            arry[:, 0] = np.median(listparatotl, 0)
            arry[:, 1] = np.percentile(listparatotl, 84., axis=0) - arry[:, 0]
            arry[:, 2] = arry[:, 0] - np.percentile(listparatotl, 16., axis=0)
            np.savetxt(pathpost, arry, delimiter=',')
            
            print('Writing the posterior derived variables to %s...' % pathpickderi)
            with open(pathpickderi, 'wb') as objtfile:
                pickle.dump(dictvarbderi, objtfile, protocol=pickle.HIGHEST_PROTOCOL)


        dictsamp = dict()
        for k, name in enumerate(listnamepara):
            dictsamp[name] = listparafitt[:, k]
        dictsamp['lpos'] = listlposwalk[:, indxsampwalkkeep].flatten()
        if pathbase is not None:
            print('Writing to %s...' % pathdict)
            pd.DataFrame.from_dict(dictsamp).to_csv(pathdict, index=False)
        for name, valu in dictvarbderi.items():
            dictsamp[name] = valu

    else:
        if typeverb > 0:
            print('A previous run has been found. Will retrieve results from this run.')
            print('Reading from %s...' % pathdict)
        dictsamp = pd.read_csv(pathdict).to_dict(orient='list')
        for name in dictsamp.keys():
            dictsamp[name] = np.array(dictsamp[name])
        
        if retr_dictderi is not None:
            objtfile = open(pathpickderi, "rb")
            if typeverb > 0:
                print('Reading from %s...' % pathpickderi)
            dictvarbderi = pickle.load(objtfile)
            for strgvarbderi, valuvarbderi in dictvarbderi.items():
                dictsamp[strgvarbderi] = valuvarbderi

    return dictsamp
