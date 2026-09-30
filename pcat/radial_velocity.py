"""Transdimensional cataloging of Keplerian signals in stellar radial-velocity (RV) time series.

PCAT treats each Keplerian signal as one element of type ``lghtlinekepl`` on its 1D data axis,
which here is time [day] with one bin per RV measurement. Element parameters are the
semi-amplitude K [m/s] (stored as ``flux``), period P [day] (stored as ``elin``), mean anomaly at
the reference time [rad] (``phas``), eccentricity (``ecce``), and argument of periastron [rad]
(``argp``). The likelihood is Gaussian. Per-instrument velocity offsets and an optional linear
trend are linear parameters, so they are marginalized analytically under flat priors, and a single
stellar jitter added in quadrature to the reported uncertainties is marginalized numerically on a
log-uniform grid.
"""

import os
import shutil

import astropy.io.fits
import numpy as np
import scipy.stats
from scipy.special import logsumexp

from .main import retr_pathrun


def retr_matrdesi(time, indxinst, boollinetren, timerefr):
    """Return the linear design matrix with one offset column per instrument and an optional trend column."""
    numbinst = int(np.amax(indxinst)) + 1
    matrdesi = (indxinst[:, None] == np.arange(numbinst)[None, :]).astype(float)
    if boollinetren:
        # trend slope per year keeps the matrix well conditioned [yr]
        matrdesi = np.column_stack((matrdesi, (time - timerefr) / 365.25))
    return matrdesi


def retr_llik_rvelmarg(resi, stdv, matrdesi, listjitt):
    """Log-likelihood of RV residuals [m/s] marginalized over linear nuisances and a jitter grid [m/s].

    The flat-prior normalization of the linear parameters is identical for every catalog, so it is
    dropped. Returns a scalar.
    """
    # inverse variances for each jitter value, shape (numbjitt, numbdata) [s^2/m^2]
    weig = 1. / (stdv[None, :]**2 + listjitt[:, None]**2)
    matrfish = np.einsum('jn,np,nq->jpq', weig, matrdesi, matrdesi)
    vectproj = np.einsum('jn,np,n->jp', weig, matrdesi, resi)
    chsqfull = np.sum(weig * resi[None, :]**2, axis=1)
    # chi-squared after projecting out the best-fitting linear nuisances
    chsqredu = chsqfull - np.einsum('jp,jp->j', vectproj, np.linalg.solve(matrfish, vectproj[:, :, None])[:, :, 0])
    _, logdfish = np.linalg.slogdet(matrfish)
    numbline = matrdesi.shape[1]
    llik = -0.5 * chsqredu - 0.5 * logdfish + 0.5 * np.sum(np.log(weig), axis=1) - \
           0.5 * (resi.size - numbline) * np.log(2. * np.pi)
    # log-uniform prior over the jitter grid
    return logsumexp(llik) - np.log(listjitt.size)


def retr_llik_rvel(gdat, strgmodl, cntpmodl):
    """PCAT likelihood callback: model RV [m/s] is the (bins,) axis of the model count cube."""
    modl = np.asarray(cntpmodl)[:, 0, 0]
    return retr_llik_rvelmarg(gdat.rvelobsv - modl, gdat.stdvrvelobsv, gdat.matrdesirvel, gdat.listjittrvel)


def retr_binstime(time):
    """Return strictly increasing bin edges [day] that contain one observation each."""
    edges = np.empty(time.size + 1)
    edges[1:-1] = 0.5 * (time[1:] + time[:-1])
    edges[0] = time[0] - 0.5 * (time[1] - time[0])
    edges[-1] = time[-1] + 0.5 * (time[-1] - time[-2])
    return edges


def retr_perdcirc(time, rvel, stdv, matrdesi, peri, timerefr, jitt=2.):
    """Weighted least-squares circular-orbit fit at each trial period [day].

    Returns the chi-squared reduction, semi-amplitude [m/s], and phase [rad] such that the
    signal is ``ampl * cos(2 pi (time - timerefr) / peri + phas)``; the linear nuisances in
    ``matrdesi`` are fit jointly. ``jitt`` [m/s] is added in quadrature to the uncertainties.
    """
    weig = 1. / (stdv**2 + jitt**2)
    # remove the linear nuisances first so that the per-period fit needs only a 2x2 solve
    matrfish = matrdesi.T @ (weig[:, None] * matrdesi)
    coef = np.linalg.solve(matrfish, matrdesi.T @ (weig * rvel))
    resi = rvel - matrdesi @ coef
    proj = matrdesi @ np.linalg.solve(matrfish, (weig[:, None] * matrdesi).T)
    phas = 2. * np.pi * (time[None, :] - timerefr) / peri[:, None]
    # basis functions with the nuisance projection removed, shape (numbperi, numbdata)
    cosi = np.cos(phas)
    sine = np.sin(phas)
    cosi -= cosi @ proj.T
    sine -= sine @ proj.T
    scc = np.sum(weig * cosi * cosi, 1)
    sss = np.sum(weig * sine * sine, 1)
    scs = np.sum(weig * cosi * sine, 1)
    scr = np.sum(weig * cosi * resi, 1)
    ssr = np.sum(weig * sine * resi, 1)
    dete = scc * sss - scs**2
    acos = (sss * scr - scs * ssr) / dete
    bsin = (scc * ssr - scs * scr) / dete
    dchi = acos * scr + bsin * ssr
    return dchi, np.hypot(acos, bsin), np.arctan2(-bsin, acos)


def retr_dictpropelem(time, rvel, stdv, matrdesi, timerefr, minmrvsa, maxmrvsa, minmperi, maxmperi,
                      numbperi=20000, expodchi=2., fracunif=0.1, fracgaus=0.5, stdvlogtrvsa=0.5, stdvphas=0.5):
    """Tabulate the data-informed element proposal on a grid uniform in log period.

    Periods are drawn with weight proportional to (chi-squared reduction)^expodchi of a circular
    fit to the data, mixed with a uniform fraction ``fracunif``. Given the period, the log
    semi-amplitude and the orbital phase are drawn around the circular fit with widths
    ``stdvlogtrvsa`` and ``stdvphas`` [rad], each mixed with the prior. Eccentricity and argument
    of periastron are drawn from the prior. The density is independent of the sampler state.
    Returns PCAT keywords that become attributes of its global object.
    """
    unitperi = (np.arange(numbperi) + 0.5) / numbperi
    peri = minmperi * (maxmperi / minmperi)**unitperi  # [day]
    dchi, ampl, phas = retr_perdcirc(time, rvel, stdv, matrdesi, peri, timerefr)
    weig = np.maximum(dchi, 0.)**expodchi
    pdfnperi = fracunif + (1. - fracunif) * weig / np.mean(weig)
    lognrvsa = np.log(maxmrvsa / minmrvsa)
    return dict(
        pdfnperipropelem=pdfnperi,
        cdfnperipropelem=np.cumsum(pdfnperi) / np.sum(pdfnperi),
        numbperipropelem=numbperi,
        unitrvsapropelem=np.clip(np.log(np.maximum(ampl, 1e-10) / minmrvsa) / lognrvsa, 0., 1.),
        stdvunitrvsapropelem=stdvlogtrvsa / lognrvsa,
        unitphaspropelem=np.mod(phas / (2. * np.pi), 1.),
        stdvunitphaspropelem=stdvphas / (2. * np.pi),
        fracgauspropelem=fracgaus,
        dchiperipropelem=dchi,
        retr_drawpropelem=retr_drawpropelem,
        retr_lpdfpropelem=retr_lpdfpropelem,
    )


def retr_pdfngausunit(unit, mean, stdv, boolwrap):
    """Normal density on the unit interval, either truncated or wrapped (periodic)."""
    if boolwrap:
        return sum(np.exp(-0.5 * ((unit + k - mean) / stdv)**2) for k in (-1., 0., 1.)) / (np.sqrt(2. * np.pi) * stdv)
    norm = scipy.stats.norm.cdf((1. - mean) / stdv) - scipy.stats.norm.cdf(-mean / stdv)
    return np.exp(-0.5 * ((unit - mean) / stdv)**2) / (np.sqrt(2. * np.pi) * stdv) / norm


def retr_drawgausunit(mean, stdv, boolwrap):
    """Draw from the truncated or wrapped normal on the unit interval."""
    if boolwrap:
        return np.mod(mean + stdv * np.random.randn(), 1.)
    return scipy.stats.truncnorm.rvs(-mean / stdv, (1. - mean) / stdv, loc=mean, scale=stdv)


def retr_drawpropelem(gdat):
    """Draw unit-cube element parameters (K, P, phase, e, omega) from the data-informed density."""
    indxperi = np.searchsorted(gdat.cdfnperipropelem, np.random.rand())
    indxperi = min(indxperi, gdat.numbperipropelem - 1)
    unit = np.random.rand(5)
    unit[1] = (indxperi + np.random.rand()) / gdat.numbperipropelem
    argp = 2. * np.pi * unit[4]
    if np.random.rand() < gdat.fracgauspropelem:
        unit[0] = retr_drawgausunit(gdat.unitrvsapropelem[indxperi], gdat.stdvunitrvsapropelem, False)
    if np.random.rand() < gdat.fracgauspropelem:
        # circular orbits have argp + phas equal to the fitted phase
        unit[2] = retr_drawgausunit(np.mod(gdat.unitphaspropelem[indxperi] - argp / (2. * np.pi), 1.), gdat.stdvunitphaspropelem, True)
    return unit


def retr_lpdfpropelem(gdat, unit):
    """Log density [unit-cube] of the data-informed element proposal."""
    indxperi = min(int(unit[1] * gdat.numbperipropelem), gdat.numbperipropelem - 1)
    # the period density is normalized so that its mean over the unit interval is one
    pdfnperi = gdat.pdfnperipropelem[indxperi] / np.mean(gdat.pdfnperipropelem)
    frac = gdat.fracgauspropelem
    pdfnrvsa = 1. - frac + frac * retr_pdfngausunit(unit[0], gdat.unitrvsapropelem[indxperi], gdat.stdvunitrvsapropelem, False)
    meanphas = np.mod(gdat.unitphaspropelem[indxperi] - unit[4], 1.)
    pdfnphas = 1. - frac + frac * retr_pdfngausunit(unit[2], meanphas, gdat.stdvunitphaspropelem, True)
    return np.log(pdfnperi) + np.log(pdfnrvsa) + np.log(pdfnphas)


def retr_dictpcatrvel(time, rvel, stdvrvel, indxinst, pathbase, strgcnfg, maxmnumbplan=4, boollinetren=False,
                      minmrvsa=0.3, maxmrvsa=300., minmperi=1.2, maxmperi=None, maxmecce=0.8,
                      minmjitt=0.1, maxmjitt=30., numbjitt=24, **dictpcat):
    """Write PCAT inputs for one star and return the keyword arguments of ``pcat.sampling.sample``.

    Parameters: time [BJD - 2450000 day], rvel and stdvrvel [m/s], indxinst integer instrument labels.
    """
    time = np.asarray(time, dtype=float)
    indxsort = np.argsort(time)
    time, rvel, stdvrvel = time[indxsort], np.asarray(rvel, float)[indxsort], np.asarray(stdvrvel, float)[indxsort]
    indxinst = np.unique(np.asarray(indxinst)[indxsort], return_inverse=True)[1]
    # drop exact duplicate epochs so that bin edges stay strictly increasing
    boolkeep = np.concatenate(([True], np.diff(time) > 1e-6))
    time, rvel, stdvrvel, indxinst = time[boolkeep], rvel[boolkeep], stdvrvel[boolkeep], indxinst[boolkeep]
    timerefr = 0.5 * (time[0] + time[-1])  # [day]
    if maxmperi is None:
        maxmperi = 2. * (time[-1] - time[0])  # [day]

    binstime = retr_binstime(time)
    widt = np.diff(binstime)  # [day]
    pathrun = retr_pathrun(pathbase, strgcnfg)
    pathinpt = os.path.join(pathrun, 'data', 'inpt')
    pathoutp = os.path.join(pathrun, 'data', 'outp', strgcnfg)
    if os.path.exists(pathoutp):
        print('Removing cached PCAT state %s...' % pathoutp)
        shutil.rmtree(pathoutp)
    os.makedirs(pathinpt, exist_ok=True)
    # PCAT's count cube equals sbrt * expo * width, so expo = 1 / width returns the velocities;
    # the data cube is only used for bookkeeping and plotting because the callback reads gdat.rvelobsv
    sbrt = rvel - np.amin(rvel) + 1.  # [m/s]
    for name, array in [('sbrt.fits', sbrt[:, None, None, None]), ('expo.fits', (1. / widt)[:, None, None])]:
        path = os.path.join(pathinpt, name)
        print('Writing to %s...' % path)
        astropy.io.fits.writeto(path, array, overwrite=True)

    dictpcatrvel = dict(
        typeexpr='fire', typedata='inpt', strgexprsbrt='sbrt.fits', typeexpo='file', strgexpo='expo.fits',
        binsenerfull=binstime, spectype=['kepl'], spatdisttype=['line'], typeelem=['lghtlinekepl'],
        # a constant background is absorbed by the marginalized instrument offsets
        dictfitt={'typeelem': ['lghtlinekepl'], 'spectype': ['kepl'],
                  'sbrtbacknorm': [np.ones((time.size, 1, 1))], 'listnamediff': ['back0000']},
        limtparaelem={'flux': (minmrvsa, maxmrvsa), 'elin': (minmperi, maxmperi), 'ecce': (0., maxmecce)},
        maxmgangdata=1e-4, anlytype='spec', fittminmnumbelempop0=0, fittmaxmnumbelempop0=maxmnumbplan,
        probspmr=0., retr_llik=retr_llik_rvel, timervel=time, timervelrefr=timerefr, rvelobsv=rvel,
        stdvrvelobsv=stdvrvel, matrdesirvel=retr_matrdesi(time, indxinst, boollinetren, timerefr),
        listjittrvel=np.geomspace(minmjitt, maxmjitt, numbjitt), booldiag=False,
        pathbase=pathbase, strgcnfg=strgcnfg,
    )
    matrdesi = dictpcatrvel['matrdesirvel']
    dictpcatrvel.update(retr_dictpropelem(time, rvel, stdvrvel, matrdesi, timerefr, minmrvsa, maxmrvsa, minmperi, maxmperi))
    dictpcatrvel.update(dictpcat)
    return dictpcatrvel
