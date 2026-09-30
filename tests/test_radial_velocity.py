import numpy as np
import pytest
from scipy.integrate import trapezoid

from pcat.radial_velocity import (
    retr_dictpropelem,
    retr_drawpropelem,
    retr_llik_rvelmarg,
    retr_lpdfpropelem,
    retr_matrdesi,
    retr_pdfngausunit,
)
from tdpy import gdatstrt


def test_marginal_likelihood_matches_numerical_integration_over_the_offset():
    rng = np.random.default_rng(0)
    resi = rng.normal(3., 2., 15)  # [m/s]
    stdv = rng.uniform(1., 3., 15)  # [m/s]
    matrdesi = retr_matrdesi(np.arange(15.), np.zeros(15, int), False, 0.)
    llik = retr_llik_rvelmarg(resi, stdv, matrdesi, np.array([1e-8]))
    offs = np.linspace(-20., 30., 20001)  # [m/s]
    lliknumb = np.sum(-0.5 * (resi[None, :] - offs[:, None])**2 / stdv**2 - 0.5 * np.log(2. * np.pi * stdv**2), 1)
    assert llik == pytest.approx(np.log(trapezoid(np.exp(lliknumb), offs)), abs=1e-6)


def test_data_informed_element_proposal_is_a_normalized_density_on_the_unit_cube():
    rng = np.random.default_rng(1)
    time = np.sort(rng.uniform(0., 500., 60))  # [day]
    rvel = 5. * np.cos(2. * np.pi * time / 17.) + rng.normal(0., 1., time.size)  # [m/s]
    stdv = np.ones(time.size)  # [m/s]
    matrdesi = retr_matrdesi(time, np.zeros(time.size, int), False, 250.)
    gdat = gdatstrt()
    for name, valu in retr_dictpropelem(time, rvel, stdv, matrdesi, 250., 0.3, 300., 1.2, 1000., numbperi=4000).items():
        setattr(gdat, name, valu)
    unitgrid = (np.arange(20000) + 0.5) / 20000
    # the period marginal, and the truncated and wrapped conditionals, each integrate to one
    assert np.mean(gdat.pdfnperipropelem) == pytest.approx(1., rel=1e-12)
    for mean in [0.02, 0.5, 0.97]:
        for boolwrap in [False, True]:
            assert np.mean(retr_pdfngausunit(unitgrid, mean, 0.05, boolwrap)) == pytest.approx(1., rel=1e-4)
    # at fixed period, integrating the joint density over semi-amplitude and phase leaves the period density
    unit = np.full((unitgrid.size, 5), 0.3)
    indxperi = int(np.argmax(gdat.pdfnperipropelem))
    unit[:, 1] = (indxperi + 0.5) / gdat.numbperipropelem
    unit[:, 0] = unitgrid
    densrvsa = np.exp([retr_lpdfpropelem(gdat, u) for u in unit])
    frac = gdat.fracgauspropelem
    densrvsacond = 1. - frac + frac * retr_pdfngausunit(unitgrid, gdat.unitrvsapropelem[indxperi], gdat.stdvunitrvsapropelem, False)
    # the joint density factorizes into the period density, the semi-amplitude conditional, and a common phase factor
    factothr = densrvsa / densrvsacond
    assert np.ptp(factothr) == pytest.approx(0., abs=1e-9 * np.amax(factothr))
    assert np.mean(densrvsacond) == pytest.approx(1., rel=1e-4)
    np.random.seed(2)
    draws = np.array([retr_drawpropelem(gdat) for _ in range(2000)])
    assert np.all((draws >= 0.) & (draws <= 1.))
    peri = 1.2 * (1000. / 1.2)**draws[:, 1]  # [day]
    # the injected period dominates the periodogram, so most proposed periods land near it
    assert np.mean(np.abs(np.log(peri / 17.)) < 0.02) > 0.5


def retr_llik_flat(gdat, strgmodl, cntpmodl):
    return 0.


def test_number_of_planets_follows_its_uniform_prior_when_the_likelihood_is_flat(tmp_path):
    from pcat import sampling
    from pcat.main import readfile
    from pcat.radial_velocity import retr_dictpcatrvel

    rng = np.random.default_rng(4)
    time = np.sort(rng.uniform(0., 300., 30))  # [day]
    rvel = rng.normal(0., 2., time.size)  # [m/s]
    dictpcat = retr_dictpcatrvel(time, rvel, np.full(time.size, 2.), np.zeros(time.size, int), str(tmp_path), 'flat',
                                 maxmnumbplan=3, retr_llik=retr_llik_flat, factpriodoff=0., probjump=0.2,
                                 numbswep=30000, numbsamp=3000, boolmakeplot=False, boolmakeplotinit=False,
                                 typeverb=0, inittype='rand', typeseed=1)
    sampling.sample(**dictpcat)
    posterior = readfile(str(tmp_path / 'pcat_runs' / 'flat' / 'data' / 'outp' / 'flat' / 'gdatfinlpost'))
    frac = np.bincount(np.asarray(posterior.listpostnumbelem).astype(int).ravel(), minlength=4) / 3000.
    assert frac == pytest.approx(np.full(4, 0.25), abs=0.06)
