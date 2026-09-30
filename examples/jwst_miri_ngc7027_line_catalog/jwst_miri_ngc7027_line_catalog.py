#!/usr/bin/env python3
"""Catalog the emission lines of NGC 7027 in a JWST MIRI MRS spectrum with PCAT.

Data: the public level-3 MIRI Medium Resolution Spectrometer channel 1 short
(4.90-5.74 um) spectrum of the planetary nebula NGC 7027 from JWST program 1523,
file jw01523-o001_t002_miri_ch1-short_x1d.fits retrieved from MAST.

PCAT fits a transdimensional catalog of Voigt emission lines on top of a
continuum template whose amplitude is a free parameter. PCAT's likelihood is
Poisson, so the flux density F [Jy] and its uncertainty are mapped to effective
counts N = F k with k = C / sigma_eff^2 per bin, where C is the continuum and
sigma_eff includes a fractional floor for calibration and fringe residuals.
"""

from tdpy.verbosity import print

import argparse
import shutil
from pathlib import Path

import astropy.io.fits
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

EXAMPLE_PATH = Path(__file__).resolve().parent
RUN_NAME = EXAMPLE_PATH.name
PATH_INPT = EXAMPLE_PATH / "data" / "inpt"
NAME_X1D = "jw01523-o001_t002_miri_ch1-short_x1d.fits"
WAVELENGTH_RANGE = (5.30, 5.535)  # [um]
FRACTIONAL_ERROR_FLOOR = 0.01
# rest wavelengths of two securely identified lines in the window [um]
LINES_IDENTIFIED = {r"[Fe II] 5.340 $\mu$m": 5.3402, r"H$_2$ 0-0 S(7) 5.511 $\mu$m": 5.5112}


def read_spectrum():
    """Download (once) and read the x1d spectrum; return wavelength [um], flux [Jy], error [Jy]."""
    path = PATH_INPT / NAME_X1D
    if not path.exists():
        from astroquery.mast import Observations

        PATH_INPT.mkdir(parents=True, exist_ok=True)
        print(f"Writing to {path}...")
        Observations.download_file(f"mast:JWST/product/{NAME_X1D}", local_path=str(path))
    print(f"Reading from {path}...")
    data = astropy.io.fits.getdata(path, "EXTRACT1D")
    wavelength, flux, error = (np.asarray(data[name], dtype=float) for name in ("WAVELENGTH", "FLUX", "FLUX_ERROR"))
    indx = (wavelength > WAVELENGTH_RANGE[0]) & (wavelength < WAVELENGTH_RANGE[1]) & np.isfinite(flux)
    return wavelength[indx], flux[indx], error[indx]


def estimate_continuum(flux, error, halfwidth=30, numbiter=5):
    """Running median with iterative masking of bins more than 3 sigma above it."""
    mask = np.ones(flux.size, dtype=bool)
    continuum = np.median(flux) * np.ones_like(flux)
    for _ in range(numbiter):
        for i in range(flux.size):
            window = slice(max(0, i - halfwidth), i + halfwidth + 1)
            valid = flux[window][mask[window]]
            continuum[i] = np.median(valid) if valid.size > 0 else np.nan
        # windows fully covered by lines are interpolated from their neighbors
        indxgood = np.isfinite(continuum)
        continuum = np.interp(np.arange(flux.size), np.flatnonzero(indxgood), continuum[indxgood])
        sigma = np.sqrt(error**2 + (FRACTIONAL_ERROR_FLOOR * continuum)**2)
        # exclude line wings by also masking the three neighbors of each flagged bin
        flagged = np.convolve((flux - continuum > 3.0 * sigma).astype(float), np.ones(7), mode="same") > 0
        mask = ~flagged
    return continuum


def write_pcat_inputs(wavelength, flux, error, continuum):
    """Write surface-brightness and exposure cubes on the wavenumber grid PCAT expects."""
    # bin edges halfway between samples, converted to wavenumber [m^-1] in increasing order
    edges = np.concatenate(([1.5 * wavelength[0] - 0.5 * wavelength[1]],
                            0.5 * (wavelength[1:] + wavelength[:-1]),
                            [1.5 * wavelength[-1] - 0.5 * wavelength[-2]]))  # [um]
    edges = 1e6 / edges[::-1]  # [m^-1]
    width = np.diff(edges)  # [m^-1]
    flux, error, continuum = flux[::-1], error[::-1], continuum[::-1]

    sigma = np.sqrt(error**2 + (FRACTIONAL_ERROR_FLOOR * continuum)**2)  # [Jy]
    countsperjansky = continuum / sigma**2  # [Jy^-1]
    counts = np.round(np.clip(flux, 0.0, None) * countsperjansky)
    exposure = countsperjansky / width  # [Jy^-1 m]
    sbrt = counts / (exposure * width)  # [Jy]

    PATH_INPT.mkdir(parents=True, exist_ok=True)
    for name, array in [("sbrt.fits", sbrt[:, None, None, None]), ("expo.fits", exposure[:, None, None])]:
        path = PATH_INPT / name
        print(f"Writing to {path}...")
        astropy.io.fits.writeto(path, array, overwrite=True)
    return edges, continuum[:, None, None]


def run_pcat(edges, template, numbswep):
    from pcat import sampling

    path_cache = EXAMPLE_PATH / "data" / "outp" / RUN_NAME
    if path_cache.exists():
        print(f"Removing cached PCAT state {path_cache}...")
        shutil.rmtree(path_cache)
    return sampling.sample(
        typeexpr="fire",
        typedata="inpt",
        strgexprsbrt="sbrt.fits",
        typeexpo="file",
        strgexpo="expo.fits",
        binsenerfull=edges,
        spectype=["voig"],
        spatdisttype=["line"],
        typeelem=["lghtlinevoig"],
        dictfitt={"typeelem": ["lghtlinevoig"], "spectype": ["voig"], "sbrtbacknorm": [template],
                  "listnamediff": ["back0000"]},
        # line flux [Jy m^-1] and widths [m^-1]; MRS resolves lines over about two 27 m^-1 bins
        limtparaelem={"flux": (3.0, 1e3), "sigm": (15.0, 80.0), "gamm": (0.5, 20.0)},
        maxmgangdata=100.0 / (3600.0 * 180.0 / np.pi),  # [rad], unused for spectra
        anlytype="spec",
        fittminmnumbelempop0=0,
        fittmaxmnumbelempop0=25,
        inittype="rand",
        typeseed=0,
        probtran=0.7,
        probspmr=0.4,
        stdvpropelemfire=[5.0e-5, 1.0e-5, 5.0e-4, 5.0e-4],
        booladaptstdp=True,
        numbswep=numbswep,
        numbsamp=numbswep // 50,
        numbswepplot=max(numbswep // 20, 1),
        makeanim=True,
        boolmakeplotinit=False,
        booldiag=False,
        typeverb=0,
        pathbase=str(EXAMPLE_PATH),
        strgcnfg=RUN_NAME,
    )


def read_posterior():
    """Return PCAT's final posterior state."""
    from pcat.main import readfile

    return readfile(str(EXAMPLE_PATH / "data" / "outp" / RUN_NAME / "gdatfinlpost"))


def configure_style(typeplotback):
    colrfore = "white" if typeplotback == "dark" else "black"
    colrback = "black" if typeplotback == "dark" else "white"
    mpl.rcParams.update({
        "font.size": 10, "text.usetex": False, "axes.grid": False,
        "figure.facecolor": colrback, "axes.facecolor": colrback, "savefig.facecolor": colrback,
        "axes.edgecolor": colrfore, "axes.labelcolor": colrfore, "xtick.color": colrfore,
        "ytick.color": colrfore, "text.color": colrfore,
        "legend.fancybox": True, "legend.framealpha": 1.0,
    })
    return colrfore


def save(figure, name, typefileplot):
    path = EXAMPLE_PATH / "visuals" / f"{name}.{typefileplot}"
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {path}...")
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_spectrum_fit(wavelength, flux, error, continuum, posterior, typefileplot, colrfore):
    """Data, continuum template, posterior model, and normalized residuals versus wavelength."""
    sigma = np.sqrt(error**2 + (FRACTIONAL_ERROR_FLOOR * continuum)**2)  # [Jy]
    countsperjansky = continuum / sigma**2  # [Jy^-1]
    # PCAT bins run in increasing wavenumber, i.e., decreasing wavelength
    modl = posterior.listpostcntpmodl[:, ::-1, 0, 0] / countsperjansky[None, :]  # [Jy]
    quantiles = np.percentile(modl, [16., 50., 84.], axis=0)
    figure, axes = plt.subplots(2, 1, figsize=(7.0, 4.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].step(wavelength, flux, where="mid", color=colrfore, lw=0.8, label="JWST MIRI MRS, NGC 7027 (PID 1523)")
    axes[0].plot(wavelength, continuum, color="C2", ls="--", lw=1.0, label="Continuum template")
    axes[0].fill_between(wavelength, quantiles[0], quantiles[2], color="C0", alpha=0.4, lw=0)
    axes[0].plot(wavelength, quantiles[1], color="C0", lw=1.0, label="PCAT posterior median model")
    for label, value in LINES_IDENTIFIED.items():
        axes[0].annotate(label, (value, np.interp(value, wavelength, flux)), xytext=(0, 12),
                         textcoords="offset points", ha="center", fontsize=8)
    axes[0].set_ylabel("Flux density [Jy]")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].step(wavelength, (flux - quantiles[1]) / sigma, where="mid", color=colrfore, lw=0.8)
    axes[1].axhline(0.0, color="0.5", lw=0.8)
    axes[1].set_ylabel(r"Residual [$\sigma$]")
    axes[1].set_xlabel(r"Wavelength [$\mu$m]")
    figure.subplots_adjust(hspace=0.05)
    save(figure, "jwst_miri_ngc7027_spectrum_fit", typefileplot)


def plot_line_catalog(wavelength, flux, posterior, typefileplot, colrfore):
    """Posterior samples of line wavelength and integrated flux over the observed spectrum."""
    listelin = np.concatenate([np.asarray(sample[0]["elin"]) for sample in posterior.listpostdictelem])  # [m^-1]
    listflux = np.concatenate([np.asarray(sample[0]["flux"]) for sample in posterior.listpostdictelem])  # [Jy m^-1]
    # 1 Jy m^-1 integrated over wavenumber is 1e-26 W m^-2 Hz^-1 times c [m s^-1]
    listfluxsi = listflux * 1e-26 * 2.998e8  # [W m^-2]
    figure, axes = plt.subplots(2, 1, figsize=(7.0, 4.4), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    axes[0].step(wavelength, flux, where="mid", color=colrfore, lw=0.8)
    axes[0].set_ylabel("Flux density [Jy]")
    axes[1].scatter(1e6 / listelin, listfluxsi, s=4, alpha=0.3, color="C0", lw=0,
                    label=f"Posterior line samples ({len(posterior.listpostdictelem)} catalogs)")
    for k, value in enumerate(LINES_IDENTIFIED.values()):
        for axis in axes:
            axis.axvline(value, color="C3", ls=":", lw=0.8, label="[Fe II] and H$_2$ S(7)" if k == 0 and axis is axes[1] else None)
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"Line flux [W m$^{-2}$]")
    axes[1].set_xlabel(r"Wavelength [$\mu$m]")
    axes[1].set_xlim(wavelength[0], wavelength[-1])
    axes[1].legend(loc="upper left", fontsize=8)
    figure.subplots_adjust(hspace=0.05)
    save(figure, "jwst_miri_ngc7027_line_catalog_samples", typefileplot)


def plot_line_count(posterior, typefileplot):
    """Posterior probability of the number of emission lines."""
    numbelem = np.asarray(posterior.listpostnumbelem).ravel().astype(int)
    values, counts = np.unique(numbelem, return_counts=True)
    figure, axis = plt.subplots(figsize=(3.4, 2.6))
    axis.bar(values, counts / counts.sum(), color="C0")
    axis.set_xlabel("Number of emission lines")
    axis.set_ylabel("Posterior probability")
    save(figure, "jwst_miri_ngc7027_line_count_posterior", typefileplot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbswep", type=int, default=200_000)
    parser.add_argument("--typefileplot", choices=("png", "pdf"), default="png")
    parser.add_argument("--typeplotback", choices=("white", "dark"), default="white")
    parser.add_argument("--skip-sampling", action="store_true", help="Replot an existing chain.")
    arguments = parser.parse_args()
    wavelength, flux, error = read_spectrum()
    continuum = estimate_continuum(flux, error)
    edges, template = write_pcat_inputs(wavelength, flux, error, continuum)
    if not arguments.skip_sampling:
        run_pcat(edges, template, arguments.numbswep)
    posterior = read_posterior()
    colrfore = configure_style(arguments.typeplotback)
    plot_spectrum_fit(wavelength, flux, error, continuum, posterior, arguments.typefileplot, colrfore)
    plot_line_catalog(wavelength, flux, posterior, arguments.typefileplot, colrfore)
    plot_line_count(posterior, arguments.typefileplot)


if __name__ == "__main__":
    main()
