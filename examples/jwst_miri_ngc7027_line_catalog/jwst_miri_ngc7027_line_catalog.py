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
import hashlib
import json
from datetime import datetime, timezone
from tdpy.cli import add_plot_arguments
from pathlib import Path

import astropy.io.fits
import matplotlib.pyplot as plt
import numpy as np
from pcat.demo import load_example_posterior
from pcat.demo import configure_example_style, save_example_figure
from pcat.main import retr_pathrun

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

    if (not np.isfinite(flux).all() or not np.isfinite(error).all()
            or not np.isfinite(continuum).all() or np.any(error <= 0.0)
            or np.any(continuum <= 0.0)):
        raise ValueError("PCAT input flux, uncertainty, and continuum must be finite with positive uncertainty and continuum")
    if np.any(flux < 0.0):
        raise ValueError(
            "The Poisson PCAT input cannot preserve negative measured flux; use an observation model that supports signed flux values"
        )

    sigma = np.sqrt(error**2 + (FRACTIONAL_ERROR_FLOOR * continuum)**2)  # [Jy]
    countsperjansky = continuum / sigma**2  # [Jy^-1]
    counts = np.round(flux * countsperjansky)
    exposure = countsperjansky / width  # [Jy^-1 m]
    sbrt = counts / (exposure * width)  # [Jy]

    PATH_INPT.mkdir(parents=True, exist_ok=True)
    for name, array in [("sbrt.fits", sbrt[:, None, None, None]), ("expo.fits", exposure[:, None, None])]:
        path = PATH_INPT / name
        print(f"Writing to {path}...")
        astropy.io.fits.writeto(path, array, overwrite=True)
    return edges, continuum[:, None, None]


def fresh_run_root():
    run_root = Path(retr_pathrun(EXAMPLE_PATH, RUN_NAME))
    path_cache = run_root / "data" / "outp"
    legacy_cache = EXAMPLE_PATH / "data" / "outp" / RUN_NAME
    if path_cache.exists() or legacy_cache.exists():
        raise FileExistsError(f"Refusing to overwrite cached JWST run {RUN_NAME}")
    return run_root


def run_pcat(edges, template, numbswep, *, number_chains=1):
    from pcat import sampling

    fresh_run_root()
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
        numbproc=number_chains,
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


def render_line_histogram_frames(posterior):
    """Show changing PCAT line-center catalogs with shared bins and complete y-limits."""
    indices = np.linspace(0, len(posterior.listpostdictelem) - 1,
                          min(12, len(posterior.listpostdictelem)), dtype=int)
    edges = np.linspace(*WAVELENGTH_RANGE, 25)  # [um]
    histograms = []
    for index in indices:
        wavenumber = np.asarray(posterior.listpostdictelem[index][0]["elin"])
        wavelengths = 1e6 / wavenumber  # [um]
        histogram = np.histogram(wavelengths, bins=edges)[0]
        if np.any(histogram):
            histograms.append((index, histogram))
    if len(histograms) < 2:
        raise ValueError("At least two sampled catalogs must contain a line in the plotted wavelength range")
    upper = max(1, *(int(histogram.max()) for _, histogram in histograms)) * 1.15
    visual_root = EXAMPLE_PATH / "visuals"
    visual_root.mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    paths = []
    for index, histogram in histograms:
        figure, axis = plt.subplots(figsize=(6.0, 4.3), facecolor="white")
        axis.bar(edges[:-1], histogram, width=np.diff(edges), align="edge", color="#176B87",
                 edgecolor="white", linewidth=0.5, label="PCAT line centers")
        for label, wavelength in LINES_IDENTIFIED.items():
            axis.axvline(wavelength, color="#A83D36", lw=1.1, ls="--", label=label)
        axis.set(xlim=WAVELENGTH_RANGE, ylim=(0, upper), xlabel="Wavelength [micrometer]",
                 ylabel="Number of spectral lines", title="JWST/MIRI NGC 7027 line catalog")
        axis.grid(False)
        axis.legend(loc="upper right", framealpha=1, facecolor="white", fontsize=8)
        path = visual_root / f"jwst_line_hist_{run_tag}_swep{index:09d}.png"
        print(f"Writing to {path}...")
        figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        paths.append(path)
    return paths


def render_spectrum_fit_frames(wavelength, flux, error, continuum, posterior):
    """Render observed spectral bins with actual retained model draws and residuals."""
    inputs = (wavelength, flux, error, continuum, posterior.listpostcntpmodl)
    if any(np.ma.is_masked(values) or np.iscomplexobj(values) for values in inputs):
        raise ValueError("Spectral animation requires unmasked real inputs")
    try:
        wavelength, flux, error, continuum, count_models = (
            np.asarray(values, dtype=float) for values in inputs
        )
    except (TypeError, ValueError) as exception:
        raise ValueError("Spectral animation requires numeric inputs") from exception
    if (wavelength.ndim != 1 or len(wavelength) < 2
            or any(values.shape != wavelength.shape for values in (flux, error, continuum))
            or any(not np.isfinite(values).all() for values in (wavelength, flux, error, continuum))
            or np.any(wavelength <= 0) or np.any(np.diff(wavelength) <= 0)
            or np.any(error <= 0) or np.any(continuum <= 0)):
        raise ValueError("Spectral animation requires aligned finite bins with increasing positive "
                         "wavelengths, positive errors and positive continuum")
    if (count_models.ndim != 4 or count_models.shape[1:] != (len(wavelength), 1, 1)
            or len(count_models) < 2):
        raise ValueError("Spectral animation requires at least two aligned model draws")
    sigma = np.sqrt(error**2 + (FRACTIONAL_ERROR_FLOOR * continuum)**2)  # [Jy]
    countsperjansky = continuum / sigma**2  # [Jy^-1]
    models = count_models[:, ::-1, 0, 0] / countsperjansky[None, :]  # [Jy]
    if not np.isfinite(models).all():
        raise ValueError("Spectral model draws must be finite")
    residuals = (flux[None, :] - models) / sigma[None, :]
    flux_minimum = min(float(np.min(flux)), float(np.min(models)))
    flux_maximum = max(float(np.max(flux)), float(np.max(models)))
    margin = max(0.01, 0.08 * (flux_maximum - flux_minimum))
    residual_limit = max(5.0, 1.1 * float(np.max(np.abs(residuals))))
    indices = np.unique(np.linspace(0, len(models) - 1, min(12, len(models)), dtype=int))
    root = EXAMPLE_PATH / "visuals"
    root.mkdir(parents=True, exist_ok=True)
    tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    paths = []
    for index in indices:
        figure, axes = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True,
                                    layout="constrained", gridspec_kw={"height_ratios": [3, 1]})
        foreground = plt.rcParams["text.color"]
        axes[0].plot(wavelength, flux, marker=".", markersize=3, linestyle="none",
                     color=foreground, label="Observed JWST/MIRI bins")
        axes[0].plot(wavelength, models[index], color="#176B87", linewidth=1.4,
                     label="Retained PCAT model")
        axes[0].set(ylabel="Flux density [Jy]", ylim=(flux_minimum - margin, flux_maximum + margin),
                    title=f"JWST/MIRI NGC 7027\nRetained draw {index + 1:04d} / {len(models):04d}")
        axes[0].title.set_fontfamily("monospace")
        axes[0].title.set_fontsize(9)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside upper center", ncol=2,
                      fancybox=True, framealpha=1.0, fontsize=9)
        axes[1].plot(wavelength, residuals[index], marker=".", markersize=3,
                     linestyle="none", color=foreground)
        axes[1].axhline(0.0, color="0.5", linewidth=0.7)
        axes[1].set(xlabel="Wavelength [micrometer]", ylabel="Residual [sigma]",
                    xlim=WAVELENGTH_RANGE, ylim=(-residual_limit, residual_limit))
        for axis in axes:
            axis.grid(False)
        path = root / f"jwst_spectrum_fit_{tag}_swep{index:09d}.png"
        print(f"Writing to {path}...")
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def plot_spectrum_fit(wavelength, flux, error, continuum, posterior, typefileplot, colrfore, time_tag=None):
    """Data, continuum template, posterior model, and normalized residuals versus wavelength."""
    sigma = np.sqrt(error**2 + (FRACTIONAL_ERROR_FLOOR * continuum)**2)  # [Jy]
    countsperjansky = continuum / sigma**2  # [Jy^-1]
    # PCAT bins run in increasing wavenumber, i.e., decreasing wavelength
    modl = posterior.listpostcntpmodl[:, ::-1, 0, 0] / countsperjansky[None, :]  # [Jy]
    quantiles = np.percentile(modl, [16., 50., 84.], axis=0)
    figure, axes = plt.subplots(2, 1, figsize=(7.0, 4.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(wavelength, flux, marker=".", markersize=2, linestyle="none",
                 color=colrfore, label="JWST MIRI MRS, NGC 7027 (PID 1523)")
    axes[0].plot(wavelength, continuum, color="C2", ls="--", lw=1.0, label="Continuum template")
    axes[0].fill_between(wavelength, quantiles[0], quantiles[2], color="C0", alpha=0.4, lw=0)
    axes[0].plot(wavelength, quantiles[1], color="C0", lw=1.0, label="PCAT posterior median model")
    for label, value in LINES_IDENTIFIED.items():
        axes[0].annotate(label, (value, np.interp(value, wavelength, flux)), xytext=(0, 12),
                         textcoords="offset points", ha="center", fontsize=8)
    axes[0].set_ylabel("Flux density [Jy]")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].plot(wavelength, (flux - quantiles[1]) / sigma, marker=".", markersize=2,
                 linestyle="none", color=colrfore)
    axes[1].axhline(0.0, color="0.5", lw=0.8)
    axes[1].set_ylabel(r"Residual [$\sigma$]")
    axes[1].set_xlabel(r"Wavelength [$\mu$m]")
    figure.subplots_adjust(hspace=0.05)
    return save_spectral_figure(figure, "jwst_miri_ngc7027_spectrum_fit", typefileplot, time_tag)


def plot_line_catalog(wavelength, flux, posterior, typefileplot, colrfore, time_tag=None):
    """Posterior samples of line wavelength and integrated flux over the observed spectrum."""
    listelin = np.concatenate([np.asarray(sample[0]["elin"]) for sample in posterior.listpostdictelem])  # [m^-1]
    listflux = np.concatenate([np.asarray(sample[0]["flux"]) for sample in posterior.listpostdictelem])  # [Jy m^-1]
    # 1 Jy m^-1 integrated over wavenumber is 1e-26 W m^-2 Hz^-1 times c [m s^-1]
    listfluxsi = listflux * 1e-26 * 2.998e8  # [W m^-2]
    figure, axes = plt.subplots(2, 1, figsize=(7.0, 4.4), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    axes[0].plot(wavelength, flux, marker=".", markersize=2, linestyle="none", color=colrfore)
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
    return save_spectral_figure(figure, "jwst_miri_ngc7027_line_catalog_samples", typefileplot, time_tag)


def plot_line_count(posterior, typefileplot, time_tag=None):
    """Posterior probability of the number of emission lines."""
    numbelem = np.asarray(posterior.listpostnumbelem).ravel().astype(int)
    values, counts = np.unique(numbelem, return_counts=True)
    figure, axis = plt.subplots(figsize=(3.4, 2.6))
    axis.bar(values, counts / counts.sum(), color="C0")
    axis.set_xlabel("Number of emission lines")
    axis.set_ylabel("Posterior probability")
    return save_spectral_figure(figure, "jwst_miri_ngc7027_line_count_posterior", typefileplot, time_tag)


def save_spectral_figure(figure, name, typefileplot, time_tag=None):
    if time_tag:
        name = f"{name}_{time_tag}"
    target = EXAMPLE_PATH / "visuals" / f"{name}.{typefileplot}"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite spectral figure {target}")
    save_example_figure(figure, EXAMPLE_PATH, name, typefileplot)
    return target


def main():
    global RUN_NAME, PATH_INPT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbswep", type=int, default=200_000)
    add_plot_arguments(parser)
    parser.add_argument("--typeplotback", choices=("white", "dark"), default="white")
    parser.add_argument("--skip-sampling", action="store_true", help="Replot an existing chain.")
    parser.add_argument("--run-name", help="Saved run to replot, or a fresh sampler run name.")
    parser.add_argument("--time-tag", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--chains", type=int, default=1)
    arguments = parser.parse_args()
    if datetime.strptime(arguments.time_tag, "%Y%m%d_%H%M%S").strftime("%Y%m%d_%H%M%S") != arguments.time_tag:
        raise ValueError("time-tag must use YYYYMMDD_HHMMSS")
    if arguments.numbswep < 100 or arguments.chains < 1:
        raise ValueError("At least 100 sweeps and one chain are required")
    RUN_NAME = arguments.run_name or (EXAMPLE_PATH.name if arguments.skip_sampling
                                      else f"{EXAMPLE_PATH.name}_{arguments.time_tag}")
    if Path(RUN_NAME).name != RUN_NAME or RUN_NAME in {".", ".."}:
        raise ValueError("run-name must be a single directory name")
    if list((EXAMPLE_PATH / "visuals").glob(f"jwst_miri_ngc7027_*_{arguments.time_tag}.*")):
        raise FileExistsError("Refusing to overwrite earlier spectral outputs")
    run_root = None if arguments.skip_sampling else fresh_run_root()
    source_path = PATH_INPT / NAME_X1D
    wavelength, flux, error = read_spectrum()
    continuum = estimate_continuum(flux, error)
    if not arguments.skip_sampling:
        PATH_INPT = run_root / "data" / "inpt"
        edges, template = write_pcat_inputs(wavelength, flux, error, continuum)
        run_pcat(edges, template, arguments.numbswep, number_chains=arguments.chains)
    posterior = load_example_posterior(
        f"{EXAMPLE_PATH.name}/data/outp/{RUN_NAME}/gdatfinlpost"
    )
    colrfore = configure_example_style(arguments.typeplotback)
    histogram_paths = render_line_histogram_frames(posterior)
    spectral_paths = render_spectrum_fit_frames(wavelength, flux, error, continuum, posterior)
    figure_paths = [
        plot_spectrum_fit(wavelength, flux, error, continuum, posterior, arguments.typefileplot, colrfore, arguments.time_tag),
        plot_line_catalog(wavelength, flux, posterior, arguments.typefileplot, colrfore, arguments.time_tag),
        plot_line_count(posterior, arguments.typefileplot, arguments.time_tag),
    ]
    metadata = {"run_name": RUN_NAME, "time_tag": arguments.time_tag, "skip_sampling": arguments.skip_sampling,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "input_file": NAME_X1D, "input_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "wavelength_range_um": WAVELENGTH_RANGE, "fractional_error_floor": FRACTIONAL_ERROR_FLOOR,
                "spectrum_bins": int(wavelength.size), "posterior_catalogs": len(posterior.listpostdictelem),
                "chains": (int(posterior.numbproc) if hasattr(posterior, "numbproc")
                           else None if arguments.skip_sampling else arguments.chains),
                "sweeps_per_chain": (int(posterior.numbswep) if hasattr(posterior, "numbswep")
                                     else None if arguments.skip_sampling else arguments.numbswep),
                "figure_paths": [path.name for path in figure_paths],
                "histogram_frame_paths": [path.name for path in histogram_paths],
                "histogram_frame_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                           for path in histogram_paths},
                "spectral_frame_paths": [path.name for path in spectral_paths],
                "spectral_frame_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                          for path in spectral_paths},
                "assumptions": "Continuum-based effective-count Poisson approximation to calibrated JWST flux density; short chains verify integration."}
    metadata_path = EXAMPLE_PATH / "visuals" / f"jwst_miri_ngc7027_run_{arguments.time_tag}.json"
    print(f"Writing to {metadata_path}...")
    with metadata_path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
