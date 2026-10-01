"""Write an example output of every public PCAT plotting routine to docs/_static/gallery.

The transdimensional figures come from a four-chain PCAT run on the simulated
Voigt-line spectrum of examples/voigt_spectral_line_catalog, the fixed-dimensional
figures from a four-chain run on a correlated Gaussian, and the lens figures from
the simulated Roman/WFI strong-lens benchmark.
"""

from tdpy.verbosity import print

import argparse
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from pcat import plot_population_grid
from pcat.diagnostics import autocorrelation_time
from pcat.main import readfile, retr_pathrun
from pcat.plotting import (
    make_image_sequence_animation, plot_autocorrelation, plot_detection_diagnostic,
    plot_gelman_rubin, plot_grid, plot_lens_image_fit, plot_lens_parameter_recovery,
    plot_posterior_convergence, plot_sampler_overview,
)
from pcat.roman_lens import (
    RomanLensConfig, render_lens, render_lens_counts, run_lens_image_pipeline, simulate_population,
)
from pcat.sampling import sample

REPOSITORY = Path(__file__).resolve().parents[1]
GALLERY = REPOSITORY / "docs" / "_static" / "gallery"
CORRELATION = 0.6


def correlated_gaussian_log_likelihood(state, model_name, parameters):
    """Log density of a unit bivariate Gaussian with correlation CORRELATION."""
    x, y = parameters
    return float(-0.5 * (x**2 - 2 * CORRELATION * x * y + y**2) / (1 - CORRELATION**2))


def run_transdimensional(root, number_sweeps):
    """Sample the simulated two-line Voigt spectrum with four chains."""
    path = REPOSITORY / "examples/voigt_spectral_line_catalog/voigt_spectral_line_catalog.py"
    specification = importlib.util.spec_from_file_location("voigt_example", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    configuration, _, _ = module.build_configurations()
    configuration.update(
        strgcnfg="gallery_voigt", pathbase=str(root), truenumbelempop0=2,
        fittminmnumbelempop0=1, fittmaxmnumbelempop0=3,
        dicttrue={"typeelem": ["lghtlinevoig"], "spectype": ["voig"]},
        dictfitt={"typeelem": ["lghtlinevoig"], "spectype": ["voig"]},
        numbproc=4, numbswep=number_sweeps, numbburn=number_sweeps // 2,
        numbsamp=min(number_sweeps // 40, 1000), boolcheckconv=False, makeanim=False,
        boolmakeplot=False, boolmakeplotinit=False, booldiag=False, typeverb=0,
    )
    sample(**configuration)
    return readfile(str(Path(retr_pathrun(str(root), "gallery_voigt")) / "data" / "outp"
                        / "gallery_voigt" / "gdatfinlpost"))


def run_fixed(root):
    """Sample a correlated bivariate Gaussian with four chains."""
    return sample(
        typeexpr="gener", retr_llik=correlated_gaussian_log_likelihood,
        parameter_names=("x", "y"), prior_types=("self", "self"),
        prior_minima=(-5.0, -5.0), prior_maxima=(5.0, 5.0), initial_values=(0.0, 0.0),
        proposal_scales=(0.1, 0.1), pathbase=str(root), strgcnfg="gallery_gaussian",
        numbproc=4, numbswep=6000, numbburn=2000, numbsamp=1000, booladaptstdp=True,
        boolmakeplot=False, typeverb=-1,
    )


def write_lens_figures(root):
    """Fit one simulated Roman lens and summarize a simulated lens population."""
    config = RomanLensConfig()
    truth = np.array([0.9, 0.06, -0.04])  # [arcsec]
    source_size, axis_ratio, source_angle = 0.09, 0.8, 0.35  # [arcsec], [], [rad]
    observed = np.random.default_rng(814).poisson(
        render_lens_counts(config, truth, source_size, axis_ratio, source_angle))
    result = run_lens_image_pipeline(config, observed, source_size, axis_ratio, source_angle,
                                     truth, root, "gallery_lens", "gallery_lens")
    plot_lens_image_fit(GALLERY / "lens_image_fit", observed, result.model_counts,
                        result.model_counts, config.pixel_scale)
    plot_lens_parameter_recovery(GALLERY / "lens_parameter_recovery", result.draws, truth)
    records, examples = simulate_population(number_lenses=40, seed=814)
    plot_detection_diagnostic(records, GALLERY / "detection_diagnostic.png", examples)
    draws = result.draws[np.linspace(0, len(result.draws) - 1, 6, dtype=int)]
    make_image_sequence_animation(
        [render_lens(config, *draw, source_size, axis_ratio, source_angle) for draw in draws],
        [f"Einstein radius {draw[0]:.3f} arcsec" for draw in draws],
        GALLERY / "image_sequence_animation.gif", image_size=320,
        title="PCAT posterior lens draws",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbswep", type=int, default=20000)
    arguments = parser.parse_args()
    GALLERY.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="pcat-gallery-") as root:
        transdimensional = run_transdimensional(Path(root), arguments.numbswep)
        plot_sampler_overview(transdimensional, GALLERY, position_label="Line energy",
                              amplitude_label="Line flux")
        plot_posterior_convergence(transdimensional, GALLERY / "convergence_transdimensional")
        plot_gelman_rubin(f"{GALLERY}/transdimensional_", np.ravel(transdimensional.gmrbstat),
                          typefileplot="png")

        fixed = run_fixed(Path(root))
        draws = np.asarray(fixed.listpostparagenrscalbase)
        plot_posterior_convergence(fixed, GALLERY / "convergence_fixed")
        plot_grid(GALLERY / "posterior", "grid", draws, ("x", "y"), truepara=(0.0, 0.0),
                  typefileplot="png")
        correlation, times = autocorrelation_time(draws[::4, 0])
        plot_autocorrelation(f"{GALLERY}/fixed_", correlation, times, typefileplot="png")

        write_lens_figures(Path(root))

    rng = np.random.default_rng(0)
    populations = [rng.standard_normal((size, 3)) + offset
                   for size, offset in ((300, 0.0), (3000, 1.5))]
    plot_population_grid(
        [[f"Feature {index + 1}", ""] for index in range(3)], listpara=populations,
        listlablpopl=["Detected", "All"], pathbase=f"{GALLERY}/", strgextn="gallery",
        boolplottria=True, boolplothistodim=False, boolplotpair=False, boolplotpies=False,
        typeverb=0,
    )


if __name__ == "__main__":
    main()
