#!/usr/bin/env python3
"""Use PCAT to infer H0 from synthetic time delays in a fixed Mejiro lens model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from tdpy.cli import add_plot_arguments, add_smoke_argument

from pcat.demo import configure_example_style, save_example_figure
from pcat.fixed import sample_fixed_chains

REFERENCE_H0 = 70.0  # [km s^-1 Mpc^-1]
TRUE_H0 = 67.4  # [km s^-1 Mpc^-1]
OMEGA_M = 0.3
H0_PRIOR = (40.0, 100.0)  # [km s^-1 Mpc^-1]
DELAY_ERROR_DAYS = 0.1  # [day], explicit synthetic measurement assumption


def make_synthetic_lensed_supernova(reference_h0: float = REFERENCE_H0):
    """Build a Mejiro LensedSupernova from an explicit four-image SIE model."""
    from mejiro.lensed_supernova import LensedSupernova

    lens_redshift = 0.5
    source_redshift = 1.5
    cosmology = FlatLambdaCDM(H0=reference_h0, Om0=OMEGA_M)
    lens_model_list = ["SIE", "SHEAR", "CONVERGENCE"]
    lens_redshift_list = [lens_redshift] * len(lens_model_list)
    kwargs_lens = [
        {"theta_E": 1.0, "center_x": 0.0, "center_y": 0.0, "e1": 0.08, "e2": -0.04},
        {"gamma1": 0.03, "gamma2": -0.02, "ra_0": 0.0, "dec_0": 0.0},
        {"kappa": 0.02, "ra_0": 0.0, "dec_0": 0.0},
    ]
    lens_model = LensModel(
        lens_model_list=lens_model_list,
        z_lens=lens_redshift,
        z_source=source_redshift,
        lens_redshift_list=lens_redshift_list,
        cosmo=cosmology,
    )
    source_position = (0.01, 0.01)  # [arcsec]
    solver = LensEquationSolver(lens_model)
    image_x, image_y = solver.image_position_from_source(
        *source_position, kwargs_lens,
        search_window=6.0, min_distance=0.02,
    )
    arrival_times = np.asarray(
        lens_model.arrival_time(image_x, image_y, kwargs_lens), dtype=float
    )  # [day]
    relative_delays = arrival_times - arrival_times[0]
    if len(relative_delays) != 4 or not np.isfinite(relative_delays).all():
        raise RuntimeError("The synthetic lens must produce four finite image delays")

    kwargs_model = {
        "cosmo": cosmology,
        "lens_light_model_list": ["SERSIC_ELLIPSE"],
        "lens_model_list": lens_model_list,
        "point_source_model_list": ["LENSED_POSITION"],
        "source_light_model_list": ["SERSIC_ELLIPSE"],
        "lens_redshift_list": lens_redshift_list,
        "source_redshift_list": [source_redshift],
        "z_source": source_redshift,
    }
    kwargs_params = {
        "kwargs_lens": kwargs_lens,
        "kwargs_lens_light": [{
            "magnitude": 20.0, "R_sersic": 1.0, "n_sersic": 4.0,
            "center_x": 0.0, "center_y": 0.0, "e1": 0.0, "e2": 0.0,
        }],
        "kwargs_source": [{
            "magnitude": 23.0, "R_sersic": 0.3, "n_sersic": 1.0,
            "center_x": source_position[0], "center_y": source_position[1],
            "e1": 0.0, "e2": 0.0,
        }],
        "kwargs_ps": [{
            "ra_image": image_x, "dec_image": image_y,
            "magnitude": [25.0, 25.5, 26.0, 26.5],
        }],
    }
    physical_params = {
        "einstein_radius": 1.0,
        "lens_stellar_mass": 1.0e11,
        "lens_velocity_dispersion": 250.0,
        "magnification": 5.0,
        "magnitudes": {
            "lens": {"F129": 20.0},
            "source": {"F129": 23.0},
            "lensed_source": {"F129": 21.5},
        },
        "sn_type": "Ia",
        "time_delays": relative_delays,
        "image_magnifications": np.ones(4),
    }
    return LensedSupernova(
        name="synthetic_fixed_sie_supernova",
        coords=None,
        kwargs_model=kwargs_model,
        kwargs_params=kwargs_params,
        physical_params=physical_params,
    )


def scale_time_delays(reference_delays_days, h0: float, reference_h0: float = REFERENCE_H0):
    """Scale fixed-lens delays with D_dt, proportional to H0^-1 at fixed Omega_m."""
    if not np.isfinite(h0) or h0 <= 0.0 or not np.isfinite(reference_h0) or reference_h0 <= 0.0:
        raise ValueError("H0 values must be finite and positive")
    delays = np.asarray(reference_delays_days, dtype=float)
    if delays.ndim != 1 or not np.isfinite(delays).all():
        raise ValueError("reference delays must be a finite one-dimensional array")
    return delays * reference_h0 / h0


def simulate_delay_measurements(lensed_supernova, *, true_h0: float = TRUE_H0,
                                error_days: float = DELAY_ERROR_DAYS,
                                seed: int = 2026) -> dict[str, object]:
    """Create seeded synthetic relative delays from the Mejiro time-delay vector."""
    if not np.isfinite(error_days) or error_days <= 0.0:
        raise ValueError("time-delay uncertainty must be finite and positive")
    reference_delays = np.asarray(lensed_supernova.get_time_delays(), dtype=float)
    reference_delays -= reference_delays[0]
    expected = scale_time_delays(reference_delays, true_h0)
    observed = expected + np.random.default_rng(seed).normal(0.0, error_days, len(expected))
    observed[0] = 0.0
    errors = np.full(len(expected), error_days)
    errors[0] = np.inf
    return {
        "reference_delays_days": reference_delays,
        "expected_delays_days": expected,
        "observed_delays_days": observed,
        "uncertainties_days": errors,
        "true_h0": true_h0,
        "reference_h0": REFERENCE_H0,
    }


def time_delay_log_likelihood(values, state) -> float:
    """Evaluate relative-delay likelihood for H0 with a fixed lens potential."""
    h0 = float(np.asarray(values, dtype=float)[0])
    if not np.isfinite(h0) or h0 <= 0.0:
        return -np.inf
    predicted = scale_time_delays(state["reference_delays_days"], h0, state["reference_h0"])
    observed = state["observed_delays_days"]
    errors = state["uncertainties_days"]
    valid = np.isfinite(errors) & (errors > 0.0)
    residual = (observed[valid] - predicted[valid]) / errors[valid]
    return float(-0.5 * residual @ residual)


def run_example(output_root: Path | None = None, *, sample_count: int = 3000,
                burn_count: int = 1000, chain_count: int = 4, seed: int = 2026,
                typefileplot: str = "png") -> dict[str, object]:
    """Run PCAT on a transparent synthetic fixed-lens time-delay data set."""
    if sample_count < 100 or burn_count < 0 or chain_count < 2:
        raise ValueError("Use at least 100 draws, nonnegative burn-in, and two chains")
    if typefileplot not in {"png", "pdf"}:
        raise ValueError("typefileplot must be 'png' or 'pdf'")
    lensed_supernova = make_synthetic_lensed_supernova()
    measurements = simulate_delay_measurements(lensed_supernova, seed=seed)
    state = {
        "reference_delays_days": measurements["reference_delays_days"],
        "observed_delays_days": measurements["observed_delays_days"],
        "uncertainties_days": measurements["uncertainties_days"],
        "reference_h0": measurements["reference_h0"],
    }
    output_root = Path(output_root) if output_root is not None else Path(__file__).resolve().parent / "runs"
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    run_directory = output_root / run_tag
    run_directory.mkdir(parents=True, exist_ok=False)
    chains, log_probability = sample_fixed_chains(
        state,
        time_delay_log_likelihood,
        None,
        ("H0",),
        ("self",),
        (H0_PRIOR[0],),
        (H0_PRIOR[1],),
        None,
        None,
        None,
        chain_count,
        sample_count,
        burn_count,
        pathbase=run_directory / "pcat",
        typeverb=0,
        seed=seed,
        booladaptstdp=True,
    )
    samples = chains[:, :, 0].reshape(-1)
    grid = np.linspace(*H0_PRIOR, 1201)
    log_likelihood = np.array([
        time_delay_log_likelihood([h0], state) for h0 in grid
    ])
    density = np.exp(log_likelihood - np.max(log_likelihood))
    density /= np.trapezoid(density, grid)
    quantiles = np.quantile(samples, (0.16, 0.5, 0.84))

    configure_example_style("white")
    figure, axis = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    axis.hist(samples, bins=40, density=True, color="#007360", alpha=0.65,
              label="PCAT posterior samples")
    axis.plot(grid, density, color="#A51430", linewidth=1.8,
              label="Likelihood grid under uniform prior")
    axis.axvline(measurements["true_h0"], color="black", linestyle="--",
                 label=fr"Injected $H_0={measurements['true_h0']:.1f}$")
    axis.set(xlabel=r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]", ylabel="Posterior density")
    axis.legend(frameon=False)
    figure_path = save_example_figure(
        figure, run_directory, f"time_delay_h0_{run_tag}", typefileplot
    )
    summary = {
        "run_tag": run_tag,
        "mejiro_object": lensed_supernova.name,
        "lens_model": "SIE + external shear + fixed convergence",
        "lens_redshift": lensed_supernova.z_lens,
        "source_redshift": lensed_supernova.z_source,
        "reference_h0_km_s_mpc": measurements["reference_h0"],
        "injected_h0_km_s_mpc": measurements["true_h0"],
        "prior_h0_km_s_mpc": list(H0_PRIOR),
        "reference_delays_days": measurements["reference_delays_days"].tolist(),
        "observed_delays_days": measurements["observed_delays_days"].tolist(),
        "time_delay_uncertainty_days": DELAY_ERROR_DAYS,
        "posterior_quantiles_km_s_mpc": quantiles.tolist(),
        "retained_samples": len(samples),
        "figure": str(figure_path),
        "assumptions": (
            "Synthetic delays are generated from a fixed lens potential. H0 changes only the time-delay distance, "
            "so relative delays scale as H0_reference/H0 at fixed Omega_m. Lens-model and mass-sheet uncertainties "
            "are omitted; this example demonstrates the PCAT likelihood path, not a cosmographic measurement."
        ),
    }
    summary_path = run_directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**summary, "run_directory": run_directory, "summary_path": summary_path,
            "figure_path": figure_path, "chains": chains, "log_probability": log_probability}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbsamp", type=int, default=3000)
    parser.add_argument("--numbburn", type=int, default=1000)
    parser.add_argument("--numbproc", type=int, default=4)
    parser.add_argument("--output-root", type=Path)
    add_smoke_argument(parser)
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    result = run_example(
        arguments.output_root,
        sample_count=250 if arguments.smoke else arguments.numbsamp,
        burn_count=200 if arguments.smoke else arguments.numbburn,
        chain_count=2 if arguments.smoke else arguments.numbproc,
        typefileplot=arguments.typefileplot,
    )
    print(f"PCAT median H0 = {result['posterior_quantiles_km_s_mpc'][1]:.2f} km s^-1 Mpc^-1")
    print(f"Wrote {result['figure_path']}")


if __name__ == "__main__":
    from tdpy.verbosity import print

    main()