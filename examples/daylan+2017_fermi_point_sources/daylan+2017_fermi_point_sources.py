#!/usr/bin/env python3
"""Run a synthetic PCAT analysis inspired by Daylan et al. (2017)."""

from __future__ import annotations

from tdpy.verbosity import print

import argparse
from tdpy.cli import add_plot_arguments
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_NAME = "daylan2017_mock"
OUTPUT_ROOT = Path(__file__).resolve().parent
FIELD_HALF_WIDTH_DEG = 20.0  # [deg]
ENERGY_BIN_EDGES_GEV = np.array([0.3, 1.0, 3.0, 10.0])  # [GeV]
NUMBER_EVENT_CLASSES = 2


def build_ngpc_diffuse_template(number_side: int) -> np.ndarray:
    """Return a normalized, deterministic proxy for high-latitude dust emission."""
    pixel_size_deg = 2.0 * FIELD_HALF_WIDTH_DEG / number_side  # [deg]
    coordinate_deg = np.linspace(  # [deg]
        -FIELD_HALF_WIDTH_DEG + 0.5 * pixel_size_deg,
        FIELD_HALF_WIDTH_DEG - 0.5 * pixel_size_deg,
        number_side,
    )
    xpos_deg, ypos_deg = np.meshgrid(coordinate_deg, coordinate_deg, indexing="ij")

    # Broad, curved filaments approximate the dust-traced morphology of the
    # Fermi diffuse model at high Galactic latitude without claiming archival data.
    filament_one = np.exp(
        -0.5 * ((ypos_deg - 0.18 * xpos_deg - 3.0 * np.sin(xpos_deg / 7.0)) / 2.8) ** 2
    )
    filament_two = np.exp(
        -0.5 * ((xpos_deg + 0.28 * ypos_deg - 7.0 * np.cos(ypos_deg / 8.0)) / 3.6) ** 2
    )
    structure = (
        0.55
        + 0.30 * filament_one
        + 0.18 * filament_two
        + 0.07 * np.sin(xpos_deg / 3.4) * np.cos(ypos_deg / 4.7)
    )
    structure /= np.mean(structure)

    template = np.broadcast_to(
        structure.reshape(1, number_side**2, 1),
        (ENERGY_BIN_EDGES_GEV.size - 1, number_side**2, NUMBER_EVENT_CLASSES),
    )
    return np.array(template, copy=True)


def build_fermi_psf() -> tuple[np.ndarray, np.ndarray]:
    """Return approximate double-King parameters and angular scales for the mock."""
    angular_scale_deg = np.array(  # [deg]
        [
            [0.80, 1.20],
            [0.32, 0.48],
            [0.16, 0.24],
        ]
    )
    shape = np.empty((ENERGY_BIN_EDGES_GEV.size - 1, NUMBER_EVENT_CLASSES, 5))
    shape[...] = np.array([0.78, 2.4, 2.2, 2.0, 0.82])
    parameters = np.empty(shape.size)
    for event_class in range(NUMBER_EVENT_CLASSES):
        for energy_bin in range(ENERGY_BIN_EDGES_GEV.size - 1):
            start = event_class * 5 * (ENERGY_BIN_EDGES_GEV.size - 1) + energy_bin * 5
            parameters[start : start + 5] = shape[energy_bin, event_class]
    return parameters, np.deg2rad(angular_scale_deg)


def build_configuration(smoke: bool = False, typefileplot: str = "png",
                        quick: bool = False) -> dict[str, object]:
    """Return the published mock-population assumptions in a runnable PCAT setup."""
    number_sources = 40 if smoke else 300
    reduced_run = smoke or quick
    number_sweeps = 10_000 if reduced_run else 1_000_000
    number_samples = 1_000 if reduced_run else 10_000
    number_side = 48 if reduced_run else 100
    diffuse_template = build_ngpc_diffuse_template(number_side)
    psf_parameters, psf_scale = build_fermi_psf()
    model_overrides = {
        "typeelemspateval": ["full"],
        "listnamediff": ["back0000", "back0001"],
        "sbrtbacknorm": [np.ones_like(diffuse_template), diffuse_template],
        "psfpexpr": psf_parameters,
    }
    return {
        "typeexpr": "ferm",
        "typedata": "simu",
        "typepixl": "cart",
        "boolforccart": True,
        "boolbindspat": True,
        "booldiag": False,
        "typeelem": ["lghtpnts"],
        "numbelempop0reg0": number_sources,
        "truenumbelempop0": number_sources,
        "truemaxmnumbelempop0": number_sources,
        "truefluxdistslop": -1.8,
        "fittminmnumbelempop0": max(1, number_sources // 2),
        "fittmaxmnumbelempop0": 2 * number_sources,
        "dicttrue": model_overrides,
        "dictfitt": model_overrides,
        "typeseed": 160704637,
        "typeseedelem": 1607,
        "numbsidecart": number_side,
        "maxmgangdata": np.deg2rad(FIELD_HALF_WIDTH_DEG),
        "strgexpo": 1.0e10,  # [cm^2 s]
        "indxdqltfull": np.arange(NUMBER_EVENT_CLASSES),
        "indxdqltincl": np.arange(NUMBER_EVENT_CLASSES),
        "indxenerincl": np.arange(ENERGY_BIN_EDGES_GEV.size - 1),
        "fermscalfact": psf_scale,
        "lablxpos": r"\theta_1",
        "lablypos": r"\theta_2",
        "numbswep": number_sweeps,
        "numbsamp": number_samples,
        "numbburn": number_sweeps // 3,
        "numbproc": 4,
        "inittype": "rand",
        "numbframanim": 24,
        "numbswepplot": 1_000 if smoke else 100_000,
        "probtran": 0.7,
        "probspmr": 0.4,
        "boolcondcatl": True,
        "boolmakeplot": True,
        "boolmakeplotinit": True,
        "boolmakeplotfram": True,
        "boolmakeplotfinlpost": True,
        "makeanim": True,
        "typefileplot": typefileplot,
        "typeverb": 0,
        "strgcnfg": RUN_NAME,
        "pathbase": str(OUTPUT_ROOT),
    }


def run_reproduction(smoke: bool = False, typefileplot: str = "png",
                     quick: bool = False) -> object:
    """Run PCAT and write its chain and figures under the example directory."""
    from pcat.sampling import sample

    return sample(**build_configuration(smoke, typefileplot, quick))


def read_posterior(run: object) -> object:
    """Load the finalized posterior produced by a PCAT run."""
    from pcat.main import readfile

    return readfile(run.pathoutpcnfg + "gdatfinlpost")


def render_source_frames(posterior: object) -> list[Path]:
    """Overlay PCAT source catalogs, from the prior draw to posterior samples, on the simulated count map."""
    import matplotlib.pyplot as plt
    from pcat.plotting import animation_phase_label, animation_states

    counts = np.asarray(posterior.cntpdata, dtype=float)[0].sum(axis=-1)
    side = int(np.sqrt(counts.size))
    if side * side != counts.size:
        raise ValueError("Fermi source frames require a square count map")
    counts = counts.reshape(side, side)
    maximum = float(np.arcsinh(counts.max()))
    output_directory = OUTPUT_ROOT / "visuals"
    output_directory.mkdir(parents=True, exist_ok=True)
    for old_frame in output_directory.glob("fermi_ngpc_sources_swep*.png"):
        print(f"Removing previous frame {old_frame}...")
        old_frame.unlink()
    paths = []
    snapshots = animation_states(posterior)
    fluxes = np.concatenate([np.asarray(snapshot["dictelem"][0]["flux"]) for snapshot in snapshots])
    low, high = np.log10(fluxes.min()), np.log10(fluxes.max())
    for snapshot in snapshots:
        sources = snapshot["dictelem"][0]
        sizes = 20 + 120 * (np.log10(sources["flux"]) - low) / max(high - low, 1e-9)  # [point^2]
        figure, axis = plt.subplots(figsize=(5.2, 5.2), facecolor="white")
        axis.imshow(np.arcsinh(counts).T, origin="lower", cmap="Greys", vmin=0,
                    vmax=max(maximum, 1.0), extent=[-FIELD_HALF_WIDTH_DEG, FIELD_HALF_WIDTH_DEG] * 2)
        axis.scatter(np.rad2deg(sources["xpos"]), np.rad2deg(sources["ypos"]), s=sizes,
                     facecolors="none", edgecolors="#c13d31", linewidths=1.5,
                     label=f"PCAT sources ({len(sources['xpos'])})")
        axis.set(xlim=(-FIELD_HALF_WIDTH_DEG, FIELD_HALF_WIDTH_DEG),
                 ylim=(-FIELD_HALF_WIDTH_DEG, FIELD_HALF_WIDTH_DEG),
                 xlabel="Field offset 1 [deg]", ylabel="Field offset 2 [deg]",
                 title=animation_phase_label(snapshot))
        axis.legend(loc="upper right", facecolor="white", framealpha=1)
        axis.grid(False)
        path = output_directory / f"fermi_ngpc_sources_swep{snapshot['cntrswep']:09d}.png"
        print(f"Writing to {path}...")
        figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run a small pipeline check.")
    parser.add_argument("--quick", action="store_true", help="Run the 300-source mock at reduced resolution and depth.")
    parser.add_argument("--fresh", action="store_true", help="Remove cached output first.")
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    if arguments.fresh:
        for path in (
            OUTPUT_ROOT / "pcat_runs" / RUN_NAME / "data" / "outp" / RUN_NAME,
            OUTPUT_ROOT / "visuals",
        ):
            if path.exists():
                print(f"Removing cached output {path}...")
                shutil.rmtree(path)
    run = run_reproduction(arguments.smoke, arguments.typefileplot, arguments.quick)
    render_source_frames(read_posterior(run))
    print(f"PCAT wrote outputs under {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())