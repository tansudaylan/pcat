#!/usr/bin/env python3
"""Run a synthetic PCAT analysis inspired by Daylan et al. (2017)."""

from __future__ import annotations

from tdpy.verbosity import print

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run a small pipeline check.")
    parser.add_argument("--quick", action="store_true", help="Run the 300-source mock at reduced resolution and depth.")
    parser.add_argument("--fresh", action="store_true", help="Remove cached output first.")
    parser.add_argument("--typefileplot", choices=("png", "pdf"), default="png")
    arguments = parser.parse_args()
    if arguments.fresh:
        for path in (
            OUTPUT_ROOT / "data" / "outp" / RUN_NAME,
            OUTPUT_ROOT / "visuals",
        ):
            if path.exists():
                print(f"Removing cached output {path}...")
                shutil.rmtree(path)
    run_reproduction(arguments.smoke, arguments.typefileplot, arguments.quick)
    print(f"PCAT wrote outputs under {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())