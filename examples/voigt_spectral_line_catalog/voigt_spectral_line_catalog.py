#!/usr/bin/env python3
"""Configure PCAT detection of simulated spectral sources with Voigt profiles."""

import argparse
import shutil
from pathlib import Path

import numpy as np

from pcat import sampling


def build_configurations():
    """Return the nominal and high-signal simulated spectral configurations."""
    angular_conversion = 3600.0 * 180.0 / np.pi  # [arcsec rad^-1]
    common = {
        "typeexpr": "fire",
        "spectype": ["voig"],
        "strgexpo": 1.0e4,
        "spatdisttype": ["line"],
        "typeelem": ["lghtlinevoig"],
        "boolmakeplotinit": True,
        "maxmgangdata": 100.0 / angular_conversion,  # [rad]
        "numbsidecart": 1,
        "anlytype": "spec",
        "numbelempop0reg0": 20,
        "probtran": 0.7,
        "probspmr": 0.4,
        "typeseed": 0,
        "typeseedelem": 17,
        "inittype": "rand",
        "numbproc": 4,
        "numbframanim": 24,
        "numbswep": 200000,
        "numbburn": 60000,
        "numbsamp": 20000,
        "numbswepplot": 10000,
        "makeanim": True,
        "boolcheckconv": True,
        "numbsampconvmin": 3000,
        "numbsampconvcheck": 500,
        "numbsampconveffc": 500.0,
        "maxmconvrhat": 1.01,
        "numbconvpass": 2,
        "stdvpropelemfire": [5.0e-4, 5.0e-4, 2.0e-3, 2.0e-3],
        # a flux floor above half the fainter line keeps one line from being fit as two partial lines
        "limtparaelem": {"flux": (0.2, 10.0)},
        "booladaptstdp": True,
        "boolburntmpr": True,
        "factburntmpr": 0.8,
    }
    names = ["nomi", "s2nrhigh"]
    variations = {name: {} for name in names}
    variations["s2nrhigh"]["strgexpo"] = 1.0e5
    return common, variations, names


def run_voigt_profile_detection(configuration="nomi", smoke=False):
    """Run the selected simulated Voigt-profile PCAT configuration."""
    common, variations, _ = build_configurations()
    common.update(variations[configuration])
    common.update(
        strgcnfg=f"voigt_{configuration}",
        pathbase=str(Path(__file__).parent),
        truenumbelempop0=2,
        fittminmnumbelempop0=1,
        fittmaxmnumbelempop0=4,
        dicttrue={"typeelem": ["lghtlinevoig"], "spectype": ["voig"]},
        dictfitt={"typeelem": ["lghtlinevoig"], "spectype": ["voig"]},
    )
    if smoke:
        common.update(
            numbswep=120000,
            numbburn=60000,
            numbsamp=4000,
            boolcheckconv=False,
            numbswepplot=10000,
            boolmakeplotfram=True,
            boolmakeplotfinlpost=True,
            makeanim=True,
            booldiag=False,
            typeverb=0,
        )
    run_root = Path(__file__).parent / "pcat_runs" / common["strgcnfg"]
    cached_state = run_root / "data" / "outp" / common["strgcnfg"]
    if cached_state.exists():
        print(f"Removing cached PCAT state {cached_state}...")
        shutil.rmtree(cached_state)
    for old_frame in (run_root / "visuals" / "post" / "fram").glob("*_swep*.png"):
        print(f"Removing previous frame {old_frame}...")
        old_frame.unlink()
    return sampling.sample(**common)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=("nomi", "s2nrhigh"), default="nomi")
    parser.add_argument(
        "--smoke", action="store_true", help="Run a short visualization smoke test."
    )
    arguments = parser.parse_args()
    run_voigt_profile_detection(arguments.configuration, smoke=arguments.smoke)


if __name__ == "__main__":
    main()
