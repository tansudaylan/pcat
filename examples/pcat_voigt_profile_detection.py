#!/usr/bin/env python3
"""Configure PCAT detection of simulated spectral sources with Voigt profiles."""

import argparse
from pathlib import Path

import numpy as np

import pcat


def build_configurations():
    """Return the nominal and high-signal simulated spectral configurations."""
    angular_conversion = 3600.0 * 180.0 / np.pi  # [arcsec rad^-1]
    common = {
        "typeexpr": "fire",
        "spectype": ["voig"],
        "strgexpo": 1.0e5,
        "spatdisttype": ["line"],
        "typeelem": ["lghtlinevoig"],
        "boolmakeplotinit": True,
        "maxmgangdata": 100.0 / angular_conversion,  # [rad]
        "numbsidecart": 1,
        "anlytype": "spec",
        "numbelempop0reg0": 20,
        "probspmr": 0.0,
        "typeseed": 0,
        "typeseedelem": 17,
        "inittype": "refr",
        "numbswep": 100000,
        "numbsamp": 1000,
    }
    names = ["nomi", "s2nrhigh"]
    variations = {name: {} for name in names}
    variations["s2nrhigh"]["strgexpo"] = 1.0e6
    return common, variations, names


def run_voigt_profile_detection(configuration="nomi", smoke=False):
    """Run the selected simulated Voigt-profile PCAT configuration."""
    common, variations, _ = build_configurations()
    common.update(variations[configuration])
    common.update(
        strgcnfg=f"voigt_{configuration}",
        pathbase=str(Path(__file__).with_name("voigt-profile")),
        truenumbelempop0=2,
        fittminmnumbelempop0=1,
        fittmaxmnumbelempop0=3,
        dicttrue={"typeelem": ["lghtlinevoig"], "spectype": ["voig"]},
        dictfitt={"typeelem": ["lghtlinevoig"], "spectype": ["voig"]},
    )
    if smoke:
        common.update(
            numbswep=30,
            numbsamp=10,
            numbswepplot=3,
            boolmakeplotfram=True,
            boolmakeplotfinlpost=True,
            makeanim=True,
            booldiag=False,
            typeverb=0,
        )
    return pcat.main.sample(**common)


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
