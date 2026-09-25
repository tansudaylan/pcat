#!/usr/bin/env python3
"""Configure PCAT detection of simulated spectral sources with Voigt profiles."""

import argparse

import numpy as np

import pcat


def build_configurations():
    """Return the nominal and high-signal simulated spectral configurations."""
    angular_conversion = 3600.0 * 180.0 / np.pi  # [arcsec rad^-1]
    common = {
        "typeexpr": "fire",
        "spectype": ["voig"],
        "strgexpo": 1.0e3,
        "spatdisttype": ["line"],
        "typeelem": ["lghtlinevoig"],
        "boolmakeplotinit": True,
        "maxmgangdata": 100.0 / angular_conversion,  # [rad]
        "numbsidecart": 1,
        "anlytype": "spec",
        "numbelempop0reg0": 20,
        "probspmr": 0.0,
        "numbswep": 100000,
        "numbsamp": 1000,
    }
    names = ["nomi", "s2nrhigh"]
    variations = {name: {} for name in names}
    variations["s2nrhigh"]["strgexpo"] = 1.0e5
    return common, variations, names


def run_voigt_profile_detection(configuration=None):
    """Run the selected simulated Voigt-profile PCAT configuration."""
    common, variations, names = build_configurations()
    return pcat.main.sample_parallel(
        variations,
        names,
        dictpcatinpt=common,
        boolexecpara=False,
        strgcnfgextnexec=configuration,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=("nomi", "s2nrhigh"), default="nomi")
    arguments = parser.parse_args()
    run_voigt_profile_detection(arguments.configuration)


if __name__ == "__main__":
    main()

