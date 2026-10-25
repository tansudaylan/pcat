#!/usr/bin/env python3
"""Configure PCAT detection of simulated spectral sources with Voigt profiles."""

import argparse
import shutil
from pathlib import Path

import numpy as np

from pcat import sampling
from pcat.spectral import LINE_PROFILE_PARAMETERS, gaussian_lsf_kernel, spectral_profile_parameters
from tdpy.verbosity import print

SPECTRAL_EXAMPLE_PROFILES = tuple(profile for profile in LINE_PROFILE_PARAMETERS if profile != "edis")


def build_configurations(spectral_profile="voig", lsftype="none",
                         lsfresolvingpower=3000.0, lsfkernel_sigma=1.0, edisintp=None):
    """Return simulated spectral configurations for a selected line and LSF profile."""
    if spectral_profile == "edis" and edisintp is None:
        raise ValueError("The 'edis' profile requires an instrument response callable via edisintp")
    angular_conversion = 3600.0 * 180.0 / np.pi  # [arcsec rad^-1]
    parameter_scales = {
        "flux": 5.0e-4, "elin": 5.0e-4, "sigm": 2.0e-3, "gamm": 2.0e-3,
        "frac": 2.0e-2, "skew": 2.0e-2,
    }
    parameter_bounds = {"flux": (0.2, 10.0)}
    if "frac" in spectral_profile_parameters(spectral_profile):
        parameter_bounds["frac"] = (0.0, 1.0)
    if "skew" in spectral_profile_parameters(spectral_profile):
        parameter_bounds["skew"] = (-10.0, 10.0)
    common = {
        "typeexpr": "fire",
        "spectype": [spectral_profile],
        "lsftype": lsftype,
        "dictfitt": {"typeelem": ["lghtlinevoig"], "spectype": [spectral_profile]},
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
        "stdvpropelemfire": [parameter_scales[name] for name in spectral_profile_parameters(spectral_profile)],
        # a flux floor above half the fainter line keeps one line from being fit as two partial lines
        "limtparaelem": parameter_bounds,
        "booladaptstdp": True,
        "boolburntmpr": True,
        "factburntmpr": 0.8,
    }
    if spectral_profile == "edis":
        common["edisintp"] = edisintp
    names = ["nomi", "s2nrhigh"]
    variations = {name: {} for name in names}
    variations["s2nrhigh"]["strgexpo"] = 1.0e5
    if lsftype == "gaus":
        common["lsfresolvingpower"] = lsfresolvingpower
    elif lsftype == "tabu":
        common["lsfkernel"] = gaussian_lsf_kernel(lsfkernel_sigma)
    elif lsftype != "none":
        raise ValueError("lsftype must be 'none', 'gaus', or 'tabu'")
    return common, variations, names


def run_voigt_profile_detection(configuration="nomi", smoke=False, spectral_profile="voig",
                                lsftype="none", lsfresolvingpower=3000.0,
                                lsfkernel_sigma=1.0, edisintp=None):
    """Run a simulated line-catalog fit with the selected profile and instrument LSF."""
    common, variations, _ = build_configurations(
        spectral_profile, lsftype, lsfresolvingpower, lsfkernel_sigma, edisintp,
    )
    common.update(variations[configuration])
    run_name = f"voigt_{configuration}"
    if spectral_profile != "voig" or lsftype != "none":
        run_name += f"_{spectral_profile}_{lsftype}"
    common.update(
        strgcnfg=run_name,
        pathbase=str(Path(__file__).parent),
        truenumbelempop0=2,
        fittminmnumbelempop0=1,
        fittmaxmnumbelempop0=4,
        dicttrue={"typeelem": ["lghtlinevoig"], "spectype": ["voig"]},
        dictfitt={"typeelem": ["lghtlinevoig"], "spectype": [spectral_profile]},
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
    parser.add_argument("--profile", choices=SPECTRAL_EXAMPLE_PROFILES, default="voig",
                        help="Intrinsic line profile; response-dependent edis is configured via Python.")
    parser.add_argument("--lsftype", choices=("none", "gaus", "tabu"), default="none")
    parser.add_argument("--lsf-resolving-power", type=float, default=3000.0)
    parser.add_argument("--lsf-kernel-sigma", type=float, default=1.0)
    parser.add_argument(
        "--smoke", action="store_true", help="Run a short visualization smoke test."
    )
    arguments = parser.parse_args()
    run_voigt_profile_detection(
        arguments.configuration,
        smoke=arguments.smoke,
        spectral_profile=arguments.profile,
        lsftype=arguments.lsftype,
        lsfresolvingpower=arguments.lsf_resolving_power,
        lsfkernel_sigma=arguments.lsf_kernel_sigma,
    )


if __name__ == "__main__":
    main()
