#!/usr/bin/env python3
"""Run PCAT on a Daylan et al. (2018)-inspired mock strong lens."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_ROOT = Path(__file__).resolve().parent


def build_configuration(one_subhalo: bool = False, smoke: bool = False,
                        typefileplot: str = "png") -> dict[str, object]:
    """Set up the simulated UVIS lens and variable- or fixed-size catalog fit."""
    number_subhalos = 3 if smoke else 25
    return {
        "typeexpr": "HST_WFC3_UVIS",
        "typeelem": ["lens"],
        "numbsidecart": 40 if smoke else 100,
        "truenumbelempop0": number_subhalos,
        "fittminmnumbelempop0": 1,
        "fittmaxmnumbelempop0": 1 if one_subhalo else (4 if smoke else 100),
        "numbelempop0": 1 if one_subhalo else (2 if smoke else 25),
        "inittype": "refr",
        "typeseed": 2018,
        "typeseedelem": 25,
        "numbswep": 2 if smoke else 20_000,
        "numbsamp": 1 if smoke else 1_000,
        "numbswepplot": 1 if smoke else 2_000,
        "makeanim": not smoke,
        "probspmr": 0.0 if one_subhalo else 0.5,
        "strgcnfg": "daylan2018_one_subhalo" if one_subhalo else "daylan2018_catalog",
        "typefileplot": typefileplot,
    }


def run_reproduction(one_subhalo: bool = False, smoke: bool = False,
                     typefileplot: str = "png", fresh: bool = False) -> None:
    """Generate PCAT image, model, residual, and catalog figures."""
    from pcat.demo import run_pipeline_demo

    configuration = build_configuration(one_subhalo, smoke, typefileplot)
    output_root = OUTPUT_ROOT / str(configuration["strgcnfg"])
    if fresh and output_root.exists():
        print(f"Removing cached output {output_root}...")
        shutil.rmtree(output_root)
    covariance_dir = output_root / "visuals/post/finl/varbscal/cova"
    print(f"Writing to {covariance_dir}...")
    covariance_dir.mkdir(parents=True, exist_ok=True)
    run_pipeline_demo(
        output_root,
        **configuration,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run a short pipeline check.")
    parser.add_argument("--fresh", action="store_true", help="Remove cached output first.")
    parser.add_argument("--one-subhalo", action="store_true", help="Fit one subhalo instead of a variable catalog.")
    parser.add_argument("--typefileplot", choices=("png", "pdf"), default="png")
    arguments = parser.parse_args()
    run_reproduction(
        arguments.one_subhalo,
        arguments.smoke,
        arguments.typefileplot,
        arguments.fresh,
    )


if __name__ == "__main__":
    main()