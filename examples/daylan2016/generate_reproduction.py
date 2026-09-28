#!/usr/bin/env python3
"""Reproduce the scaled mock-catalog analysis of Daylan et al. (2016)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_ROOT = Path(__file__).resolve().parent / "pcat-output"
RUN_NAME = "daylan2016_mock"


def build_configuration(smoke: bool = False, typefileplot: str = "png") -> dict[str, object]:
    """Return the published mock-population assumptions in a runnable PCAT setup."""
    number_sources = 12 if smoke else 300
    number_sweeps = 10_000 if smoke else 1_000_000
    number_samples = 1_000 if smoke else 10_000
    number_side = 12 if smoke else 100
    return {
        "typeexpr": "chan",
        "typedata": "simu",
        "typepixl": "cart",
        "boolbindspat": True,
        "booldiag": False,
        "typeelem": ["lghtpnts"],
        "numbelempop0reg0": number_sources,
        "truenumbelempop0": number_sources,
        "truemaxmnumbelempop0": number_sources,
        "truefluxdistslop": -1.8,
        "fittminmnumbelempop0": max(1, number_sources // 2),
        "fittmaxmnumbelempop0": 2 * number_sources,
        "dicttrue": {"typeelemspateval": ["full"]},
        "dictfitt": {"typeelemspateval": ["full"]},
        "typeseed": 160704637,
        "typeseedelem": 1607,
        "numbsidecart": number_side,
        "numbswep": number_sweeps,
        "numbsamp": number_samples,
        "numbswepplot": 1_000 if smoke else 100_000,
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


def run_reproduction(smoke: bool = False, typefileplot: str = "png") -> object:
    """Run PCAT and write its chain and figures under the example directory."""
    import pcat

    return pcat.main.sample(**build_configuration(smoke, typefileplot))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run a small pipeline check.")
    parser.add_argument("--fresh", action="store_true", help="Remove cached output first.")
    parser.add_argument("--typefileplot", choices=("png", "pdf"), default="png")
    arguments = parser.parse_args()
    if arguments.fresh:
        for path in (
            OUTPUT_ROOT / "data" / "outp" / RUN_NAME,
            OUTPUT_ROOT / "visuals" / RUN_NAME,
        ):
            if path.exists():
                print(f"Removing cached output {path}...")
                shutil.rmtree(path)
    run_reproduction(arguments.smoke, arguments.typefileplot)
    print(f"PCAT wrote outputs under {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())