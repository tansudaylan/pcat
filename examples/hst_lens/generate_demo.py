#!/usr/bin/env python
"""Run a genuine HST lensing image example so the pipeline writes its own figures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_ROOT = Path(__file__).resolve().parent / "pcat-output"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["PCAT_DATA_PATH"] = str(OUTPUT_ROOT)


def main() -> None:
    from pcat import main as pcat_main

    cfg = {
        "typeexpr": "HST_WFC3_IR",
        "typedata": "simu",
        "typepixl": "cart",
        "boolbindspat": True,
        "numbsidecart": 80,
        "numbpixl": 80**2,
        "numbpixlcart": 80**2,
        "booldiag": False,
        "typeverb": 0,
        "numbswep": 2,
        "numbsamp": 1,
        "boolmakeplot": True,
        "boolmakeplotinit": True,
        "boolmakeplotfram": True,
        "pathbase": str(OUTPUT_ROOT),
        "strgcnfg": "hst_lens_demo",
        "inittype": "refr",
        "numbelempop0": 1,
        "fittminmnumbelem": 1,
        "fittminmnumbelempop0": 1,
        "fittmaxmnumbelempop0": 1,
        "fittminmdefs": 0.005 / (3600.0 * 180.0 / 3.141592653589793),
        "minmdefs": 0.005 / (3600.0 * 180.0 / 3.141592653589793),
    }
    pcat_main.sample(**cfg)
    print(f"PCAT wrote outputs under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
