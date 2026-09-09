#!/usr/bin/env python
"""Run a genuine PCAT Gaussian-mixture pipeline example and let it write its own figures."""

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
        "typeexpr": "gmix",
        "typedata": "simu",
        "typepixl": "cart",
        "boolbindspat": True,
        "numbsidecart": 8,
        "numbspatdims": 2,
        "booldiag": False,
        "typeverb": 0,
        "numbswep": 2,
        "numbsamp": 1,
        "boolmakeplot": True,
        "boolmakeplotinit": True,
        "boolmakeplotfram": True,
        "pathbase": str(OUTPUT_ROOT),
        "strgcnfg": "gmix_demo",
    }
    pcat_main.sample(**cfg)
    print(f"PCAT wrote outputs under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
