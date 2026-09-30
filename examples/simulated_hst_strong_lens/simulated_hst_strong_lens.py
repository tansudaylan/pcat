#!/usr/bin/env python
"""Run a genuine HST lensing image example so the pipeline writes its own figures."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_NAME = "simulated_hst_strong_lens"
OUTPUT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    from pcat.demo import run_pipeline_demo

    cfg = {
        "typeexpr": "HST_WFC3_IR",
        "typeelem": ["lens"],
        "numbsidecart": 80,
        "numbpixl": 80**2,
        "numbpixlcart": 80**2,
        "strgcnfg": RUN_NAME,
        "inittype": "refr",
        "truenumbelempop0": 3,
        "numbelempop0": 2,
        "fittminmnumbelem": 1,
        "fittminmnumbelempop0": 1,
        "fittmaxmnumbelempop0": 3,
        "fittminmdefs": 0.005 / (3600.0 * 180.0 / 3.141592653589793),
        "minmdefs": 0.005 / (3600.0 * 180.0 / 3.141592653589793),
    }
    run_pipeline_demo(OUTPUT_ROOT, **cfg)


if __name__ == "__main__":
    main()
