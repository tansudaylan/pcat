#!/usr/bin/env python
"""Run a genuine PCAT Chandra-style source-detection pipeline example and let it produce the figures."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_NAME = "chandra_point_source_catalog"
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / RUN_NAME


def main() -> None:
    from pcat.demo import run_pipeline_demo

    cfg = {
        "typeexpr": "chan",
        "typeelem": ["lghtpnts"],
        "truenumbelempop0": 2,
        "fittminmnumbelempop0": 1,
        "fittmaxmnumbelempop0": 3,
        "dicttrue": {"typeelemspateval": ["full"]},
        "dictfitt": {"typeelemspateval": ["full"]},
        "probspmr": 0.4,
        "numbswep": 250,
        "numbsamp": 50,
        "numbswepplot": 25,
        "numbsidecart": 8,
        "strgcnfg": RUN_NAME,
    }
    run_pipeline_demo(OUTPUT_ROOT, **cfg)


if __name__ == "__main__":
    main()
