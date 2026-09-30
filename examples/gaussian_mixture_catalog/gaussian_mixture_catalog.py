#!/usr/bin/env python
"""Run a genuine PCAT Gaussian-mixture pipeline example and let it write its own figures."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_NAME = "gmix_demo"
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / RUN_NAME


def main() -> None:
    from pcat.demo import run_pipeline_demo

    cfg = {
        "typeexpr": "gmix",
        "typeelem": ["clusvari"],
        "truenumbelempop0": 2,
        "fittminmnumbelempop0": 1,
        "fittmaxmnumbelempop0": 3,
        "dicttrue": {"typeelem": ["clusvari"], "typeelemspateval": ["full"]},
        "dictfitt": {"typeelem": ["clusvari"], "typeelemspateval": ["full"]},
        "numbsidecart": 8,
        "numbspatdims": 2,
        "strgexpo": 50.0,  # [arbitrary exposure units]
        "typeseedelem": 2_293,
        "inittype": "refr",
        "probspmr": 0.0,
        "numbswep": 1_000,
        "numbsamp": 100,
        "numbswepplot": 100,
        "strgcnfg": RUN_NAME,
    }
    run_pipeline_demo(OUTPUT_ROOT, **cfg)


if __name__ == "__main__":
    main()
