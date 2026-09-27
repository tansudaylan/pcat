#!/usr/bin/env python
"""Run a genuine PCAT Chandra-style source-detection pipeline example and let it produce the figures."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_ROOT = Path(__file__).resolve().parent / "pcat-output"


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
        "numbswep": 4,
        "numbsamp": 2,
        "numbsidecart": 8,
        "strgcnfg": "chan_demo",
    }
    run_pipeline_demo(OUTPUT_ROOT, **cfg)


if __name__ == "__main__":
    main()
