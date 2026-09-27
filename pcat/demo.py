"""Shared orchestration for lightweight PCAT pipeline demonstrations."""

from __future__ import annotations

import os
from pathlib import Path

from . import main as pcat_main


DEMO_DEFAULTS = {
    "typedata": "simu",
    "typepixl": "cart",
    "boolbindspat": True,
    "booldiag": False,
    "typeverb": 0,
    "numbswep": 2,
    "numbsamp": 1,
    "boolcondcatl": False,
    "boolmakeplot": True,
    "boolmakeplotinit": True,
    "boolmakeplotfram": True,
    "boolmakeplotfinlpost": True,
    "makeanim": True,
    "numbswepplot": 1,
    "typefileplot": "png",
}


def run_pipeline_demo(output_root: Path, **configuration: object) -> None:
    """Run a lightweight simulated pipeline with shared plotting defaults."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ["PCAT_DATA_PATH"] = str(output_root)
    sample_configuration = {**DEMO_DEFAULTS, **configuration, "pathbase": str(output_root)}
    pcat_main.sample(**sample_configuration)
    print(f"PCAT wrote outputs under {output_root}")