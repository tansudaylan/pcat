#!/usr/bin/env python
"""Run all PCAT example scripts and generate figure outputs in each example folder."""

from __future__ import annotations

import os
import runpy
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EXAMPLES = [
    (ROOT / "gaussian_mixture" / "generate_demo.py", "gmix_demo"),
    (ROOT / "chandra_point_source" / "generate_demo.py", "chan_demo"),
    (ROOT / "hst_lens" / "generate_demo.py", "hst_lens_demo"),
]


def main() -> None:
    repo_root = ROOT.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    for script, run_name in EXAMPLES:
        print(f"\n=== Running {script.relative_to(ROOT)} ===")
        output_root = script.parent / "pcat-output"
        for path in [output_root / "data" / "outp" / run_name, output_root / "visuals" / run_name]:
            if path.exists():
                print(f"Removing cached example output {path}...")
                shutil.rmtree(path)
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
