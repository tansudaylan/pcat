#!/usr/bin/env python
"""Run all PCAT example scripts and generate figure outputs in each example folder."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EXAMPLES = [
    ROOT / "gaussian_mixture" / "generate_demo.py",
    ROOT / "chandra_point_source" / "generate_demo.py",
    ROOT / "hst_lens" / "generate_demo.py",
]


def main() -> None:
    repo_root = ROOT.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    for script in EXAMPLES:
        print(f"\n=== Running {script.relative_to(ROOT)} ===")
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
