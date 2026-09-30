"""Shared orchestration for lightweight PCAT pipeline demonstrations."""

from __future__ import annotations

import os
import runpy
import sys
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


def run_example_script(relative_path: str, *arguments: str) -> None:
    """Run a repository example from a notebook without leaking CLI arguments."""
    script = Path(__file__).resolve().parents[1] / "examples" / relative_path
    print(f"Reading from {script}...")
    previous_arguments = sys.argv
    try:
        sys.argv = [str(script), *arguments]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous_arguments


def display_example_visuals(relative_directory: str, *patterns: str) -> None:
    """Show one saved figure per requested scientific product in a notebook."""
    from IPython.display import Image, display

    example_root = Path(__file__).resolve().parents[1] / "examples" / relative_directory
    for pattern in patterns:
        matches = sorted(example_root.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No example figure matches {example_root / pattern}")
        figure_path = matches[-1]
        print(f"Reading from {figure_path}...")
        display(Image(filename=str(figure_path)))