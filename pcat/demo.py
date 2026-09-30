"""Shared orchestration for lightweight PCAT pipeline demonstrations."""

from __future__ import annotations

from tdpy.verbosity import print

import os
import runpy
import sys
from pathlib import Path

import matplotlib as mpl

from . import main as pcat_main


DEMO_DEFAULTS = {
    "typedata": "simu",
    "typepixl": "cart",
    "boolbindspat": True,
    "booldiag": False,
    "typeverb": 0,
    "numbswep": 100,
    "numbsamp": 20,
    "numbswepplot": 10,
    "probtran": 0.7,
    "probspmr": 0.4,
    "boolcondcatl": False,
    "boolmakeplot": True,
    "boolmakeplotinit": True,
    "boolmakeplotfram": True,
    "boolmakeplotfinlpost": True,
    "makeanim": True,
    "typefileplot": "png",
}


def run_pipeline_demo(output_root: Path, **configuration: object) -> None:
    """Run a lightweight simulated pipeline with shared plotting defaults."""

    mpl.rcParams["text.usetex"] = False
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


def load_example_namespace(relative_path: str) -> dict[str, object]:
    """Load an example's definitions without running its command-line entry point."""
    script = Path(__file__).resolve().parents[1] / "examples" / relative_path
    print(f"Reading from {script}...")
    return runpy.run_path(str(script), run_name=f"pcat_example_{script.stem}")


def load_example_posterior(relative_path: str) -> object:
    """Load a finalized PCAT posterior by its extensionless example-relative path."""
    path = Path(__file__).resolve().parents[1] / "examples" / relative_path
    print(f"Reading from {path}...")
    return pcat_main.readfile(str(path))


def display_example_visuals(relative_directory: str, *patterns: str) -> None:
    """Show one saved figure per requested scientific product in a notebook."""
    from IPython.display import Image, display

    example_root = Path(__file__).resolve().parents[1] / "examples" / relative_directory
    listpathdisplay = set()
    for pattern in patterns:
        matches = sorted(example_root.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No example figure matches {example_root / pattern}")
        figure_path = matches[-1]
        print(f"Reading from {figure_path}...")
        display(Image(filename=str(figure_path)))
        listpathdisplay.add(figure_path)

    listrootvisual = {example_root / "visuals"}
    listrootvisual.update(example_root.glob("*/visuals"))
    listrootvisual.update(example_root.glob("pcat_runs/*/visuals"))
    for pathvisual in sorted(listrootvisual):
        for name in ("proposal_activity.gif", "proposal_sequence.gif", "proposal_candidates.gif"):
            figure_path = pathvisual / "post" / "anim" / name
            if not figure_path.is_file() or figure_path in listpathdisplay:
                continue
            print(f"Reading from {figure_path}...")
            display(Image(filename=str(figure_path)))