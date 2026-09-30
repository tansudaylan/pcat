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


def display_posterior_maps(
    posterior: object,
    relative_directory: str,
    filename_prefix: str,
    channels: tuple[int, ...] | None = None,
) -> list[Path]:
    """Plot observed counts, posterior median models, and residuals from a state."""
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.display import display

    counts = np.asarray(posterior.cntpdata, dtype=float)
    models = np.asarray(posterior.listpostcntpmodl, dtype=float)
    grid_shape = tuple(int(size) for size in posterior.shapcart)
    if counts.ndim != 3 or models.ndim != 4 or models.shape[1:] != counts.shape:
        raise ValueError("posterior count arrays must have shapes (channels, pixels, classes) and (samples, channels, pixels, classes)")
    if len(grid_shape) != 2 or int(np.prod(grid_shape)) != counts.shape[1]:
        raise ValueError("posterior.shapcart must describe the flattened pixel axis")
    if channels is None:
        channels = tuple(np.unique(np.linspace(0, counts.shape[0] - 1, min(3, counts.shape[0])).astype(int)))
    if not channels or any(index < 0 or index >= counts.shape[0] for index in channels):
        raise ValueError("channels must select at least one available image channel")

    example_root = Path(__file__).resolve().parents[1] / "examples" / relative_directory
    visuals_path = example_root / "visuals"
    visuals_path.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for channel in channels:
        data_map = np.sum(counts[channel], axis=-1).reshape(grid_shape)
        model_map = np.median(np.sum(models[:, channel], axis=-1), axis=0).reshape(grid_shape)
        residual_map = (data_map - model_map) / np.sqrt(np.maximum(model_map, 1.0))
        stretch_scale = max(float(np.median(data_map[data_map > 0.0])) if np.any(data_map > 0.0) else 1.0, 1.0)
        stretch_maximum = np.arcsinh(max(float(data_map.max()), float(model_map.max())) / stretch_scale)
        figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.3), constrained_layout=True)
        for axis, image, title in zip(
            axes[:2], (data_map, model_map), ("Observed counts", "Posterior median model")
        ):
            artist = axis.imshow(
                np.arcsinh(image / stretch_scale), origin="lower", cmap="magma",
                vmin=0.0, vmax=max(stretch_maximum, np.finfo(float).eps),
            )
            axis.set_title(title)
            axis.set_xlabel("Pixel [pixel]")
            axis.set_ylabel("Pixel [pixel]")
            figure.colorbar(artist, ax=axis, label=f"arcsinh(counts / {stretch_scale:.2g})")
        artist = axes[2].imshow(
            residual_map, origin="lower", cmap="coolwarm", vmin=-5.0, vmax=5.0
        )
        axes[2].set_title("Posterior residual")
        axes[2].set_xlabel("Pixel [pixel]")
        axes[2].set_ylabel("Pixel [pixel]")
        figure.colorbar(artist, ax=axes[2], label="Residual [model standard deviations]")
        output_path = visuals_path / f"{filename_prefix}_channel{channel:02d}.png"
        print(f"Writing to {output_path}...")
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        display(figure)
        plt.close(figure)
        output_paths.append(output_path)
    return output_paths