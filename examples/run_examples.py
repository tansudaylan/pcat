#!/usr/bin/env python
"""Run all PCAT example scripts and generate figure outputs in each example folder."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent

PIPELINE_EXAMPLES = [
    (ROOT / "gmix_demo" / "generate_demo.py", (), ROOT / "gmix_demo", "gmix_demo", True, True),
    (ROOT / "chan_demo" / "generate_demo.py", (), ROOT / "chan_demo", "chan_demo", True, False),
    (ROOT / "Daylan+2017" / "generate_reproduction.py", ("--smoke",), ROOT / "Daylan+2017", "daylan2017_mock", True, True),
    (ROOT / "hst_lens" / "generate_demo.py", (), ROOT / "hst_lens", "hst_lens_demo", True, True),
    (ROOT / "voigt-profile" / "pcat_voigt_profile_detection.py", ("--smoke",), ROOT / "voigt-profile", "voigt_nomi", False, True),
]

UTILITY_EXAMPLES = [
    (
        ROOT / "catalog_association" / "catalog_association.py",
        ROOT / "catalog_association" / "visuals" / "catalog_association.png",
    ),
    (
        ROOT / "psf_subpixel" / "psf_subpixel.py",
        ROOT / "psf_subpixel" / "visuals" / "psf_subpixel.png",
    ),
    (
        ROOT / "roman_lens_catalog" / "roman_lens_catalog_diagnostic.py",
        ROOT / "roman_lens_catalog" / "visuals" / "roman_lens_catalog_diagnostic.png",
    ),
]


def verify_pipeline_outputs(
    output_root: Path, run_name: str, require_initial: bool = True, require_multiframe: bool = True
) -> None:
    """Require static posterior products and at least one genuine animation."""
    visual_root = output_root / "visuals"
    phases = {
        "posterior frames": list((visual_root / "post" / "fram").rglob("*.png")),
        "final plots": list((visual_root / "post" / "finl").rglob("*.png")),
        "animations": list((visual_root / "post" / "anim").rglob("*.gif")),
    }
    if require_initial:
        phases["initial plots"] = list((visual_root / "init").rglob("*.png"))
    missing = [name for name, paths in phases.items() if not paths]
    if missing:
        raise RuntimeError(f"{run_name} did not produce: {', '.join(missing)}")
    if len(phases["posterior frames"]) < 2:
        raise RuntimeError(f"{run_name} produced fewer than two posterior frames")
    if require_multiframe:
        for animation_path in phases["animations"]:
            print(f"Reading from {animation_path}...")
            with Image.open(animation_path) as animation:
                if animation.n_frames > 1:
                    break
        else:
            raise RuntimeError(f"{run_name} did not produce a multi-frame animation")


def run_script(script: Path, arguments: tuple[str, ...] = ()) -> None:
    """Execute an example in an isolated interpreter."""
    print(f"\n=== Running {script.relative_to(ROOT)} ===")
    subprocess.run([sys.executable, str(script), *arguments], cwd=ROOT.parent, check=True)


def main() -> None:
    repo_root = ROOT.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from pcat.plotting import make_posterior_animation_collage

    for script, arguments, output_root, run_name, require_initial, require_multiframe in PIPELINE_EXAMPLES:
        visual_root = output_root / "visuals"
        for path in [output_root / "data" / "outp" / run_name, visual_root]:
            if path.exists():
                print(f"Removing cached example output {path}...")
                shutil.rmtree(path)
        run_script(script, arguments)
        verify_pipeline_outputs(
            output_root,
            run_name,
            require_initial=require_initial,
            require_multiframe=require_multiframe,
        )

    make_posterior_animation_collage(
        output_path=ROOT / "pcat_posterior_samples.gif",
        examples_root=ROOT,
    )

    for script, output_path in UTILITY_EXAMPLES:
        if output_path.exists():
            output_path.unlink()
        run_script(script)
        if not output_path.is_file():
            raise RuntimeError(f"{script.name} did not produce {output_path}")


if __name__ == "__main__":
    main()
