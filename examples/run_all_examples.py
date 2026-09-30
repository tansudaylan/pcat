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
    (ROOT / "gaussian_mixture_catalog" / "gaussian_mixture_catalog.py", (), ROOT / "gaussian_mixture_catalog", "gaussian_mixture_catalog", True, True),
    (ROOT / "chandra_point_source_catalog" / "chandra_point_source_catalog.py", (), ROOT / "chandra_point_source_catalog", "chandra_point_source_catalog", True, False),
    (ROOT / "daylan_2017_fermi_point_sources" / "daylan_2017_fermi_point_sources.py", ("--smoke",), ROOT / "daylan_2017_fermi_point_sources", "daylan2017_mock", True, True),
    (ROOT / "simulated_hst_strong_lens" / "simulated_hst_strong_lens.py", (), ROOT / "simulated_hst_strong_lens", "simulated_hst_strong_lens", True, True),
    (ROOT / "voigt_spectral_line_catalog" / "voigt_spectral_line_catalog.py", ("--smoke",), ROOT / "voigt_spectral_line_catalog", "voigt_nomi", False, True),
]

UTILITY_EXAMPLES = [
    (
        ROOT / "catalog_association_completeness_purity" / "catalog_association_completeness_purity.py",
        ROOT / "catalog_association_completeness_purity" / "visuals" / "catalog_association_completeness_purity.png",
    ),
    (
        ROOT / "subpixel_psf_reconstruction" / "subpixel_psf_reconstruction.py",
        ROOT / "subpixel_psf_reconstruction" / "visuals" / "subpixel_psf_reconstruction.png",
    ),
    (
        ROOT / "roman_strong_lens_perturber_catalog" / "roman_strong_lens_perturber_catalog.py",
        ROOT / "roman_strong_lens_perturber_catalog" / "visuals" / "roman_strong_lens_perturber_catalog.png",
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
