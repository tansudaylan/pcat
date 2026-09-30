#!/usr/bin/env python
"""Run all PCAT example scripts and generate figure outputs in each example folder."""

from __future__ import annotations

from tdpy.verbosity import print

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent

PIPELINE_EXAMPLES = [
    (
        ROOT / "gaussian_mixture_catalog" / "gaussian_mixture_catalog.py",
        (),
        ROOT / "gaussian_mixture_catalog" / "gaussian_mixture_catalog",
        "gaussian_mixture_catalog",
        True,
        True,
    ),
    (
        ROOT / "chandra_point_source_catalog" / "chandra_point_source_catalog.py",
        (),
        ROOT / "chandra_point_source_catalog" / "chandra_point_source_catalog",
        "chandra_point_source_catalog",
        True,
        True,
    ),
    (
        ROOT / "daylan+2017_fermi_point_sources" / "daylan+2017_fermi_point_sources.py",
        ("--smoke",),
        ROOT / "daylan+2017_fermi_point_sources" / "pcat_runs" / "daylan2017_mock",
        "daylan2017_mock",
        True,
        True,
    ),
    (
        ROOT / "daylan+2018_strong_lens_subhalos" / "daylan+2018_strong_lens_subhalos.py",
        ("--smoke", "--fresh"),
        ROOT / "daylan+2018_strong_lens_subhalos" / "daylan2018_catalog",
        "daylan2018_catalog",
        True,
        True,
    ),
    (
        ROOT / "simulated_hst_strong_lens" / "simulated_hst_strong_lens.py",
        (),
        ROOT / "simulated_hst_strong_lens",
        "simulated_hst_strong_lens",
        True,
        True,
    ),
    (
        ROOT / "portillo+2017_crowded_sdss_m2" / "portillo+2017_crowded_sdss_m2.py",
        (),
        ROOT / "portillo+2017_crowded_sdss_m2" / "pcat_runs" / "portillo2017_sdss_m2",
        "portillo2017_sdss_m2",
        True,
        True,
    ),
    (
        ROOT / "feder+2020_multiband_sdss_deblending" / "feder+2020_multiband_sdss_deblending.py",
        (),
        ROOT / "feder+2020_multiband_sdss_deblending" / "pcat_runs" / "feder2020_multiband_sdss",
        "feder2020_multiband_sdss",
        True,
        True,
    ),
    (
        ROOT / "butler+2022_spire_sz_component_separation" / "butler+2022_spire_sz_component_separation.py",
        (),
        ROOT / "butler+2022_spire_sz_component_separation" / "pcat_runs" / "butler2022_spire_sz",
        "butler2022_spire_sz",
        True,
        True,
    ),
    (
        ROOT / "feder+2023_point_diffuse_spire" / "feder+2023_point_diffuse_spire.py",
        (),
        ROOT / "feder+2023_point_diffuse_spire" / "pcat_runs" / "feder2023_point_diffuse_spire",
        "feder2023_point_diffuse_spire",
        True,
        True,
    ),
    (
        ROOT / "hall+2026_herschel_dsfg_multiplicity" / "hall+2026_herschel_dsfg_multiplicity.py",
        (),
        ROOT / "hall+2026_herschel_dsfg_multiplicity" / "pcat_runs" / "hall2026_dsfg_multiplicity",
        "hall2026_dsfg_multiplicity",
        True,
        True,
    ),
    (
        ROOT / "variable_number_stellar_flares" / "variable_number_stellar_flares.py",
        ("--smoke",),
        ROOT / "variable_number_stellar_flares",
        "variable_number_stellar_flares",
        False,
        True,
    ),
    (
        ROOT / "voigt_spectral_line_catalog" / "voigt_spectral_line_catalog.py",
        ("--smoke",),
        ROOT / "voigt_spectral_line_catalog" / "pcat_runs" / "voigt_nomi",
        "voigt_nomi",
        False,
        True,
    ),
]

UTILITY_EXAMPLES = [
    (
        ROOT / "catalog_association_completeness_purity" / "catalog_association_completeness_purity.py",
        ROOT / "catalog_association_completeness_purity" / "visuals" / "catalog_association_completeness_purity.png",
    ),
    (
        ROOT / "roman_strong_lens_perturber_catalog" / "roman_strong_lens_perturber_catalog.py",
        ROOT / "roman_strong_lens_perturber_catalog" / "visuals" / "roman_strong_lens_perturber_catalog.png",
    ),
    (
        ROOT / "proposal_profiling" / "proposal_profiling.py",
        ROOT / "proposal_profiling" / "visuals" / "proposal_time_per_sweep.png",
    ),
    (
        ROOT / "sampler_comparison_emcee_dynesty" / "sampler_comparison_emcee_dynesty.py",
        ROOT / "sampler_comparison_emcee_dynesty" / "visuals" / "sampler_comparison_efficiency.png",
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
    proposal_animation = visual_root / "post" / "anim" / "proposal_activity.gif"
    if not proposal_animation.is_file():
        raise RuntimeError(f"{run_name} did not produce proposal activity animation")
    print(f"Reading from {proposal_animation}...")
    with Image.open(proposal_animation) as animation:
        if animation.n_frames < 2:
            raise RuntimeError(f"{run_name} proposal activity animation has fewer than two frames")
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
