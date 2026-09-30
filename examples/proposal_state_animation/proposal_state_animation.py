#!/usr/bin/env python3
"""Generate retained-state and every-proposal animations from one PCAT run.

The data are a seeded simulated spectrum containing two Voigt emission lines.
PCAT fits one to three lines with within-model, birth, death, split, and merge
proposals. The short run demonstrates output semantics rather than convergence.
"""

from tdpy.verbosity import print

import argparse
from pathlib import Path
import shutil

from pcat import sampling
from pcat.demo import load_example_namespace


EXAMPLE_PATH = Path(__file__).resolve().parent
RUN_NAME = EXAMPLE_PATH.name


def run_example(number_sweeps=24):
    """Run the simulated Voigt problem and return both animation paths."""
    namespace = load_example_namespace(
        "voigt_spectral_line_catalog/voigt_spectral_line_catalog.py"
    )
    configuration, variations, _ = namespace["build_configurations"]()
    configuration.update(variations["nomi"])
    configuration.update(
        strgcnfg=RUN_NAME,
        pathbase=str(EXAMPLE_PATH),
        truenumbelempop0=2,
        fittminmnumbelempop0=1,
        fittmaxmnumbelempop0=3,
        dicttrue={"typeelem": ["lghtlinevoig"], "spectype": ["voig"]},
        dictfitt={"typeelem": ["lghtlinevoig"], "spectype": ["voig"]},
        probtran=0.8,
        probspmr=0.5,
        numbswep=number_sweeps,
        numbburn=max(2, number_sweeps // 6),
        numbsamp=max(4, number_sweeps // 3),
        numbswepplot=max(1, number_sweeps // 6),
        boolcheckconv=False,
        boolmakeplot=True,
        boolmakeplotinit=False,
        boolmakeplotfram=True,
        boolmakeplotfinlpost=False,
        makeanim=True,
        boolmakeanimprop=True,
        booldiag=False,
        numbproc=1,
        typeverb=0,
    )
    for path in (EXAMPLE_PATH / "data" / "outp" / RUN_NAME, EXAMPLE_PATH / "visuals"):
        if path.exists():
            print(f"Removing cached example output {path}...")
            shutil.rmtree(path)
    sampling.sample(**configuration)

    animation_root = EXAMPLE_PATH / "visuals" / "post" / "anim"
    posterior_path = animation_root / "proposal_sequence.gif"
    proposal_path = animation_root / "proposal_candidates.gif"
    if not posterior_path.is_file() or not proposal_path.is_file():
        raise RuntimeError("PCAT did not produce both requested animations.")
    return {
        "posterior_animation": posterior_path,
        "proposal_animation": proposal_path,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbswep", type=int, default=24)
    arguments = parser.parse_args()
    products = run_example(arguments.numbswep)
    for label, path in products.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()