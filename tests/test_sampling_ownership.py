"""Enforce PCAT as the active sampler owner in the first-party ecosystem."""

from pathlib import Path

import numpy as np
from PIL import Image
import pcat.main
from pcat import sampling


def test_pcat_exports_fixed_and_transdimensional_sampling():
    for name in (
        "sample",
        "sample_fixed",
        "sample_fixed_chains",
        "sample_parallel",
        "sample_posterior",
    ):
        assert callable(getattr(sampling, name))


def test_fixed_sampling_dispatches_through_native_pipeline(monkeypatch):
    captured = {}
    expected = object()

    def fake_sample(**configuration):
        captured.update(configuration)
        return expected

    monkeypatch.setattr(pcat.main, "sample", fake_sample)
    result = sampling.sample_fixed(retr_llik="likelihood", strgcnfg="fixed")

    assert result is expected
    assert captured == {
        "typeexpr": "gener",
        "retr_llik": "likelihood",
        "strgcnfg": "fixed",
    }


def candidate_animation_likelihood(gdat, strgmodl, values):
    return -0.5 * values[0] ** 2


def test_fixed_sampling_records_every_proposal_candidate(tmp_path):
    sampling.sample_fixed(
        retr_llik=candidate_animation_likelihood,
        parameter_names=("x",),
        prior_types=("self",),
        prior_minima=(-2.0,),
        prior_maxima=(2.0,),
        initial_values=(0.0,),
        proposal_scales=(0.2,),
        pathbase=tmp_path,
        strgcnfg="candidate_animation",
        numbswep=4,
        numbburn=0,
        numbsamp=4,
        numbswepplot=1,
        boolmakeplot=False,
        boolmakeplotfram=False,
        boolmakeanimprop=True,
        typeverb=0,
    )

    visual_root = tmp_path / "pcat_runs" / "candidate_animation" / "visuals" / "post"
    frames = sorted((visual_root / "fram").glob("proposal_candidates_swep*.png"))
    animation_path = visual_root / "anim" / "proposal_candidates.gif"
    assert len(frames) == 4
    assert animation_path.is_file()
    assert len({Image.open(path).size for path in frames}) == 1
    with Image.open(animation_path) as animation:
        assert animation.n_frames == 4


def test_legacy_tdpy_sampler_copies_are_removed():
    deprecated_directory = Path(__file__).resolve().parents[1] / "pcat" / "depr"
    assert not list(deprecated_directory.glob("tdpy_mcmc*_to_be_deleted.py"))