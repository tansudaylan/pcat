"""Enforce PCAT as the active sampler owner in the first-party ecosystem."""

from pathlib import Path

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


def test_legacy_tdpy_sampler_copies_are_removed():
    deprecated_directory = Path(__file__).resolve().parents[1] / "pcat" / "depr"
    assert not list(deprecated_directory.glob("tdpy_mcmc*_to_be_deleted.py"))