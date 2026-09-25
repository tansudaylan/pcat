import numpy as np

from pcat.roman_lens import (
    RomanLensConfig,
    infer_catalog_probability,
    render_lens,
    simulate_population,
)


def test_injected_perturber_raises_catalog_probability():
    config = RomanLensConfig(number_side=21, source_counts=2.0e5, number_candidates=1)
    parameters = (config, 0.8, 0.05, -0.03, 0.09, 0.7, 0.4)
    macro = render_lens(*parameters)
    perturbed = render_lens(*parameters, subhalo=(0.8, 0.0, 0.03))
    variance = perturbed + config.background + config.read_noise**2
    template = (perturbed - macro)[None, :, :]
    null_probability = infer_catalog_probability(macro + config.background, macro + config.background, template, variance)[0]
    perturbed_probability = infer_catalog_probability(perturbed + config.background, macro + config.background, template, variance)[0]

    assert np.isfinite(perturbed_probability)
    assert perturbed_probability > null_probability


def test_population_records_generated_scene_diagnostics():
    records, _ = simulate_population(number_lenses=4, seed=814)

    assert records[0]["injected_signal_to_noise"] > 0.0
    assert records[-1]["injected_signal_to_noise"] == 0.0
    assert 0.6 <= records[0]["macro_einstein_radius_arcsec"] <= 1.2
    assert 0.06 <= records[0]["source_size_arcsec"] <= 0.14