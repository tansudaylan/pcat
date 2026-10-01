import numpy as np
import pytest
import pickle
from types import SimpleNamespace

from pcat.roman_lens import (
    RomanLensConfig,
    gaussian_lens_log_likelihood,
    infer_catalog_probability,
    poisson_lens_log_likelihood,
    render_lens,
    render_lens_counts,
    run_lens_image_pipeline,
    simulate_population,
    summarize_population,
)


def test_lens_image_pipeline_writes_fit_and_parameter_visuals(tmp_path, monkeypatch):
    from pcat import main

    config = RomanLensConfig(number_side=24, pixel_scale=0.2, psf_fwhm=0.7)
    truth = np.array((1.2, 0.1, -0.2))  # [arcsec]
    observed = render_lens_counts(config, truth, 0.3, 0.7, 0.4)
    draws = np.tile(truth, (20, 1)) + np.linspace(-0.02, 0.02, 20)[:, None]
    posterior = SimpleNamespace(listpostparagenrscalbase=draws)
    def sample_with_visible_ring(**configuration):
        assert configuration["prior_minima"][0] < truth[0] < configuration["prior_maxima"][0]
        assert configuration["prior_maxima"][0] < config.number_side * config.pixel_scale / 2
        return posterior

    monkeypatch.setattr(main, "sample", sample_with_visible_ring)

    result = run_lens_image_pipeline(
        config=config,
        observed_counts=observed,
        source_size=0.3,
        source_axis_ratio=0.7,
        source_angle=0.4,
        true_parameters=truth,
        output_root=tmp_path,
        run_name="lens_test",
        visual_stem="lens_test",
    )

    assert result.posterior is posterior
    assert result.image_fit_path.is_file()
    assert result.parameter_path.is_file()
    assert result.model_counts.shape == observed.shape


def test_cluster_poisson_likelihood_prefers_the_injected_lens():
    config = RomanLensConfig(number_side=32, pixel_scale=0.2, psf_fwhm=0.7)
    true_parameters = np.array((1.2, 0.1, -0.2))  # [arcsec]
    source_size = 0.3  # [arcsec]
    observed = render_lens_counts(config, true_parameters, source_size, 0.7, 0.4)
    state = SimpleNamespace(
        lens_config=config,
        lens_observed_counts=observed,
        lens_source_size=source_size,
        lens_source_axis_ratio=0.7,
        lens_source_angle=0.4,
    )

    assert poisson_lens_log_likelihood(state, "fitt", true_parameters) > poisson_lens_log_likelihood(
        state, "fitt", np.array((1.6, 0.1, -0.2))
    )
    assert pickle.loads(pickle.dumps(poisson_lens_log_likelihood)) is poisson_lens_log_likelihood


def test_simulated_roman_source_produces_bright_einstein_arc():
    config = RomanLensConfig()
    image = render_lens(config, 0.9, 0.06, -0.04, 0.09, 0.8, 0.35)
    row, column = np.indices(image.shape)
    radius = np.hypot(row - 15, column - 15) * config.pixel_scale  # [arcsec]
    assert image[(radius > 0.8) & (radius < 1.2)].mean() > 3 * image[radius < 0.2].mean()


def test_gaussian_lens_likelihood_uses_variance_and_ignores_invalid_pixels():
    config = RomanLensConfig(number_side=24, background=0.0)
    true_parameters = np.array((0.9, 0.04, -0.03))  # [arcsec]
    image = render_lens_counts(config, true_parameters, 0.1, 0.8, 0.2)
    variance = np.full_like(image, 4.0)  # [nJy^2 pixel^-2]
    variance[0, 0] = np.nan
    state = SimpleNamespace(
        lens_config=config,
        lens_observed_image=image,
        lens_variance=variance,
        lens_source_size=0.1,
        lens_source_axis_ratio=0.8,
        lens_source_angle=0.2,
    )

    assert gaussian_lens_log_likelihood(state, "fitt", true_parameters) > gaussian_lens_log_likelihood(
        state, "fitt", np.array((1.2, 0.04, -0.03))
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


def test_zero_information_template_has_finite_catalog_probability():
    data = np.ones((5, 5))
    templates = np.zeros((1, 5, 5))

    probability, best_index, amplitude = infer_catalog_probability(
        data, data, templates, np.ones_like(data)
    )

    assert np.isfinite(probability)
    assert probability < 0.5
    assert best_index == 0
    assert amplitude == 0.0


def test_population_records_generated_scene_diagnostics():
    records, examples = simulate_population(number_lenses=4, seed=814)
    summary = summarize_population(records)

    assert records[0]["injected_signal_to_noise"] > 0.0
    assert records[-1]["injected_signal_to_noise"] == 0.0
    assert 0.6 <= records[0]["macro_einstein_radius_arcsec"] <= 1.2
    assert 0.06 <= records[0]["source_size_arcsec"] <= 0.14
    assert examples["lens_0_data"].shape == (31, 31)
    assert examples["lens_0_macro"].shape == (31, 31)
    assert np.any(examples["lens_0_residual"] != 0.0)
    assert summary["true_positive_rate_lower"] <= summary["true_positive_rate"]
    assert summary["true_positive_rate"] <= summary["true_positive_rate_upper"]
    assert summary["false_positive_rate_lower"] <= summary["false_positive_rate"]
    assert summary["false_positive_rate"] <= summary["false_positive_rate_upper"]


def test_population_threshold_controls_detection_rates():
    records, _ = simulate_population(number_lenses=20, seed=814)
    permissive = summarize_population(records, detection_threshold=0.1)
    conservative = summarize_population(records, detection_threshold=0.9)

    assert permissive["true_positive_rate"] >= conservative["true_positive_rate"]
    assert permissive["false_positive_rate"] >= conservative["false_positive_rate"]


def test_population_rejects_invalid_detection_threshold():
    records, _ = simulate_population(number_lenses=4, seed=814)

    with pytest.raises(ValueError, match="between zero and one"):
        summarize_population(records, detection_threshold=1.1)


def test_population_requires_injected_and_null_cohorts():
    with pytest.raises(ValueError, match="injected and null"):
        simulate_population(number_lenses=1)
