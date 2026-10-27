import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "neural_proposal_efficiency"
    / "neural_posterior_diffusion.py"
)
SPEC = importlib.util.spec_from_file_location("neural_posterior_diffusion", MODULE_PATH)
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)

IMAGE_SPEC = importlib.util.spec_from_file_location(
    "point_source_detection", MODULE_PATH.with_name("point_source_detection.py")
)
IMAGE_BENCHMARK = importlib.util.module_from_spec(IMAGE_SPEC)
IMAGE_SPEC.loader.exec_module(IMAGE_BENCHMARK)


def test_point_source_templates_preserve_finite_image_footprint():
    templates = IMAGE_BENCHMARK.point_source_templates()
    assert templates.shape == (81, 2)
    assert np.all((templates.sum(axis=0) > 0.99) & (templates.sum(axis=0) < 1.0))
    for index, (column, row) in enumerate(IMAGE_BENCHMARK.SOURCE_PIXELS):
        larger = np.zeros((25, 25))
        larger[row + 8, column + 8] = 1.0
        rendered = IMAGE_BENCHMARK.forward_model_image(larger, IMAGE_BENCHMARK.PSF_SIGMA)
        np.testing.assert_allclose(
            templates[:, index].reshape(9, 9), rendered["observed_image"][8:17, 8:17], atol=1e-12
        )
    assert np.argmax(templates[:, 0]) != np.argmax(templates[:, 1])


def test_point_source_reference_is_normalized_and_grid_converged():
    observed, expected = IMAGE_BENCHMARK.simulate_image(17, [[0.25, 180.0], [0.75, 120.0]])
    original = observed.copy()
    coarse = IMAGE_BENCHMARK.reference_count_probabilities(observed, grid_size=51)
    fine = IMAGE_BENCHMARK.reference_count_probabilities(observed, grid_size=101)
    assert observed.shape == expected.shape == (9, 9)
    assert np.isfinite(fine).all()
    assert fine.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(coarse, fine, atol=2e-3)
    np.testing.assert_array_equal(observed, original)


def test_blind_pixel_grid_reference_and_source_selection():
    pytest.importorskip("torch")
    pixels = tuple((column, row) for row in range(3, 6) for column in range(3, 6))
    observed, expected = IMAGE_BENCHMARK.simulate_image(71, [[0.35, 180.0]], source_pixels=pixels)
    reference = IMAGE_BENCHMARK.reference_count_probabilities(observed, grid_size=31, source_pixels=pixels)
    assert reference.sum() == pytest.approx(1.0)
    assert IMAGE_BENCHMARK.point_source_templates(pixels).shape == (81, 9)
    model = IMAGE_BENCHMARK.NeuralCatalogSelection.fit(96, 2, 72, source_pixels=pixels)
    draw, density = model.bind(observed)
    random = np.random.default_rng(73)
    for _ in range(20):
        base, catalog = draw(random)
        assert np.isfinite(density(base, catalog))
        assert np.all((catalog[:, 0] >= 0.0) & (catalog[:, 0] <= 1.0))
    nodes, weights = np.polynomial.legendre.leggauss(80)
    feature = np.r_[IMAGE_BENCHMARK.image_features(observed, IMAGE_BENCHMARK.point_source_templates(pixels)), 0.5]
    total = sum(np.dot(weights / (2.0 * len(pixels)), [
        np.exp(model.sources.log_prob(feature, [(pixel + 0.5) / len(pixels), (node + 1.0) / 2.0]))
        for node in nodes]) for pixel in range(len(pixels)))
    assert total == pytest.approx(1.0, abs=1e-6)


def test_point_source_training_and_sampler_use_image_pixels():
    pytest.importorskip("torch")
    conditions, objects = IMAGE_BENCHMARK.training_pairs(48, 7)
    assert conditions.shape == (48, 2)
    assert objects.shape == (48, 2)
    network = IMAGE_BENCHMARK.NeuralElementProposal.fit(
        conditions, objects, hidden_sizes=(8,), number_components=2,
        epochs=2, batch_size=24, seed=8,
    )
    observed, _ = IMAGE_BENCHMARK.simulate_image(9, [])
    result, arrays = IMAGE_BENCHMARK.compare_image(observed, network, number_steps=80, chains=1, seed=10)
    assert len(result["reference_count_probability"]) == 3
    assert {run["method"] for run in result["runs"]} == {"prior", "neural"}
    assert arrays["neural_chain0_count"].size == 30


def test_point_source_selection_integrates_candidate_coordinate_without_changing_pixel_flux_density():
    pytest.importorskip("torch")
    conditions, objects = IMAGE_BENCHMARK.training_pairs(96, 27)
    network = IMAGE_BENCHMARK.NeuralElementProposal.fit(
        conditions, objects, hidden_sizes=(8,), number_components=2,
        epochs=2, batch_size=32, seed=28,
    )
    selection = IMAGE_BENCHMARK.PointSourceSelection(network)
    nodes, weights = np.polynomial.legendre.leggauss(80)
    condition = conditions[0]
    for pixel in range(2):
        coordinates = pixel / 2.0 + (nodes + 1.0) / 4.0
        for flux in (0.1, 0.4, 0.8):
            unit = np.column_stack((coordinates, np.full(coordinates.size, flux)))
            native_integral = np.dot(weights / 4.0, np.exp(network.log_prob_batch(condition, unit)))
            density = np.exp(selection.log_prob(condition, [(pixel + 0.5) / 2.0, flux]))
            assert density / 2.0 == pytest.approx(native_integral, rel=2e-5, abs=1e-7)
    flux = (nodes + 1.0) / 2.0
    total = sum(np.dot(weights / 4.0, [np.exp(selection.log_prob(condition, [coordinate, value])) for value in flux])
                for coordinate in (0.25, 0.75))
    assert total == pytest.approx(1.0, abs=1e-6)
    random = np.random.default_rng(29)
    samples = np.asarray([selection.sample(condition, random) for _ in range(6000)])
    expected_pixel = np.dot(weights / 4.0, [np.exp(selection.log_prob(condition, [0.25, value])) for value in flux])
    assert np.mean(samples[:, 0] < 0.5) == pytest.approx(expected_pixel, abs=0.025)
    assert np.all((samples >= 0.0) & (samples <= 1.0))


def test_complete_neural_catalog_draws_and_finite_densities():
    pytest.importorskip("torch")
    model = IMAGE_BENCHMARK.NeuralCatalogSelection.fit(96, 2, 51)
    observed, _ = IMAGE_BENCHMARK.simulate_image(52, [[0.25, 180.0]])
    draw, density = model.bind(observed)
    random = np.random.default_rng(53)
    for _ in range(30):
        base, catalog = draw(random)
        assert 90.0 <= base[0] <= 110.0
        assert catalog.shape[1] == 2 and len(catalog) <= 2
        assert np.isfinite(density(base, catalog))
    assert IMAGE_BENCHMARK.finite_rhat([np.zeros(20), np.zeros(20)]) is None


def test_prior_catalog_suggestions_have_known_density_without_trained_weights():
    from scipy.stats import truncnorm

    model = IMAGE_BENCHMARK.NeuralCatalogSelection()
    observed, _ = IMAGE_BENCHMARK.simulate_image(52, [[0.25, 180.0]])
    draw, density = model.bind(observed, learned=False)
    templates = IMAGE_BENCHMARK.point_source_templates()
    random = np.random.default_rng(59)
    counts = []
    for _ in range(1200):
        base, catalog = draw(random)
        counts.append(len(catalog))
        assert 90.0 <= base[0] <= 110.0
        assert np.all((catalog[:, 0] >= 0.0) & (catalog[:, 0] <= 1.0))
        assert np.all((catalog[:, 1] >= 0.0) & (catalog[:, 1] <= 500.0))
        mean = np.mean(observed.ravel() - IMAGE_BENCHMARK.source_image(catalog, templates))
        scale = 10.0 / 9.0
        expected = np.log([0.4, 0.4, 0.2][len(catalog)]) - len(catalog) * np.log(500.0)
        expected += truncnorm.logpdf(base[0], (90.0 - mean) / scale, (110.0 - mean) / scale,
                                     loc=mean, scale=scale)
        assert density(base, catalog) == pytest.approx(expected, abs=1e-10)
    np.testing.assert_allclose(np.bincount(counts, minlength=3) / len(counts), [0.4, 0.4, 0.2], atol=0.04)


def test_image_efficiency_summary_reports_paired_ratios_and_training_cost():
    cases = []
    for count in range(3):
        rows = []
        for method, multiplier in (("prior", 1.0), ("catalog", 2.0)):
            rows.append({
                "method": method, "combined_count_probability": [0.4, 0.4, 0.2],
                "count_ess": 100.0 * multiplier, "sampling_seconds": 10.0,
                "count_ess_per_second": 10.0 * multiplier,
                "log_posterior_ess_per_second": 20.0 * multiplier,
                "total_variation_from_reference": 0.01, "count_rhat": 1.0, "log_posterior_rhat": 1.0,
            })
        cases.append({"injected_source_count": count, "runs": rows})
    summary = IMAGE_BENCHMARK.summarize_efficiency(cases, {"catalog": 30.0}, 67)["catalog"]
    assert summary["median_count_ess_per_second_ratio"] == 2.0
    assert summary["median_count_ess_per_second_ratio_95pct_image_bootstrap_interval"] == [2.0, 2.0]
    assert summary["median_training_inclusive_count_ess_per_second_ratio"] == 1.0
    assert summary["median_training_inclusive_log_posterior_ess_per_second_ratio"] == 1.0
    assert summary["median_total_variation_difference_from_prior"] == 0.0
    cases[0]["runs"][0]["count_ess_per_second"] = 0.0
    summary = IMAGE_BENCHMARK.summarize_efficiency(cases, {"catalog": 30.0}, 67)["catalog"]
    assert summary["median_count_ess_per_second_ratio"] is None
    assert not summary["all_count_statistics_defined"]


@pytest.mark.parametrize("typefileplot", ["png", "pdf"])
def test_point_source_count_plot_pads_endpoint_markers(monkeypatch, tmp_path, typefileplot):
    import tdpy.plotting
    cases = [{"injected_source_count": index, "reference_count_probability": [1.0, 0.0, 0.0],
              "runs": [{"method": method, "combined_count_probability": [1.0, 0.0, 0.0]}
                       for method in ("prior", "neural")]} for index in range(3)]
    arrays = {f"case{index}_observed": np.full((9, 9), 100.0 + index) for index in range(3)}

    def check_figure(figure, output, output_type, **kwargs):
        assert output_type == typefileplot
        for axis in figure.axes[3:6]:
            assert axis.get_xlim()[0] < -0.1 and axis.get_xlim()[1] > 2.1
            assert axis.get_ylim()[0] < -0.04 and axis.get_ylim()[1] > 1.04
            for line in axis.lines:
                np.testing.assert_array_equal(line.get_xdata(), [0, 1, 2])
                np.testing.assert_array_equal(line.get_ydata(), [1.0, 0.0, 0.0])
        assert kwargs["font_size"] == 12

    monkeypatch.setattr(tdpy.plotting, "save_figure", check_figure)
    IMAGE_BENCHMARK.plot_results(
        {"cases": cases}, arrays, tmp_path / f"counts.{typefileplot}", typefileplot
    )


def test_replicated_count_plot_shows_catalog_proposal_and_scales_tv(monkeypatch, tmp_path):
    import tdpy.plotting

    methods = ("prior", "neural", "catalog")
    cases, arrays = [], {}
    for count in range(3):
        for scene in range(4):
            index = 4 * count + scene
            cases.append({
                "injected_source_count": count,
                "runs": [{
                    "method": method,
                    "total_variation_from_reference": 0.001 * (count + scene + offset),
                } for offset, method in enumerate(methods, start=1)],
            })
            arrays[f"case{index}_observed"] = np.full((9, 9), 100.0 + index)

    def check_figure(figure, output, output_type, **kwargs):
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        assert figure.get_size_inches()[0] >= 6.0
        for axis in figure.axes[3:6]:
            assert len(axis.collections) == 3
            assert axis.get_ylim()[0] == 0.0
            assert axis.get_ylim()[1] < 0.1
            assert axis.get_xticks().size == 0
        FigureCanvasAgg(figure)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        label_boxes = [axis.xaxis.label.get_window_extent(renderer) for axis in figure.axes[3:6]]
        legend_box = figure.legends[0].get_window_extent(renderer)
        assert all(not first.overlaps(second)
                   for index, first in enumerate(label_boxes)
                   for second in label_boxes[index + 1:])
        assert all(not legend_box.overlaps(label_box) for label_box in label_boxes)
        legend_labels = [text.get_text() for text in figure.legends[0].get_texts()]
        assert legend_labels == ["Standard PCAT", "Single source", "Whole catalog", "Median"]
        assert all(text.get_fontsize() == 12 for text in figure.legends[0].get_texts())
        assert output_type == "pdf"
        assert kwargs["font_size"] == 12

    monkeypatch.setattr(tdpy.plotting, "save_figure", check_figure)
    IMAGE_BENCHMARK.plot_replicated_results(
        {"cases": cases}, arrays, tmp_path / "complete_catalog.pdf", "pdf"
    )


def test_exact_grid_posterior_is_normalized_and_centered_for_zero_data():
    design = BENCHMARK._design_matrix()
    observations = BENCHMARK.BACKGROUND + BENCHMARK.BOUND * design.sum(axis=1)
    axis, probability = BENCHMARK._exact_posterior(observations, design)

    assert axis.shape == (BENCHMARK.GRID_SIZE,)
    assert probability.shape == (BENCHMARK.GRID_SIZE, BENCHMARK.GRID_SIZE)
    assert np.isfinite(probability).all()
    assert probability.sum() == pytest.approx(1.0)
    assert np.sum(probability * np.meshgrid(axis, axis, indexing="ij")[0]) == pytest.approx(
        0.0, abs=1e-10
    )


def test_diffusion_sampler_returns_reproducible_in_bounds_draws():
    pytest.importorskip("torch")
    design = BENCHMARK._design_matrix()
    conditions, coefficients = BENCHMARK._simulate_pairs(design, 96, 17)
    model = BENCHMARK.ConditionalDiffusionPosterior.fit(
        conditions, coefficients, steps=8, epochs=2, batch_size=32, seed=19
    )

    first = model.sample(conditions[0], 24, 23)
    second = model.sample(conditions[0], 24, 23)

    assert first.shape == (24, 2)
    assert np.isfinite(first).all()
    assert np.all(first >= -BENCHMARK.BOUND)
    assert np.all(first <= BENCHMARK.BOUND)
    np.testing.assert_array_equal(first, second)


def test_grid_reference_draws_remain_within_prior_support():
    axis = np.linspace(-BENCHMARK.BOUND, BENCHMARK.BOUND, BENCHMARK.GRID_SIZE)
    probability = np.zeros((axis.size, axis.size))
    probability[0, -1] = 0.5
    probability[-1, 0] = 0.5

    samples = BENCHMARK._sample_exact(axis, probability, 2000, seed=31)

    assert np.all(samples >= -BENCHMARK.BOUND)
    assert np.all(samples <= BENCHMARK.BOUND)


def test_total_variation_is_bounded_for_finite_samples():
    axis = np.linspace(-1.0, 1.0, 21)
    probability = np.ones((axis.size, axis.size)) / axis.size**2
    samples = np.column_stack((np.zeros(100), np.zeros(100)))

    distance = BENCHMARK._total_variation(samples, axis, probability)

    assert 0.0 <= distance <= 1.0


def test_coverage_and_scene_bootstrap_intervals_are_valid_and_seeded():
    interval = BENCHMARK._wilson_interval(18, 20)
    first = BENCHMARK._bootstrap_median_interval([0.04, 0.06, 0.08], seed=29)
    second = BENCHMARK._bootstrap_median_interval([0.04, 0.06, 0.08], seed=29)

    assert len(interval) == len(first) == 2
    assert interval[0] < 0.9 < interval[1]
    assert first[0] <= 0.06 <= first[1]
    assert first == second