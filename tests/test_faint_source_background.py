import numpy as np
import pytest

from examples.chandra_point_source_catalog.chandra_point_source_catalog import (
    make_faint_source_background_scene,
    make_single_star_scene,
    run_background_cut_comparison,
    run_single_star_15x15,
)


def test_faint_source_background_scene_separates_cut_models_deterministically():
    first = make_faint_source_background_scene(seed=31)
    repeated = make_faint_source_background_scene(seed=31)
    different = make_faint_source_background_scene(seed=32)

    assert first["observed_counts"].shape == (24, 24)
    assert np.array_equal(first["observed_counts"], repeated["observed_counts"])
    assert not np.array_equal(first["observed_counts"], different["observed_counts"])
    assert np.allclose(
        first["source_counts_by_cut"][1.0], first["true_source_counts"]
    )
    omitted_mean = np.mean(
        first["source_counts_by_cut"][1.0]
        - first["source_counts_by_cut"][10.0]
    )
    assert np.isclose(omitted_mean, first["omitted_surface_brightness"])
    assert 0.35 < omitted_mean < 0.5


def test_single_star_scene_is_centered_in_a_15_by_15_image():
    first = make_single_star_scene(seed=17)
    repeated = make_single_star_scene(seed=17)

    assert first["image_shape"] == (15, 15)
    assert first["source_position_pixels"] == (7.0, 7.0)
    assert first["source_parameters"].shape == (1, 3)
    assert np.unravel_index(np.argmax(first["source_counts"]), first["image_shape"]) == (7, 7)
    assert first["observed_counts"].shape == (15, 15)
    assert np.array_equal(first["observed_counts"], repeated["observed_counts"])


@pytest.mark.parametrize("float_psf", [False, True])
def test_single_star_inference_runs_pcat_and_saves_summary(tmp_path, float_psf):
    import json

    result = run_single_star_15x15(
        output_root=tmp_path / "runs", example_root=tmp_path / "examples",
        seed=17, numbswep=1000, float_psf=float_psf,
    )
    summary_path = result["run_directory"] / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert result["figure_path"].is_file()
    assert summary["image_shape"] == [15, 15]
    assert summary["source_count"] == 1
    assert summary["source_position_pixels"] == [7.0, 7.0]
    assert summary["retained_draws"] > 0
    assert len(summary["posterior_quantiles"]["source_flux_counts"]) == 3
    assert summary["psf_mode"] == ("sampled" if float_psf else "fixed")
    if float_psf:
        assert len(summary["posterior_quantiles"]["psf_sigma_pixels"]) == 3


def test_background_cut_example_runs_pcat_and_matches_grid_reference(tmp_path):
    result = run_background_cut_comparison(
        output_root=tmp_path / "runs",
        example_root=tmp_path,
        seed=173,
        numbswep=3000,
    )
    fits = result["summary"]["fits"]

    assert result["figure_path"].is_file()
    assert (result["run_directory"] / "summary.json").is_file()
    assert np.allclose(
        fits["1.0"]["sample_quantiles"],
        fits["1.0"]["grid_reference_quantiles"],
        atol=0.08,
    )
    assert np.allclose(
        fits["10.0"]["sample_quantiles"],
        fits["10.0"]["grid_reference_quantiles"],
        atol=0.08,
    )
    assert 0.35 < fits["10.0"]["sample_quantiles"][1] - fits["1.0"]["sample_quantiles"][1] < 0.55