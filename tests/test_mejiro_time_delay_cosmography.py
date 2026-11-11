import numpy as np
import pytest

from examples.mejiro_time_delay_cosmography.mejiro_time_delay_cosmography import (
    REFERENCE_H0,
    TRUE_H0,
    make_synthetic_lensed_supernova,
    run_example,
    scale_time_delays,
)


def test_synthetic_mejiro_supernova_has_four_consistent_image_delays():
    pytest.importorskip("mejiro")
    lens = make_synthetic_lensed_supernova()
    delays = np.asarray(lens.get_time_delays(), dtype=float)

    assert len(delays) == 4
    assert delays[0] == pytest.approx(0.0)
    assert np.isfinite(delays).all()
    assert np.all(np.diff(np.sort(delays)) > 0.0)
    assert lens.z_lens == pytest.approx(0.5)
    assert lens.z_source == pytest.approx(1.5)


def test_fixed_lens_delay_scaling_is_inverse_in_h0():
    reference = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        scale_time_delays(reference, 2.0 * REFERENCE_H0),
        reference / 2.0,
    )


def test_pcat_infers_synthetic_h0_and_writes_figure(tmp_path):
    pytest.importorskip("mejiro")
    result = run_example(
        tmp_path, sample_count=800, burn_count=400, chain_count=2, seed=2026,
    )

    assert result["chains"].shape[0] == 2
    assert result["chains"].shape[2] == 1
    assert np.isfinite(result["chains"]).all()
    assert abs(result["posterior_quantiles_km_s_mpc"][1] - TRUE_H0) < 5.0
    assert result["figure_path"].is_file()
    assert result["summary_path"].is_file()