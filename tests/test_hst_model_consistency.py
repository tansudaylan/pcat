from types import SimpleNamespace

import numpy as np

from pcat.main import _rescale_hst_mock_model, _retr_sbrt_from_cntp, retr_cntp


def hst_mock_state():
    observed_counts = np.full((1, 2, 1), 100.0)  # [counts]
    return SimpleNamespace(
        typeexpr="HST_WFC3_IR",
        typedata="simu",
        expo=np.ones((1, 2, 1)),
        apix=1.0,
        enerdiff=False,
        cntpdata=observed_counts,
        typeverb=-1,
    )


def test_hst_count_floor_updates_brightness_and_likelihood_count_map():
    gdat = hst_mock_state()
    brightness = np.ones((1, 2, 1))
    model_counts = retr_cntp(gdat, brightness)

    scaled_brightness, likelihood_counts = _rescale_hst_mock_model(
        gdat, "fitt", "this", brightness, model_counts
    )

    assert np.sum(likelihood_counts) == np.sum(gdat.cntpdata)
    np.testing.assert_allclose(retr_cntp(gdat, scaled_brightness), likelihood_counts)
    np.testing.assert_allclose(_retr_sbrt_from_cntp(gdat, likelihood_counts), scaled_brightness)


def test_hst_count_floor_does_not_change_non_fitted_states():
    gdat = hst_mock_state()
    brightness = np.ones((1, 2, 1))
    model_counts = retr_cntp(gdat, brightness)

    unchanged_brightness, unchanged_counts = _rescale_hst_mock_model(
        gdat, "true", "this", brightness, model_counts
    )

    np.testing.assert_array_equal(unchanged_brightness, brightness)
    np.testing.assert_array_equal(unchanged_counts, model_counts)