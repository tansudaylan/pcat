import numpy as np
import pytest
from types import SimpleNamespace

from pcat import forward_model_image
from pcat import main


def test_zero_poisson_realization_is_kept_without_retry(monkeypatch):

    calls = []

    def draw(rate):
        calls.append(rate.copy())
        return np.zeros_like(rate, dtype=int)

    monkeypatch.setattr(np.random, "poisson", draw)
    counts = main.draw_simulated_counts(np.array([[[0.1]]]))
    assert counts.sum() == 0
    assert len(calls) == 1


def test_invalid_poisson_expectation_is_rejected():
    for rate in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            main.draw_simulated_counts(np.array([rate]))


def test_zero_counts_survive_data_processing(monkeypatch):
    monkeypatch.setattr(main, 'retr_spatmean', lambda *args, **kwargs: (0., 0.))
    monkeypatch.setattr(main, 'setp_varb', lambda *args, **kwargs: None)
    state = SimpleNamespace(typedata='simu', cntpdata=np.zeros((1, 1, 1)),
                            liststrgmodl=['true'], true=SimpleNamespace(),
                            indxdqlt=[0], numbpixl=1, indxener=[0],
                            blimpara=SimpleNamespace(cntpdata=np.array([-0.5, 0.5])),
                            typepixl='heal')

    main.proc_cntpdata(state)

    assert state.cntpdata[0, 0, 0] == 0
    assert state.varidata[0, 0, 0] == 1
    assert state.llikoffs[0, 0, 0] == 0
    assert state.histcntpdataevt0.tolist() == [1]


def test_forward_model_image_conserves_centered_source_flux():
    source_image = np.zeros((51, 51))
    source_image[25, 25] = 1.0

    products = forward_model_image(source_image, psf_sigma_pixels=1.5)

    assert products["observed_image"].shape == source_image.shape
    np.testing.assert_allclose(products["psf_kernel"].sum(), 1.0)
    np.testing.assert_allclose(products["observed_image"].sum(), 1.0)
    assert products["observed_image"].max() < source_image.max()
    assert np.unravel_index(
        np.argmax(products["observed_image"]), source_image.shape
    ) == (25, 25)


def test_forward_model_image_adds_uniform_background():
    source_image = np.zeros((11, 11))

    products = forward_model_image(
        source_image,
        psf_sigma_pixels=1.0,
        background=4.0,
    )

    np.testing.assert_allclose(products["observed_image"], 4.0)


@pytest.mark.parametrize("psf_sigma_pixels", [0.0, -1.0, np.nan])
def test_forward_model_image_rejects_invalid_psf_width(psf_sigma_pixels):
    with pytest.raises(ValueError, match="finite and positive"):
        forward_model_image(np.ones((3, 3)), psf_sigma_pixels)