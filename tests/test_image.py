import numpy as np
import pytest

from pcat import forward_model_image


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