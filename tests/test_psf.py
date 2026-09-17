from types import SimpleNamespace

import numpy as np
import pytest

from pcat.psf import psf_poly_fit


def test_psf_poly_fit_reconstructs_cubic_profile():
	factusam = 4
	indxusam = np.arange(12, dtype=np.float32)
	psfnusam = (12. - indxusam) * (1. + 0.1 * indxusam + 0.01 * indxusam**2)
	gdat = SimpleNamespace()

	coefspix = psf_poly_fit(gdat, psfnusam, factusam)
	offs = np.linspace(0., 1., factusam + 1, dtype=np.float32)
	design = np.column_stack((np.ones_like(offs), offs, offs**2, offs**3))
	reconstructed = (design @ coefspix).T.reshape(-1)[
		np.r_[0:5, 6:10, 11:15]
	]
	expected = np.pad(psfnusam, (0, 1))

	assert coefspix.shape == (4, 3)
	assert coefspix.dtype == np.float32
	assert np.allclose(reconstructed, expected, rtol=2e-5, atol=2e-5)
	assert np.array_equal(gdat.enerpsfnusam, np.arange(13) - 6.)


@pytest.mark.parametrize('factusam', [0, 2, 3.5])
def test_psf_poly_fit_rejects_invalid_oversampling(factusam):
	with pytest.raises(ValueError):
		psf_poly_fit(SimpleNamespace(), np.ones(12), factusam)