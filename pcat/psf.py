"""Point-spread-function utilities."""

import numpy as np


def psf_poly_fit(gdat, psfnusam, factusam):
	"""Fit a cubic subpixel model to each pixel of an oversampled 1D PSF.

	The returned columns contain coefficients in ascending polynomial order.
	Evaluating a column at offsets from zero through one reconstructs the
	oversampled PSF values across the corresponding detector pixel.
	"""
	psfnusam = np.asarray(psfnusam, dtype=np.float32)
	if psfnusam.ndim != 1:
		raise ValueError('psfnusam must be one-dimensional.')
	if not isinstance(factusam, (int, np.integer)) or factusam < 3:
		raise ValueError('factusam must be an integer of at least 3.')
	if psfnusam.size % factusam:
		raise ValueError('psfnusam.size must be divisible by factusam.')

	gdat.enerpsfnusam = np.arange(psfnusam.size + 1) - psfnusam.size / 2.
	psfnusampadd = np.pad(psfnusam, (0, 1))
	offs = np.linspace(0., 1., factusam + 1, dtype=np.float32)
	design = np.column_stack((np.ones_like(offs), offs, offs**2, offs**3))
	numbsidepsfn = psfnusam.size // factusam
	coefspix = np.empty((design.shape[1], numbsidepsfn), dtype=np.float32)

	for indxpixl in range(numbsidepsfn):
		strt = indxpixl * factusam
		stop = strt + factusam + 1
		coefspix[:, indxpixl] = np.linalg.lstsq(
			design, psfnusampadd[strt:stop], rcond=None,
		)[0]

	return coefspix