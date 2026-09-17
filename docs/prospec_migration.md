# Prospec migration

PCAT is the intended active implementation of probabilistic line cataloging.
The legacy `prospec_tobedistilledintopcat` repository is not yet safe to
delete because its complete workflow has not been reproduced in PCAT.

## Migrated functionality

- `pcat.psf.psf_poly_fit` provides the cubic subpixel PSF interpolation that
  was unique to `prospec`.
- `pcat.associate.associate_catalogs` preserves the reusable many-to-many
  position and value matching from the M2 analysis scripts, including optional
  confidence and significance summaries.
- Light-line element models in `pcat.main` provide the spectral line model.
- PCAT's birth, death, split, and merge proposals provide transdimensional
  catalog inference.
- PCAT's catalog post-processing provides posterior and condensed catalogs.

## Candidates for retirement

- The Python 2 sampler duplicated PCAT's inference machinery.
- `blas.so` and `blas.c` provided a platform-specific model evaluator. PCAT's
  maintained NumPy and Numba paths replace it.
- `plot_pcat.py` and `m2plots.py` duplicated plotting and association tools.
- `make_mock.py` depended on local `LION_PATH` data and an incompatible image
  model signature. New mock workflows belong in PCAT tests or examples with
  explicit input data.

Tests for migrated utilities live in `tests/test_psf.py`.

## Deletion gates

- Repair and validate a PCAT example for the legacy one-dimensional line-catalog
  workflow, including mock generation, sampling, and catalog condensation. A
  minimal `typeexpr='fire'` run currently either initializes zero fitted
  populations or fails while expanding per-element `deltllik` names because
  `gmod.maxmpara.numbelemtotl` is unavailable.
- Decide whether the time-series imaging prototype in `make_mock.py` has
  scientific value. Migrate it or record its explicit retirement.
- Decide whether the M2 catalog-comparison analyses in `m2plots.py` and
  `plot_pcat.py` must remain reproducible. Their general completeness and
  false-discovery calculations exist in PCAT, but the dataset-specific
  analyses do not.
- Preserve the legacy Git history and any uncommitted work before removing a
  working tree.