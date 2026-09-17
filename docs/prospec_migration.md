# Prospec migration

PCAT is the active implementation of probabilistic cataloging. The reusable
content from `prospec_tobedistilledintopcat` has been migrated or classified
below. The legacy working tree may be removed after the current changes are
committed or otherwise preserved.

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

## Retired functionality

- The Python 2 line sampler is retired. It duplicates PCAT's maintained
  transdimensional inference machinery and depends on an obsolete runtime.
- `blas.so` and `blas.c` provided a platform-specific model evaluator. PCAT's
  maintained NumPy and Numba paths replace it. The C source also contains
  undefined indices and is not retained as a reference implementation.
- The dataset-specific plotting in `plot_pcat.py` and `m2plots.py` is retired.
  PCAT provides completeness and false-discovery diagnostics, and the reusable
  many-to-many matcher has been migrated.
- The time-series imaging prototype in `make_mock.py` is retired. It depends on
  untracked local `LION_PATH` data, Python 2 behavior, and an incompatible model
  signature. It has no reproducible inputs or tests.
- `coll_garb.py`, `sh/down.sh`, and `sh/load.sh` are environment-specific file
  deletion, download, and module-loading wrappers with no reusable logic.

Tests for migrated utilities live in `tests/test_psf.py` and
`tests/test_associate.py`.

## Removal checklist

- Commit or otherwise preserve the PCAT migration changes.
- Preserve the legacy Git history. The repository currently has an `origin`
  remote and two historical commits.
- Resolve or intentionally discard the uncommitted migration edits in the
  legacy working tree before deleting its local directory.

The dormant PCAT `typeexpr='fire'` path is not the replacement criterion for
the retired Python 2 application. A repair attempt showed that its absorption
line population lacks a complete generative parameter definition. It should be
treated as separate future PCAT work, not as a reason to retain the legacy
working tree indefinitely.