# PCAT example figures

These notebooks apply probabilistic cataloging to point-source images, Gaussian mixtures, strong gravitational lenses, and Voigt spectral lines. They also demonstrate catalog association and subpixel point-spread-function modeling. Open a notebook in each directory and run its code cells to see selected data, model, residual, and posterior figures inline. The matching Python scripts remain available for command-line automation and existing tests. The Fermi-LAT filter and manual batch-command notebooks have no generated figures to display.

## Layout

- `gaussian_mixture_catalog/`: transdimensional inference of a compact Gaussian mixture
- `chandra_point_source_catalog/`: point-source inference in a simulated Chandra-style image
- `daylan_2017_fermi_point_sources/`: a scaled mock-catalog reproduction of Daylan et al. (2017)
- `daylan_2018_strong_lens_subhalos/`: a simulated HST strong-lens catalog and one-subhalo comparison inspired by Daylan et al. (2018)
- `simulated_hst_strong_lens/`: catalog inference in a simulated Hubble Space Telescope lens image
- `roman_strong_lens_perturber_catalog/`: a seeded Roman strong-lens benchmark and its generated diagnostic
- `simulated_rubin_cluster_lens/`: a synthetic Rubin-like cluster-scale lens fitted with PCAT's Poisson image sampler
- `rubin_dp1_confirmed_strong_lenses/`: an RSP-only workflow that crossmatches confirmed SIMBAD lens systems to real Rubin DP1 imaging, retrieves all catalog-footprint matches, and runs demonstration PCAT fits
- `catalog_association_completeness_purity/`: completeness and purity across catalog-matching radii
- `subpixel_psf_reconstruction/`: cubic reconstruction of an oversampled point-spread function
- `voigt_spectral_line_catalog/`: nominal and high-signal Voigt-profile line detection in simulated spectra
- `fermi_lat_pg1553_event_filter/`: a Fermi Large Area Telescope event-filter configuration for PG 1553+113

## Running

From the repository root:

```bash
cd /path/to/pcat
python examples/run_all_examples.py
```

The equivalent interactive entry point is [run_all_examples.ipynb](run_all_examples.ipynb). Both aggregate entry points replace cached figure outputs for the included analyses. They evaluate the image analyses, the Roman strong-lens benchmark, the two utility examples, and the reduced Voigt configuration. The output includes static posterior diagnostics and posterior animations.

Run either Voigt configuration at its full sampling depth with:

```bash
python examples/voigt_spectral_line_catalog/voigt_spectral_line_catalog.py --configuration nomi
python examples/voigt_spectral_line_catalog/voigt_spectral_line_catalog.py --configuration s2nrhigh
```
