# PCAT example figures

These notebooks apply probabilistic cataloging to point-source images, Gaussian mixtures, strong gravitational lenses, and Voigt spectral lines. They also demonstrate catalog association and subpixel point-spread-function modeling. Open a notebook in each directory and run its code cells to see selected data, model, residual, and posterior figures inline. The matching Python scripts remain available for command-line automation and existing tests. The Fermi-LAT filter and manual batch-command notebooks have no generated figures to display.

## Layout

- `gmix_demo/`: transdimensional inference of a compact Gaussian mixture
- `chan_demo/`: point-source inference in a simulated Chandra-style image
- `Daylan+2017/`: a scaled mock-catalog reproduction of Daylan et al. (2017)
- `Daylan+2018/`: a simulated HST strong-lens catalog and one-subhalo comparison inspired by Daylan et al. (2018)
- `hst_lens/`: catalog inference in a simulated Hubble Space Telescope lens image
- `roman_lens_catalog/`: a seeded Roman strong-lens benchmark and its generated diagnostic
- `rubin_cluster_lens/`: a synthetic Rubin-like cluster-scale lens fitted with PCAT's Poisson image sampler
- `catalog_association/`: completeness and purity across catalog-matching radii
- `psf_subpixel/`: cubic reconstruction of an oversampled point-spread function
- `voigt-profile/`: nominal and high-signal Voigt-profile line detection in simulated spectra
- `fermi_lat_pg1553/`: a Fermi Large Area Telescope event-filter configuration for PG 1553+113

## Running

From the repository root:

```bash
cd /path/to/pcat
python examples/run_examples.py
```

The equivalent interactive entry point is [run_examples.ipynb](run_examples.ipynb). Both aggregate entry points replace cached figure outputs for the included analyses. They evaluate the image analyses, the Roman strong-lens benchmark, the two utility examples, and the reduced Voigt configuration. The output includes static posterior diagnostics and posterior animations.

Run either Voigt configuration at its full sampling depth with:

```bash
python examples/voigt-profile/pcat_voigt_profile_detection.py --configuration nomi
python examples/voigt-profile/pcat_voigt_profile_detection.py --configuration s2nrhigh
```
