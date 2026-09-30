# PCAT example figures

These notebooks apply probabilistic cataloging to point-source images, Gaussian mixtures, strong gravitational lenses, diffuse emission, and Voigt spectral lines. They also demonstrate catalog association. Open a notebook in each directory and run its code cells to see selected data, model, residual, and posterior figures inline. The matching Python scripts remain available for command-line automation and existing tests. The Fermi-LAT filter and manual batch-command notebooks have no generated figures to display.

## Publication coverage

| Publication | PCAT application | Example | Scope |
| --- | --- | --- | --- |
| Daylan et al. (2017), [10.3847/1538-4357/aa679e](https://doi.org/10.3847/1538-4357/aa679e) | Fermi-LAT point-source populations | `daylan_2017_fermi_point_sources/` | Scaled synthetic analog |
| Portillo et al. (2017), [10.3847/1538-3881/aa8565](https://doi.org/10.3847/1538-3881/aa8565) | Crowded SDSS M2 field | `portillo_2017_crowded_sdss_m2/` | Scaled synthetic analog |
| Daylan et al. (2018), [10.3847/1538-4357/aaa1f2](https://doi.org/10.3847/1538-4357/aaa1f2) | Strong-lens subhalo catalogs | `daylan_2018_strong_lens_subhalos/` | Scaled synthetic analog |
| Feder et al. (2020), [10.3847/1538-3881/ab74cf](https://doi.org/10.3847/1538-3881/ab74cf) | Multiband SDSS deblending | `feder_2020_multiband_sdss_deblending/` | Scaled synthetic analog |
| Butler et al. (2022), [10.3847/1538-4357/ac6c04](https://doi.org/10.3847/1538-4357/ac6c04) | SPIRE CIB, cirrus, and rSZ separation | `butler_2022_spire_sz_component_separation/` | Scaled synthetic analog |
| Feder et al. (2023), [10.3847/1538-3881/ace69b](https://doi.org/10.3847/1538-3881/ace69b) | SPIRE point-plus-diffuse inference | `feder_2023_point_diffuse_spire/` | Scaled synthetic analog |
| Hall et al. (2026), [10.3847/1538-4357/ae1e7a](https://doi.org/10.3847/1538-4357/ae1e7a) | Herschel DSFG multiplicity | `hall_2026_herschel_dsfg_multiplicity/` | Scaled synthetic analog |

These examples showcase the PCAT model families used by the papers. They do not claim numerical reproduction unless the paper inputs and converged sampling configuration are explicitly supplied.

## Layout

- `gaussian_mixture_catalog/`: transdimensional inference of a compact Gaussian mixture
- `chandra_point_source_catalog/`: point-source inference in a simulated Chandra-style image
- `daylan_2017_fermi_point_sources/`: a scaled mock-catalog reproduction of Daylan et al. (2017)
- `daylan_2018_strong_lens_subhalos/`: a simulated HST strong-lens catalog and one-subhalo comparison inspired by Daylan et al. (2018)
- `portillo_2017_crowded_sdss_m2/`: a scaled crowded-field SDSS catalog analysis inspired by Portillo et al. (2017)
- `feder_2020_multiband_sdss_deblending/`: five-band SDSS probabilistic deblending inspired by Feder et al. (2020)
- `butler_2022_spire_sz_component_separation/`: point-source and diffuse-template separation inspired by Butler et al. (2022)
- `feder_2023_point_diffuse_spire/`: three-band point-plus-diffuse inference inspired by Feder et al. (2023)
- `hall_2026_herschel_dsfg_multiplicity/`: crowded three-band source multiplicity inspired by Hall et al. (2026)
- `simulated_hst_strong_lens/`: catalog inference in a simulated Hubble Space Telescope lens image
- `roman_strong_lens_perturber_catalog/`: a seeded Roman strong-lens benchmark and its generated diagnostic
- `simulated_rubin_cluster_lens/`: a synthetic Rubin-like cluster-scale lens fitted with PCAT's Poisson image sampler
- `rubin_dp1_confirmed_strong_lenses/`: an RSP-only workflow that crossmatches confirmed SIMBAD lens systems to real Rubin DP1 imaging, retrieves all catalog-footprint matches, and runs demonstration PCAT fits
- `catalog_association_completeness_purity/`: completeness and purity across catalog-matching radii
- `voigt_spectral_line_catalog/`: nominal and high-signal Voigt-profile line detection in simulated spectra
- `fermi_lat_pg1553_event_filter/`: a Fermi Large Area Telescope event-filter configuration for PG 1553+113

## Running

From the repository root:

```bash
cd /path/to/pcat
python examples/run_all_examples.py
```

The aggregate script replaces cached figure outputs for the included analyses. It evaluates the image analyses, the Roman strong-lens benchmark, the two utility examples, and the reduced Voigt configuration. The output includes static posterior diagnostics and posterior animations.

Run either Voigt configuration at its full sampling depth with:

```bash
python examples/voigt_spectral_line_catalog/voigt_spectral_line_catalog.py --configuration nomi
python examples/voigt_spectral_line_catalog/voigt_spectral_line_catalog.py --configuration s2nrhigh
```
