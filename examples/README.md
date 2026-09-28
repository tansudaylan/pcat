# PCAT example figures

These examples apply probabilistic cataloging to point-source images, Gaussian mixtures, strong gravitational lenses, and Voigt spectral lines. The aggregate command uses reduced sampling depths for rapid calculations, while the Voigt commands below run the complete sampling configurations.

## Layout

- `gmix_demo/`: transdimensional inference of a compact Gaussian mixture
- `chan_demo/`: point-source inference in a simulated Chandra-style image
- `Daylan+2017/`: a scaled mock-catalog reproduction of Daylan et al. (2017)
- `hst_lens/`: catalog inference in a simulated Hubble Space Telescope lens image
- `roman_lens_catalog/`: a seeded Roman strong-lens benchmark and its generated diagnostic
- `voigt-profile/`: nominal and high-signal Voigt-profile line detection in simulated spectra
- `fermi_lat_pg1553/`: a Fermi Large Area Telescope event-filter configuration for PG 1553+113

## Running

From the repository root:

```bash
cd /path/to/pcat
python examples/run_examples.py
```

The aggregate calculation evaluates the three image analyses, the Roman strong-lens benchmark, and the reduced Voigt configuration. It produces static posterior diagnostics for every analysis and posterior animations for the Gaussian-mixture and Hubble lens calculations.

Run either Voigt configuration at its full sampling depth with:

```bash
python examples/voigt-profile/pcat_voigt_profile_detection.py --configuration nomi
python examples/voigt-profile/pcat_voigt_profile_detection.py --configuration s2nrhigh
```
