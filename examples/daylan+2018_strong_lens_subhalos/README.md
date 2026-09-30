# Daylan et al. 2018 strong-lens mock

This example runs PCAT's simulated HST/WFC3 UVIS lens model to explore the
variable-subhalo-catalog and one-subhalo comparison in
[Daylan et al. (2018)](https://doi.org/10.3847/1538-4357/aaaa1e),
ApJ 854, 141. The publication injects 25 subhalos into a mock photon-count
image with 0.04 arcsec pixels. The example uses PCAT's built-in simulated lens
and writes image, residual, deflection, convergence, and parameter plots.

From the PCAT root, run:

```bash
python 'examples/daylan+2018_strong_lens_subhalos/daylan+2018_strong_lens_subhalos.py'
python 'examples/daylan+2018_strong_lens_subhalos/daylan+2018_strong_lens_subhalos.py' --one-subhalo
```

Use `--typefileplot pdf` for PDF output or `--smoke` for a short pipeline check.
The two fits write separately under `examples/daylan+2018_strong_lens_subhalos/daylan2018_catalog/`
and `examples/daylan+2018_strong_lens_subhalos/daylan2018_one_subhalo/`. A full run may take a long
time; inspect chain convergence before interpreting its posterior figures.

The built-in simulation uses a different random seed and instrument/background
defaults from the published realization. The run therefore recreates the
transdimensional versus fixed-catalog *analysis*, not the paper's numerical
posterior values or its exact Figures 5-19. PCAT currently skips some
specialized final lens posterior products, including the mass-function and
mass-fraction figures, when the required derived quantities are unavailable.