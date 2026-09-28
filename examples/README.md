# PCAT example figures

This folder contains demonstration scripts that exercise the PCAT package and write generated figures into each example directory.

The examples are intentionally lightweight and focus on the package’s path conventions, plotting workflow, and configuration patterns rather than expensive MCMC runs.

## Layout

- `gmix_demo/`: source and generated products for a compact Gaussian-mixture example
- `chan_demo/`: source and generated products for a point-source mock inspired by Chandra-style source detection
- `Daylan+2017/`: a scaled mock-catalog reproduction of Daylan et al. (2017)
- `hst_lens/`: a lensing-style image example illustrating catalog-level inference concepts
- `roman_lens_catalog/`: a seeded Roman strong-lens benchmark and its generated diagnostic
- `voigt-profile/`: simulated nominal and high-signal Voigt-profile detection configurations and products
- `fermi_lat_pg1553/`: a Fermi Large Area Telescope event-filter configuration for PG 1553+113
- `run_examples.py`: runs all maintained examples in isolated processes and verifies their plots

## Running

From the repository root:

```bash
cd /path/to/pcat
python examples/run_examples.py
```

The aggregate runner removes cached products, executes all three image demonstrations, runs the Roman diagnostic, and runs the reduced Voigt smoke configuration. It fails if a pipeline example omits required static posterior plots or animations. Gaussian and HST additionally require a genuine multi-frame animation because their deterministic posterior frames differ.

Each example keeps its source and generated products in one direct child of `examples/`. Configuration-specific run tags remain below `data/outp/`, while all plots are written directly under the example's `visuals/` directory without another run-tag subfolder.

Run either Voigt configuration at its full sampling depth with:

```bash
python examples/voigt-profile/pcat_voigt_profile_detection.py --configuration nomi
python examples/voigt-profile/pcat_voigt_profile_detection.py --configuration s2nrhigh
```
