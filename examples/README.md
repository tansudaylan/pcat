# PCAT example figures

This folder contains demonstration scripts that exercise the PCAT package and write generated figures into each example directory.

The examples are intentionally lightweight and focus on the package’s path conventions, plotting workflow, and configuration patterns rather than expensive MCMC runs.

## Layout

- `gaussian_mixture/`: a compact Gaussian-mixture toy example
- `chandra_point_source/`: a point-source mock example inspired by Chandra-style source detection
- `hst_lens/`: a lensing-style image example illustrating catalog-level inference concepts
- `roman_lens_catalog_diagnostic.py`: a seeded Roman strong-lens benchmark showing simulated input, macro-model and residual stages, and population catalog probabilities
- `pcat_voigt_profile_detection.py`: simulated nominal and high-signal Voigt-profile detection configurations
- `run_examples.py`: runs all maintained examples in isolated processes and verifies their plots

## Running

From the repository root:

```bash
cd /path/to/pcat
python examples/run_examples.py
```

The aggregate runner removes cached products, executes all three image demonstrations, runs the Roman diagnostic, and runs the Voigt example in its two-sweep smoke mode. It fails if a pipeline example omits required static posterior plots or animations. Gaussian and HST additionally require a genuine multi-frame animation because their deterministic posterior frames differ.

The image examples write under their local `pcat-output/` directories. The Voigt example writes under `examples/voigt-profile-output/`, and the Roman diagnostic writes `examples/roman_lens_catalog_diagnostic.png`.

Run either Voigt configuration at its full sampling depth with:

```bash
python examples/pcat_voigt_profile_detection.py --configuration nomi
python examples/pcat_voigt_profile_detection.py --configuration s2nrhigh
```
