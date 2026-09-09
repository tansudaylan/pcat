# PCAT example figures

This folder contains a small set of demonstration scripts that exercise the PCAT package and write generated figures into each example directory.

The examples are intentionally lightweight and focus on the package’s path conventions, plotting workflow, and configuration patterns rather than expensive MCMC runs.

## Layout

- `gaussian_mixture/`: a compact Gaussian-mixture toy example
- `chandra_point_source/`: a point-source mock example inspired by Chandra-style source detection
- `hst_lens/`: a lensing-style image example illustrating catalog-level inference concepts
- `run_examples.py`: runs all example scripts in sequence

## Running

From the repository root:

```bash
cd /path/to/pcat
python examples/run_examples.py
```

Each script writes its outputs under its own folder under the example-specific `pcat-output/` directory using the PCAT data/visuals convention.
