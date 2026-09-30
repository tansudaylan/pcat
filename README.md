# PCAT

## Purpose

PCAT is a transdimensional, hierarchical Bayesian framework for inferring a catalog-level posterior from Poisson-distributed data. It is designed for problems where the number of sources and their parameters are not fixed a priori, and where the model space itself is a mixture of competing catalog configurations.

The primary scientific use cases are:

- probabilistic source catalogs from image or photon-count data;
- transdimensional inference over source populations;
- membership or detection uncertainty in crowded fields;
- forward modeling and posterior diagnostics for catalog-level summaries.

The core method is introduced in Daylan, Portillo & Finkbeiner (2017) and extended for different image and catalog analysis workflows.

## Posterior samples

![Posterior samples from four PCAT example problems](examples/pcat_posterior_samples.gif)

Each frame combines posterior-chain draws from simulated Gaussian-mixture,
point-source imaging, strong-lensing imaging, and spectral-line examples. The
animation demonstrates PCAT's shared inference and visualization pipeline. The
simulations are pipeline examples rather than measurements of observed systems.

## Catalog inference

PCAT compares catalog configurations with different numbers of sources, samples source and population parameters jointly, quantifies detection and membership probabilities, and evaluates posterior predictions against image or photon-count data.

Fixed-dimensional models with custom likelihoods use the same proposal,
acceptance, persistence, convergence, and final-processing pipeline as
transdimensional catalog models:

```python
from pcat.main import sample

result = sample(
	typeexpr="gener",
	retr_llik=log_likelihood,
	parameter_names=("mean", "scale"),
	prior_types=("self", "gaus"),
	prior_minima=(-5.0, 0.0),
	prior_maxima=(5.0, 5.0),
	prior_means=(0.0, 1.0),
	prior_stdvs=(1.0, 0.2),
	initial_values=(0.0, 1.0),
	proposal_scales=(0.05, 0.05),
	proposal_correlation=((1.0, 0.3), (0.3, 1.0)),
	pathbase="/path/to/run-root",
	strgcnfg="fixed-example",
)
```

``"self"`` parameters have uniform priors between their minima and maxima;
``"gaus"`` parameters use the specified means and standard deviations. PCAT
proposes in unit-prior coordinates and applies its existing inverse-CDF
transforms. Generic runs set birth/death and split/merge probabilities to zero,
so only the native type-0 within-model proposal is active. The returned object
is the normal persisted ``gdatfinlpost`` state.

By default, type-0 moves perturb one parameter at a time. Set
``probpropblock`` above zero to mix in correlated block moves using
``proposal_correlation``; ``factpropblock`` scales those block increments.
These moves retain PCAT's standard acceptance, adaptation, persistence, and
final-processing machinery.

## Installation

A modern installation path is:

```bash
cd /path/to/pcat
pip install -e .
export PCAT_PATH=/path/to/pcat
```

`PCAT_PATH` identifies the Git repository root. Its `data/` and `visuals/` directories are ignored by Git. The separate `PCAT_DATA_PATH` variable remains the working-data and pipeline-output root used by scientific runs.

This repository expects the shared library `tdpy` to be installed in the same Python environment. For local development, this is usually easiest with:

```bash
cd /path/to/tdpy
pip install -e .
cd /path/to/pcat
pip install -e .
```

## Roman strong-lens example

The executable example uses the compact PCAT Roman benchmark to simulate 100 strong-lens images. Half contain one dark-matter perturber drawn at a candidate position around the macro Einstein ring, and half contain no perturber. The catalog approximation compares zero- and one-perturber models after Roman point-spread-function convolution and Poisson plus read noise. Its diagnostic follows one representative lens from simulated detector input through the macro-only model and residual, then summarizes the final population-level catalog probabilities.

```bash
python examples/roman_lens_catalog/roman_lens_catalog_diagnostic.py --typefileplot png
```

![Simulated Roman strong-lens catalog diagnostic](examples/roman_lens_catalog/roman_lens_catalog_diagnostic.png)

For the fixed seed, the approximate catalog classifier recovers 38% of injected perturbers above a posterior threshold of 0.5, with a 68% Wilson interval of 31% to 45%. It produces no false positives among the 50 null lenses, with an upper interval bound of 2%, and localizes 84% of injected perturbers to the correct candidate position. The mean one-perturber posterior probability is 0.37 for injected systems and $4.6\times10^{-5}$ for null systems. These values characterize this clearly labeled simulation and are not forecasts from real Roman observations.

Full transdimensional runs initialize a model configuration, define a likelihood and data product, and call `pcat.main.init(...)` with a populated configuration dictionary.

## Daylan et al. 2017 reproduction

The synthetic reproduction of the original PCAT point-source analysis is in
[`examples/Daylan+2017`](examples/Daylan+2017). It preserves the published 300-source
population and flux-distribution slope while clearly separating this scaled,
self-contained run from the archival Fermi-LAT data analysis.

## Outputs

PCAT writes serialized posterior states under `data/` and figures under `visuals/`, rooted at `PCAT_DATA_PATH` or `TDGU_DATA_PATH`. The figures show the input data, model realization, residual, source population, parameter distributions, and convergence diagnostics.

## Posterior plotting

PCAT can render every one- and two-parameter posterior projection without an
external corner-plot package:

```python
from pcat import plot_grid

plot_grid(
	"/path/to/posterior",
	"model",
	result.listpostparagenrscalbase,
	("mean", "scale"),
	typefileplot="pdf",
)
```

## Documentation

The user guide begins at [`docs/index.rst`](docs/index.rst) and covers
installation, runnable examples, output products, diagnostics, and the public
API. Build the HTML documentation with:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -E -a -W --keep-going -b html docs docs/_build/html
```

## References

- Daylan, Portillo, & Finkbeiner (2017), transdimensional Bayesian catalog inference
