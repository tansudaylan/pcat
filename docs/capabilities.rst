Capabilities
============

PCAT applies one posterior-sampling and output pipeline to fixed-dimensional
models and transdimensional catalogs. A run can use simulated data or supplied
data, and its model can contain a fixed set of parameters, a variable number of
elements, or both.

Functionality overview
----------------------

.. list-table:: Maintained PCAT workflows
   :header-rows: 1
   :widths: 18 24 21 37

   * - Workflow
     - Model
     - Inference
     - Maintained example
   * - Arbitrary likelihood
     - User callback and configured priors
     - Fixed-dimensional scalar or correlated proposals
     - ``typeexpr="gener"`` example below
   * - Point-source imaging
     - Catalog, background, exposure, and point-spread function
     - Variable source count and source parameters
    - ``chandra_point_source_catalog`` and ``daylan+2017_fermi_point_sources``
   * - Gaussian mixtures
     - Variable-width components in two-dimensional data
     - Birth and death catalog transitions
     - ``gaussian_mixture_catalog``
   * - Strong-lens imaging
     - Lens mass, foreground light, source light, and lensed emission
     - Fixed or variable perturber catalog
    - ``simulated_hst_strong_lens``, ``daylan+2018_strong_lens_subhalos``, and ``roman_strong_lens_perturber_catalog``
   * - Spectral lines
     - Voigt profiles in spectral data
     - Variable line count and profile parameters
     - ``voigt_spectral_line_catalog``
   * - Catalog association
     - Positions, values, confidence, and significance
     - Completeness and purity versus matching criteria
     - ``catalog_association_completeness_purity``

Inference and proposal engine
-----------------------------

PCAT supports fixed-dimensional parameters and populations of exchangeable
elements. Within-model proposals update parameters already in the state. Birth
and death proposals change the catalog size, while supported element models can
also use split and merge proposals. Multiple independent workers can sample a
configuration, and :func:`pcat.sampling.sample_parallel` can execute related
configurations for controlled comparisons.

Parameters are sampled in unit-prior coordinates and transformed to their
physical priors. Fixed-dimensional runs can mix single-parameter updates with
correlated block proposals. Catalog runs can include population
hyperparameters, spatial priors, spectral models, backgrounds, exposure,
point-spread functions, and instrument-specific response settings.

User-supplied likelihoods
-------------------------

``typeexpr="gener"`` runs a fixed-dimensional model with an arbitrary
user-supplied log-likelihood function. The callback receives the run state, the
model name, and the current parameter values, and returns one scalar log
likelihood:

.. code-block:: python

   import numpy as np

   from pcat.main import sample

   def log_likelihood(gdat, strgmodl, values):
       residual = values - np.array([1.0, 2.0])
       return -0.5 * residual @ residual

   result = sample(
       typeexpr="gener",
       retr_llik=log_likelihood,
       parameter_names=("x", "y"),
       prior_types=("self", "self"),
       prior_minima=(-5.0, -5.0),
       prior_maxima=(5.0, 5.0),
       proposal_scales=(0.05, 0.05),
       pathbase="analysis/custom_likelihood",
       strgcnfg="custom_likelihood",
   )

The callback supplies only the log likelihood. PCAT applies the configured
priors through its unit-coordinate transforms and provides proposal,
acceptance, persistence, convergence, and final-processing machinery. Generic
runs currently use within-model proposals. They do not activate birth, death,
split, or merge proposals.

Point sources in imaging data
-----------------------------

The ``lghtpnts`` element represents a point source with inferred position and
flux. PCAT forward-models an image from a catalog of these elements, including
the configured point-spread function and background, and infers the catalog
size and source parameters jointly. This supports crowded-field and
photon-count analyses in which detections, memberships, and blends are
uncertain.

The compact Chandra-style example is directly runnable:

.. code-block:: bash

   python examples/chandra_point_source_catalog/chandra_point_source_catalog.py

The :doc:`getting_started` page gives its configuration, while the
``daylan+2017_fermi_point_sources`` example demonstrates a larger synthetic point-source catalog.

Lenses and lensed emission in imaging data
------------------------------------------

The ``lens`` element supports strong-lensing image models. PCAT jointly models
the lens mass parameters, foreground lens-galaxy emission, source-plane
emission, its lensed image, point-spread-function convolution, and image
residuals. A transdimensional lens population can represent uncertain
additional deflectors or substructure while fixed parameters describe the
macro lens and emission components.

Run the Hubble Space Telescope Wide Field Camera 3 demonstration with:

.. code-block:: bash

   python examples/simulated_hst_strong_lens/simulated_hst_strong_lens.py

The ``roman_strong_lens_perturber_catalog`` example provides a seeded population benchmark for
catalog-level perturber detection. These examples use simulations and validate
the pipeline. They are not measurements or performance forecasts for observed
systems.

Spectral lines in spectral data
-------------------------------

Spectral-line elements use the same transdimensional catalog machinery along a
spectral axis. PCAT infers the number of lines together with line locations,
amplitudes, and profile parameters. The maintained ``lghtlinevoig``
configuration models Voigt-profile emission lines in simulated spectral data.

Run the nominal line-detection analysis with:

.. code-block:: bash

   python examples/voigt_spectral_line_catalog/voigt_spectral_line_catalog.py --configuration nomi

This example compares catalogs containing different numbers of lines. Its
short modes are pipeline checks, while scientific analyses require adequate
sampling, convergence assessment, and problem-specific prior validation.

Shared inference products
-------------------------

All four model classes use PCAT's persisted run state and final-processing
pipeline. Depending on the configuration, products include posterior samples,
model-count probabilities, catalog summaries, associations, convergence and
proposal diagnostics, posterior-predictive models, residuals, static figures,
and animations. See :doc:`outputs` for the file layout and interpretation
requirements.

Catalog summaries and diagnostics
---------------------------------

PCAT can condense posterior catalogs, associate inferred elements with a
reference catalog, and evaluate completeness and false-discovery summaries.
The public :func:`pcat.associate_catalogs` utility performs coordinate-and-value
matching for external catalogs. :func:`pcat.posterior_convergence` reports
Gelman-Rubin statistics and effective sample sizes from a completed state.
For fixed-dimensional posterior samples with normalized bounded or Gaussian
priors, :func:`pcat.diagnostics.estimate_evidence` provides a defensive
importance-sampling evidence estimate and its effective sample size.

Persistence, plotting, and reproducibility
------------------------------------------

Every full run can persist its initialized, worker, and final state as paired
pickle and HDF5 files. Final processing aggregates workers and creates model,
residual, parameter, catalog, association, and convergence products. The
plotting API creates one- and two-parameter posterior projections, while the
animation API assembles chain frames into GIF files. See :doc:`outputs` for the
directory layout and :doc:`api` for the callable interfaces.