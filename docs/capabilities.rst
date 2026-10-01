Capabilities
============

PCAT applies one posterior-sampling and output pipeline to fixed-dimensional
models and transdimensional catalogs. A run can use simulated data or supplied
data, and its model can contain a fixed set of parameters, a variable number of
elements, or both.

Functionality overview
----------------------

Maintained workflows include:

* **Arbitrary likelihoods.** User callbacks with fixed-dimensional scalar or
  correlated proposals. The ``typeexpr="gener"`` example is described below.
* **Poisson image likelihood.** Point sources, extended emission, and lensed
  sources are element populations in one image model with backgrounds,
  exposure, and point-spread functions. The Chandra, Fermi-LAT, Hubble Space
  Telescope (HST), and Roman examples exercise different combinations of
  these components.
* **Unbinned Gaussian-mixture likelihood.** A distinct likelihood class models
  observed coordinates directly as draws from a variable-component mixture,
  without first accumulating them into image pixels. The current public
  ``gaussian_mixture_catalog`` example remains a binned Poisson image workflow
  until the unbinned dispatcher is completed.
* **Spectral and time-series likelihoods.** Variable-count spectral lines with
  eight intrinsic profile choices, instrument line-spread convolution, four
  flare profiles, and Keplerian radial-velocity models.
* **Transdimensional catalog products.** Every transdimensional likelihood
  produces samples of catalogs. By default, final processing condenses those
  samples into a catalog before optional association with a reference catalog
  when the elements have supported two-dimensional spatial coordinates. Set
  ``boolcondcatl=False`` to skip this potentially expensive summary. Catalog
  association is a posterior product rather than a likelihood family.

Inference and proposal engine
-----------------------------

PCAT supports fixed-dimensional parameters and populations of exchangeable
elements. Within-model proposals update parameters already in the state. Birth
and death proposals change the catalog size, while supported element models can
also use split and merge proposals. Multiple independent workers can sample a
configuration, and :func:`pcat.sampling.sample_parallel` can execute related
configurations for controlled comparisons.

See :doc:`sampling` for the proposal acceptance ratio, move-selection settings,
proposal-scale adaptation, burn-in, and automatic convergence monitoring.

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

Poisson image likelihood
------------------------

Point sources, extended emission, foreground light, and gravitational lenses
are alternative model components within the same image-data likelihood. The
catalog may contain one or several supported element populations; it does not
change the likelihood family. With the default ``boolcondcatl=True``, final
processing creates a condensed catalog from the sampled catalogs before any
optional association with a reference catalog.

Point-source and extended emission
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``lghtpnts`` element represents a point source with inferred position and
flux. PCAT forward-models an image from a catalog of these elements, including
the configured point-spread function and background, and infers the catalog
size and source parameters jointly. This supports crowded-field and
photon-count analyses in which detections, memberships, and blends are
uncertain.

For deterministic Gaussian-PSF image formation outside the sampler,
:func:`pcat.image.forward_model_image` convolves a two-dimensional scene,
normalizes the kernel, and adds a uniform background. It returns the intrinsic
scene, PSF kernel, and predicted image for inspection or a user-defined
likelihood.

The compact Chandra-style example is directly runnable:

.. code-block:: bash

   python examples/chandra_point_source_catalog/chandra_point_source_catalog.py

The :doc:`getting_started` page gives its configuration, while the
``daylan+2017_fermi_point_sources`` example demonstrates a larger synthetic point-source catalog.
See :doc:`likelihoods` for image geometry, backgrounds, extended-source
components, lensing options, exposure, and PSF modes.

Lenses and lensed emission
~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Spectral and time-series catalogs
---------------------------------

Spectral-line elements use the same transdimensional catalog machinery along a
spectral axis. PCAT infers the number of lines together with line locations,
amplitudes, and profile parameters. Gaussian, Lorentzian, Voigt, pseudo-Voigt,
sinc-squared, skew-Gaussian, top-hat, and configured energy-dispersion profiles
are available. A Gaussian line-spread function at constant resolving power or
a measured tabulated kernel can convolve the intrinsic profiles while
preserving line flux. The maintained ``lghtlinevoig``
configuration models Voigt-profile emission lines in simulated spectral data
and in the public JWST MIRI spectrum of NGC 7027. The observed-spectrum example
uses measured flux uncertainties and a stated error floor to construct the
effective Poisson counts consumed by PCAT.

Run the nominal line-detection analysis with:

.. code-block:: bash

   python examples/voigt_spectral_line_catalog/voigt_spectral_line_catalog.py --configuration nomi

This example compares catalogs containing different numbers of lines. Its
short modes are pipeline checks, while scientific analyses require adequate
sampling, convergence assessment, and problem-specific prior validation.

The same one-dimensional machinery supports binned time-series catalogs.
Gaussian, instantaneous-rise exponential-decay, fast-rise exponential-decay
(FRED), and Davenport empirical flare templates have profile-specific sampled
parameters. The ``variable_number_stellar_flares`` example fits native FRED
components with independent rise and decay times to simulated FRED flares. The
``variable_number_stellar_spots`` example fits a transdimensional catalog of
periodic Gaussian spot dips to simulated stellar photometry. It samples the
number, phase, depth, and width of spots for a fixed rotation period. The
``variable_number_exoplanets_radial_velocity`` example instead uses dedicated
Keplerian elements, a radial-velocity likelihood, analytically marginalized
instrument offsets, and numerically marginalized stellar jitter for a simulated
two-instrument data set.

See :doc:`likelihoods` for the available spectral profiles, response settings,
line parameters, and input options.

Shared inference products
-------------------------

All supported likelihood and model families use PCAT's persisted run state and
final-processing pipeline. Depending on the configuration, products include posterior samples,
model-count probabilities, catalog summaries, associations, convergence and
proposal diagnostics, posterior-predictive models, residuals, static figures,
and animations. See :doc:`outputs` for the file layout and interpretation
requirements.

Catalog summaries and diagnostics
---------------------------------

Every transdimensional run samples catalogs whose element labels can vary
between states. With ``boolcondcatl=True``, PCAT condenses those samples into a
persistent posterior catalog before associating inferred elements with a
reference catalog or evaluating completeness and false-discovery summaries.
The current condensation algorithm clusters elements by ``xpos`` and ``ypos``.
One-dimensional spectral, flare, and radial-velocity catalogs retain their raw
catalog samples but do not yet receive this condensed spatial summary.
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