Capabilities
============

PCAT applies one posterior-sampling and output pipeline to fixed-dimensional
models and transdimensional catalogs. A run can use simulated data or supplied
data, and its model can contain a fixed set of parameters, a variable number of
elements, or both.

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

   python examples/chan_demo/generate_demo.py

The :doc:`getting_started` page gives its configuration, while the
``Daylan+2017`` example demonstrates a larger synthetic point-source catalog.

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

   python examples/hst_lens/generate_demo.py

The ``roman_lens_catalog`` example provides a seeded population benchmark for
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

   python examples/voigt-profile/pcat_voigt_profile_detection.py --configuration nomi

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