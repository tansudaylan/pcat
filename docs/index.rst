PCAT documentation
==================

The Probabilistic Cataloger (PCAT) is a hierarchical, transdimensional Bayesian
inference framework for datasets whose number of sources or other model elements
is unknown. It samples an ensemble of catalogs rather than returning only one
best-fit catalog. The framework was introduced by `Daylan, Portillo, and
Finkbeiner (2017) <https://doi.org/10.3847/1538-4357/aa679e>`_ in ApJ 839, 4.

PCAT supports arbitrary user-supplied likelihood functions, a unified Poisson
image likelihood with point, extended, and lensed components, spectral-line
models, time-series flare catalogs, and Keplerian radial-velocity catalogs. It
also runs fixed-dimensional models through the same sampler and persistence
pipeline. Its transdimensional models infer the number and parameters of
catalog elements jointly. Simulated and supplied data, population-level priors,
posterior catalog summaries, convergence diagnostics, static figures, and
posterior animations use a shared pipeline.

.. image:: ../examples/pcat_posterior_samples.gif
   :alt: Posterior samples for Gaussian mixtures, point sources, strong lenses, and spectral lines
   :width: 760px
   :align: center

The animation shows twelve changing inference views from seven maintained
example workflows. Each panel uses fixed source-plot limits and the GIF uses one
shared color palette across all frames. The examples use simulated data except
for the explicitly identified JWST MIRI spectrum.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting_started
   capabilities
   likelihoods
   sampling
   examples
   outputs
   troubleshooting
   api

Scientific scope
----------------

PCAT represents a metamodel as the union of models with different numbers of
elements. Each element is a group of parameters that appears or disappears
together, such as the position and flux of a point source or the parameters of a
spectral line. Fixed-dimensional parameters describe shared quantities such as
backgrounds, population hyperparameters, or instrumental response.

Within-model proposals update existing parameters. Birth and death proposals
move between catalog dimensions while respecting detailed balance. Individual
element labels are exchangeable, so scientific conclusions should generally use
population summaries, condensed catalogs, associations to reference objects, or
predictions in data space rather than raw element indices.

Current status
--------------

PCAT is research software under active development. Reproducible analyses should
record the repository revision, complete run configuration, random seeds, and
the final serialized state. The output tree stores these products together for
each run.
