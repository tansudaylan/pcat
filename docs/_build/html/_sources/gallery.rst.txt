Plotting gallery
================

Each figure below is an unedited output of one public plotting routine. Run
``python docs/make_plot_gallery.py`` to regenerate all of them. The
transdimensional figures come from a four-chain PCAT run on the simulated
two-line Voigt spectrum of ``examples/voigt_spectral_line_catalog`` (20,000
sweeps per chain, half discarded as burn-in), the fixed-dimensional figures from
a four-chain run on a correlated bivariate Gaussian, and the lens figures from
the simulated Roman/WFI strong-lens benchmark. All data are simulated.

Sampler operation
-----------------

:func:`pcat.plotting.plot_sampler_overview` writes every view in this section
and the next that applies to a run; :func:`pcat.sampling.sample` calls it for
each run with final plots enabled.

:func:`pcat.plotting.plot_proposal_ledger` follows each move type through
burn-in and sampling and summarizes its acceptance after burn-in.

.. image:: _static/gallery/proposal_ledger.png
   :alt: Acceptance fraction per move type through the run and per-move totals
   :width: 100%

:func:`pcat.plotting.plot_acceptance_decomposition` splits each log
acceptance ratio into the posterior change, the move-selection ratio, and the
split/merge Jacobian.

.. image:: _static/gallery/acceptance_decomposition.png
   :alt: Distributions of Metropolis-Hastings terms per move type
   :width: 100%

:func:`pcat.plotting.plot_compute_budget` locates where each move spends its
wall time.

.. image:: _static/gallery/compute_budget.png
   :alt: Mean wall time per sampler stage and move type
   :width: 100%

Catalogs and fits
-----------------

:func:`pcat.plotting.plot_catalog_trace` draws every element of every retained
catalog of one chain, so births, deaths, splits, and merges appear as tracks.

.. image:: _static/gallery/catalog_trace.png
   :alt: Element positions of every retained catalog with catalog size
   :width: 100%

:func:`pcat.plotting.plot_posterior_predictive` compares the data with data
replicated from the posterior.

.. image:: _static/gallery/posterior_predictive.png
   :alt: Data, replicated-data bands, residuals, and tail probabilities
   :width: 100%

:func:`pcat.plotting.plot_lens_image_fit` and
:func:`pcat.plotting.plot_lens_parameter_recovery` summarize one simulated
Roman/WFI lens fit.

.. image:: _static/gallery/lens_image_fit.png
   :alt: Observed lens image, model, and standardized residual
   :width: 100%

.. image:: _static/gallery/lens_parameter_recovery.png
   :alt: Posterior histograms of the lens parameters with injected values
   :width: 100%

:func:`pcat.plotting.plot_detection_diagnostic` summarizes perturber detection
over a simulated lens population.

.. image:: _static/gallery/detection_diagnostic.png
   :alt: Lens images and perturber probability against signal-to-noise
   :width: 80%

Convergence
-----------

:func:`pcat.plotting.plot_posterior_convergence` writes one set of figures per
run. The transdimensional run adds catalog-size diagnostics.

.. image:: _static/gallery/convergence_transdimensional/element_count_trace.png
   :alt: Catalog-size trace per chain
   :width: 100%

.. image:: _static/gallery/convergence_transdimensional/element_count_transitions.png
   :alt: Catalog-size transition matrix
   :width: 60%

.. image:: _static/gallery/convergence_transdimensional/element_parameter_pop0_elin_stability.png
   :alt: Line-energy distribution in the first and second halves of the samples
   :width: 70%

.. image:: _static/gallery/convergence_fixed/fixed_parameter_trace.png
   :alt: Parameter traces of four chains
   :width: 100%

.. image:: _static/gallery/convergence_fixed/fixed_parameter_mixing.png
   :alt: Effective sample size and R-hat per parameter
   :width: 70%

:func:`pcat.plotting.plot_gelman_rubin` histograms the potential scale
reduction factor of the model in every data bin, and
:func:`pcat.plotting.plot_autocorrelation` plots one autocorrelation sequence.

.. image:: _static/gallery/transdimensional_gmrb.png
   :alt: Distribution of potential scale reduction factors
   :width: 60%

.. image:: _static/gallery/fixed_atcr.png
   :alt: Autocorrelation of one parameter
   :width: 60%

Posterior projections
---------------------

:func:`pcat.plotting.plot_grid` draws all one- and two-parameter projections,
here with the injected values marked.

.. image:: _static/gallery/posterior_grid.png
   :alt: Corner plot of a correlated bivariate Gaussian posterior
   :width: 60%

:func:`pcat.plot_population_grid` compares labeled sample populations.

.. image:: _static/gallery/pmar_gallery_scatscat.png
   :alt: Corner plot comparing two simulated populations
   :width: 80%

Animations
----------

:func:`pcat.plotting.make_image_sequence_animation` animates images with a
shared stretch, here lensed arcs from six posterior draws.

.. image:: _static/gallery/image_sequence_animation.gif
   :alt: Lensed arcs of posterior draws
   :width: 50%

:func:`pcat.plotting.make_posterior_animation_collage` combines the posterior
frames of several examples; ``examples/run_all_examples.py`` writes the
collage shown on the :doc:`index` page.

.. image:: ../examples/pcat_posterior_samples.gif
   :alt: Collage of posterior frames from several examples
   :width: 100%
