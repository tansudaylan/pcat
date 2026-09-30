Public API
==========

Sampling
--------

.. py:function:: pcat.main.sample(**configuration)

   Dispatch an image, catalog, or fixed-dimensional PCAT configuration, execute
   sampling and final processing, and return the final global state object.

   Frequently used configuration keys include:

   * ``typeexpr`` for the experiment family, including ``"chan"``, ``"gmix"``,
     ``"fire"``, and HST Wide Field Camera 3 identifiers.
   * ``typedata`` set to ``"simu"`` for generated data or ``"inpt"`` for
     supplied data.
   * ``pathbase`` for the parent project root and ``strgcnfg`` for the
     filesystem-safe run under its common ``pcat_runs`` directory.
   * ``typeelem`` for the element types in each population.
   * ``numbswep`` and ``numbsamp`` for sampler sweeps and retained posterior
     samples.
   * ``probtran`` and ``probspmr`` for transdimensional and split/merge proposal
       probabilities. The listed image configurations disable split/merge where
       that proposal is unsupported.
   * ``boolmakeplotinit``, ``boolmakeplotfram``, ``boolmakeplotfinlpost``, and
     ``makeanim`` for visual-output controls.

   ``typeexpr="gener"`` registers ``parameter_names`` as ordinary base
   parameters. ``prior_types`` accepts ``"self"`` for bounded uniform priors
   and ``"gaus"`` for Gaussian priors. ``prior_minima``, ``prior_maxima``,
   ``prior_means``, ``prior_stdvs``, and ``proposal_scales`` each contain one
   value per parameter. ``initial_values`` optionally supplies a finite
   physical initial state through PCAT's native initializer. The callback
   ``retr_llik(gdat, strgmodl, values)``
   returns the log likelihood only. PCAT's unit-coordinate transforms encode
   the priors. Generic runs use the standard type-0 proposal and set
   transdimensional proposal probabilities to zero.

   ``proposal_correlation`` optionally defines correlated type-0 block moves.
   Set ``probpropblock`` to their mixture probability and ``factpropblock`` to
   their scale relative to the adapted per-parameter proposal scales.
   See :doc:`capabilities` for a complete example and the image and spectral
   model families.

.. py:function:: pcat.main.init(configuration)

   Lower-level initialization and execution engine. New image-analysis code
   should normally call :func:`pcat.main.sample`, which performs experiment
   dispatch before invoking this function.

.. py:function:: pcat.main.sample_parallel(dictargsvari, listnamecnfgextn, dictpcatinpt=None, **options)

   Execute a family of related configurations, optionally in separate
   processes. ``dictpcatinpt`` contains shared settings and ``dictargsvari``
   contains per-configuration overrides keyed by configuration name.

.. py:function:: pcat.main.retr_listgdat(liststrgcnfg, typegdat="finlpost")

   Load the requested persisted state for each run tag in a configuration
   family.

Plotting
--------

.. py:function:: pcat.plotting.plot_grid(path, name, listpara, listlablparatotl, truepara=None, listvarbdraw=None, typefileplot="pdf")

   Render the diagonal marginal distributions and every lower-triangle pairwise
   posterior projection. ``listpara`` contains one row per sample and one
   column per parameter. ``truepara`` marks injected values and
   ``listvarbdraw`` can mark maximum-likelihood or other reference vectors.
   The output path is ``path_name.pdf`` or ``path_name.png``.

Instrument response
-------------------

.. py:function:: pcat.psf_poly_fit(gdat, psfnusam, factusam)

   Fit a piecewise-cubic subpixel model to an oversampled one-dimensional
   point-spread function. The returned coefficient columns reconstruct each
   detector pixel as a polynomial in subpixel offset.

Example analyses
----------------

.. py:function:: pcat.demo.run_pipeline_demo(output_root, **configuration)

   Apply the reduced sampling and plotting settings used by the simulated image
   analyses, set ``pathbase``, and call :func:`pcat.main.sample`.

.. py:function:: pcat.demo.run_example_script(relative_path, *arguments)

   Run a repository example from a notebook while isolating its command-line
   arguments.

Catalog analysis
----------------

.. py:function:: pcat.associate_catalogs(coordinates_source, values_source, coordinates_target, values_target, distance_maximum, value_difference_maximum, confidence_target=None, significance_target=None)

   Match each source to compatible nearby targets. The result reports source
   matches and can include the best target confidence and significance.

.. py:function:: pcat.posterior_convergence(state, max_rhat=1.05, min_effective_sample_size=200.0)

   Summarize Gelman-Rubin statistics and autocorrelation-based effective sample
   sizes for a completed posterior state, including a combined pass flag.

.. py:function:: pcat.diagnostics.estimate_evidence(posterior, log_likelihood, prior_types, prior_minima, prior_maxima, prior_means=None, prior_stdvs=None, sample_count=4000, seed=None)

   Estimate normalized evidence for fixed-dimensional posterior samples using
   independent defensive importance draws. The result includes the log
   evidence, relative error estimate, and importance effective sample size.

Roman lens benchmark
--------------------

.. py:function:: pcat.roman_lens.simulate_population(number_lenses=500, seed=814, config=None)

   Simulate a seeded population of strong lenses with and without a perturber,
   then evaluate the zero-versus-one-perturber catalog probability.

.. py:function:: pcat.roman_lens.summarize_population(records, detection_threshold=0.5)

   Report detection, false-positive, and localization rates with Wilson
   intervals for the simulated population.

.. py:function:: pcat.roman_lens.plot_detection_diagnostic(records, output_path, examples=None)

   Plot representative detector, macro-model, and residual images together
   with the population catalog probabilities.

Persistence
-----------

.. py:function:: pcat.main.readfile(path)

   Reconstruct a PCAT state from its pickle and HDF5 companions. Pass the path
   without ``.p`` or ``.h5``.

.. py:function:: pcat.main.writfile(state, path)

   Persist a PCAT state to pickle and HDF5 companions.

.. py:function:: pcat.main.proc_finl(gdat=None, strgcnfg=None, strgpdfn="post", listnamevarbproc=None, forcplot=False)

   Aggregate worker states and create final posterior products. Normal calls to
   :func:`pcat.main.sample` invoke this automatically.

.. py:function:: pcat.main.proc_anim(strgcnfg, pathbase=None)

   Build GIF animations from posterior frame plots for an existing run.

Runtime path helpers
--------------------

``pcat.paths`` exports ``get_repository_path()``, ``get_data_path()``, and
``get_visuals_path()`` for repository-local tooling. These helpers use
``PCAT_PATH`` and are distinct from the per-analysis ``pathbase`` used by the
sampler.