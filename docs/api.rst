Public API
==========

Sampling
--------

New integrations should use ``pcat.sampling``. The implementation currently
remains in ``pcat.main`` for backward compatibility while the sampler engine is
decomposed into smaller modules.

.. py:function:: pcat.sampling.sample(**configuration)

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
       ``makeanim`` for visual-output controls. Set ``boolmakeanimprop=True`` to
       render every proposed candidate before its accept/reject decision and
       assemble ``proposal_candidates.gif``. This expensive diagnostic is
       disabled by default.

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

.. py:function:: pcat.sampling.init(dictglob, **options)

   Lower-level initialization and execution engine. ``dictglob`` contains
   model and data configuration overrides, while keyword options control
   sampling, diagnostics, persistence, and plotting. Most analyses should call
   :func:`pcat.sampling.sample`, which dispatches the experiment before
   invoking this function.

.. py:function:: pcat.sampling.init_image(**configuration)

   Initialize built-in image, spectral, or time-series configuration and
   construct the populated state passed to :func:`pcat.sampling.init`. The
   historical function name predates the one-dimensional workflows.

.. py:function:: pcat.sampling.sample_parallel(dictpcatinptvari, listlablcnfg, dictpcatinpt=None, **options)

   Execute a family of related configurations, optionally in separate
   processes. ``dictpcatinpt`` contains shared settings,
   ``dictpcatinptvari`` contains per-configuration overrides, and
   ``listlablcnfg`` supplies the configuration labels.

.. py:function:: pcat.sampling.sample_fixed(**configuration)

   Run a fixed-dimensional likelihood through the same native PCAT sampling
   and persistence pipeline.

.. py:function:: pcat.sampling.sample_fixed_chains(state, log_likelihood, log_prior, names, scales, minima, maxima, means, stdvs, initial, chain_count, sample_count, burn_count, **options)

   Run ``chain_count`` independent native PCAT chains of a fixed-dimensional
   likelihood ``log_likelihood(values, state)`` and return them as arrays of
   shape (chains, samples, parameters) with their log posteriors. Extra
   options, such as ``booladaptstdp``, pass through to PCAT.

.. py:function:: pcat.sampling.sample_posterior(gdat, numbsampwalk, retr_llik, listnamepara, listlablpara, scalpara, minmpara, maxmpara, **options)

   Sample a fixed-dimensional posterior with PCAT chains and return a
   dictionary of post-burn-in samples keyed by parameter name. Options add
   Gaussian priors, derived variables, trace and corner plots, and a saved
   posterior summary that later runs reuse.

.. py:function:: pcat.sampling.sample_allesfitter_pcat(datadir)

   Run the PCAT-backed fixed-dimensional adapter for an allesfitter data
   directory. This compatibility entry point is specialized for that external
   package; new likelihood integrations should use
   :func:`pcat.sampling.sample_fixed` or
   :func:`pcat.sampling.sample_fixed_chains`.

.. py:function:: pcat.main.retr_listgdat(liststrgcnfg, typegdat="finlpost")

   Load the requested persisted state for each run tag in a configuration
   family.

Image formation
---------------

.. py:function:: pcat.image.forward_model_image(source_image, psf_sigma_pixels, background=0.0)

   Convolve a finite two-dimensional source scene with a normalized circular
   Gaussian point-spread function and add a finite uniform background. The
   returned dictionary contains ``source_image``, ``psf_kernel``, and
   ``observed_image`` arrays. This deterministic utility does not run the
   sampler or draw Poisson noise.

   ``psf_sigma_pixels`` is the Gaussian standard deviation in pixels. The
   source scene and background use the same image units.

.. code-block:: python

   from pcat.image import forward_model_image

   products = forward_model_image(source_image, psf_sigma_pixels=1.5, background=5.0)

Plotting
--------

.. py:function:: pcat.plotting.plot_grid(path, name, listpara, listlablparatotl, scalpara=None, truepara=None, join=False, listvarbdraw=None, typefileplot="pdf", **kwargs)

   Render the diagonal marginal distributions and every lower-triangle pairwise
   posterior projection. ``listpara`` contains one row per sample and one
   column per parameter. ``truepara`` marks injected values and
   ``listvarbdraw`` can mark maximum-likelihood or other reference vectors.
   The output path is ``path_name.pdf`` or ``path_name.png``. ``scalpara``,
   ``join``, and extra keyword arguments are accepted for compatibility with
   earlier callers but do not alter the current native grid rendering.

.. py:function:: pcat.plot_population_grid(listlablpara, listpara=None, dictpara=None, pathbase=None, strgextn=None, typefileplot="png", **options)

   Plot marginal distributions and pairwise projections for one or more
   populations supplied either as arrays or a parameter dictionary. Optional
   settings control triangular grids, histograms, pair plots, population
   labels, markers, limits, and annotations. The
   ``examples/population_grid`` workflow demonstrates the maintained interface.

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
   analyses, set ``pathbase``, and call :func:`pcat.sampling.sample`.

.. py:function:: pcat.demo.run_example_script(relative_path, *arguments)

   Run a repository example from a notebook while isolating its command-line
   arguments.

Catalog analysis
----------------

.. py:function:: pcat.binomial_wilson_interval(successes, trials, z_score=1.0)

   Return the lower and upper Wilson interval for a binomial fraction. Counts
   must satisfy ``0 <= successes <= trials`` and ``z_score`` must be positive.

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
   :func:`pcat.sampling.sample` invoke this automatically.

.. py:function:: pcat.main.proc_anim(strgcnfg, pathbase=None)

   Build GIF animations from posterior frame plots for an existing run.

Runtime path helpers
--------------------

``pcat.paths`` exports ``get_repository_path()``, ``get_data_path()``, and
``get_visuals_path()`` for repository-local tooling. These helpers use
``PCAT_PATH`` and are distinct from the per-analysis ``pathbase`` used by the
sampler.