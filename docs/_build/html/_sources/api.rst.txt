Public API
==========

Sampling
--------

PCAT exposes its sampling interfaces in ``pcat.sampling``. These functions
support inference within one model, transdimensional inference over metamodels,
and multiple-chain runs.

.. py:function:: pcat.sampling.sample(**configuration)

   Dispatch an image, catalog, or arbitrary-likelihood PCAT configuration,
   execute sampling and final processing, and return the final global state object.

   Frequently used configuration keys include:

   * ``typeexpr`` for the experiment family, including ``"chan"``, ``"gmix"``,
     ``"fire"``, and HST Wide Field Camera 3 identifiers.
   * ``typedata`` set to ``"simu"`` for generated data or ``"inpt"`` for
     supplied data.
   * ``pathbase`` for the parent project root and ``strgcnfg`` for the
     filesystem-safe run under its common ``pcat_runs`` directory.
   * ``typeelem`` for the element types in each population.
   * ``numbswep`` and ``numbsamp`` for sampler sweeps and retained posterior
       samples. ``numbburn`` sets discarded initial sweeps,
       ``booladaptstdp`` enables burn-in-only proposal adaptation, and
       ``boolburntmpr`` with ``factburntmpr`` enables likelihood-tempered burn-in.
   * ``probtran`` and ``probspmr`` for transdimensional and split/merge proposal
       probabilities. The listed image configurations disable split/merge where
       that proposal is unsupported.
   * ``boolmakeplotinit``, ``boolmakeplotfram``, ``boolmakeplotfinlpost``, and
       ``makeanim`` for visual-output controls. Set ``boolmakeanimprop=True`` to
       render every proposed candidate before its accept/reject decision and
       assemble ``proposal_candidates.gif``. This expensive diagnostic is
       disabled by default.
   * ``numbframanim`` to record that many chain-0 states for animations, one
       third spread over burn-in starting from the initial state and two
       thirds over sampling. With ``inittype="rand"`` (the default) the first
       state is a random draw from the prior, including the catalog size. The
       final state stores them in ``listanimstate``, and PCAT's own frames are
       drawn at the same sweeps.

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

Neural transdimensional proposals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: pcat.NeuralElementProposal

   Optional PyTorch-backed conditional mixture density for proposing catalog
   elements in prior-transformed coordinates. Install ``pcat[neural]`` to use
   it. Fit the model with paired observation features and unit-cube element
   parameters, then pass ``proposal.callbacks()`` and the matching
   ``neural_proposal_features`` to :func:`pcat.sampling.sample`. The fitted
   density includes a uniform component and is used in PCAT's exact birth/death
   and prior-redraw jump corrections. See :doc:`sampling` for the training
   example and assumptions.

   ``state_callbacks(retr_condition)`` supplies callbacks conditioned on both
   the observation and an immutable :class:`pcat.ProposalContext`. The feature
   builder must use the same catalog summaries used to create training
   conditions. PCAT supplies the current catalog for births and the catalog
   with the removed element omitted for death reverse densities.

   The proposal can be checkpointed with ``save(path)`` and restored with
   ``NeuralElementProposal.load(path)``. Fit it before sampling and keep its
   parameters fixed during each chain.

.. py:class:: pcat.ProposalContext

   Immutable snapshot of active unit-prior element coordinates for each
   population. State-conditioned callbacks receive this object for a birth,
   death, or prior-redraw jump. Death contexts omit the element being removed.

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

   The returned sample axis follows the completed state's retained draws.
   Convergence-based stopping can make it shorter than ``sample_count``.
   Parameter draws and log posteriors retain the same worker and sweep order.

.. py:function:: pcat.sampling.sample_posterior(gdat, numbsampwalk, retr_llik, listnamepara, listlablpara, scalpara, minmpara, maxmpara, **options)

   Sample a fixed-dimensional posterior with PCAT chains and return a
   dictionary of post-burn-in samples keyed by parameter name. Options add
   Gaussian priors, derived variables, trace and corner plots, and a saved
   posterior summary that later runs reuse.

.. py:function:: pcat.sampling.sample_allesfitter_pcat(datadir)

   Run the PCAT-backed fixed-dimensional adapter for an allesfitter data
   directory. This compatibility entry point is specialized for that external
   package. New likelihood integrations should use
   :func:`pcat.sampling.sample_fixed` or
   :func:`pcat.sampling.sample_fixed_chains`.

.. py:function:: pcat.main.retr_listgdat(liststrgcnfg, typegdat="finlpost")

   Load the requested persisted state for each run tag in a configuration
   family.

Image formation
---------------

.. py:function:: pcat.image.make_gaussian_scene(image_shape, source_parameters, intrinsic_sigma_pixels)

   Render a two-dimensional scene from finite Gaussian sources. Each source is
   given by ``(x_pixel, y_pixel, flux)`` with coordinates inside the image.
   Every sampled profile is normalized to its requested in-frame flux. The
   returned array uses the caller's flux units.

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

Static figure-writing routines support PNG or PDF output and log their output
paths. Animation and state helpers may instead return snapshots or write GIFs.
Their return values and products are described individually below. The
:doc:`gallery` shows representative outputs produced by
``docs/make_plot_gallery.py`` from PCAT runs.

The ``state`` argument of the sampler views is the final posterior object
returned by :func:`pcat.sampling.sample` or read with
:func:`pcat.main.readfile` from ``gdatfinlpost``. Per-proposal arrays such as
``listpostindxproptype`` hold every sweep including burn-in, ordered
sweep-major across chains. Per-sample arrays such as ``listpostnumbelem`` hold
retained samples ordered sample-major across chains.

The :doc:`plotting gallery <gallery>` shows rendered output for every
figure-producing routine documented below.

Sampler operation
~~~~~~~~~~~~~~~~~

.. py:function:: pcat.plotting.plot_sampler_overview(state, output_directory, typefileplot="png", **catalog_options)

   Write every operation, catalog, and predictive view that applies to
   ``state`` and return a dictionary from view name to path. :func:`pcat.sampling.sample`
   calls it for each run with final plots enabled and writes to
   ``visuals/post/operation``. ``catalog_options`` are passed to
   :func:`pcat.plotting.plot_catalog_trace`. Its returned ``sampler_trajectory``
   view aligns proposal-level acceptance terms and likelihood temperature.

.. py:function:: pcat.plotting.plot_sampler_trajectory(state, output_path, typefileplot="png")

   Plot the signed log-compressed log-acceptance ratio for every evaluated
   proposal, colored by move type and marked as accepted or rejected. Filtered
   proposals appear along the panel baseline. The lower panel shows the
   likelihood inverse temperature by sweep. The shaded interval marks burn-in.
   See the :ref:`rendered sampler trajectory <sampler-trajectory-example>`.

.. py:function:: pcat.plotting.plot_proposal_ledger(state, output_path, typefileplot="png", number_windows=120)

   Plot the accepted fraction of each move type (within-model, birth, death,
   split, merge, jump) in ``number_windows`` sweep windows pooled over chains,
   the burn-in interval, and the likelihood inverse temperature when tempered
   burn-in is active. A second panel shows each move's share of proposals, its
   acceptance after burn-in with 1-sigma Wilson intervals, and the fraction
   rejected before the likelihood is evaluated.

.. py:function:: pcat.plotting.plot_acceptance_decomposition(state, output_path, typefileplot="png")

   Split each evaluated post-burn-in log acceptance ratio [nat] into the
   posterior and auxiliary-density change, the move-selection ratio, and the
   split/merge Jacobian, and draw their distributions per move type on a signed
   logarithmic axis. This shows whether a move fails because of the data or
   because of the proposal bookkeeping.

.. py:function:: pcat.plotting.plot_compute_budget(state, output_path, typefileplot="png")

   Map the mean wall time [ms] of each timed sampler stage for each move type.
   Stages nest, so their times do not add to each move's total. Stages timed
   once per sweep, such as frame plotting, are omitted.

Catalogs and fits
~~~~~~~~~~~~~~~~~

.. py:function:: pcat.plotting.plot_catalog_trace(state, output_path, position="elin", amplitude="flux", population=0, chain=0, position_label=None, amplitude_label=None, typefileplot="png")

   Draw every element of every retained catalog of one chain at its
   ``position`` parameter, colored by ``amplitude``, so births, deaths, splits,
   and merges appear as tracks that start, stop, fork, or join. A lower strip
   shows the catalog size and a side panel the expected number of elements per
   position bin over all chains. Use ``position="xpos"`` or ``"ypos"`` for
   image catalogs.

.. py:function:: pcat.plotting.plot_posterior_predictive(state, output_path, axis_values=None, axis_label="Data axis", data_label="Counts per bin", typefileplot="png", seed=0)

   For one-dimensional data, draw replicated data from every retained model
   with the run's Poisson or Gaussian noise and plot the data against the 68%
   and 95% replicated bands, the standardized residual, and the posterior
   predictive tail probability P(replicated > data) per bin.
   :func:`pcat.plotting.has_one_dimensional_prediction` reports whether a
   state qualifies.

.. py:function:: pcat.plotting.plot_lens_image_fit(output_path, observed, model, variance, pixel_scale_arcsec, typefileplot="png")

   Plot an observed lens image, a model image, and the standardized residual
   on a shared angular grid [arcsec].

.. py:function:: pcat.plotting.plot_lens_parameter_recovery(output_path, draws, true_parameters, typefileplot="png", labels=("Einstein radius [arcsec]", "Source x [arcsec]", "Source y [arcsec]"))

   Plot one posterior histogram per parameter with the injected value and
   posterior median, using three panels per row. The default labels describe
   the Einstein radius and source position [arcsec]. Draws must be a nonempty
   sample-by-parameter array with one injected value and one label per column.
   Supply custom labels for other parameter sets. To plot a subset, slice the
   sample columns and injected values together and supply matching labels.
   Each column must contain finite real unmasked samples. Histograms and
   medians use the same unmasked samples independently for each parameter.
   Injected values must be finite, real, and unmasked so every recovery
   panel has a defined injected-value marker.

.. py:function:: pcat.plotting.plot_detection_diagnostic(records, output_path, examples=None, detection_threshold=0.5, example_prefix=None)

   Plot representative detector, macro-model, and residual images with the
   per-lens posterior perturber probability against injected signal-to-noise
   and Wilson intervals on the detection rates. ``records`` and ``examples``
   come from :func:`pcat.roman_lens.simulate_population`.
   Each ``injected_signal_to_noise`` value must be an unmasked finite
   nonnegative real scalar. Zero values remain valid.
   Set ``example_prefix="lens_2"`` to select the supplied ``lens_2_data``,
   ``lens_2_macro``, and ``lens_2_residual`` images. The default uses the first
   sorted data-image prefix. An explicit selection requires matching examples.
   Representative data, macro-model, and residual images must have matching
   nonempty two-dimensional shapes on the detector pixel grid.
   Each panel must contain at least one unmasked pixel, and all unmasked
   values must be finite and real. Color limits exclude explicitly masked
   values while the displayed images retain their masks.
   Integer residuals use floating-point color-limit arithmetic without
   changing the supplied pixel arrays.
   Posterior probabilities must be unmasked finite real scalars in [0, 1], as
   required by :func:`pcat.roman_lens.summarize_population`.
   ``detection_threshold`` defaults to 0.5 and controls both the horizontal
   reference line and the reported detection rates and Wilson intervals.
   It is passed to the shared population summary and must be an unmasked,
   finite real scalar in [0, 1].

Convergence
~~~~~~~~~~~

.. py:function:: pcat.plotting.plot_posterior_convergence(state, output_directory, typefileplot="png")

   Write traces, autocorrelations, effective sample sizes, and multi-chain
   R-hat for the fixed-dimensional parameters, and catalog-size traces,
   occupancy, and transition matrices for each population. Element parameter
   distributions are compared between the first and second halves of the
   retained samples. Returns a dictionary from figure name to path.

.. py:function:: pcat.diagnostics.gelman_rubin(chains)

   Compute classic, unsplit R-hat for a matrix with one row per draw and one
   column per chain, requiring at least two draws and two chains. Masked,
   nonfinite, complex, or undersized traces return ``NaN``. Missing samples
   are not filled or dropped. Identical constant chains also return ``NaN``.
   Separated constant chains return infinity.
   Extreme magnitudes use power-of-two normalization before variance
   calculations, shared with :func:`pcat.diagnostics.autocorrelation_time`.

.. py:function:: pcat.plotting.plot_gelman_rubin(path, statistics, typefileplot="pdf", typeplotback="norm")

   Plot the distribution of potential scale reduction factors, for example
   ``state.gmrbstat`` over every data bin, to ``path + "gmrb"``.
   All supplied statistics must be unmasked finite real values, with at least
   one entry. Histogram bins include finite values below one and handle
   identical values without dropping entries. Undefined results are rejected
   before figure creation.

.. py:function:: pcat.diagnostics.autocorrelation_time(samples, typeverb=0, atcrtype="maxm", verbtype=None)

   Compute autocorrelation sequences and integrated times with draws on the
   first axis and parameter channels on the remaining axes. Correlations have
   the lag axis last. Times have the parameter-channel shape. Masked,
   nonfinite, complex, or constant channels return ``NaN`` outputs without
   discarding valid neighboring channels or filling or dropping samples.
   Extreme input magnitudes use power-of-two normalization before forming
   correlation products.
   Compatibility arguments do not change the native estimator.

.. py:function:: pcat.plotting.plot_autocorrelation(path, autocorrelation, correlation_time, strgextn="", typefileplot="pdf", typeplotback="norm")

   Plot one autocorrelation sequence and its integrated time to
   ``path + "atcr" + strgextn``. :func:`pcat.diagnostics.autocorrelation_time`
   returns both inputs.

Posterior projections
~~~~~~~~~~~~~~~~~~~~~

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

Animations
~~~~~~~~~~

.. py:function:: pcat.plotting.animation_states(state)

   Return the snapshots recorded with ``numbframanim``, ordered by sweep. Each
   is a dictionary with ``cntrswep``, ``boolburn``, ``facttmpr``,
   ``paragenrscalfull``, ``lliktotl``, ``lpostotl``, and, when available,
   ``numbelem``, ``cntpmodl``, and ``dictelem``.

.. py:function:: pcat.plotting.animation_phase_label(snapshot)

   Describe a snapshot as the initial prior draw, a burn-in state, or a
   posterior sample, for frame titles.

.. py:function:: pcat.plotting.make_image_sequence_animation(images, labels, output_path, duration_ms=800, image_size=640, title="Rubin DP1 lens cutouts")

   Write a GIF with one frame per two-dimensional image, a shared intensity
   stretch (1st to 99.5th percentile of all finite pixels), ``title`` above
   and each label below its image.

.. py:function:: pcat.plotting.make_posterior_animation_collage(output_path=DEFAULT_POSTERIOR_COLLAGE, examples_root=EXAMPLES_ROOT, panels=POSTERIOR_ANIMATION_PANELS, frame_count=24, duration_ms=80, panel_size=720)

   Combine posterior frame sequences of several examples into one synchronized
   GIF, as in ``examples/pcat_posterior_samples.gif``. Each
   :class:`pcat.plotting.PosteriorAnimationPanel` names a label and either a
   glob ``pattern`` of frames or a ``static_image``. A sequence whose frames do
   not change raises an error.

Instrument response
-------------------

.. py:function:: pcat.spectral.evaluate_line_profile(axis, profile, flux, center, gaussian_width=None, lorentz_width=None, mixing_fraction=None, skewness=None)

   Evaluate integrated-flux-normalized Gaussian, Lorentzian, Voigt,
   pseudo-Voigt, sinc-squared, skew-Gaussian, top-hat, or configured
   energy-dispersion line profiles.

.. py:function:: pcat.spectral.apply_gaussian_resolving_power(axis, spectrum, centers, resolving_power)

   Convolve each line column with a Gaussian line-spread function at constant
   resolving power while preserving sampled flux.

.. py:function:: pcat.spectral.apply_line_spread_function(spectrum, kernel)

   Convolve one or more spectra with an odd-length tabulated kernel while
   preserving each column's sampled flux.

.. py:function:: pcat.time_series.evaluate_flare_profile(time, profile, amplitude, peak_time, rise_time=None, decay_time=None, fwhm=None)

   Evaluate peak-normalized Gaussian, one-sided exponential-decay, FRED, or
   Davenport empirical flare profiles.

.. py:function:: pcat.time_series.log_likelihood_transit_times(observed_times, predicted_times, timing_errors)

   Return the uncertainty-weighted Gaussian log likelihood of matching
   one-dimensional transit-time arrays, up to a parameter-independent
   normalization constant. Timing errors must be positive. Nonfinite predicted
   times return negative infinity.

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
   must be unmasked integer-valued scalars satisfying
   ``0 <= successes <= trials`` with positive ``trials``. The ``z_score``
   must be an unmasked positive finite real scalar.
   Closed-form boundary limits give an exact lower endpoint of 0 for zero
   counts and an exact upper endpoint of 1 for full counts.
   Very large finite scores use reciprocal scaling to avoid overflow when
   squaring the score.
   Large trial counts use a scaled form of the same equation without
   squaring the trial count.

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

.. py:function:: pcat.diagnostics.estimate_kl_divergence(samples, log_density_p, log_density_q)

   Estimate forward ``D_KL(P || Q)`` as the Monte Carlo mean of
   ``log_density_p - log_density_q`` for draws from ``P``. Both callbacks must
   return normalized log densities and accept a matrix of draws. Samples may
   have shape ``(draws, parameters)`` or ``(chains, draws, parameters)``. The
   result includes an autocorrelation-aware standard error and effective sample
   size. Finite-sample estimates may be negative. If ``Q`` has zero density at
   any draw from ``P``, the estimated divergence is infinite.

   Samples and log-density outputs must be unmasked real values. Each callback
   must return one value per draw, rather than a scalar or column matrix.

.. py:function:: pcat.diagnostics.estimate_mutual_information(samples, log_density_joint, log_density_x, log_density_y, split_index=1)

   Estimate ``I(X;Y) = D_KL(p(X,Y) || p(X)p(Y))`` in nats using draws from the
   joint distribution. Columns before ``split_index`` belong to ``X`` and the
   remaining columns belong to ``Y``. Both groups must be nonempty. The sample
   shape is ``(draws, parameters)`` or ``(chains, draws, parameters)``.

   Supply normalized joint and marginal log-density callbacks for the same
   distribution. The joint callback receives all columns. Marginal callbacks
   receive only their respective columns and must each return one value per
   row. The function does not estimate marginal densities from a sample cloud.

   The returned dictionary contains ``mutual_information``, ``standard_error``,
   and ``effective_sample_size``. Uncertainty accounts for within-chain
   autocorrelation and variation between independent chains, as in the KL
   estimator. Finite-sample estimates are not clipped at zero. Error estimates
   do not establish that the chains have converged or that supplied density
   callbacks are mutually consistent.

Roman lens benchmark
--------------------

.. py:function:: pcat.roman_lens.simulate_population(number_lenses=500, seed=814, config=None)

   Simulate a seeded population of strong lenses with and without a perturber,
   then evaluate the zero-versus-one-perturber catalog probability.

.. py:function:: pcat.roman_lens.summarize_population(records, detection_threshold=0.5)

   Report true-positive, false-positive, and localization rates for the
   simulated population. Wilson intervals apply to the true-positive and
   false-positive rates. Each ``posterior_one`` value must be an unmasked
   finite real scalar in [0, 1]. Each ``has_subhalo`` value must be an unmasked
   Boolean or real scalar equal to 0 or 1. Text flags such as ``"False"``
   are rejected. Both localization indices must be unmasked integer-valued
   scalars. Exact integer-valued reals and integer strings are accepted,
   including the ``-1`` null-injection sentinel. Values must fit the NumPy
   platform integer type (``np.int_``). Fractional inputs are
   rejected. ``detection_threshold`` must be an unmasked, finite real scalar
   in [0, 1]. Plot the population with
   :func:`pcat.plotting.plot_detection_diagnostic`.

mejiro lens inference
---------------------

These functions import mejiro and lenstronomy only when called.

.. py:function:: pcat.mejiro_lens.simulate_mejiro_exposure(strong_lens=None, band="F129", exposure_time=146.0, fov_arcsec=5.0, supersampling_factor=3, seed=2026)

   Render a mejiro strong lens (default ``SampleGG``) with mejiro's Roman
   instrument, add its zodiacal and thermal sky, and draw one Poisson exposure.
   Returns a plain-data specification with the observed counts and injected
   parameters.

.. py:function:: pcat.mejiro_lens.render_mejiro_counts(specification, parameters)

   Return expected counts per pixel for the sampled SIE, shear, and source
   parameters listed in ``pcat.mejiro_lens.PARAMETERS``.

.. py:function:: pcat.mejiro_lens.run_mejiro_lens_inference(specification, output_root, run_name, numbswep=60000, numbburn=40000, numbsamp=2000, numbproc=4, numbframanim=None)

   Sample the Poisson likelihood with PCAT from a prior draw, with tempered
   burn-in, adaptive scales, and differential-evolution jumps.

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
