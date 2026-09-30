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
   * ``pathbase`` and ``strgcnfg`` for the output root and filesystem-safe run
     tag.
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
   transdimensional proposal probabilities to zero. See :doc:`capabilities`

   ``proposal_correlation`` optionally defines correlated type-0 block moves.
   Set ``probpropblock`` to their mixture probability and ``factpropblock`` to
   their scale relative to the adapted per-parameter proposal scales.
   for a complete example and the image and spectral model families.

.. py:function:: pcat.main.init(configuration)

   Lower-level initialization and execution engine. New image-analysis code
   should normally call :func:`pcat.main.sample`, which performs experiment
   dispatch before invoking this function.

Plotting
--------

.. py:function:: pcat.plotting.plot_grid(path, name, listpara, listlablparatotl, truepara=None, listvarbdraw=None, typefileplot="pdf")

   Render the diagonal marginal distributions and every lower-triangle pairwise
   posterior projection. ``listpara`` contains one row per sample and one
   column per parameter. ``truepara`` marks injected values and
   ``listvarbdraw`` can mark maximum-likelihood or other reference vectors.
   The output path is ``path_name.pdf`` or ``path_name.png``.

Example analyses
----------------

.. py:function:: pcat.demo.run_pipeline_demo(output_root, **configuration)

   Apply the reduced sampling and plotting settings used by the simulated image
   analyses, set ``pathbase``, and call :func:`pcat.main.sample`.

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