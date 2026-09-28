Public API
==========

Sampling
--------

.. py:function:: pcat.main.sample(**configuration)

   Dispatch an image or general PCAT configuration, execute sampling and final
   processing, and return the final global state object.

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

.. py:function:: pcat.main.init(configuration)

   Lower-level initialization and execution engine. New image-analysis code
   should normally call :func:`pcat.main.sample`, which performs experiment
   dispatch before invoking this function.

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