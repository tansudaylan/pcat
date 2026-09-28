Troubleshooting
===============

Output path errors
------------------

``No output data path is configured`` means neither ``pathbase`` nor a supported
environment variable was provided. Pass ``pathbase`` explicitly or set
``PCAT_DATA_PATH``. ``PCAT_PATH`` identifies the repository for local utilities
and does not select a sampler output directory.

If final processing or animation reports that ``PCAT_DATA_PATH`` is not set,
pass the same ``pathbase`` used for sampling. A run cannot be reconstructed from
``visuals/`` alone because final processing reads serialized states under
``data/outp/<run tag>/``.

Incomplete runs
---------------

Inspect ``data/outp/<run tag>/stat.txt`` from bottom to top. A complete posterior
run normally records these stages in order:

.. code-block:: text

   gdatinit written.
   gdatmodipost written.
   gdatfinlpost written.
   plotfinlpost written.
   animfinl written.

The animation marker is expected only when animation generation is enabled.
Missing ``gdatinit`` indicates initialization did not finish. Missing
``gdatmodipost`` indicates a worker failed before writing its chain. Missing
``gdatfinlpost`` means worker products were not aggregated. Read the original
traceback before rerunning so a configuration defect is not hidden by cached
partial state.

Stale caches
------------

PCAT can reuse existing state for a matching run tag. After changing model
structure, output paths, or plotting configuration, make a fresh run. Commands
that support fresh initialization accept ``--fresh``. Otherwise remove both
``data/outp/<run tag>`` and the corresponding ``visuals/`` directory, preserving
any products needed for provenance first.

Missing figures or animations
-----------------------------

Check ``boolmakeplot``, ``boolmakeplotinit``, ``boolmakeplotfram``,
``boolmakeplotfinlpost``, and ``makeanim``. Animations require at least two frame
plots, controlled by chain length and ``numbswepplot``. Frames that are visually
identical may collapse to a single-frame GIF; such a file is a static image and
should not be reported as an animation.

For headless systems, set a writable Matplotlib cache before running:

.. code-block:: bash

   MPLCONFIGDIR=/tmp/pcat-mplconfig python examples/chan_demo/generate_demo.py

Sampling configuration failures
-------------------------------

``Sampling failed due to incomplete or inconsistent model configuration`` wraps
a lower-level attribute, key, index, or type error. Preserve the chained
traceback and compare the configuration with the closest example.
Explicitly set model-specific values in ``dicttrue`` and ``dictfitt`` when an
experiment default would otherwise replace them.

Split and merge proposals are not implemented for every element model. For
those models, set ``probspmr=0.0``. Birth and death proposals still provide
genuine transdimensional model-count changes when ``probtran`` is nonzero.

Convergence
-----------

When ``boolcheckconv`` is enabled, inspect ``boolconv``,
``maxmconvrhatcalc``, and ``numbsampconveffccalc`` in the final state. A process
that reaches its sweep limit and writes all files is operationally complete but
may still be statistically unconverged. Increase chain length or improve the
proposal and prior configuration based on acceptance, trace, residual, and
model-count diagnostics rather than weakening thresholds solely to terminate a
run.

Minimal diagnostic sequence
---------------------------

1. Reproduce the issue with a deterministic seed and a small configuration.
2. Run the closest example in the same Python environment.
3. Compare ``cmndargs.txt``, ``stat.txt``, and the final completed stage.
4. Confirm every pickle state has its matching HDF5 companion.
5. Rerun under a new tag after correcting the root cause.