Outputs and diagnostics
=======================

Directory layout
----------------

A run with project root ``pathbase`` and tag ``strgcnfg`` writes:

.. code-block:: text

   pathbase/
   `-- pcat_runs/
       `-- strgcnfg/
           |-- data/
           |   `-- outp/
           |       `-- strgcnfg/
           |           |-- cmndargs.txt
           |           |-- stat.txt
           |           |-- gdatinit.p
           |           |-- gdatinit.h5
           |           |-- gdatmodi0000post.p
           |           |-- gdatmodi0000post.h5
           |           |-- gdatfinlpost.p
           |           `-- gdatfinlpost.h5
           `-- visuals/
               |-- init/
               `-- post/
                   |-- fram/
                   |-- finl/
                   `-- anim/

Each run is a self-contained child of ``pathbase/pcat_runs``. PCAT also accepts
``pathbase`` values that already identify either ``pcat_runs`` or the run root
without repeating either directory.

State files
-----------

``gdatinit`` stores initialized configuration and model metadata.
``gdatmodi####post`` stores worker-chain products. ``gdatfinlpost`` stores the
aggregated posterior state and derived summaries. PCAT splits each state between
a pickle file and an HDF5 companion, and :func:`pcat.main.readfile` reconstructs
the object from the extension-free path:

.. code-block:: python

   from pathlib import Path

   from pcat.main import readfile

   project_root = Path("examples")
   run_root = project_root / "pcat_runs" / "gaussian_mixture_catalog"
   state = readfile(
       str(run_root / "data" / "outp" / "gaussian_mixture_catalog" / "gdatfinlpost")
   )
   retained_sample_count = state.numbsamp

Completion markers
------------------

``stat.txt`` is append-only and records completed stages. A fully processed run
normally contains ``gdatfinlpost written.``, ``plotfinlpost written.``, and,
when animations are enabled, ``animfinl written.``. Validate the corresponding
files as well as the markers before treating a run as complete.

Visual products
---------------

``init`` contains simulated data, reference-model, and initial diagnostics.
``post/fram`` contains snapshots across the chain. ``post/finl`` contains final
posterior summaries and diagnostics. ``post/anim`` contains GIFs assembled from
frame plots. When proposal metadata is present, it also contains
``proposal_activity.gif`` with cumulative attempts and acceptances and
``proposal_sequence.gif`` with proposal labels on saved chain-state frames.
With ``boolmakeanimprop=True``, ``post/fram`` also contains one
``proposal_candidates_swep#########.png`` per sweep of worker zero and
``post/anim/proposal_candidates.gif`` contains all of those attempted states,
including rejected candidates. Candidate frames are not posterior samples.
Empty optional directories are removed at successful completion.

Condensed catalogs and association
----------------------------------

Transdimensional runs always retain the sampled catalogs in
``listpostdictelem``, ``listpostindxelemfull``, and related final-state arrays.
With ``boolcondcatl=True`` and supported two-dimensional spatial elements,
final processing groups samples by ``xpos`` and ``ypos`` into a condensed
catalog. ``dictglob["liststkscond"]`` stores the grouped samples,
``dictglob["poststkscond"]`` stores their median and 16th/84th percentiles, and
``prvl`` stores the prevalence of each condensed element across retained
catalogs. Reference-catalog association and completeness products use this
post-sampling representation. They are not likelihood terms.

The current condensation algorithm is spatial. Spectral lines, stellar flares,
and Keplerian radial-velocity elements do not expose the required ``xpos`` and
``ypos`` pair, so those workflows retain the sampled catalogs without a
condensed summary. Set ``boolcondcatl=False`` to skip condensation explicitly,
as the lightweight demo defaults do.

Convergence and interpretation
------------------------------

Inspect effective sample sizes, split-chain or multi-chain convergence metrics,
proposal acceptance, likelihood traces, model-count transitions, and posterior
predictive residuals. A completed process is not necessarily a converged or
scientifically adequate inference. Transdimensional parameters also require
population-level or association-aware summaries because raw element labels are
not persistent.

The final state stores per-sweep proposal IDs, accept/reject flags, and
acceptance probabilities, together with acceptance fractions by proposal type.
Interpret these alongside chain movement and convergence rather than optimizing
acceptance fraction alone. See :doc:`sampling` for the move-specific acceptance
terms, tunable settings, burn-in, and stopping rules.

Reproducibility checklist
-------------------------

Before publishing or comparing runs, preserve and report:

* the PCAT and TDpy repository revisions and Python environment;
* ``cmndargs.txt`` and all explicit random seeds;
* input filenames, versions, selections, units, and checksums;
* ``gdatfinlpost.p`` and its matching ``gdatfinlpost.h5`` companion;
* retained sample count, convergence thresholds, and measured diagnostics;
* proposal acceptance and model-count transition summaries;
* posterior predictive data, model, and residual plots; and
* whether the run used real data, a controlled simulation, or a smoke-test
    configuration.

Do not compare serialized states copied without their HDF5 companions. Avoid
overwriting a completed run with a different configuration under the same run
tag; use a new ``strgcnfg`` or remove both cached data and visuals deliberately.