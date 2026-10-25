Outputs and plots
=======================

Directory layout
----------------

A run with project root ``pathbase`` and tag ``strgcnfg`` writes files in this layout.

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
                   |-- convergence/
                   |-- operation/
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
the object from the extension-free path shown below.

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

``init`` contains simulated data, reference models, and initial-state figures.
``post/fram`` contains snapshots across the chain. ``post/finl`` contains final
posterior summaries and their plots. ``post/anim`` contains GIFs assembled from
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
It also stores the per-sweep inverse likelihood temperature in
``listpostfacttmpr``. This value is one for ordinary sampling and after tempered
burn-in reaches the posterior target.
Read acceptance fractions alongside chain movement and convergence measures.
Maximizing acceptance alone can still leave the chain exploring poorly. See
:doc:`sampling` for move-specific acceptance terms, tunable settings, burn-in,
and stopping rules.

When final plots are enabled, ``post/convergence`` holds the convergence
figures from :func:`pcat.plotting.plot_posterior_convergence`, while
``post/operation`` holds the sampler-activity views from
:func:`pcat.plotting.plot_sampler_overview`. These include the proposal ledger,
acceptance-term decomposition, compute budget, catalog trace, and, for
one-dimensional data, the posterior-predictive check. The :doc:`gallery` shows
examples of each.

Animation output
----------------

After building a model-specific ``configuration``, enable both retained-state
and proposal-candidate animations. ``makeanim`` assembles the retained-state
GIF from saved chain frames at ``numbswepplot`` cadence. ``boolmakeanimprop``
records one current-versus-candidate frame per sweep on worker zero before each
proposal is accepted or rejected. Enabling it also enables ``makeanim``.
Candidate recording must be
enabled during sampling because the proposed states cannot be recovered from
the final posterior alone.

.. code-block:: python

   from pcat import sampling

   configuration.update(
       boolmakeplotfram=True,
       makeanim=True,
       boolmakeanimprop=True,
   )
   result = sampling.sample(**configuration)

After a completed, converged run, ``proposal_sequence.gif`` contains the saved
retained states from sweeps at or after ``numbburn``. It visualizes the
post-burn-in chain, whose states remain correlated.
``proposal_candidates.gif`` contains every recorded worker-zero proposal,
including rejected candidates and burn-in transitions. Both files are written
under ``<run>/visuals/post/anim/``. The maintained
``examples/proposal_state_animation/`` run enables both and displays both GIFs.

If the GIFs are absent but the run still has ``gdatinit``, ``gdatfinlpost``,
and its saved frame images, call :func:`pcat.main.proc_anim` with the run tag
and project root to assemble them again.

.. code-block:: python

    from pcat.main import proc_anim

    proc_anim(configuration["strgcnfg"], pathbase=configuration["pathbase"])

Reproducibility checklist
-------------------------

For reproducible comparisons, keep these details with each run.

* PCAT and TDpy revisions and the Python environment.
* Command-line arguments from ``cmndargs.txt`` and all random seeds.
* Input filenames, versions, selections, units, and checksums.
* ``gdatfinlpost.p`` and its matching ``gdatfinlpost.h5`` companion.
* The retained sample count, convergence thresholds, and measured convergence statistics.
* Proposal acceptance and model-count transition summaries.
* Posterior-predictive data, model, and residual plots.
* Whether the run used observations, simulated inputs, or the smoke-test configuration.

Do not compare serialized states copied without their HDF5 companions. Avoid
overwriting a completed run with a different configuration under the same run
tag. Use a new ``strgcnfg`` for a separate run. Remove cached data and visuals
only when you intend to replace that run.