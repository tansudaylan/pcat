Sampling, proposals, and convergence
====================================

PCAT uses one Metropolis-Hastings sampler for fixed-dimensional models and
transdimensional catalogs. A sweep selects a proposal, evaluates its candidate
state, and accepts or rejects that candidate. The realized proposal type,
acceptance probability, and accept/reject result are recorded for every sweep.

Proposal types
--------------

.. list-table:: PCAT proposal families
   :header-rows: 1
   :widths: 18 38 44

   * - Type
     - State change
     - Main settings
   * - Within-model
     - Updates existing parameters without changing the number of elements.
       It is the only proposal used by default for fixed-dimensional models.
     - ``stdvpropbase`` and ``stdvpropelemfire`` set unit-coordinate step
       scales. ``booladaptstdp=True`` adapts these scales during burn-in.
   * - Birth
     - Adds an element drawn from its prior or a configured data-informed
       proposal.
     - ``probtran`` enables transdimensional proposals. ``factpriodoff``
       changes the prior penalty for added dimensions.
   * - Death
     - Removes one existing element.
     - Selected with birth moves under ``probtran`` and the model-count
       bounds.
   * - Split
     - Replaces an element with two elements when the element model supports
       reversible-jump splitting.
     - ``probspmr`` controls the split/merge share. ``radispmr`` sets the
       scale of the split displacement.
   * - Merge
     - Replaces a pair of elements with one element when the model supports
       reversible-jump merging.
     - Uses the reverse-proposal probability and Jacobian correction of the
       split move.
   * - Jump
     - Redraws one existing element from its prior without changing catalog
       size.
     - ``probjump`` controls the probability of attempting this move when a
       catalog is nonempty.
   * - Differential evolution
     - Proposes a symmetric within-model step from the difference between two
       states in a bounded history.
     - ``probdemc`` enables the move, ``numbdemchist`` sets the history length,
       and ``factdemcgamma`` optionally sets its jump scale.
   * - Correlated block
     - Updates a configured set of fixed-dimensional parameters together.
     - ``probpropblock`` sets its mixture probability;
       ``proposal_correlation`` and ``proposal_blocks`` define its shape, and
       ``factpropblock`` scales its step.

Birth, death, split, and merge moves are available only for supported element
models. Their direction can be restricted by the model's minimum and maximum
element counts. If split or merge is unsupported, set ``probspmr=0``. At a
catalog boundary, PCAT selects only moves that can leave the current state.

Acceptance probability
----------------------

For a current state ``x`` and candidate ``x'``, the log Metropolis-Hastings
ratio is

.. math::

   \log r = \log \pi(x'\mid d) - \log \pi(x\mid d)
            + \log q(x\mid x') - \log q(x'\mid x)
            + \log |J|,

where ``pi`` is the posterior density, ``q`` is the proposal density, and
``J`` is the reversible-jump Jacobian. PCAT accepts with probability

.. math::

   \alpha = \min(1, \exp(\log r)).

The within-model random walk and differential-evolution proposals are
symmetric, so their proposal-density ratio is one and their Jacobian is one.
Birth and death moves include model-count and auxiliary-parameter corrections.
Split and merge moves include reverse-move probabilities and the split
Jacobian. For a prior-redraw jump, the element prior cancels its proposal
density, leaving the likelihood change plus any Hastings correction for a
data-informed draw.

The final state stores per-sweep proposal IDs in ``listpostindxproptype``,
accept/reject flags in ``listpostboolpropaccp``, and acceptance probabilities in
``listpostaccpprob``. Final processing also computes ``accpwith``, ``accpbrth``,
``accpdeth``, ``accpsplt``, ``accpmerg``, and ``accpjump`` from attempted moves.
An acceptance fraction is accepted attempts divided by attempts of that type.
It is a tuning diagnostic, not a target to maximize. Very small steps can have
high acceptance but explore the posterior slowly; large steps can have low
acceptance and waste evaluations.

.. figure:: ../examples/proposal_profiling/visuals/proposal_acceptance_and_cost.png
   :alt: Acceptance fraction and computational cost by PCAT proposal type
   :width: 100%
   :align: center

   Proposal acceptance and evaluation cost in the simulated Voigt-line
   profiling example. The values depend on the model, prior, data, and tuning.

Tuning proposals
----------------

``probtran`` defaults to ``0.4`` when a variable population is present and
``0`` otherwise. It is the probability of selecting a transdimensional move.
``probspmr`` defaults to half of ``probtran`` and sets the fraction of eligible
transdimensional selections allocated to split and merge; the remaining share
is allocated to birth and death. Split and merge directions are balanced when
both are available, with boundary restrictions taking precedence.
``probjump`` and ``probdemc`` both default to zero, so those optional moves must
be enabled explicitly.

For an existing model configuration, proposal controls can be added before
calling the sampler:

.. code-block:: python

   configuration.update(
       probtran=0.7,
       probspmr=0.4,
       probjump=0.1,
       probdemc=0.1,
       numbdemchist=1000,
       booladaptstdp=True,
   )
   result = pcat.sampling.sample(**configuration)

Start with the model's native proposal scales and inspect acceptance by move
type, trace behavior, model-count transitions, residuals, and effective sample
sizes. Adjust ``stdvpropbase`` or ``stdvpropelemfire`` when within-model moves
are too large or too small. Set ``probpropblock`` only when correlated blocks
and a matching ``proposal_correlation`` are configured. ``probdemc`` uses at
least ten history states before it can propose; its default history length is
1,000 and its default scale is ``2.38 / sqrt(2 d)`` for a ``d``-parameter
update.

Burn-in and scale adaptation
----------------------------

``numbburn`` is the number of initial sweeps excluded from retained samples. If
it is omitted, PCAT uses ten percent of ``numbswep``. Set ``numbsamp`` or
``factthin`` to control the number of retained samples after burn-in.

``booladaptstdp=True`` adapts within-model proposal scales during burn-in only.
The sampler uses a target acceptance of 0.44 for one-dimensional updates and
0.234 for multivariate updates. Adaptation stops once burn-in ends. Report the
burn-in length and proposal settings with scientific results, and check that
posterior summaries are stable to a longer run or a different seed.

Automatic convergence monitoring
--------------------------------

By default PCAT stops after the configured sweep count. Set
``boolcheckconv=True`` to stop after repeated convergence checks. The defaults
are ``numbsampconvmin=1000`` retained samples before the first check,
``numbsampconvcheck=250`` samples between checks, ``maxmconvrhat=1.01``,
``numbsampconveffc=200``, and ``numbconvpass=2`` consecutive passing checks.
For one worker PCAT estimates R-hat by splitting the chain in half. With several
workers, it calculates convergence across independent worker chains and stops
them together after they all observe the shared stop signal.

Convergence is assessed using log posterior, element count, persistent element
parameters, or modeled data pixels when no persistent element parameter is
available. Inspect the values saved in ``gdatfinlpost`` and the convergence
plots. Automatic stopping does not establish that the likelihood, model, or
priors are scientifically adequate.

Animations and saved diagnostics
--------------------------------

Set ``makeanim=True`` and ``numbswepplot`` so at least two chain frames are
saved. With frame plotting enabled, PCAT writes the usual posterior animations
from these retained chain states. It also writes ``proposal_activity.gif``,
which shows cumulative proposal attempts and acceptances, and
``proposal_sequence.gif``, which labels each saved state frame with its sweep,
proposal type, and acceptance result. The sequence animation uses saved frame
intervals rather than every sweep. A rejected proposal remains represented by
the retained chain state.

Set ``boolmakeanimprop=True`` to additionally write one candidate frame for
every sweep of worker zero. PCAT renders the frame after evaluating the
candidate and acceptance probability but before an accepted candidate replaces
the current state. ``proposal_candidates.gif`` therefore shows accepted and
rejected candidates, including moves outside prior support. Each frame compares
current and candidate predictions and unit-prior coordinates and reports the
proposal type, acceptance probability, decision, log posterior values,
proposal-density ratio, and Jacobian.

Candidate frames are proposal diagnostics, not posterior samples. The option is
off by default because it performs plotting and writes one PNG per sweep. Its
runtime and storage scale linearly with ``numbswep``. Use short runs for proposal
inspection, then disable it for production inference. The
``proposal_state_animation`` example produces both retained-state and
every-candidate animations from the same simulated Voigt-line run.

The proposal-profiling example includes a visual comparison of acceptance and
cost above and writes proposal activity with its run. See :doc:`outputs` for
the output directory and persisted diagnostics, and :doc:`capabilities` for
the maintained model families.