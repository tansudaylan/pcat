# Split/merge Jacobian example

This example runs PCAT on a seeded simulated spectrum with two Voigt lines and a one-to-three-line metamodel. The 120-sweep run exercises within-model, birth, death, split, and merge proposals. It is a short demonstration, not a converged posterior analysis.

A split maps a parent line's amplitude-coordinate value $F$ to $rF$ and $(1-r)F$, while the proposal auxiliary variables describe the displacement and split fraction. For this amplitude-fraction transform, PCAT adds $\log|J|=\log F$ to the split log acceptance ratio. A reverse merge uses the negative term. More generally,

$$
\log\alpha = \Delta\log\pi + \log(q_{\rm reverse}/q_{\rm forward}) + \log|J|.
$$

The example records the unclipped log acceptance ratio as well as the clipped probability. Its no-Jacobian values subtract only $\log|J|$ from each saved log ratio, holding the proposal, target-density change, and Hastings term fixed. They are counterfactuals for the same proposed states, not a second chain. The figure also plots the acceptance curves using the median split and merge Jacobians from the run.

Run the example and notebook from the PCAT repository root:

```bash
python examples/proposal_state_animation/proposal_state_animation.py
```

![Retained model, every proposed candidate, and Jacobian acceptance comparison](visuals/jacobian_acceptance_effect.png)

PCAT writes `proposal_sequence.gif` for retained states and `proposal_candidates.gif` for every candidate, including rejected proposals. The static comparison is saved as `jacobian_acceptance_effect.png`.
