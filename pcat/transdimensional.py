"""Birth/death posterior sampling for variable catalogs with custom likelihoods."""

import numpy as np
from scipy.special import gammaln, logsumexp


def sample_catalog(
    log_likelihood, base_bounds, component_bounds,
    *, maximum_components=3, count_prior_mean=1.0, number_steps=12000,
    burn_steps=4000, thinning=10, proposal_fraction=0.05, seed=None,
    initial_base=None, draw_component=None, log_component_density=None,
    draw_catalog=None, log_catalog_density=None, catalog_move_probability=0.8,
):
    """Sample a custom-likelihood catalog with finite uniform parameter priors.

    The count prior is Poisson, normalized over 0..maximum_components. Catalog
    rows are exchangeable. Births draw a row from its normalized prior and insert
    it at a random position; deaths choose a row uniformly. Hastings factors
    include state-dependent move probabilities, row-selection probabilities,
    and the birth density. The augmentation Jacobian is one. Within-dimension
    proposals reflect at prior bounds. This routine is single-chain and does
    not perform evidence estimation or establish convergence.

    Optional ``draw_component(base, catalog, generator)`` returns one component
    in physical coordinates. Its paired ``log_component_density(base, catalog,
    component)`` returns the normalized log density in those coordinates. A
    removal evaluates this density using the catalog after removing the object.

    Optional independent ``draw_catalog(generator)`` returns a base vector and
    complete ordered catalog. ``log_catalog_density(base, catalog)`` evaluates
    its normalized joint density, including the probability of the source count.
    These fixed-distribution suggestions replace a fraction of ordinary steps.
    """

    if (draw_component is None) != (log_component_density is None):
        raise ValueError('Component selection requires both draw and density functions')
    if draw_component is not None and not (callable(draw_component) and callable(log_component_density)):
        raise ValueError('Component draw and density functions must be callable')
    if (draw_catalog is None) != (log_catalog_density is None):
        raise ValueError('Catalog selection requires both draw and density functions')
    if draw_catalog is not None and not (callable(draw_catalog) and callable(log_catalog_density)):
        raise ValueError('Catalog draw and density functions must be callable')
    if not np.isfinite(catalog_move_probability) or not 0.0 <= catalog_move_probability <= 1.0:
        raise ValueError('catalog_move_probability must lie between zero and one')
    base_bounds = np.asarray(base_bounds, dtype=float)
    component_bounds = np.asarray(component_bounds, dtype=float)
    for bounds in (base_bounds, component_bounds):
        if (bounds.ndim != 2 or bounds.shape[1] != 2 or bounds.shape[0] == 0
                or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0])):
            raise ValueError('Parameter bounds must be finite increasing [minimum, maximum] pairs')
    for value in (maximum_components, number_steps, burn_steps, thinning):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError('Counts and iteration settings must be integers')
    if maximum_components < 0 or not 0 <= burn_steps < number_steps or thinning < 1:
        raise ValueError('Provide nonnegative count bounds, burn < steps, and positive thinning')
    if not np.isfinite(count_prior_mean) or count_prior_mean <= 0:
        raise ValueError('count_prior_mean must be finite and positive')
    if not np.isfinite(proposal_fraction) or proposal_fraction <= 0:
        raise ValueError('proposal_fraction must be finite and positive')
    generator = np.random.default_rng(seed)
    base = generator.uniform(base_bounds[:, 0], base_bounds[:, 1]) if initial_base is None else np.asarray(initial_base, dtype=float).copy()
    if (base.shape != (base_bounds.shape[0],) or not np.isfinite(base).all()
            or np.any(base < base_bounds[:, 0]) or np.any(base > base_bounds[:, 1])):
        raise ValueError('Initial base parameters must lie inside their bounds')
    catalog = np.empty((0, component_bounds.shape[0]))
    counts = np.arange(maximum_components + 1)
    log_count_prior = counts * np.log(count_prior_mean) - gammaln(counts + 1)
    log_count_prior -= logsumexp(log_count_prior)
    log_component_volume = np.log(np.diff(component_bounds, axis=1)[:, 0]).sum()
    log_birth_density = -log_component_volume

    def posterior(current_base, current_catalog):
        likelihood = float(log_likelihood(current_base, current_catalog))
        if np.isnan(likelihood) or likelihood == np.inf:
            raise ValueError('Likelihood must return a finite value or negative infinity')
        return likelihood + log_count_prior[len(current_catalog)] - len(current_catalog) * log_component_volume

    def move_probabilities(count):
        weights = np.array([0.5, 0.25 if count < maximum_components else 0.0,
                            0.25 if count > 0 else 0.0])
        return weights / weights.sum()

    def reflected(value, lower, upper):
        width = upper - lower
        phase = (value - lower) % (2.0 * width)
        return lower + min(phase, 2.0 * width - phase)

    def component_density(current_base, current_catalog, component):
        if log_component_density is None:
            return log_birth_density
        density = log_component_density(current_base.copy(), current_catalog.copy(), component.copy())
        if np.ma.is_masked(density) or np.iscomplexobj(density):
            raise ValueError('Component log density must be a real scalar below positive infinity')
        density = np.asarray(density)
        if (density.shape != () or density.dtype.kind not in 'iuf'
                or np.isnan(density) or density == np.inf):
            raise ValueError('Component log density must be a real scalar below positive infinity')
        return float(density)

    current_posterior = posterior(base, catalog)
    if not np.isfinite(current_posterior):
        raise ValueError('Initial empty-catalog state must have a finite likelihood')
    base_samples, catalogs, log_samples, count_trace = [], [], [], []
    move_names = ('within', 'birth', 'death') + (('catalog',) if draw_catalog is not None else ())
    proposals = np.zeros(len(move_names), dtype=int)
    acceptances = np.zeros(len(move_names), dtype=int)
    current_catalog_density = None
    for step in range(number_steps):
        count = len(catalog)
        if draw_catalog is not None and generator.random() < catalog_move_probability:
            move = 3
            probabilities = None
        else:
            probabilities = move_probabilities(count)
            move = generator.choice(3, p=probabilities)
        proposals[move] += 1
        proposed_base, proposed_catalog = base.copy(), catalog.copy()
        log_hastings = 0.0
        if move == 3:
            proposed_base, proposed_catalog = draw_catalog(generator)
            if any(np.ma.is_masked(value) or np.iscomplexobj(value) for value in (proposed_base, proposed_catalog)):
                raise ValueError('Suggested catalog must contain finite real parameters inside the bounds')
            proposed_base = np.asarray(proposed_base, dtype=float)
            proposed_catalog = np.asarray(proposed_catalog, dtype=float)
            if (proposed_base.shape != base.shape or proposed_catalog.ndim != 2
                    or proposed_catalog.shape[1] != component_bounds.shape[0]
                    or len(proposed_catalog) > maximum_components
                    or not np.isfinite(proposed_base).all() or not np.isfinite(proposed_catalog).all()
                    or np.any(proposed_base < base_bounds[:, 0]) or np.any(proposed_base > base_bounds[:, 1])
                    or np.any(proposed_catalog < component_bounds[:, 0]) or np.any(proposed_catalog > component_bounds[:, 1])):
                raise ValueError('Suggested catalog must contain finite real parameters inside the bounds')
            densities = [current_catalog_density]
            candidates = [(proposed_base, proposed_catalog)]
            if current_catalog_density is None:
                densities = []
                candidates.insert(0, (base, catalog))
            for candidate_base, candidate_catalog in candidates:
                value = log_catalog_density(candidate_base.copy(), candidate_catalog.copy())
                if np.ma.is_masked(value) or np.iscomplexobj(value):
                    raise ValueError('Catalog selection log density must be finite and real')
                value = np.asarray(value)
                if value.shape != () or value.dtype.kind not in 'iuf' or not np.isfinite(value):
                    raise ValueError('Catalog selection log density must be finite and real')
                densities.append(float(value))
            current_catalog_density = densities[0]
            log_hastings = densities[0] - densities[1]
        elif move == 1:
            if draw_component is None:
                row = generator.uniform(component_bounds[:, 0], component_bounds[:, 1])
            else:
                row = draw_component(base.copy(), catalog.copy(), generator)
                if np.ma.is_masked(row) or np.iscomplexobj(row):
                    raise ValueError('Suggested component must contain finite real parameters inside the bounds')
                row = np.asarray(row, dtype=float)
                if (row.shape != (component_bounds.shape[0],) or not np.isfinite(row).all()
                        or np.any(row < component_bounds[:, 0]) or np.any(row > component_bounds[:, 1])):
                    raise ValueError('Suggested component must contain finite real parameters inside the bounds')
            birth_density = component_density(base, catalog, row)
            if not np.isfinite(birth_density):
                raise ValueError('A generated component must have finite selection log density')
            position = generator.integers(count + 1)
            proposed_catalog = np.insert(catalog, position, row, axis=0)
            log_hastings = np.log(move_probabilities(count + 1)[2]) - np.log(probabilities[1]) - birth_density
        elif move == 2:
            position = generator.integers(count)
            proposed_catalog = np.delete(catalog, position, axis=0)
            birth_density = component_density(proposed_base, proposed_catalog, catalog[position])
            log_hastings = np.log(move_probabilities(count - 1)[1]) - np.log(probabilities[2]) + birth_density
        else:
            coordinate = generator.integers(base.size + catalog.size)
            if coordinate < base.size:
                bounds = base_bounds[coordinate]
                value = base[coordinate] + generator.normal() * proposal_fraction * (bounds[1] - bounds[0])
                proposed_base[coordinate] = reflected(value, *bounds)
            else:
                row, column = divmod(coordinate - base.size, component_bounds.shape[0])
                bounds = component_bounds[column]
                value = catalog[row, column] + generator.normal() * proposal_fraction * (bounds[1] - bounds[0])
                proposed_catalog[row, column] = reflected(value, *bounds)
        proposed_posterior = posterior(proposed_base, proposed_catalog)
        if np.log(generator.random()) < proposed_posterior - current_posterior + log_hastings:
            base, catalog, current_posterior = proposed_base, proposed_catalog, proposed_posterior
            acceptances[move] += 1
            current_catalog_density = densities[1] if move == 3 else None
        count_trace.append(len(catalog))
        if step >= burn_steps and (step - burn_steps) % thinning == 0:
            base_samples.append(base.copy())
            catalogs.append(catalog.copy())
            log_samples.append(current_posterior)
    return {
        'base_samples': np.asarray(base_samples), 'catalog_samples': catalogs,
        'component_counts': np.asarray([len(sample) for sample in catalogs]),
        'count_trace': np.asarray(count_trace), 'log_posterior': np.asarray(log_samples),
        'proposal_counts': dict(zip(move_names, proposals)),
        'acceptance_counts': dict(zip(move_names, acceptances)),
    }