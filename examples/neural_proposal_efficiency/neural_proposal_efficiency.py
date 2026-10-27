#!/usr/bin/env python3
"""Historical RV comparisons, retained for interpreting earlier saved results.

Training systems and validation systems use disjoint random seeds. Both samplers
target the same 0-to-3-planet posterior and differ only in the element density
used for births and their reverse deaths. Prior-redraw jumps are disabled to
isolate the birth proposal. The script reports count and log-likelihood ESS,
posterior agreement, wall time, and move acceptance. It does not assume that a
higher acceptance fraction is a gain.
"""

from tdpy.verbosity import print

import argparse
import json
import os
import platform
import time
from pathlib import Path
from importlib.metadata import version as package_version

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pcat import NeuralElementProposal, sampling
from pcat.demo import configure_example_style
from tdpy.cli import add_smoke_argument
from pcat.main import retr_pathrun
from pcat.radial_velocity import retr_dictpcatrvel, retr_llik_rvel, retr_matrdesi, retr_perdcirc
from tdpy.exoplanet import keplerian_radial_velocity

EXAMPLE_PATH = Path(__file__).resolve().parent
RUNS_PATH = EXAMPLE_PATH / "data"
VISUALS_PATH = EXAMPLE_PATH / "visuals"
MINIMUM_AMPLITUDE = 0.3  # [m s^-1]
MAXIMUM_AMPLITUDE = 300.0  # [m s^-1]
MINIMUM_PERIOD = 1.2  # [day]
MAXIMUM_ECCENTRICITY = 0.8
MEASUREMENT_UNCERTAINTY = 1.5  # [m s^-1]
MAXIMUM_PLANETS = 3
TRUE_PLANETS = np.array([[9.1, 6.0, 0.1, 1.0, 0.5], [61.0, 3.0, 0.0, 0.0, 2.0]])
METHOD_COLORS = {"prior": "#6D7478", "neural": "#087F8C"}
FIGURE_FONT_SIZE = 12.0  # [pt]


def _draw_planets(unit, maximum_period):
    """Map element coordinates ordered as (K, P, phase, e, omega) to physical values."""
    unit = np.asarray(unit, dtype=float).reshape(-1, 5)
    amplitude = MINIMUM_AMPLITUDE * (MAXIMUM_AMPLITUDE / MINIMUM_AMPLITUDE) ** unit[:, 0]
    period = MINIMUM_PERIOD * (maximum_period / MINIMUM_PERIOD) ** unit[:, 1]
    eccentricity = MAXIMUM_ECCENTRICITY * unit[:, 3]
    argument = 2.0 * np.pi * unit[:, 4]
    anomaly = 2.0 * np.pi * unit[:, 2]
    return np.column_stack((period, amplitude, eccentricity, argument, anomaly))


def _render_planets(time_observed, planets, reference_time):
    if len(planets) == 0:
        return np.zeros_like(time_observed, dtype=float)
    return np.sum(
        keplerian_radial_velocity(time_observed[:, None], *np.asarray(planets).T, reference_time),
        axis=1,
    )


def _condition_feature(time_observed, velocity, reference_time, maximum_period):
    """Return a fixed log-periodogram feature vector for the data-conditioned proposal."""
    velocity = np.asarray(velocity, dtype=float)
    period_grid = MINIMUM_PERIOD * (maximum_period / MINIMUM_PERIOD) ** (
        (np.arange(128) + 0.5) / 128
    )
    design = retr_matrdesi(time_observed, np.zeros(time_observed.size, dtype=int), False, reference_time)
    improvement, amplitude, phase = retr_perdcirc(
        time_observed,
        velocity,
        np.full(time_observed.size, MEASUREMENT_UNCERTAINTY),
        design,
        period_grid,
        reference_time,
    )
    log_power = np.log1p(np.maximum(improvement, 0.0))
    unit_amplitude = np.clip(
        np.log(np.maximum(amplitude, MINIMUM_AMPLITUDE) / MINIMUM_AMPLITUDE)
        / np.log(MAXIMUM_AMPLITUDE / MINIMUM_AMPLITUDE),
        0.0,
        1.0,
    )
    return np.stack((log_power, unit_amplitude, np.sin(phase), np.cos(phase)), axis=1).reshape(-1)


def _training_data(time_observed, reference_time, maximum_period, scene_count, seed):
    random = np.random.default_rng(seed)
    conditions = []
    elements = []
    standard_deviation = np.full(time_observed.size, MEASUREMENT_UNCERTAINTY)
    for _ in range(scene_count):
        count = int(random.integers(1, MAXIMUM_PLANETS + 1))
        unit = random.uniform(size=(count, 5))
        velocity = _render_planets(time_observed, _draw_planets(unit, maximum_period), reference_time)
        velocity += random.normal(0.0, standard_deviation)
        feature = _condition_feature(time_observed, velocity, reference_time, maximum_period)
        conditions.extend([feature] * count)
        elements.extend(unit)
    return np.asarray(conditions), np.asarray(elements)


def _catalog_state_summary(unit_elements):
    """Return count, mean, and spread features for a permutation-invariant catalog summary."""
    unit_elements = np.asarray(unit_elements, dtype=float).reshape(-1, 5)
    if unit_elements.shape[0] > MAXIMUM_PLANETS:
        raise ValueError("Catalog context exceeds the benchmark planet-count limit.")
    if not unit_elements.shape[0]:
        return np.zeros(1 + 2 * 5, dtype=float)
    return np.concatenate((
        np.array([unit_elements.shape[0] / MAXIMUM_PLANETS]),
        np.mean(unit_elements, axis=0),
        np.std(unit_elements, axis=0),
    ))


def _state_condition_features(gdat, context):
    """Build data-only or leave-one-out residual features for the live catalog."""
    feature_mode = getattr(gdat, "neural_state_feature_mode", "catalog")
    if feature_mode == "catalog":
        observation_features = np.asarray(gdat.neural_proposal_features, dtype=float)
    elif feature_mode == "residual":
        maximum_period = float(gdat.neural_maximum_period)
        planets = _draw_planets(context.unit_elements, maximum_period)
        residual = np.asarray(gdat.rvelobsv, dtype=float) - _render_planets(
            np.asarray(gdat.timervel, dtype=float), planets, float(gdat.timervelrefr)
        )
        observation_features = _condition_feature(
            np.asarray(gdat.timervel, dtype=float),
            residual,
            float(gdat.timervelrefr),
            maximum_period,
        )
    else:
        raise ValueError("neural_state_feature_mode must be 'catalog' or 'residual'.")
    return np.concatenate((observation_features, _catalog_state_summary(context.unit_elements)))


def _training_data_state(
    time_observed, reference_time, maximum_period, scene_count, seed, feature_mode="catalog"
):
    """Build catalog- or residual-conditioned training pairs with each target omitted."""
    if feature_mode not in {"catalog", "residual"}:
        raise ValueError("feature_mode must be 'catalog' or 'residual'.")
    random = np.random.default_rng(seed)
    conditions = []
    elements = []
    standard_deviation = np.full(time_observed.size, MEASUREMENT_UNCERTAINTY)
    for _ in range(scene_count):
        planet_count = int(random.integers(1, MAXIMUM_PLANETS + 1))
        scene_elements = random.uniform(size=(planet_count, 5))
        velocity = _render_planets(
            time_observed, _draw_planets(scene_elements, maximum_period), reference_time
        )
        velocity += random.normal(0.0, standard_deviation)
        observation_features = None
        if feature_mode == "catalog":
            observation_features = _condition_feature(
                time_observed, velocity, reference_time, maximum_period
            )
        for target_index, target_element in enumerate(scene_elements):
            other_elements = np.delete(scene_elements, target_index, axis=0)
            if feature_mode == "catalog":
                include_context = random.random(other_elements.shape[0]) < 0.5
                context_elements = other_elements[include_context]
                condition_features = observation_features
            else:
                context_elements = other_elements
                context_planets = _draw_planets(context_elements, maximum_period)
                residual = velocity - _render_planets(
                    time_observed, context_planets, reference_time
                )
                condition_features = _condition_feature(
                    time_observed, residual, reference_time, maximum_period
                )
            conditions.append(np.concatenate((
                condition_features, _catalog_state_summary(context_elements)
            )))
            elements.append(target_element)
    return np.asarray(conditions), np.asarray(elements)


def _validation_data(time_observed, reference_time, seed):
    random = np.random.default_rng(seed)
    velocity = _render_planets(time_observed, TRUE_PLANETS, reference_time)
    velocity += random.normal(0.0, MEASUREMENT_UNCERTAINTY, time_observed.size)
    return velocity, np.full(time_observed.size, MEASUREMENT_UNCERTAINTY)


def _count_trace(values, statistic):
    """Normalize one count trace without discarding unavailable samples."""
    if np.ma.is_masked(values):
        raise ValueError(f"{statistic} requires samples without masked values")
    values = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError(f"{statistic} requires finite samples")
    return values


def _count_ess(values):
    """Estimate scalar-chain ESS with Geyer's initial-positive sequence."""
    values = _count_trace(values, "Count ESS")
    count = values.size
    if count == 0 or np.all(values == values[0]):
        return 0.0
    if count < 4:
        return float(count)
    centered = values - np.mean(values)
    fft_size = 1 << (2 * count - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size)
    autocovariance = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)[:count]
    autocovariance /= np.arange(count, 0, -1)
    if autocovariance[0] <= 0.0:
        return float(count)
    correlation = autocovariance / autocovariance[0]
    paired_sum = 0.0
    for lag in range(1, count - 1, 2):
        pair = correlation[lag] + correlation[lag + 1]
        if pair <= 0.0:
            break
        paired_sum += pair
    return float(min(count, count / max(1.0, 1.0 + 2.0 * paired_sum)))


def _count_rhat(chains):
    """Return classical between-chain R-hat for equal-length scalar count chains."""
    chains = [_count_trace(chain, "Count R-hat") for chain in chains]
    if len(chains) < 2:
        return float("nan")
    sample_count = min(chain.size for chain in chains)
    if sample_count < 2:
        return float("nan")
    values = np.stack([chain[-sample_count:] for chain in chains])
    within = float(np.mean(np.var(values, axis=1, ddof=1)))
    between = float(sample_count * np.var(np.mean(values, axis=1), ddof=1))
    if within == 0.0:
        return float("nan") if between == 0.0 else float("inf")
    variance = (sample_count - 1.0) / sample_count * within + between / sample_count
    return float(np.sqrt(max(variance / within, 0.0)))


def _count_acceptance(run_root):
    path = run_root / "data" / "outp" / run_root.name / "gdatmodi0000post.h5"
    print(f"Reading from {path}...")
    with h5py.File(path, "r") as output:
        proposal_type = np.asarray(output["listpostindxproptype"][()]).reshape(-1)
        accepted = np.asarray(output["listpostboolpropaccp"][()]).reshape(-1)
    if proposal_type.shape != accepted.shape:
        raise ValueError("Proposal types and acceptance flags must have aligned trace lengths")
    if (proposal_type.dtype.kind not in 'iuf' or not np.isfinite(proposal_type).all()
            or np.any(proposal_type < 0)
            or (proposal_type.dtype.kind == 'f' and np.any(proposal_type != np.floor(proposal_type)))):
        raise ValueError("Proposal types must be finite nonnegative integer IDs")
    if accepted.dtype.kind not in 'biuf' or not np.isin(accepted, [0, 1]).all():
        raise ValueError("Acceptance flags must contain only binary zero/one values")
    accepted = accepted.astype(bool)
    result = {}
    for proposal_id, name in ((1, "birth"), (2, "death"), (5, "jump")):
        selected = proposal_type == proposal_id
        result[f"{name}_acceptance"] = float(np.mean(accepted[selected])) if selected.any() else 0.0
    return result


def _read_posterior_traces(path):
    """Read aligned posterior traces without truncating invalid count values."""
    print(f"Reading from {path}...")
    with h5py.File(path, "r") as output:
        counts = np.asarray(output["listpostnumbelem"][()]).reshape(-1)
        likelihood = np.asarray(output["listpostlliktotl"][()]).reshape(-1)
    if not counts.size or counts.shape != likelihood.shape:
        raise ValueError("Posterior count and likelihood traces must be nonempty and aligned")
    if (counts.dtype.kind not in 'iuf' or not np.isfinite(counts).all()
            or np.any(counts < 0) or np.any(counts > MAXIMUM_PLANETS)
            or (counts.dtype.kind == 'f' and np.any(counts != np.floor(counts)))):
        raise ValueError("Posterior counts must be finite integers within the supported planet range")
    if likelihood.dtype.kind not in 'iuf':
        raise ValueError("Posterior likelihood samples must be real numeric values")
    likelihood = _count_trace(likelihood, "Posterior likelihood")
    return counts.astype(int), likelihood


def _run_sampler(
    method, run_id, time_observed, velocity, stdv, features, proposal, sweeps, seed,
    proposal_conditioning="data", state_feature_mode="catalog",
):
    global ACTIVE_LIKELIHOOD_CALLS
    setup_start = time.perf_counter()
    path_run = Path(retr_pathrun(str(RUNS_PATH), run_id))
    path_output = path_run / "data" / "outp" / run_id
    if path_output.exists():
        raise FileExistsError(
            f"Run output already exists at {path_output}; choose a new --run-id."
        )

    ACTIVE_LIKELIHOOD_CALLS[0] = 0

    configuration = retr_dictpcatrvel(
        time_observed,
        velocity,
        stdv,
        np.zeros(time_observed.size, dtype=int),
        str(RUNS_PATH),
        run_id,
        maxmnumbplan=MAXIMUM_PLANETS,
        maxmperi=2.0 * (time_observed[-1] - time_observed[0]),
        minmrvsa=MINIMUM_AMPLITUDE,
        maxmrvsa=MAXIMUM_AMPLITUDE,
        maxmecce=MAXIMUM_ECCENTRICITY,
        factpriodoff=0.0,
        boolmakepropelem=method != "prior" and not (
            method == "neural" and proposal_conditioning == "state"
        ),
        probtran=0.7,
        probspmr=0.0,
        probjump=0.0,
        retr_llik=counted_rv_log_likelihood,
    )
    if method == "prior":
        configuration.pop("retr_drawpropelem", None)
        configuration.pop("retr_lpdfpropelem", None)
    configuration.update(
        typeseed=seed,
        numbproc=1,
        numbswep=sweeps,
        numbburn=sweeps // 4,
        numbsamp=sweeps - sweeps // 4,
        boolcheckconv=False,
        boolmakeplot=False,
        boolmakeplotinit=False,
        makeanim=False,
        typeverb=0,
    )
    if method == "neural":
        configuration["neural_proposal_features"] = features
        if proposal_conditioning == "state":
            configuration["neural_state_feature_mode"] = state_feature_mode
            configuration["neural_maximum_period"] = 2.0 * (time_observed[-1] - time_observed[0])
            configuration.update(proposal.state_callbacks(_state_condition_features))
        else:
            configuration.update(proposal.callbacks())

    setup_seconds = time.perf_counter() - setup_start
    sampling_start = time.perf_counter()
    sampling.sample(**configuration)
    sampling_seconds = time.perf_counter() - sampling_start
    elapsed = setup_seconds + sampling_seconds
    trace_path = path_output / "gdatmodi0000post.h5"
    counts, likelihood = _read_posterior_traces(trace_path)
    ess = _count_ess(counts)
    likelihood_ess = _count_ess(likelihood)
    return {
        "method": method,
        "proposal_conditioning": proposal_conditioning if method == "neural" else None,
        "state_feature_mode": state_feature_mode if method == "neural" and proposal_conditioning == "state" else None,
        "seed": int(seed),
        "count_samples": counts.tolist(),
        "likelihood_samples": likelihood.tolist(),
        "count_probability": (
            np.bincount(counts, minlength=MAXIMUM_PLANETS + 1).astype(float) / counts.size
        ).tolist(),
        "count_ess": ess,
        "likelihood_ess": likelihood_ess,
        "likelihood_calls": int(ACTIVE_LIKELIHOOD_CALLS[0]),
        "elapsed_seconds": float(elapsed),
        "setup_seconds": float(setup_seconds),
        "sampling_seconds": float(sampling_seconds),
        "ess_per_call": float(ess / ACTIVE_LIKELIHOOD_CALLS[0]),
        "ess_per_second": float(ess / elapsed),
        "likelihood_ess_per_call": float(likelihood_ess / ACTIVE_LIKELIHOOD_CALLS[0]),
        "likelihood_ess_per_second": float(likelihood_ess / elapsed),
        **_count_acceptance(path_run),
    }


ACTIVE_LIKELIHOOD_CALLS = [0]


def counted_rv_log_likelihood(gdat, model_name, model_data):
    """PCAT callback that counts exact likelihood evaluations in this worker."""
    ACTIVE_LIKELIHOOD_CALLS[0] += 1
    return retr_llik_rvel(gdat, model_name, model_data)


def _check_chain_scalar(value, name, upper):
    """Reject unavailable or out-of-range scalar metadata without repairing it."""
    if np.ma.is_masked(value):
        raise ValueError(f"Chain {name} must not be masked")
    value = np.asarray(value)
    if (value.ndim != 0 or value.dtype.kind not in 'iuf'
            or not np.isfinite(value) or value < 0 or value > upper):
        raise ValueError(f"Chain {name} must be a finite real scalar between zero and {upper}")


def _aggregate_chain_runs(runs):
    if not runs:
        raise ValueError("Chain aggregation requires at least one run")
    probabilities = []
    for result in runs:
        calls = result["likelihood_calls"]
        if isinstance(calls, (bool, np.bool_)) or not np.isfinite(calls) or calls <= 0 or calls != int(calls):
            raise ValueError("Chain likelihood_calls must be a positive finite integer")
        elapsed = result["elapsed_seconds"]  # [s]
        overhead = result.get("training_overhead_seconds", 0.0)  # [s]
        if not np.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Chain elapsed_seconds must be finite and positive")
        if not np.isfinite(overhead) or overhead < 0:
            raise ValueError("Chain training_overhead_seconds must be finite and nonnegative")
        for ess_name, samples_name in (("count_ess", "count_samples"), ("likelihood_ess", "likelihood_samples")):
            samples = _count_trace(result[samples_name], f"Chain {samples_name}")
            _check_chain_scalar(result[ess_name], ess_name, samples.size)
        for acceptance_name in ("birth_acceptance", "death_acceptance"):
            _check_chain_scalar(result[acceptance_name], acceptance_name, 1.0)
        probability = result["count_probability"]
        if np.ma.is_masked(probability):
            raise ValueError("Chain count_probability must not contain masked entries")
        probability = np.asarray(probability)
        if probability.shape != (MAXIMUM_PLANETS + 1,) or probability.dtype.kind not in 'biuf':
            raise ValueError("Chain count_probability must be a real vector over the supported counts")
        if not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
            raise ValueError("Chain count_probability entries must be finite and between zero and one")
        tolerance = np.finfo(probability.dtype).eps * probability.size if probability.dtype.kind == 'f' else 0.0
        if not np.isclose(np.sum(probability), 1.0, rtol=0.0, atol=tolerance):
            raise ValueError("Chain count_probability must sum to one")
        probabilities.append(probability)
    count_chains = [result["count_samples"] for result in runs]
    likelihood_chains = [result["likelihood_samples"] for result in runs]
    count_ess = float(sum(result["count_ess"] for result in runs))
    likelihood_ess = float(sum(result["likelihood_ess"] for result in runs))
    likelihood_calls = sum(int(result["likelihood_calls"]) for result in runs)
    elapsed_seconds = float(sum(result["elapsed_seconds"] for result in runs))
    total_seconds = elapsed_seconds + float(sum(result.get("training_overhead_seconds", 0.0) for result in runs))
    return {
        "count_probability": np.mean(probabilities, axis=0).tolist(),
        "count_ess": count_ess,
        "count_rhat": _count_rhat(count_chains),
        "likelihood_ess": likelihood_ess,
        "likelihood_rhat": _count_rhat(likelihood_chains),
        "likelihood_calls": likelihood_calls,
        "elapsed_seconds": elapsed_seconds,
        "ess_per_call": count_ess / likelihood_calls,
        "ess_per_second": count_ess / elapsed_seconds,
        "ess_per_end_to_end_second": count_ess / total_seconds,
        "likelihood_ess_per_call": likelihood_ess / likelihood_calls,
        "likelihood_ess_per_second": likelihood_ess / elapsed_seconds,
        "likelihood_ess_per_end_to_end_second": likelihood_ess / total_seconds,
        "count_ess_per_end_to_end_second": count_ess / total_seconds,
        "birth_acceptance": float(np.mean([result["birth_acceptance"] for result in runs])),
        "death_acceptance": float(np.mean([result["death_acceptance"] for result in runs])),
    }


def _efficiency_ratio(value, baseline):
    """Leave comparisons undefined when the baseline has zero effective samples."""
    return float("nan") if baseline == 0.0 else value / baseline


def _paired_summary(results):
    by_scene = {}
    for result in results:
        by_scene.setdefault(result["validation_seed"], {}).setdefault(result["method"], []).append(result)
    paired = []
    for validation_seed, runs_by_method in sorted(by_scene.items()):
        methods = {name: _aggregate_chain_runs(runs) for name, runs in runs_by_method.items()}
        prior = methods["prior"]
        data_informed = methods["data-informed"]
        neural = methods["neural"]
        paired.append({
            "validation_seed": validation_seed,
            "neural_to_prior_ess_per_call_ratio": _efficiency_ratio(neural["ess_per_call"], prior["ess_per_call"]),
            "neural_to_data_informed_ess_per_call_ratio": _efficiency_ratio(neural["ess_per_call"], data_informed["ess_per_call"]),
            "neural_to_prior_ess_per_second_ratio": _efficiency_ratio(neural["ess_per_second"], prior["ess_per_second"]),
            "neural_to_data_informed_ess_per_second_ratio": _efficiency_ratio(neural["ess_per_second"], data_informed["ess_per_second"]),
            "neural_to_prior_ess_per_end_to_end_second_ratio": (
                _efficiency_ratio(neural["ess_per_end_to_end_second"], prior["ess_per_end_to_end_second"])
            ),
            "neural_to_data_informed_ess_per_end_to_end_second_ratio": (
                _efficiency_ratio(neural["ess_per_end_to_end_second"], data_informed["ess_per_end_to_end_second"])
            ),
            "neural_to_prior_likelihood_ess_per_call_ratio": (
                _efficiency_ratio(neural["likelihood_ess_per_call"], prior["likelihood_ess_per_call"])
            ),
            "neural_to_data_informed_likelihood_ess_per_call_ratio": (
                _efficiency_ratio(neural["likelihood_ess_per_call"], data_informed["likelihood_ess_per_call"])
            ),
            "neural_to_prior_likelihood_ess_per_second_ratio": (
                _efficiency_ratio(neural["likelihood_ess_per_second"], prior["likelihood_ess_per_second"])
            ),
            "neural_to_data_informed_likelihood_ess_per_second_ratio": (
                _efficiency_ratio(
                    neural["likelihood_ess_per_second"], data_informed["likelihood_ess_per_second"]
                )
            ),
            "neural_to_prior_likelihood_ess_per_end_to_end_second_ratio": (
                _efficiency_ratio(
                    neural["likelihood_ess_per_end_to_end_second"],
                    prior["likelihood_ess_per_end_to_end_second"],
                )
            ),
            "neural_to_data_informed_likelihood_ess_per_end_to_end_second_ratio": (
                _efficiency_ratio(
                    neural["likelihood_ess_per_end_to_end_second"],
                    data_informed["likelihood_ess_per_end_to_end_second"],
                )
            ),
            "neural_to_prior_ess_per_end_to_end_second_ratio": (
                _efficiency_ratio(
                    neural["count_ess_per_end_to_end_second"],
                    prior["count_ess_per_end_to_end_second"],
                )
            ),
            "neural_to_data_informed_ess_per_end_to_end_second_ratio": (
                _efficiency_ratio(
                    neural["count_ess_per_end_to_end_second"],
                    data_informed["count_ess_per_end_to_end_second"],
                )
            ),
            "neural_to_prior_likelihood_ess_per_second_ratio": (
                _efficiency_ratio(neural["likelihood_ess_per_second"], prior["likelihood_ess_per_second"])
            ),
            "neural_to_data_informed_likelihood_ess_per_second_ratio": (
                _efficiency_ratio(neural["likelihood_ess_per_second"], data_informed["likelihood_ess_per_second"])
            ),
            "neural_prior_count_posterior_total_variation": 0.5 * float(np.sum(np.abs(
                np.asarray(neural["count_probability"]) - np.asarray(prior["count_probability"])
            ))),
            "neural_data_informed_count_posterior_total_variation": 0.5 * float(np.sum(np.abs(
                np.asarray(neural["count_probability"]) - np.asarray(data_informed["count_probability"])
            ))),
            "prior_probability_two_planets": prior["count_probability"][2],
            "data_informed_probability_two_planets": data_informed["count_probability"][2],
            "neural_probability_two_planets": neural["count_probability"][2],
            "prior_count_rhat": prior["count_rhat"],
            "data_informed_count_rhat": data_informed["count_rhat"],
            "neural_count_rhat": neural["count_rhat"],
            "prior_likelihood_rhat": prior["likelihood_rhat"],
            "data_informed_likelihood_rhat": data_informed["likelihood_rhat"],
            "neural_likelihood_rhat": neural["likelihood_rhat"],
        })
    return paired


def _bootstrap_interval(values, seed, draws=5000):
    if isinstance(draws, (bool, np.bool_)) or not isinstance(draws, (int, np.integer)) or draws <= 0:
        raise ValueError("Bootstrap draws must be a positive integer")
    if np.ma.is_masked(values):
        return [float("nan"), float("nan")]
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("Bootstrap comparisons must be one-dimensional")
    if not values.size or not np.isfinite(values).all():
        return [float("nan"), float("nan")]
    random = np.random.default_rng(seed)
    samples = random.choice(values, size=(draws, values.size), replace=True)
    medians = np.median(samples, axis=1)
    return [float(value) for value in np.quantile(medians, [0.025, 0.975])]


def _efficiency_gain_supported(
    ratios,
    interval,
    total_variations,
    proposal_rhats,
    reference_rhats,
    likelihood_ratios,
    likelihood_interval,
    proposal_likelihood_rhats,
    reference_likelihood_rhats,
    end_to_end_ratios,
    end_to_end_interval,
):
    """Require ESS-rate gains, count/log-likelihood mixing, and close count posteriors."""
    evidence = (
        ratios, interval, total_variations, proposal_rhats, reference_rhats,
        likelihood_ratios, likelihood_interval, proposal_likelihood_rhats,
        reference_likelihood_rhats, end_to_end_ratios, end_to_end_interval,
    )
    if any(np.ma.is_masked(values) for values in evidence):
        return False
    ratios = np.asarray(ratios, dtype=float)
    interval = np.asarray(interval, dtype=float)
    total_variations = np.asarray(total_variations, dtype=float)
    proposal_rhats = np.asarray(proposal_rhats, dtype=float)
    reference_rhats = np.asarray(reference_rhats, dtype=float)
    likelihood_ratios = np.asarray(likelihood_ratios, dtype=float)
    likelihood_interval = np.asarray(likelihood_interval, dtype=float)
    proposal_likelihood_rhats = np.asarray(proposal_likelihood_rhats, dtype=float)
    reference_likelihood_rhats = np.asarray(reference_likelihood_rhats, dtype=float)
    end_to_end_ratios = np.asarray(end_to_end_ratios, dtype=float)
    end_to_end_interval = np.asarray(end_to_end_interval, dtype=float)
    scene_values = (
        total_variations, proposal_rhats, reference_rhats, likelihood_ratios,
        proposal_likelihood_rhats, reference_likelihood_rhats, end_to_end_ratios,
    )
    if ratios.ndim != 1 or any(values.shape != ratios.shape for values in scene_values):
        return False
    if any(bounds.shape != (2,) or bounds[0] > bounds[1]
           for bounds in (interval, likelihood_interval, end_to_end_interval)):
        return False
    return bool(
        ratios.size >= 2
        and np.isfinite(ratios).all()
        and np.isfinite(interval).all()
        and interval[0] > 1.0
        and likelihood_ratios.size >= 2
        and np.isfinite(likelihood_ratios).all()
        and np.isfinite(likelihood_interval).all()
        and likelihood_interval[0] > 1.0
        and end_to_end_ratios.size >= 2
        and np.isfinite(end_to_end_ratios).all()
        and np.isfinite(end_to_end_interval).all()
        and end_to_end_interval[0] > 1.0
        and np.isfinite(total_variations).all()
        and np.median(total_variations) <= 0.1
        and np.isfinite(proposal_rhats).all()
        and np.all(proposal_rhats <= 1.1)
        and np.isfinite(reference_rhats).all()
        and np.all(reference_rhats <= 1.1)
        and np.isfinite(proposal_likelihood_rhats).all()
        and np.all(proposal_likelihood_rhats <= 1.1)
        and np.isfinite(reference_likelihood_rhats).all()
        and np.all(reference_likelihood_rhats <= 1.1)
    )


def _json_safe(value):
    """Convert benchmark results to strict JSON, mapping undefined numbers to null."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _plot_results(results, run_id, *, typefileplot="png"):
    paired = _paired_summary(results)
    neural_result = next(
        (result.get("proposal_conditioning") for result in results if result["method"] == "neural"),
        None,
    )
    proposal_conditioning = neural_result or "data"
    state_feature_mode = (
        next(
            (result.get("state_feature_mode") or "catalog"
             for result in results if result["method"] == "neural"),
            "catalog",
        )
        if proposal_conditioning == "state" else None
    )
    neural_label = (
        f"State-conditioned, {state_feature_mode} summary"
        if proposal_conditioning == "state" else "Data-only"
    )
    if typefileplot not in {"png", "pdf"}:
        raise ValueError("typefileplot must be 'png' or 'pdf'.")
    figure, axes = plt.subplots(3, 2, figsize=(6.3, 7.8))
    figure.suptitle(f"{neural_label} neural proposal comparison", fontsize=FIGURE_FONT_SIZE)
    axes = axes.ravel()
    for axis, key, label in (
        (axes[0], "neural_to_prior_ess_per_call_ratio", "Count ESS/call\nvs prior"),
        (axes[1], "neural_to_data_informed_ess_per_call_ratio", "Count ESS/call\nvs data-informed"),
        (axes[2], "neural_to_prior_likelihood_ess_per_call_ratio", "Log-likelihood ESS/call\nvs prior"),
        (axes[3], "neural_to_data_informed_likelihood_ess_per_call_ratio", "Log-likelihood ESS/call\nvs data-informed"),
        (axes[4], "neural_prior_count_posterior_total_variation", "Count posterior TV\nvs prior"),
        (axes[5], "neural_data_informed_count_posterior_total_variation", "Count posterior TV\nvs data-informed"),
    ):
        values = [item[key] for item in paired]
        axis.axhline(1.0 if "ratio" in key else 0.0, color="0.45", lw=0.8, ls="--")
        axis.scatter(np.arange(len(values)), values, color="#087F8C", s=28, zorder=3)
        axis.set_xticks(np.arange(len(values)), [str(item["validation_seed"]) for item in paired])
        axis.set_xlabel("Scene seed", fontsize=FIGURE_FONT_SIZE)
        axis.set_ylabel("ESS/call ratio" if "ratio" in key else "Count total variation",
                        fontsize=FIGURE_FONT_SIZE)
        axis.set_title(label, fontsize=FIGURE_FONT_SIZE)
        axis.tick_params(labelsize=FIGURE_FONT_SIZE)
        axis.grid(False)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    path = VISUALS_PATH / f"{run_id}_efficiency.{typefileplot}"
    VISUALS_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {path}...")
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def run_benchmark(arguments):
    methods = ("prior", "data-informed", "neural")
    typefileplot = getattr(arguments, "typefileplot", "png")
    proposal_conditioning = getattr(arguments, "proposal_conditioning", "data")
    state_feature_mode = getattr(arguments, "state_feature_mode", "catalog")
    if proposal_conditioning not in {"data", "state"}:
        raise ValueError("proposal_conditioning must be 'data' or 'state'.")
    if state_feature_mode not in {"catalog", "residual"}:
        raise ValueError("state_feature_mode must be 'catalog' or 'residual'.")
    if proposal_conditioning == "data" and state_feature_mode != "catalog":
        raise ValueError("state_feature_mode applies only when proposal_conditioning='state'.")
    artifacts = [
        RUNS_PATH / f"{arguments.run_id}_proposal.pt",
        RUNS_PATH / f"{arguments.run_id}_results.json",
        VISUALS_PATH / f"{arguments.run_id}_efficiency.{typefileplot}",
    ]
    for validation_seed in range(arguments.seed + 10, arguments.seed + 10 + arguments.validation_scenes):
        for chain_index in range(arguments.chains_per_scene):
            for method in methods:
                run_id = f"{arguments.run_id}_{method}_{validation_seed}_chain{chain_index}"
                artifacts.append(Path(retr_pathrun(str(RUNS_PATH), run_id)) / "data" / "outp" / run_id)
    for artifact in artifacts:
        if artifact.exists():
            raise FileExistsError(f"Benchmark output already exists at {artifact}. Choose a new --run-id.")

    time_random = np.random.default_rng(arguments.seed)
    time_observed = np.sort(time_random.uniform(0.0, 800.0, 60)) + 8000.0  # [day]
    reference_time = 0.5 * (time_observed[0] + time_observed[-1])  # [day]
    maximum_period = 2.0 * (time_observed[-1] - time_observed[0])  # [day]
    training_start = time.perf_counter()
    training_function = _training_data_state if proposal_conditioning == "state" else _training_data
    if proposal_conditioning == "state":
        training_conditions, training_elements = training_function(
            time_observed, reference_time, maximum_period,
            arguments.training_scenes, arguments.seed + 1000,
            feature_mode=state_feature_mode,
        )
    else:
        training_conditions, training_elements = training_function(
            time_observed, reference_time, maximum_period,
            arguments.training_scenes, arguments.seed + 1000,
        )
    proposal = NeuralElementProposal.fit(
        training_conditions,
        training_elements,
        number_components=arguments.mixture_components,
        hidden_sizes=(64, 64),
        epochs=arguments.epochs,
        batch_size=128,
        uniform_fraction=arguments.uniform_fraction,
        seed=arguments.seed + 2000,
    )
    training_seconds = time.perf_counter() - training_start
    import torch

    checkpoint = RUNS_PATH / f"{arguments.run_id}_proposal.pt"
    proposal.save(checkpoint)

    results = []
    for scene_index, validation_seed in enumerate(
        range(arguments.seed + 10, arguments.seed + 10 + arguments.validation_scenes)
    ):
        velocity, stdv = _validation_data(time_observed, reference_time, validation_seed)
        features = _condition_feature(time_observed, velocity, reference_time, maximum_period)
        for chain_index in range(arguments.chains_per_scene):
            chain_seed = arguments.seed + 20000 + scene_index * arguments.chains_per_scene + chain_index
            for method in methods:
                run_id = f"{arguments.run_id}_{method}_{validation_seed}_chain{chain_index}"
                print(
                    f"Running {method} proposal on validation scene {validation_seed}, "
                    f"chain {chain_index}..."
                )
                result = _run_sampler(
                    method,
                    run_id,
                    time_observed,
                    velocity,
                    stdv,
                    features,
                    proposal,
                    arguments.sweeps,
                    chain_seed,
                    proposal_conditioning=proposal_conditioning,
                    state_feature_mode=state_feature_mode,
                )
                result["validation_seed"] = validation_seed
                result["chain_index"] = chain_index
                result["training_overhead_seconds"] = (
                    training_seconds / (arguments.validation_scenes * arguments.chains_per_scene)
                    if method == "neural" else 0.0
                )
                results.append(result)
                print(
                    f"{method}: {result['likelihood_calls']} likelihood calls, "
                    f"ESS(N) {result['count_ess']:.1f}, ESS/call {result['ess_per_call']:.4g}, "
                    f"ESS/s {result['ess_per_second']:.3g}"
                )

    paired = _paired_summary(results)
    ratios_per_call_prior = [item["neural_to_prior_ess_per_call_ratio"] for item in paired]
    ratios_per_call_data = [item["neural_to_data_informed_ess_per_call_ratio"] for item in paired]
    likelihood_ratios_per_call_prior = [item["neural_to_prior_likelihood_ess_per_call_ratio"] for item in paired]
    likelihood_ratios_per_call_data = [
        item["neural_to_data_informed_likelihood_ess_per_call_ratio"] for item in paired
    ]
    ratios_per_second_prior = [item["neural_to_prior_ess_per_second_ratio"] for item in paired]
    ratios_per_second_data = [item["neural_to_data_informed_ess_per_second_ratio"] for item in paired]
    ratios_end_to_end_prior = [item["neural_to_prior_ess_per_end_to_end_second_ratio"] for item in paired]
    ratios_end_to_end_data = [item["neural_to_data_informed_ess_per_end_to_end_second_ratio"] for item in paired]
    summary = {
        "run_id": arguments.run_id,
        "typefileplot": typefileplot,
        "training_seed": arguments.seed + 1000,
        "validation_seeds": [item["validation_seed"] for item in paired],
        "chains_per_scene": arguments.chains_per_scene,
        "training_scenes": arguments.training_scenes,
        "training_elements": int(training_elements.shape[0]),
        "proposal_conditioning": proposal_conditioning,
        "state_feature_mode": state_feature_mode if proposal_conditioning == "state" else None,
        "training_condition_dimension": int(training_conditions.shape[1]),
        "training_seconds": float(training_seconds),
        "final_training_negative_log_density": float(proposal.loss_history[-1]),
        "sweeps_per_chain": arguments.sweeps,
        "mixture_components": arguments.mixture_components,
        "uniform_fraction": arguments.uniform_fraction,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": package_version("torch"),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
        },
        "paired_results": paired,
        "neural_to_prior_median_ess_per_call_ratio": float(np.median(ratios_per_call_prior)),
        "neural_to_prior_ess_per_call_95pct_bootstrap_interval": _bootstrap_interval(
            ratios_per_call_prior, arguments.seed + 3000
        ),
        "neural_to_data_informed_median_ess_per_call_ratio": float(np.median(ratios_per_call_data)),
        "neural_to_data_informed_ess_per_call_95pct_bootstrap_interval": _bootstrap_interval(
            ratios_per_call_data, arguments.seed + 4000
        ),
        "neural_to_prior_median_loglik_ess_per_call_ratio": float(np.median(likelihood_ratios_per_call_prior)),
        "neural_to_prior_loglik_ess_per_call_95pct_bootstrap_interval": _bootstrap_interval(
            likelihood_ratios_per_call_prior, arguments.seed + 9000
        ),
        "neural_to_data_informed_median_loglik_ess_per_call_ratio": float(
            np.median(likelihood_ratios_per_call_data)
        ),
        "neural_to_data_informed_loglik_ess_per_call_95pct_bootstrap_interval": _bootstrap_interval(
            likelihood_ratios_per_call_data, arguments.seed + 10000
        ),
        "neural_to_prior_median_ess_per_second_ratio": float(np.median(ratios_per_second_prior)),
        "neural_to_prior_ess_per_second_95pct_bootstrap_interval": _bootstrap_interval(
            ratios_per_second_prior, arguments.seed + 5000
        ),
        "neural_to_data_informed_median_ess_per_second_ratio": float(np.median(ratios_per_second_data)),
        "neural_to_data_informed_ess_per_second_95pct_bootstrap_interval": _bootstrap_interval(
            ratios_per_second_data, arguments.seed + 6000
        ),
        "neural_to_prior_median_ess_per_end_to_end_second_ratio": float(np.median(ratios_end_to_end_prior)),
        "neural_to_prior_ess_per_end_to_end_second_95pct_bootstrap_interval": _bootstrap_interval(
            ratios_end_to_end_prior, arguments.seed + 7000
        ),
        "neural_to_data_informed_median_ess_per_end_to_end_second_ratio": float(
            np.median(ratios_end_to_end_data)
        ),
        "neural_to_data_informed_ess_per_end_to_end_second_95pct_bootstrap_interval": _bootstrap_interval(
            ratios_end_to_end_data, arguments.seed + 8000
        ),
        "median_neural_prior_count_posterior_total_variation": float(np.median([
            item["neural_prior_count_posterior_total_variation"] for item in paired
        ])),
        "median_neural_data_informed_count_posterior_total_variation": float(np.median([
            item["neural_data_informed_count_posterior_total_variation"] for item in paired
        ])),
        "median_prior_count_rhat": float(np.median([item["prior_count_rhat"] for item in paired])),
        "median_data_informed_count_rhat": float(np.median([
            item["data_informed_count_rhat"] for item in paired
        ])),
        "median_neural_count_rhat": float(np.median([item["neural_count_rhat"] for item in paired])),
        "proposal_checkpoint": str(checkpoint.relative_to(EXAMPLE_PATH)),
    }
    summary["supported_efficiency_gain_vs_prior"] = _efficiency_gain_supported(
        ratios_per_call_prior,
        summary["neural_to_prior_ess_per_call_95pct_bootstrap_interval"],
        [item["neural_prior_count_posterior_total_variation"] for item in paired],
        [item["neural_count_rhat"] for item in paired],
        [item["prior_count_rhat"] for item in paired],
        likelihood_ratios_per_call_prior,
        summary["neural_to_prior_loglik_ess_per_call_95pct_bootstrap_interval"],
        [item["neural_likelihood_rhat"] for item in paired],
        [item["prior_likelihood_rhat"] for item in paired],
        ratios_end_to_end_prior,
        summary["neural_to_prior_ess_per_end_to_end_second_95pct_bootstrap_interval"],
    )
    summary["supported_efficiency_gain_vs_data_informed"] = _efficiency_gain_supported(
        ratios_per_call_data,
        summary["neural_to_data_informed_ess_per_call_95pct_bootstrap_interval"],
        [item["neural_data_informed_count_posterior_total_variation"] for item in paired],
        [item["neural_count_rhat"] for item in paired],
        [item["data_informed_count_rhat"] for item in paired],
        likelihood_ratios_per_call_data,
        summary["neural_to_data_informed_loglik_ess_per_call_95pct_bootstrap_interval"],
        [item["neural_likelihood_rhat"] for item in paired],
        [item["data_informed_likelihood_rhat"] for item in paired],
        ratios_end_to_end_data,
        summary["neural_to_data_informed_ess_per_end_to_end_second_95pct_bootstrap_interval"],
    )
    result_path = RUNS_PATH / f"{arguments.run_id}_results.json"
    RUNS_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {result_path}...")
    result_path.write_text(
        json.dumps(_json_safe({"summary": summary, "runs": results}), indent=2, allow_nan=False) + "\n"
    )
    figure_path = _plot_results(results, arguments.run_id, typefileplot=typefileplot)
    print(
        "Paired median neural/prior ESS per call: "
        f"{summary['neural_to_prior_median_ess_per_call_ratio']:.3g} "
        f"(95% bootstrap interval {summary['neural_to_prior_ess_per_call_95pct_bootstrap_interval']})"
    )
    print(
        "Paired median neural/data-informed ESS per call: "
        f"{summary['neural_to_data_informed_median_ess_per_call_ratio']:.3g} "
        f"(95% bootstrap interval {summary['neural_to_data_informed_ess_per_call_95pct_bootstrap_interval']})"
    )
    print(
        "Paired median neural/prior ESS per second: "
        f"{summary['neural_to_prior_median_ess_per_second_ratio']:.3g} "
        f"(95% bootstrap interval {summary['neural_to_prior_ess_per_second_95pct_bootstrap_interval']})"
    )
    print(
        "Paired median neural/data-informed ESS per second: "
        f"{summary['neural_to_data_informed_median_ess_per_second_ratio']:.3g} "
        f"(95% bootstrap interval {summary['neural_to_data_informed_ess_per_second_95pct_bootstrap_interval']})"
    )
    print(
        "Training-amortized neural/prior ESS per second: "
        f"{summary['neural_to_prior_median_ess_per_end_to_end_second_ratio']:.3g}"
    )
    print(
        "Training-amortized neural/data-informed ESS per second: "
        f"{summary['neural_to_data_informed_median_ess_per_end_to_end_second_ratio']:.3g}"
    )
    print(
        "Median count-posterior total variation: "
        f"prior {summary['median_neural_prior_count_posterior_total_variation']:.3f}, "
        f"data-informed {summary['median_neural_data_informed_count_posterior_total_variation']:.3f}"
    )
    print(
        "Median count R-hat across paired chains: "
        f"prior {summary['median_prior_count_rhat']:.3f}, "
        f"data-informed {summary['median_data_informed_count_rhat']:.3f}, "
        f"neural {summary['median_neural_count_rhat']:.3f}"
    )
    print(f"Efficiency gain supported versus prior element proposals: {summary['supported_efficiency_gain_vs_prior']}")
    print(
        "Efficiency gain supported versus data-informed element proposals: "
        f"{summary['supported_efficiency_gain_vs_data_informed']}"
    )
    print(
        f"Proposal conditioning: {proposal_conditioning}; training time: {training_seconds:.1f} s; "
        f"checkpoint: {checkpoint.relative_to(EXAMPLE_PATH)}"
    )
    print(f"Figure: {figure_path.relative_to(EXAMPLE_PATH)}")
    return summary


def historical_radial_velocity_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="neural_proposal_efficiency")
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--training-scenes", type=int, default=300)
    parser.add_argument("--validation-scenes", type=int, default=5)
    parser.add_argument("--chains-per-scene", type=int, default=2)
    parser.add_argument("--sweeps", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--mixture-components", type=int, default=8)
    parser.add_argument("--uniform-fraction", type=float, default=0.05)
    parser.add_argument(
        "--proposal-conditioning",
        choices=("data", "state"),
        default="data",
        help="Condition the neural proposal on observation features alone or also on the live catalog.",
    )
    parser.add_argument(
        "--state-feature-mode",
        choices=("catalog", "residual"),
        default="catalog",
        help="Use catalog summaries alone or periodogram features of the leave-one-out residual.",
    )
    add_smoke_argument(parser, help="Use a small run to check the pipeline.")
    parser.add_argument("--typeplotback", choices=("white", "dark"), default="white")
    parser.add_argument("--typefileplot", choices=("png", "pdf"), default="png")
    arguments = parser.parse_args()
    if arguments.smoke:
        arguments.training_scenes = 36
        arguments.validation_scenes = 1
        arguments.sweeps = 400
        arguments.chains_per_scene = 2
        arguments.epochs = 8
        arguments.mixture_components = 3
    if (arguments.training_scenes < 2 or arguments.validation_scenes < 1
            or arguments.chains_per_scene < 2 or arguments.sweeps < 40):
        parser.error(
            "training_scenes >= 2, validation_scenes >= 1, chains_per_scene >= 2, "
            "and sweeps >= 40 are required."
        )
    configure_example_style(arguments.typeplotback)
    run_benchmark(arguments)


def main():
    from examples.neural_proposal_efficiency.point_source_detection import main as point_source_main

    point_source_main()


if __name__ == "__main__":
    main()
