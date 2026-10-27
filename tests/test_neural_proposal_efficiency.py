import importlib.util
from pathlib import Path

import numpy as np
import h5py
import pytest


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "neural_proposal_efficiency"
    / "neural_proposal_efficiency.py"
)
SPEC = importlib.util.spec_from_file_location("neural_proposal_efficiency", EXAMPLE_PATH)
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def test_count_ess_rejects_constant_chain_and_is_bounded_by_sample_count():
    assert BENCHMARK._count_ess(np.ones(100)) == 0.0
    assert BENCHMARK._count_ess(np.ones(3)) == 0.0

    random = np.random.default_rng(5)
    values = random.normal(size=2000)
    effective_size = BENCHMARK._count_ess(values)

    assert 0.0 < effective_size <= values.size
    assert effective_size > 0.5 * values.size


def test_rv_condition_feature_has_fixed_finite_periodogram_shape():
    time = np.linspace(8000.0, 8600.0, 60)  # [day]
    reference_time = 0.5 * (time[0] + time[-1])  # [day]
    maximum_period = 2.0 * (time[-1] - time[0])  # [day]
    velocity = 3.0 * np.sin(2.0 * np.pi * time / 17.0)  # [m/s]

    feature = BENCHMARK._condition_feature(time, velocity, reference_time, maximum_period)

    assert feature.shape == (512,)
    assert np.isfinite(feature).all()


def test_catalog_state_summary_is_permutation_invariant_and_handles_empty_catalogs():
    elements = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.7, 0.6, 0.5, 0.4, 0.3]])
    first = BENCHMARK._catalog_state_summary(elements)
    second = BENCHMARK._catalog_state_summary(elements[::-1])
    empty = BENCHMARK._catalog_state_summary(np.empty((0, 5)))

    assert first.shape == (11,)
    np.testing.assert_allclose(first, second)
    assert first[0] == 2.0 / BENCHMARK.MAXIMUM_PLANETS
    np.testing.assert_array_equal(empty, np.zeros(11))


def test_state_conditioned_training_pairs_have_consistent_features_and_elements():
    time = np.linspace(8000.0, 8600.0, 30)
    reference_time = 0.5 * (time[0] + time[-1])
    conditions, elements = BENCHMARK._training_data_state(
        time, reference_time, 1200.0, scene_count=12, seed=57
    )

    assert conditions.ndim == elements.ndim == 2
    assert conditions.shape[0] == elements.shape[0]
    assert conditions.shape[1] == 512 + 11
    assert elements.shape[1] == 5
    assert np.isfinite(conditions).all() and np.isfinite(elements).all()
    assert np.all((elements >= 0.0) & (elements <= 1.0))


def test_residual_state_features_subtract_the_context_catalog_model():
    from types import SimpleNamespace
    from pcat import ProposalContext

    time = np.linspace(8000.0, 8600.0, 60)
    reference_time = 0.5 * (time[0] + time[-1])
    maximum_period = 1200.0
    context_elements = np.array([[0.15, 0.4, 0.2, 0.1, 0.7]])
    target_element = np.array([0.8, 0.6, 0.8, 0.05, 0.3])
    context_planets = BENCHMARK._draw_planets(context_elements, maximum_period)
    target_planets = BENCHMARK._draw_planets(target_element[None, :], maximum_period)
    velocity = BENCHMARK._render_planets(
        time, np.concatenate((context_planets, target_planets)), reference_time
    ) + np.random.default_rng(9).normal(0.0, BENCHMARK.MEASUREMENT_UNCERTAINTY, time.size)
    gdat = SimpleNamespace(
        neural_state_feature_mode="residual",
        neural_maximum_period=maximum_period,
        timervel=time,
        timervelrefr=reference_time,
        rvelobsv=velocity,
        neural_proposal_features=np.zeros(512),
    )
    context = ProposalContext(0, (context_elements,))

    actual = BENCHMARK._state_condition_features(gdat, context)
    residual = velocity - BENCHMARK._render_planets(time, context_planets, reference_time)
    expected_data_features = BENCHMARK._condition_feature(
        time, residual, reference_time, maximum_period
    )

    np.testing.assert_allclose(actual[:512], expected_data_features)
    np.testing.assert_allclose(actual[512:], BENCHMARK._catalog_state_summary(context_elements))


def test_residual_training_conditions_encode_the_leave_one_out_catalog():
    time = np.linspace(8000.0, 8600.0, 30)
    reference_time = 0.5 * (time[0] + time[-1])
    maximum_period = 1200.0
    scene_count = 3
    seed = 61
    conditions, elements = BENCHMARK._training_data_state(
        time, reference_time, maximum_period, scene_count, seed, feature_mode="residual"
    )
    random = np.random.default_rng(seed)
    expected_contexts = []
    for _ in range(scene_count):
        planet_count = int(random.integers(1, BENCHMARK.MAXIMUM_PLANETS + 1))
        scene_elements = random.uniform(size=(planet_count, 5))
        random.normal(0.0, np.full(time.size, BENCHMARK.MEASUREMENT_UNCERTAINTY))
        expected_contexts.extend(
            BENCHMARK._catalog_state_summary(np.delete(scene_elements, target_index, axis=0))
            for target_index in range(planet_count)
        )

    np.testing.assert_allclose(conditions[:, -11:], expected_contexts)
    np.testing.assert_allclose(elements.shape, (conditions.shape[0], 5))


def test_count_ess_rejects_masked_and_nonfinite_values():
    masked = np.ma.array([0.0, 1.0, 2.0], mask=[False, True, False])
    with pytest.raises(ValueError, match="masked"):
        BENCHMARK._count_ess(masked)
    with pytest.raises(ValueError, match="finite"):
        BENCHMARK._count_ess([0.0, np.nan, 1.0])


def test_count_rhat_detects_offset_chains_and_accepts_matching_chains():
    random = np.random.default_rng(9)
    first = random.normal(size=2000)
    second = random.normal(size=2000)

    assert BENCHMARK._count_rhat([first, second]) < 1.01
    assert BENCHMARK._count_rhat([first, second + 1.0]) > 1.1
    assert np.isnan(BENCHMARK._count_rhat([first]))


@pytest.mark.parametrize('sample_count', [0, 1, 2, 4, 64])
def test_count_rhat_identical_constant_chains_are_undefined(sample_count):
    chains = [np.ones(sample_count), np.ones(sample_count)]
    with np.errstate(all='raise'):
        actual = BENCHMARK._count_rhat(chains)
    assert np.isnan(actual)


@pytest.mark.parametrize('sample_count', [2, 64])
def test_count_rhat_disagreeing_constant_chains_remain_infinite(sample_count):
    with np.errstate(all='raise'):
        actual = BENCHMARK._count_rhat([np.ones(sample_count), np.full(sample_count, 2)])
    assert actual == np.inf


@pytest.mark.parametrize('proposal_undefined', [False, True])
def test_efficiency_claim_rejects_undefined_constant_count_rhat(proposal_undefined):
    undefined = BENCHMARK._count_rhat([np.ones(8), np.ones(8)])
    proposal = [undefined, 1.01] if proposal_undefined else [1.01, 1.02]
    reference = [1.01, 1.02] if proposal_undefined else [undefined, 1.01]

    assert not BENCHMARK._efficiency_gain_supported(
        [1.3, 1.5], [1.1, 1.7], [0.05, 0.07], proposal, reference,
        [1.3, 1.5], [1.1, 1.7], [1.02, 1.03], [1.01, 1.02],
        [1.2, 1.3], [1.1, 1.4],
    )


@pytest.mark.parametrize('sample_count', [2, 8, 65])
def test_count_rhat_preserves_classical_varying_chain_formula(sample_count):
    values = np.random.default_rng(53).normal(size=(3, sample_count))
    within = float(np.mean(np.var(values, axis=1, ddof=1)))
    between = float(sample_count * np.var(np.mean(values, axis=1), ddof=1))
    variance = (sample_count - 1.0) / sample_count * within + between / sample_count
    expected = float(np.sqrt(max(variance / within, 0.0)))

    with np.errstate(all='raise'):
        actual = BENCHMARK._count_rhat(values)

    assert actual == expected


@pytest.mark.parametrize('bad_chain', [
    [np.nan, 0, 1, 2, 1],
    [np.inf, 0, 1, 2, 1],
    [-np.inf, 0, 1, 2, 1],
    [0, 1, np.nan, 2],
    np.ma.array([0, 1, 2, 1], mask=[False, True, False, False]),
    np.ma.masked_all(4),
    [np.nan],
])
@pytest.mark.parametrize('bad_first', [False, True])
def test_count_rhat_rejects_unavailable_samples_before_tail_alignment(bad_chain, bad_first):
    valid = [0, 1, 2, 1]
    chains = [bad_chain, valid] if bad_first else [valid, bad_chain]
    with np.errstate(all='raise'), pytest.raises(ValueError, match='Count R-hat requires'):
        BENCHMARK._count_rhat(chains)


@pytest.mark.parametrize('container', [list, tuple, np.asarray, np.ma.asarray])
def test_count_rhat_preserves_valid_unequal_length_tail_alignment(container):
    first = np.array([3, 0, 1, 2, 1], dtype=float)
    second = np.array([0, 2, 1, 1], dtype=float)
    expected = BENCHMARK._count_rhat([first[-4:], second])

    with np.errstate(all='raise'):
        actual = BENCHMARK._count_rhat([container(first), container(second)])

    assert actual == expected


@pytest.mark.parametrize(('values', 'expected'), [
    ([np.nan], np.nan),
    ([np.nan, np.nan], np.nan),
    ([np.nan, 1.01], np.nan),
    ([1.01, np.nan, 1.02], np.nan),
    ([1.01, 1.03], 1.02),
    ([1.01, np.inf], np.inf),
    ([np.inf, np.inf], np.inf),
])
def test_benchmark_rhat_summary_preserves_every_scene_without_nanmedian_warnings(values, expected):
    import ast
    import warnings

    print(f'Reading from {EXAMPLE_PATH}...')
    tree = ast.parse(EXAMPLE_PATH.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_benchmark')
    summary = next(node.value for node in function.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'summary' for target in node.targets))
    names = {'median_prior_count_rhat', 'median_data_informed_count_rhat', 'median_neural_count_rhat'}
    fields = {key.value: value for key, value in zip(summary.keys, summary.values) if isinstance(key, ast.Constant) and key.value in names}
    assert set(fields) == names
    paired = [dict(prior_count_rhat=value, data_informed_count_rhat=value, neural_count_rhat=value) for value in values]

    with warnings.catch_warnings(), np.errstate(all='raise'):
        warnings.simplefilter('error', RuntimeWarning)
        actual = {name: eval(compile(ast.Expression(body=expression), str(EXAMPLE_PATH), 'eval'), {'np': np, 'paired': paired}) for name, expression in fields.items()}

    for value in actual.values():
        if np.isnan(expected):
            assert np.isnan(value)
        else:
            assert value == expected


@pytest.fixture
def favorable_efficiency_evidence():
    return {
        'ratios': [1.3, 1.5], 'interval': [1.1, 1.7], 'total_variations': [0.05, 0.07],
        'proposal_rhats': [1.02, 1.03], 'reference_rhats': [1.01, 1.02],
        'likelihood_ratios': [1.3, 1.5], 'likelihood_interval': [1.1, 1.7],
        'proposal_likelihood_rhats': [1.02, 1.03], 'reference_likelihood_rhats': [1.01, 1.02],
        'end_to_end_ratios': [1.3, 1.5], 'end_to_end_interval': [1.1, 1.7],
    }


@pytest.mark.parametrize('field', [
    'ratios', 'interval', 'total_variations', 'proposal_rhats', 'reference_rhats',
    'likelihood_ratios', 'likelihood_interval', 'proposal_likelihood_rhats',
    'reference_likelihood_rhats', 'end_to_end_ratios', 'end_to_end_interval',
])
@pytest.mark.parametrize('fully_masked', [False, True])
def test_efficiency_claim_rejects_masked_evidence_before_conversion(favorable_efficiency_evidence, field, fully_masked):
    evidence = favorable_efficiency_evidence
    assert BENCHMARK._efficiency_gain_supported(**evidence)
    evidence[field] = np.ma.array(evidence[field], mask=[True, fully_masked])

    with np.errstate(all='raise'):
        assert not BENCHMARK._efficiency_gain_supported(**evidence)

    assert np.ma.is_masked(evidence[field])


@pytest.mark.parametrize('container', [list, tuple, np.asarray, np.ma.asarray])
def test_efficiency_claim_preserves_available_evidence_containers(favorable_efficiency_evidence, container):
    evidence = {name: container(values) for name, values in favorable_efficiency_evidence.items()}

    with np.errstate(all='raise'):
        assert BENCHMARK._efficiency_gain_supported(**evidence)


@pytest.mark.parametrize('field', [
    'ratios', 'total_variations', 'proposal_rhats', 'reference_rhats',
    'likelihood_ratios', 'proposal_likelihood_rhats', 'reference_likelihood_rhats',
    'end_to_end_ratios',
])
@pytest.mark.parametrize('values', [[], [1.02], [[1.02], [1.03]], 1.02])
def test_efficiency_claim_rejects_unaligned_scene_evidence(favorable_efficiency_evidence, field, values):
    evidence = {**favorable_efficiency_evidence, field: values}

    with np.errstate(all='raise'):
        assert not BENCHMARK._efficiency_gain_supported(**evidence)


@pytest.mark.parametrize('field', ['interval', 'likelihood_interval', 'end_to_end_interval'])
@pytest.mark.parametrize('bounds', [[], [1.1], [1.1, 1.7, 2.0], [[1.1, 1.7]], 1.1, [1.7, 1.1]])
def test_efficiency_claim_rejects_malformed_or_reversed_intervals(favorable_efficiency_evidence, field, bounds):
    evidence = {**favorable_efficiency_evidence, field: bounds}

    with np.errstate(all='raise'):
        assert not BENCHMARK._efficiency_gain_supported(**evidence)


def test_efficiency_gain_requires_mixing_and_posterior_agreement():
    assert BENCHMARK._efficiency_gain_supported(
        [1.3, 1.5], [1.1, 1.7], [0.05, 0.07], [1.02, 1.03], [1.01, 1.02],
        [1.2, 1.4], [1.1, 1.5], [1.02, 1.03], [1.01, 1.02],
        [1.2, 1.3], [1.1, 1.4],
    )
    assert not BENCHMARK._efficiency_gain_supported(
        [2.0, 2.0], [1.5, 2.5], [0.05, 0.07], [1.3, 1.02], [1.01, 1.02],
        [2.0, 2.0], [1.5, 2.5], [1.02, 1.03], [1.01, 1.02],
        [1.2, 1.3], [1.1, 1.4],
    )
    assert not BENCHMARK._efficiency_gain_supported(
        [2.0, 2.0], [1.5, 2.5], [0.5, 0.07], [1.02, 1.03], [1.01, 1.02],
        [2.0, 2.0], [1.5, 2.5], [1.02, 1.03], [1.01, 1.02],
        [1.2, 1.3], [1.1, 1.4],
    )
    assert not BENCHMARK._efficiency_gain_supported(
        [2.0, 2.0], [1.5, 2.5], [0.05, 0.07], [1.02, 1.03], [1.01, 1.02],
        [0.5, 0.7], [0.4, 0.9], [1.02, 1.03], [1.01, 1.02],
        [1.2, 1.3], [1.1, 1.4],
    )


@pytest.fixture
def acceptance_trace_file(tmp_path):
    root = tmp_path / 'synthetic'
    path = root / 'data' / 'outp' / root.name / 'gdatmodi0000post.h5'
    path.parent.mkdir(parents=True)

    def write(proposal_types, flags):
        print(f'Writing to {path}...')
        with h5py.File(path, 'w') as output:
            output['listpostindxproptype'] = proposal_types
            output['listpostboolpropaccp'] = flags
        return root

    return write


@pytest.mark.parametrize('flags', [
    [np.nan, 0.0], [np.inf, 0.0], [0.5, 0.0], [2, 0], [-1, 0],
    np.array([b'1', b'0']), np.array([1.0, 0.0], dtype=complex),
])
def test_acceptance_reader_rejects_nonbinary_saved_flags(acceptance_trace_file, flags):
    root = acceptance_trace_file([1, 2], flags)
    with np.errstate(all='raise'), pytest.raises(ValueError, match='Acceptance flags'):
        BENCHMARK._count_acceptance(root)


@pytest.mark.parametrize('proposal_types', [
    [np.nan, 1.0], [np.inf, 1.0], [-1, 1], [1.5, 1.0],
    [True, False], np.array([b'1', b'2']), np.array([1.0, 2.0], dtype=complex),
])
def test_acceptance_reader_rejects_invalid_saved_proposal_ids(acceptance_trace_file, proposal_types):
    root = acceptance_trace_file(proposal_types, [True, False])
    with np.errstate(all='raise'), pytest.raises(ValueError, match='Proposal types'):
        BENCHMARK._count_acceptance(root)


def test_acceptance_reader_rejects_unaligned_saved_traces(acceptance_trace_file):
    root = acceptance_trace_file([1, 2], [True])
    with pytest.raises(ValueError, match='aligned trace lengths'):
        BENCHMARK._count_acceptance(root)


@pytest.mark.parametrize('flag_dtype', [bool, np.int8, np.float64])
@pytest.mark.parametrize('proposal_dtype', [np.int64, np.float64])
def test_acceptance_reader_preserves_valid_rates_and_other_proposal_ids(acceptance_trace_file, flag_dtype, proposal_dtype):
    proposal_types = np.array([0, 1, 1, 2, 2, 5, 5, 99], dtype=proposal_dtype).reshape(-1, 1)
    flags = np.array([1, 1, 0, 0, 1, 1, 1, 1], dtype=flag_dtype)
    root = acceptance_trace_file(proposal_types, flags)

    with np.errstate(all='raise'):
        actual = BENCHMARK._count_acceptance(root)

    assert actual == {'birth_acceptance': 0.5, 'death_acceptance': 0.5, 'jump_acceptance': 1.0}


def test_acceptance_reader_keeps_empty_trace_zero_attempt_policy(acceptance_trace_file):
    root = acceptance_trace_file(np.array([], dtype=int), np.array([], dtype=bool))
    assert BENCHMARK._count_acceptance(root) == {'birth_acceptance': 0.0, 'death_acceptance': 0.0, 'jump_acceptance': 0.0}


@pytest.fixture
def posterior_trace_file(tmp_path):
    path = tmp_path / 'posterior.h5'

    def write(counts, likelihood):
        print(f'Writing to {path}...')
        with h5py.File(path, 'w') as output:
            output['listpostnumbelem'] = counts
            output['listpostlliktotl'] = likelihood
        return path

    return write


@pytest.mark.parametrize('counts', [
    [0.0, 1.9], [-1, 1], [0, 4], [np.nan, 1], [np.inf, 1],
    [True, False], np.array([b'0', b'1']), np.array([0, 1], dtype=complex),
])
def test_posterior_reader_rejects_invalid_counts_before_integer_conversion(posterior_trace_file, counts):
    path = posterior_trace_file(counts, [-3.0, -2.0])
    with np.errstate(all='raise'), pytest.raises(ValueError, match='Posterior counts'):
        BENCHMARK._read_posterior_traces(path)


@pytest.mark.parametrize('likelihood', [
    [np.nan, -2.0], [np.inf, -2.0], [-np.inf, -2.0],
    [True, False], np.array([b'-3', b'-2']), np.array([-3, -2], dtype=complex),
])
def test_posterior_reader_rejects_unavailable_or_nonreal_likelihood(posterior_trace_file, likelihood):
    path = posterior_trace_file([0, 1], likelihood)
    with np.errstate(all='raise'), pytest.raises(ValueError, match='Posterior likelihood'):
        BENCHMARK._read_posterior_traces(path)


@pytest.mark.parametrize(('counts', 'likelihood'), [([], []), ([0, 1], [-3.0]), ([], [-3.0])])
def test_posterior_reader_rejects_empty_or_unaligned_traces(posterior_trace_file, counts, likelihood):
    path = posterior_trace_file(counts, likelihood)
    with pytest.raises(ValueError, match='nonempty and aligned'):
        BENCHMARK._read_posterior_traces(path)


@pytest.mark.parametrize('count_dtype', [np.int64, np.float64])
def test_posterior_reader_preserves_valid_values_and_flattened_layout(posterior_trace_file, count_dtype):
    counts = np.array([0, 1, 2, 3], dtype=count_dtype).reshape(-1, 1)
    likelihood = np.array([-4.0, -3.0, -2.0, -1.0])
    path = posterior_trace_file(counts, likelihood)
    print(f'Reading from {path}...')
    original = path.read_bytes()

    with np.errstate(all='raise'):
        actual_counts, actual_likelihood = BENCHMARK._read_posterior_traces(path)

    np.testing.assert_array_equal(actual_counts, [0, 1, 2, 3])
    np.testing.assert_array_equal(actual_likelihood, likelihood)
    assert actual_counts.dtype.kind == 'i'
    print(f'Reading from {path}...')
    assert path.read_bytes() == original


@pytest.mark.parametrize('invalid_count', [False, True])
def test_actual_run_sampler_validates_saved_posterior_before_metrics(monkeypatch, tmp_path, invalid_count):
    from types import SimpleNamespace
    from unittest.mock import Mock

    root = tmp_path / 'data'
    run_id = 'synthetic'
    trace = Path(BENCHMARK.retr_pathrun(str(root), run_id)) / 'data' / 'outp' / run_id / 'gdatmodi0000post.h5'
    counts = np.array([0.0, 1.5 if invalid_count else 1.0, 2.0, 1.0])

    def sample(**configuration):
        trace.parent.mkdir(parents=True)
        print(f'Writing to {trace}...')
        with h5py.File(trace, 'w') as output:
            output['listpostnumbelem'] = counts
            output['listpostlliktotl'] = [-4.0, -3.0, -2.0, -3.0]
            output['listpostindxproptype'] = [1, 1, 2, 5]
            output['listpostboolpropaccp'] = [True, False, True, False]
        BENCHMARK.ACTIVE_LIKELIHOOD_CALLS[0] = 10

    reader = Mock(wraps=BENCHMARK._read_posterior_traces)
    ess = Mock(wraps=BENCHMARK._count_ess)
    monkeypatch.setattr(BENCHMARK, 'RUNS_PATH', root)
    monkeypatch.setattr(BENCHMARK, 'ACTIVE_LIKELIHOOD_CALLS', [0])
    monkeypatch.setattr(BENCHMARK, 'retr_dictpcatrvel', Mock(return_value={}))
    monkeypatch.setattr(BENCHMARK, 'sampling', SimpleNamespace(sample=sample))
    monkeypatch.setattr(BENCHMARK, '_read_posterior_traces', reader)
    monkeypatch.setattr(BENCHMARK, '_count_ess', ess)
    arguments = ('prior', run_id, np.arange(4.0), np.zeros(4), np.ones(4), np.zeros(4), None, 40, 53)

    if invalid_count:
        with pytest.raises(ValueError, match='Posterior counts'):
            BENCHMARK._run_sampler(*arguments)
        ess.assert_not_called()
    else:
        result = BENCHMARK._run_sampler(*arguments)
        assert result['count_samples'] == [0, 1, 2, 1]
        assert result['likelihood_samples'] == [-4.0, -3.0, -2.0, -3.0]
        assert result['birth_acceptance'] == 0.5
        assert result['death_acceptance'] == 1.0
        assert result['likelihood_calls'] == 10
        assert ess.call_count == 2
    reader.assert_called_once_with(trace)


def test_state_mode_run_sampler_disables_data_only_fallback(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock

    root = tmp_path / "data"
    run_id = "state_conditioned"
    trace = Path(BENCHMARK.retr_pathrun(str(root), run_id)) / "data" / "outp" / run_id / "gdatmodi0000post.h5"
    captured = {}

    def sample(**configuration):
        captured.update(configuration)
        trace.parent.mkdir(parents=True)
        with h5py.File(trace, "w") as output:
            output["listpostnumbelem"] = [0, 1, 2, 1]
            output["listpostlliktotl"] = [-4.0, -3.0, -2.0, -3.0]
            output["listpostindxproptype"] = [1, 1, 2, 5]
            output["listpostboolpropaccp"] = [True, False, True, False]
        BENCHMARK.ACTIVE_LIKELIHOOD_CALLS[0] = 10

    state_callbacks = {
        "retr_drawpropelem_state": object(),
        "retr_lpdfpropelem_state": object(),
    }
    proposal = SimpleNamespace(state_callbacks=Mock(return_value=state_callbacks))
    configuration_builder = Mock(return_value={})
    monkeypatch.setattr(BENCHMARK, "RUNS_PATH", root)
    monkeypatch.setattr(BENCHMARK, "ACTIVE_LIKELIHOOD_CALLS", [0])
    monkeypatch.setattr(BENCHMARK, "retr_dictpcatrvel", configuration_builder)
    monkeypatch.setattr(BENCHMARK, "sampling", SimpleNamespace(sample=sample))

    BENCHMARK._run_sampler(
        "neural", run_id, np.arange(4.0), np.zeros(4), np.ones(4), np.zeros(4),
        proposal, 40, 53, proposal_conditioning="state",
    )

    assert configuration_builder.call_args.kwargs["boolmakepropelem"] is False
    assert captured["retr_drawpropelem_state"] is state_callbacks["retr_drawpropelem_state"]
    assert captured["retr_lpdfpropelem_state"] is state_callbacks["retr_lpdfpropelem_state"]
    assert "retr_drawpropelem" not in captured
    assert "retr_lpdfpropelem" not in captured


@pytest.fixture
def chain_cost_record():
    return {
        'count_samples': [0, 1, 2, 1], 'likelihood_samples': [-3.0, -2.0, -1.0, -2.0],
        'count_ess': 2.0, 'likelihood_ess': 2.0,
        'likelihood_calls': 10, 'elapsed_seconds': 1.0,
        'count_probability': [0.25, 0.5, 0.25, 0.0],
        'birth_acceptance': 0.1, 'death_acceptance': 0.2,
    }


@pytest.mark.parametrize(('field', 'value'), [
    ('likelihood_calls', 0), ('likelihood_calls', -1), ('likelihood_calls', 2.5),
    ('likelihood_calls', np.nan), ('likelihood_calls', np.inf), ('likelihood_calls', -np.inf),
    ('likelihood_calls', True),
    ('elapsed_seconds', 0.0), ('elapsed_seconds', -1.0), ('elapsed_seconds', np.nan),
    ('elapsed_seconds', np.inf), ('elapsed_seconds', -np.inf),
    ('training_overhead_seconds', -0.5), ('training_overhead_seconds', np.nan),
    ('training_overhead_seconds', np.inf), ('training_overhead_seconds', -np.inf),
])
@pytest.mark.parametrize('bad_first', [False, True])
def test_chain_aggregation_rejects_invalid_cost_metadata_before_reduction(chain_cost_record, field, value, bad_first):
    valid = dict(chain_cost_record)
    invalid = {**chain_cost_record, field: value}
    runs = [invalid, valid] if bad_first else [valid, invalid]

    with np.errstate(all='raise'), pytest.raises(ValueError, match=field):
        BENCHMARK._aggregate_chain_runs(runs)


def test_chain_aggregation_requires_at_least_one_run():
    with np.errstate(all='raise'), pytest.raises(ValueError, match='at least one run'):
        BENCHMARK._aggregate_chain_runs([])


@pytest.mark.parametrize('calls', [10, 10.0, np.int64(10)])
@pytest.mark.parametrize('overhead', [None, 0.0, 0.5])
def test_chain_aggregation_preserves_valid_costs_and_rates(chain_cost_record, calls, overhead):
    first = {**chain_cost_record, 'likelihood_calls': calls}
    if overhead is not None:
        first['training_overhead_seconds'] = overhead
    second = {**chain_cost_record, 'likelihood_calls': 20, 'elapsed_seconds': 2.0}

    with np.errstate(all='raise'):
        actual = BENCHMARK._aggregate_chain_runs([first, second])

    assert actual['likelihood_calls'] == 30
    assert actual['elapsed_seconds'] == 3.0
    assert actual['count_ess'] == actual['likelihood_ess'] == 4.0
    assert actual['ess_per_call'] == actual['likelihood_ess_per_call'] == 4.0 / 30
    assert actual['ess_per_second'] == actual['likelihood_ess_per_second'] == 4.0 / 3
    expected = 4.0 / (3.0 + (0.0 if overhead is None else overhead))
    assert actual['ess_per_end_to_end_second'] == actual['likelihood_ess_per_end_to_end_second'] == expected


def test_chain_aggregation_sums_integer_call_counts_without_numpy_overflow(chain_cost_record):
    calls = np.iinfo(np.int64).max
    runs = [{**chain_cost_record, 'likelihood_calls': np.int64(calls)} for _ in range(2)]

    with np.errstate(all='raise'):
        actual = BENCHMARK._aggregate_chain_runs(runs)

    assert actual['likelihood_calls'] == 2 * int(calls)
    assert isinstance(actual['likelihood_calls'], int)
    assert actual['ess_per_call'] == 4.0 / (2 * int(calls))


@pytest.mark.parametrize('field', ['count_ess', 'likelihood_ess'])
@pytest.mark.parametrize('value', [-1.0, np.nan, np.inf, -np.inf, 4.01, True, [2.0], 2.0 + 0.0j, np.ma.masked])
@pytest.mark.parametrize('bad_first', [False, True])
def test_chain_aggregation_rejects_invalid_effective_sizes(chain_cost_record, field, value, bad_first):
    valid = dict(chain_cost_record)
    invalid = {**chain_cost_record, field: value}
    runs = [invalid, valid] if bad_first else [valid, invalid]

    with np.errstate(all='raise'), pytest.raises(ValueError, match=field):
        BENCHMARK._aggregate_chain_runs(runs)


@pytest.mark.parametrize('field', ['count_ess', 'likelihood_ess'])
@pytest.mark.parametrize('value', [0.0, 1.5, 4.0])
def test_chain_aggregation_preserves_valid_effective_sizes(chain_cost_record, field, value):
    runs = [{**chain_cost_record, field: value}, dict(chain_cost_record)]

    with np.errstate(all='raise'):
        actual = BENCHMARK._aggregate_chain_runs(runs)

    assert actual[field] == value + 2.0
    metric = 'ess_per_call' if field == 'count_ess' else 'likelihood_ess_per_call'
    assert actual[metric] == (value + 2.0) / 20
    assert runs[0][field] == value


@pytest.mark.parametrize('field', ['birth_acceptance', 'death_acceptance'])
@pytest.mark.parametrize('value', [-0.2, 1.2, np.nan, np.inf, -np.inf, True, [0.5], 0.5 + 0.0j, np.ma.masked])
@pytest.mark.parametrize('bad_first', [False, True])
def test_chain_aggregation_rejects_invalid_acceptance_rates(chain_cost_record, field, value, bad_first):
    valid = dict(chain_cost_record)
    invalid = {**chain_cost_record, field: value}
    runs = [invalid, valid] if bad_first else [valid, invalid]

    with np.errstate(all='raise'), pytest.raises(ValueError, match=field):
        BENCHMARK._aggregate_chain_runs(runs)


@pytest.mark.parametrize('field', ['birth_acceptance', 'death_acceptance'])
@pytest.mark.parametrize('value', [0.0, 0.5, 1.0])
def test_chain_aggregation_preserves_acceptance_endpoints_and_means(chain_cost_record, field, value):
    first = {**chain_cost_record, field: value}
    second = {**chain_cost_record, field: 0.25}

    with np.errstate(all='raise'):
        actual = BENCHMARK._aggregate_chain_runs([first, second])

    assert actual[field] == (value + 0.25) / 2
    assert first[field] == value


@pytest.mark.parametrize('probability', [
    [], 1.0, [0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 0.0],
    [[0.25, 0.5, 0.25, 0.0]], [-1.0, 2.0, 0.0, 0.0],
    [0.0, 1.1, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2],
    [np.nan, 0.5, 0.5, 0.0], [np.inf, 0.0, 0.0, 0.0],
    np.ma.array([0.25, 0.5, 0.25, 0.0], mask=[True, False, False, False]),
    np.ma.masked_all(4), np.array([None, 0, 1, 0], dtype=object),
    np.array([0.25, 0.5, 0.25, 0.0], dtype=complex),
])
@pytest.mark.parametrize('bad_first', [False, True])
def test_chain_aggregation_rejects_invalid_count_probability(chain_cost_record, probability, bad_first):
    valid = dict(chain_cost_record)
    invalid = {**chain_cost_record, 'count_probability': probability}
    runs = [invalid, valid] if bad_first else [valid, invalid]

    with np.errstate(all='raise'), pytest.raises(ValueError, match='count_probability'):
        BENCHMARK._aggregate_chain_runs(runs)


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int64, bool])
@pytest.mark.parametrize('unmasked', [False, True])
def test_chain_aggregation_preserves_probability_types_and_equal_chain_weights(chain_cost_record, dtype, unmasked):
    first = np.array([0, 1, 0, 0], dtype=dtype)
    second = np.array([0, 0, 0, 1], dtype=dtype)
    if unmasked:
        first = np.ma.asarray(first)
        second = np.ma.asarray(second)
    runs = [
        {**chain_cost_record, 'count_probability': first, 'count_samples': [1] * 4},
        {**chain_cost_record, 'count_probability': second, 'count_samples': [3] * 8, 'likelihood_samples': [-3.0] * 8},
    ]

    with np.errstate(all='raise'):
        actual = BENCHMARK._aggregate_chain_runs(runs)

    np.testing.assert_array_equal(actual['count_probability'], [0.0, 0.5, 0.0, 0.5])


@pytest.mark.parametrize(('prior_ess', 'data_informed_ess', 'neural_ess'), [
    (0.0, 2.0, 3.0), (2.0, 0.0, 3.0), (0.0, 0.0, 3.0),
    (0.0, 0.0, 0.0), (2.0, 3.0, 0.0), (2.0, 3.0, 4.0),
])
def test_paired_efficiency_handles_zero_baselines_without_inventing_gains(prior_ess, data_informed_ess, neural_ess):
    runs = []
    efficiencies = {}
    for method, ess, calls, elapsed in (
        ('prior', prior_ess, 10, 1.0),
        ('data-informed', data_informed_ess, 20, 2.0),
        ('neural', neural_ess, 30, 3.0),
    ):
        overhead = 0.5 if method == 'neural' else 0.0  # [s]
        efficiencies[method] = {
            'ess_per_call': ess / calls,
            'ess_per_second': ess / elapsed,
            'ess_per_end_to_end_second': ess / (elapsed + overhead),
            'likelihood_ess_per_call': (ess / 2.0) / calls,
            'likelihood_ess_per_second': (ess / 2.0) / elapsed,
            'likelihood_ess_per_end_to_end_second': (ess / 2.0) / (elapsed + overhead),
        }
        for seed in (1, 2):
            runs.append({
                'validation_seed': 63, 'method': method, 'seed': seed,
                'count_samples': [1, 1, 1, 1] if ess == 0 else [0, 1, 2, 1],
                    'likelihood_samples': [1, 1, 1, 1] if ess == 0 else [0.0, -1.0, 0.5, 1.0],
                    'count_ess': ess, 'likelihood_ess': ess / 2.0,
                    'likelihood_calls': calls, 'elapsed_seconds': elapsed,
                'training_overhead_seconds': overhead,
                'count_probability': [0.0, 1.0, 0.0, 0.0] if ess == 0 else [0.25, 0.5, 0.25, 0.0],
                'birth_acceptance': 0.1, 'death_acceptance': 0.2,
            })

    with np.errstate(all='raise'):
        paired = BENCHMARK._paired_summary(runs)

    assert len(paired) == 1
    result = paired[0]
    assert result['validation_seed'] == 63
    for method, label in (('prior', 'prior'), ('data-informed', 'data_informed')):
        for metric in efficiencies[method]:
            actual = result[f'neural_to_{label}_{metric}_ratio']
            baseline = efficiencies[method][metric]
            if baseline == 0.0:
                assert np.isnan(actual)
            else:
                assert actual == efficiencies['neural'][metric] / baseline
    for label in ('prior', 'data_informed'):
        assert np.isfinite(result[f'neural_{label}_count_posterior_total_variation'])


@pytest.mark.parametrize('values', [[np.nan], [np.nan, 2.0], [np.nan, np.nan]])
def test_efficiency_bootstrap_keeps_missing_comparisons_undefined(values):
    with np.errstate(all='raise'):
        interval = BENCHMARK._bootstrap_interval(values, seed=53, draws=64)

    assert np.isnan(interval).all()


def test_json_safe_maps_undefined_metrics_to_null():
    value = BENCHMARK._json_safe({"undefined": np.nan, "infinite": np.inf, "nested": [1.0, np.float64(2.0)]})

    assert value == {"undefined": None, "infinite": None, "nested": [1.0, 2.0]}


@pytest.mark.parametrize('values', [
    [], [np.inf], [-np.inf, 1.5], [np.nan, 1.5],
    np.ma.array([1.3, 1.5], mask=[True, False]),
    np.ma.masked_all(2), np.ma.masked,
])
def test_bootstrap_unavailable_comparisons_stay_undefined_without_resampling(monkeypatch, values):
    from unittest.mock import Mock
    import warnings

    random = Mock(side_effect=AssertionError('Unavailable evidence reached resampling'))
    monkeypatch.setattr(BENCHMARK.np.random, 'default_rng', random)

    with warnings.catch_warnings(), np.errstate(all='raise'):
        warnings.simplefilter('error', RuntimeWarning)
        actual = BENCHMARK._bootstrap_interval(values, seed=53, draws=64)

    assert np.isnan(actual).all()
    random.assert_not_called()


@pytest.mark.parametrize('draws', [0, -1, 1.5, True, np.nan, np.inf])
def test_bootstrap_rejects_invalid_resampling_budget(draws):
    with pytest.raises(ValueError, match='Bootstrap draws'):
        BENCHMARK._bootstrap_interval([1.3, 1.5], seed=53, draws=draws)


@pytest.mark.parametrize('values', [1.3, [[1.3, 1.5]], [[1.3], [1.5]]])
def test_bootstrap_rejects_nonvector_comparisons(values):
    with pytest.raises(ValueError, match='one-dimensional'):
        BENCHMARK._bootstrap_interval(values, seed=53, draws=64)


@pytest.mark.parametrize('container', [list, tuple, np.asarray, np.ma.asarray])
@pytest.mark.parametrize('draws', [64, np.int64(65)])
def test_bootstrap_preserves_valid_seeded_intervals_exactly(container, draws):
    values = np.array([1.2, 1.3, 1.5, 1.7])
    sampled = np.random.default_rng(53).choice(values, size=(draws, values.size), replace=True)
    expected = np.quantile(np.median(sampled, axis=1), [0.025, 0.975]).tolist()

    with np.errstate(all='raise'):
        actual = BENCHMARK._bootstrap_interval(container(values), seed=53, draws=draws)

    assert actual == expected


def test_paired_efficiency_plot_handles_undefined_ratios(monkeypatch, tmp_path):
    from matplotlib import pyplot as plt

    runs = []
    for method in ('prior', 'data-informed', 'neural'):
        for seed in (1, 2):
            runs.append({
                'validation_seed': 63, 'method': method, 'seed': seed,
                'count_samples': [1, 1, 1, 1], 'count_ess': 0.0,
                'likelihood_samples': [1.0, 1.0, 1.0, 1.0], 'likelihood_ess': 0.0,
                'likelihood_calls': 10, 'elapsed_seconds': 1.0,
                'count_probability': [0.0, 1.0, 0.0, 0.0],
                'birth_acceptance': 0.0, 'death_acceptance': 0.0,
            })
    monkeypatch.setattr(BENCHMARK, 'VISUALS_PATH', tmp_path)
    before = set(plt.get_fignums())

    path = BENCHMARK._plot_results(runs, 'undefined_ratios')

    print(f'Reading from {path}...')
    image = plt.imread(path)
    assert image.shape[0] > 100 and image.shape[1] > 100
    assert np.ptp(image[..., :3]) > 0
    assert set(plt.get_fignums()) == before


def test_paired_efficiency_plot_labels_state_conditioning(monkeypatch, tmp_path):
    from matplotlib import pyplot as plt

    figures_before = set(plt.get_fignums())
    runs = []
    for method in ("prior", "data-informed", "neural"):
        runs.append({
            "validation_seed": 63,
            "method": method,
            "proposal_conditioning": "state" if method == "neural" else None,
            "seed": 20053,
            "count_samples": [0, 1, 2, 1],
            "count_ess": 2.0,
            "likelihood_samples": [-3.0, -2.0, -1.0, -2.0],
            "likelihood_ess": 2.0,
            "likelihood_calls": 10,
            "elapsed_seconds": 1.0,
            "count_probability": [0.25, 0.5, 0.25, 0.0],
            "birth_acceptance": 0.1,
            "death_acceptance": 0.2,
        })
    original_subplots = plt.subplots
    figures = []

    def capture_figure(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        figures.append(figure)
        return figure, axes

    monkeypatch.setattr(plt, "subplots", capture_figure)
    monkeypatch.setattr(BENCHMARK, "VISUALS_PATH", tmp_path)

    path = BENCHMARK._plot_results(runs, "state_mode", typefileplot="pdf")

    assert path.is_file()
    assert path.suffix == ".pdf"
    assert path.read_bytes().startswith(b"%PDF-")
    assert figures[0]._suptitle.get_text() == "State-conditioned, catalog summary neural proposal comparison"
    assert figures[0].axes[0].get_title() == "Count ESS/call\nvs prior"
    assert figures[0].axes[0].title.get_fontsize() == pytest.approx(12.0)
    assert figures[0].axes[0].xaxis.label.get_fontsize() == pytest.approx(12.0)
    assert set(plt.get_fignums()) == figures_before
