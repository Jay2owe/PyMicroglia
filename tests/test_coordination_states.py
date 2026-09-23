"""Saved categorical states, physical exposure and persistence-aware pair evidence."""
import copy
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import pymicroglia.states.behaviour_models as behaviour_models
import pymicroglia.measure.relationship_statistics as native
import pymicroglia.pipelines.behaviour.durations as duration, pymicroglia.pipelines.coordination.states as states
import pymicroglia.pipelines.coordination.state_inputs as source_inputs, pymicroglia.pipelines._screening as screening
from pymicroglia.pipelines.behaviour.options import ObservationKey
from pymicroglia.pipelines._contracts import CellKey, Settings
from pymicroglia.pipelines.coordination.options import resolve_request, run_request
from tests.test_behaviour_durations import sequence, SETTINGS as TIME_SETTINGS
from tests.test_coordination_options import request
SUPPORT = Settings({'min_observations': 2, 'min_span_hours': 0.0, 'max_gap_hours': 1.0, 'matching': 'exact', 'matching_tolerance_hours': 0.0})
FORMAL = {'method': 'truncated_time_shift', 'radius_hours': 24.0, 'stationary_series': 'reference', 'stationarity_justification': 'Controlled stationary two-state process, initialized from its stationary distribution'}

def fixture(a, b, hours=None, frames=None, formal=False, settings=None):
    hours = np.arange(len(a)) * 0.5 + 50.0 if hours is None else hours
    rows, definitions, inventory = sequence(a, hours, frames=frames, movie='paired')
    other, _, second = sequence(b, hours, frames=frames, movie='paired')
    other['identity'] = 8
    second['identity'] = 8
    other['observation_id'] = [ObservationKey(CellKey('controlled-source', 'paired', 8), int(frame), float(hour)).record_id for frame, hour in zip(other.frame_index, other.hours)]
    assignments = pd.concat([rows, other], ignore_index=True)
    inventory = pd.concat([inventory, second], ignore_index=True)
    time = duration.statistics(assignments, definitions, inventory, TIME_SETTINGS)
    source = {'status': 'accepted', 'model_id': 'controlled-model', 'bindings': {'support': {'scientific_id': 'support'}}, 'assignments': assignments, 'definitions': definitions, 'inventory': inventory, 'time': time}
    pair = {'source_run': 'controlled-source', 'movie': 'paired', 'reference_identity': 7, 'target_identity': 8, 'sample': 'paired', 'sample_confirmed': True, 'condition': 'one', 'geometry_population_eligible': True, 'pair_id': 'first'}
    prepared = {'inventory': pd.DataFrame([pair, {**pair, 'pair_id': 'second-measurement'}])}
    question = {'enabled': True, 'statistic': 'phi', 'evidence': FORMAL if formal else {'method': 'none'}, 'settings': settings or {}}
    resolved = SimpleNamespace(request=SimpleNamespace(questions={'states': question}, support=SUPPORT))
    return (resolved, prepared, source)

def analyse(*args, **kwargs):
    return states.analyse(*fixture(*args, **kwargs), 'controlled-coordination')[0]

def test_joint_unknown_and_gap_exposure_preserve_both_denominators():
    result = analyse([0, 0, None, 1, 1], [0, 0, 1, 1, 1])
    summary = result['pair_summary'].iloc[0]
    assert len(result['pair_summary']) == 1 and len(result['pair_membership']) == 2
    assert summary.joint_observed_hours == 2.0 and summary.joint_assigned_hours == 1.5
    assert summary.joint_unknown_hours == 0.5 and summary.same_state_hours == 1.5
    assert result['joint_occupancy'].fraction_observed.sum() == 0.75
    assert result['joint_occupancy'].fraction_assigned.sum() == 1.0
    assert summary.reference_events == 0 and summary.matched_events == 0
    assert result['switch_matches'].empty
    gap = analyse([0, 0, 1, 1], [0, 0, 1, 1], frames=[0, 1, 3, 4], hours=[50.0, 50.5, 51.5, 52.0])
    summary = gap['pair_summary'].iloc[0]
    assert summary.joint_unobserved_hours == 1.0 and summary.joint_observed_hours == 1.0
    assert summary.reference_events == summary.target_events == 0

def test_exact_switches_count_original_events_once_and_keep_interval_bounds():
    result = analyse([0, 0, 1, 1, 0], [0, 0, 1, 1, 0])
    summary = result['pair_summary'].iloc[0]
    assert summary.reference_events == summary.target_events == summary.matched_events == 2
    assert result['switch_events'].groupby('endpoint_role').event_id.nunique().tolist() == [2, 2]
    assert result['switch_matches'].reference_event_id.is_unique and result['switch_matches'].target_event_id.is_unique
    assert (result['switch_events'].latest_hours - result['switch_events'].earliest_hours).eq(0.5).all()
    assert pd.isna(summary.coincidence_p_value)
    assert len(result['pair_effects']) == 5
    assert result['pair_effects'].result_id.is_unique

def test_tolerance_ties_and_collisions_never_reuse_a_switch():

    def events(times):
        return pd.DataFrame({'event_id': [str(i) for i in range(len(times))], 'event_hours': times, 'eligible_for_coincidence': True})
    matched, summary = states.event_coincidence(events([1.0]), events([0.9, 1.1]), 0.2)
    assert not matched and summary['ambiguous_reference_events'] == 1
    matched, summary = states.event_coincidence(events([0.9, 1.1]), events([1.0]), 0.2)
    assert not matched and summary['ambiguous_reference_events'] == 2
    matched, summary = states.event_coincidence(events([1.0, 2.0]), events([1.1, 2.1]), 0.2)
    assert len(matched) == 2 and all((row['target_minus_reference_hours'] == pytest.approx(0.1) for row in matched))

def test_common_persistent_state_is_not_evidence_for_coordinated_quiet():
    result = analyse([0] * 150, [0] * 150, formal=True)
    assert result['pair_summary'].same_state_fraction_assigned.iloc[0] == 1.0
    assert result['pair_effects'].p_value.isna().all() and result['pair_effects'].effect.isna().all()
    assert result['pair_effects'].status.eq('unresolvable').all()
    assert pd.isna(result['pair_summary'].reference_matched_fraction.iloc[0])

def persistent(seed, n=300):
    rng = np.random.default_rng(seed)
    return (int(rng.integers(2)) + np.cumsum(rng.random(n) < 0.08)) % 2

def test_occupancy_and_switching_dependence_are_distinct_original_processes():
    a = persistent(710)
    result = analyse(a, 1 - a, formal=True)
    summary = result['pair_summary'].iloc[0]
    assert summary.same_state_fraction_assigned == 0.0 and summary.reference_matched_fraction == 1.0
    evidence = result['pair_effects']
    assert evidence.p_value.notna().all() and evidence.p_value.le(0.05).all()
    switch = evidence.loc[evidence.evidence_kind.eq('switch_process')].iloc[0]
    assert switch.effect == pytest.approx(1.0)
    matching = evidence.evidence_kind.eq('state_cooccupancy') & evidence.reference_state_id.eq(evidence.target_state_id)
    np.testing.assert_allclose(evidence.loc[matching, 'effect'], -1.0)
    assert summary.agreement_p_value is None and summary.coincidence_p_value is None

def test_original_unknowns_and_frames_refuse_temporal_compression():
    a = persistent(710).astype(object)
    b = a.copy()
    a[100] = None
    result = analyse(a, b, formal=True)
    assert result['pair_effects'].p_value.isna().all()
    assert result['pair_effects'].status.eq('untestable').all()
    frames = np.arange(len(a))
    frames[100:] += 1
    result = analyse(b, b, formal=True, frames=frames)
    assert result['pair_effects'].p_value.isna().all()
    assert result['pair_effects'].reason.str.contains('original acquisition').all()

def test_model_development_samples_cannot_supply_unused_application_evidence():
    a = persistent(710)
    resolved, prepared, source = fixture(a, a, formal=True)
    source['assignments'].loc[source['assignments'].identity.eq(7), 'role'] = 'development'
    result = states.analyse(resolved, prepared, source, 'controlled')[0]
    assert result['pair_effects'].status.eq('descriptive_model_used').all()
    source['assignments']['role'] = 'assignment_only'
    source['inventory'] = pd.concat([source['inventory'], source['inventory'].iloc[:1].assign(movie='other', role='learning')])
    result = states.analyse(resolved, prepared, source, 'controlled')[0]
    assert result['pair_effects'].status.eq('descriptive_model_used').all()

def test_explicit_state_labels_require_same_accepted_model_and_empty_cases_stay_visible():
    resolved, prepared, source = fixture([0], [1])
    result = states.analyse(resolved, prepared, source, 'controlled')[0]
    assert result['pair_summary'].status.eq('insufficient').all()
    assert result['pair_effects'].status.eq('insufficient').all() and result['joint_exposure'].empty
    q = {**resolved.request.questions['states'], 'settings': {'state_pairs': [['0', '1']]}}
    with pytest.raises(ValueError, match='another model'):
        states.options(q, SUPPORT, source['definitions'])
    q['settings'] = {'state_pairs': 'matching'}
    assert len(states.options(q, SUPPORT, source['definitions'])['resolved_state_pairs']) == 2
    source.update(status='unaccepted_model', reason='No supported states', definitions=None)
    with patch.object(states, 'indicator_evidence', side_effect=AssertionError('No comparison')):
        result = states.analyse(resolved, prepared, source, 'negative')[0]
    assert result['pair_summary'].status.eq('unaccepted_model').all() and result['pair_effects'].empty

def test_disabled_adapter_never_opens_optional_state_results():
    context = SimpleNamespace(request=SimpleNamespace(request=SimpleNamespace(questions={'states': {'enabled': False}})), step=SimpleNamespace(name='state-coordination'), scientific_id='disabled')
    with patch.object(states, 'load_states', side_effect=AssertionError('No source I/O')):
        assert states.identity(context) and states.produce(context).status == 'skipped-empty'
