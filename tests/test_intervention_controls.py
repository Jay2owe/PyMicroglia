"""Original sample units, matching and native permutation reference checks."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from pymicroglia.measure.sample_contrasts import mean_difference
from pymicroglia.pipelines import parse
import pymicroglia.pipelines.intervention.control_statistics as native
from pymicroglia.pipelines.intervention.options import resolve_request, run_request
from pymicroglia.pipelines.intervention.windows import prepare
from pymicroglia.pipelines.intervention.evidence import analyse
from pymicroglia.pipelines.intervention.controls import policy, compare_units, correct, read_controls
from pymicroglia.pipelines.intervention.control_inputs import aggregate
from pymicroglia.pipelines._screening import file_hash

def fixture(tmp_path, matched=False, count=4, extra=False, missing=False, unconfirmed=False):
    rows = []
    anchors = {}
    samples = {}
    conditions = {}
    matching = []
    for i in range(count):
        for role in ['reference', 'target']:
            movie = role + str(i)
            anchor = 100.0 + 10 * i
            sample = role + '-sample-' + str(i)
            anchors[movie] = {'hours': anchor, 'kind': 'control' if role == 'reference' else 'intervention', 'label': 'Software event'}
            samples[movie] = sample
            conditions[movie] = role
            for frame in range(16):
                t = frame - 8
                change = 2.0 + i * 0.1 + (6.0 if role == 'target' else 0)
                value = 10.0 + change * (t >= 0)
                if missing and i == 0 and (role == 'target') and (t >= 0):
                    value = np.nan
                rows.append({'stem': movie, 'identity': 7, 'frame_index': frame, 'hours': anchor + t, 'custom_signal': value})
        if matched:
            matching.append({'match_id': 'original-' + str(i), 'reference_sample': 'reference-sample-' + str(i), 'target_sample': 'target-sample-' + str(i)})
    if extra:
        movie = 'reference-extra'
        anchors[movie] = deepcopy(anchors['reference0'])
        samples[movie] = samples['reference0']
        conditions[movie] = 'reference'
        for identity in [7, 8, 9]:
            for frame in range(16):
                rows.append({'stem': movie, 'identity': identity, 'frame_index': frame, 'hours': 100.0 + frame - 8, 'custom_signal': 10.0 + 10.0 * (frame >= 8)})
    if unconfirmed:
        samples.pop('target' + str(count - 1))
    frame = pd.DataFrame(rows)
    tables = {'cell_frame': frame, 'cell_summary': frame[['stem', 'identity']].drop_duplicates()}
    paths = {}
    for name, table in tables.items():
        paths[name] = tmp_path / (name + '.csv')
        table.to_csv(paths[name], index=False)
    evidence = {'method': 'matched_sample_permutation', 'matched_samples': 'Independent software pairs, labels exchangeable within original pairs'} if matched else {'method': 'independent_sample_permutation', 'independent_samples': 'Independent exchangeable software sample aggregates'}
    request = {'pipeline': 'intervention-response', 'measurements': ['custom_signal'], 'summary': 'mean', 'anchors': anchors, 'windows': [{'name': 'baseline', 'coordinate': 'relative_hours', 'start': -8, 'end': 0}, {'name': 'followup', 'coordinate': 'relative_hours', 'start': 0, 'end': 8, 'baseline': 'baseline'}], 'support': {'max_gap_hours': 1.1}, 'biological_samples': samples, 'conditions': conditions, 'matching': matching, 'evidence': {'method': 'none'}, 'inference': {'alpha': 0.05, 'multiple_testing': 'bonferroni', 'correction_scope': 'all'}, 'controls': {'enabled': True, 'aggregation': 'mean', 'evidence': evidence, 'comparisons': [{'name': 'original-change', 'measurement': 'custom_signal', 'baseline': 'baseline', 'target_window': 'followup', 'reference_condition': 'reference', 'target_condition': 'target', 'quantity': 'absolute_change', 'design': 'matched' if matched else 'independent'}]}}
    resolved = resolve(request, tables, paths)
    return (request, tables, paths, resolved)

def resolve(request, tables, paths):
    return resolve_request(parse([request])[0], source_run='software-source', tables=tables, input_hashes={key: file_hash(path) for key, path in paths.items()})

def data_for(resolved, tables):
    prepared = prepare(resolved, tables)
    evidence = analyse(prepared, resolved)
    settings = policy(resolved)
    return (aggregate(prepared, evidence, resolved, settings), settings)

def test_native_paired_labels_and_joint_bootstrap_match_public_scipy():
    a = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0])
    b = a + np.array([1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0])
    settings = {'method': 'matched_sample_permutation', 'matched_samples': 'Independent original software pairs', 'interval': {'method': 'paired_bootstrap', 'confidence': 0.95, 'resamples': 399, 'seed': 85}}
    result = native.compare(a, b, settings, matched=True)
    reference = stats.permutation_test((a, b), mean_difference, permutation_type='samples', alternative='two-sided', vectorized=True, batch=256, n_resamples=9999, random_state=np.random.default_rng(29))
    interval = stats.bootstrap((a, b), mean_difference, paired=True, vectorized=True, method='percentile', batch=256, n_resamples=399, confidence_level=0.95, random_state=np.random.default_rng(85))
    assert result['effect'] == reference.statistic and result['p_value'] == reference.pvalue and result['exact']
    assert np.array_equal(result['null_distribution'], reference.null_distribution)
    assert result['effect_interval'] == [interval.confidence_interval.low, interval.confidence_interval.high]
    assert result['effect_interval'][1] - result['effect_interval'][0] < 3.0

def test_matched_arrays_declaration_and_number_of_actual_pairs_are_required():
    setting = {'method': 'matched_sample_permutation', 'matched_samples': 'Independent original software pairs'}
    with pytest.raises(ValueError):
        native.compare([1, 2], [3], setting, matched=True)
    with pytest.raises(ValueError):
        native.compare([1, np.nan], [3, 4], setting, matched=True)
    with pytest.raises(ValueError):
        native.compare([1, 2], [3, 4], setting, matched=False)
    assert native.compare([1, 2], [3, 4], setting, matched=True)['p_value'] is None
    assert native.compare(range(8), range(8), {'method': 'matched_sample_permutation'}, matched=True)['p_value'] is None

def test_repeat_recordings_have_equal_weight_and_nonresponders_remain(tmp_path):
    _, tables, _, resolved = fixture(tmp_path, extra=True)
    data, settings = data_for(resolved, tables)
    units = data['unit_summaries']
    one = units.loc[units['sample'].eq('reference-sample-0')].iloc[0]
    assert one.value == 6.0 and one.available_recordings == 2 and (one.available_cells == 4)
    assert len(data['unit_inventory']) == 8 and len(data['recording_inventory']) == 9
    assert not data['cell_changes'].within_cell_supported.any() and data['cell_changes'].eligible.all()
    rows, _, _, details = compare_units(data, resolved, settings)
    row = rows[0]
    assert row['reference_samples'] == 4 and row['reference_recordings'] == 5 and (row['reference_cells'] == 7)
    assert row['effect'] == pytest.approx(5.0) and details[0]['native_result']['permutation_count'] == 70

def test_missing_matched_partner_retains_original_identity_and_full_family(tmp_path):
    _, tables, _, resolved = fixture(tmp_path, matched=True, count=8, missing=True)
    data, settings = data_for(resolved, tables)
    rows, members, matches, details = compare_units(data, resolved, settings)
    assert len(matches) == 8 and rows[0]['complete_matches'] == 7 and (len(members) == 16)
    lost = next((row for row in matches if row['match_id'] == 'original-0'))
    assert not lost['complete'] and lost['reference_unit_id'] and lost['target_unit_id']
    assert lost['reference_value'] == 2.0 and lost['target_value'] is None
    assert len(details[0]['actual_matching']) == 7
    rows, families = correct(rows, resolved.request.inference.as_dict())
    assert rows[0]['supported'] and families[0]['evidence_level'] == 'biological_sample'
    assert all((row['population'] == 'All original cells; no within-cell significance filter' for row in data['cell_changes'].to_dict('records')))

def test_matching_cannot_be_ignored_or_invented(tmp_path):
    request, tables, paths, resolved = fixture(tmp_path, matched=True)
    request['controls']['comparisons'][0]['design'] = 'independent'
    request['controls']['evidence'] = {'method': 'independent_sample_permutation', 'independent_samples': 'Independent software units'}
    resolved = resolve(request, tables, paths)
    data, settings = data_for(resolved, tables)
    rows, _, matches, _ = compare_units(data, resolved, settings)
    assert rows[0]['p_value'] is None and 'cannot be ignored' in rows[0]['reason'] and all((not row['included'] for row in matches))
    request['matching'] = []
    request['controls']['comparisons'][0]['design'] = 'matched'
    request['controls']['evidence'] = {'method': 'matched_sample_permutation', 'matched_samples': 'Software pairs'}
    resolved = resolve(request, tables, paths)
    data, settings = data_for(resolved, tables)
    rows, _, matches, _ = compare_units(data, resolved, settings)
    assert rows[0]['p_value'] is None and (not matches) and ('No original biological-sample matching' in rows[0]['reason'])

def test_unconfirmed_samples_and_unavailable_quantities_do_not_disappear(tmp_path):
    request, tables, paths, _ = fixture(tmp_path, unconfirmed=True)
    request['controls']['comparisons'].append({**request['controls']['comparisons'][0], 'name': 'unavailable-model', 'quantity': 'estimate'})
    resolved = resolve(request, tables, paths)
    data, settings = data_for(resolved, tables)
    rows, members, _, _ = compare_units(data, resolved, settings)
    rows, families = correct(rows, resolved.request.inference.as_dict())
    assert len(data['unit_inventory']) == 8 and len(data['unit_summaries']) == 16 and (len(members) == 16)
    assert families[0]['requested'] == 2 and families[0]['tested'] == 0
    assert all((row['p_value'] is None and row['q_value'] is None for row in rows))
    descriptive = data['unit_summaries'].loc[lambda x: ~x.sample_confirmed & x.value.notna()]
    assert len(descriptive) == 1 and (not descriptive.iloc[0].formal_eligible)

def test_incompatible_original_windows_are_not_silently_pooled(tmp_path):
    request, tables, paths, _ = fixture(tmp_path)
    request['recording_windows'] = {'target0': [{'name': 'baseline', 'coordinate': 'relative_hours', 'start': -4, 'end': 0}, {'name': 'followup', 'coordinate': 'relative_hours', 'start': 0, 'end': 4, 'baseline': 'baseline'}]}
    resolved = resolve(request, tables, paths)
    data, settings = data_for(resolved, tables)
    rows, _, _, _ = compare_units(data, resolved, settings)
    assert len(data['definitions']) == 2 and rows[0]['status'] == 'untestable' and (rows[0]['p_value'] is None)
    assert 'incompatible original window' in rows[0]['reason']

def test_native_frame_windows_require_an_established_common_clock(tmp_path):
    request, tables, paths, _ = fixture(tmp_path)
    request['windows'] = [{'name': 'baseline', 'coordinate': 'frames', 'start': 0, 'end': 8}, {'name': 'followup', 'coordinate': 'frames', 'start': 8, 'end': 16, 'baseline': 'baseline'}]
    resolved = resolve(request, tables, paths)
    data, settings = data_for(resolved, tables)
    assert len(data['definitions']) == 1 and data['cell_changes'].eligible.all()
    tables['cell_frame'].loc[tables['cell_frame'].stem.eq('target0'), 'hours'] += 0.5
    tables['cell_frame'].to_csv(paths['cell_frame'], index=False)
    resolved = resolve(request, tables, paths)
    data, settings = data_for(resolved, tables)
    rows, _, _, _ = compare_units(data, resolved, settings)
    assert len(data['definitions']) == 2 and rows[0]['p_value'] is None

def test_untestable_quantity_retains_its_correction_slot(tmp_path):
    request, tables, paths, _ = fixture(tmp_path)
    request['controls']['comparisons'].append({**request['controls']['comparisons'][0], 'name': 'unavailable-model', 'quantity': 'estimate'})
    resolved = resolve(request, tables, paths)
    data, settings = data_for(resolved, tables)
    rows, _, _, _ = compare_units(data, resolved, settings)
    rows, families = correct(rows, resolved.request.inference.as_dict())
    assert families[0]['requested'] == 2 and families[0]['tested'] == 1
    tested = next((row for row in rows if row['p_value'] is not None))
    assert tested['q_value'] == pytest.approx(2 * tested['p_value'])
    assert next((row for row in rows if row['quantity'] == 'estimate'))['q_value'] is None

def test_control_only_settings_reuse_original_windows_and_evidence(tmp_path, monkeypatch):
    request, tables, paths, resolved = fixture(tmp_path, extra=True)
    root = tmp_path / 'pipeline'
    first = run_request(resolved, paths, root, only=['control-comparisons'])
    assert first.successful
    saved = first.results['control-comparisons']
    read_controls(saved, first.results['response-evidence'].outcome.scientific_id)
    originals = {str(path): file_hash(path) for path in paths.values()}
    before = {str(item.artifact(ref.name)): file_hash(item.artifact(ref.name)) for item in first.results.values() for ref in item.outcome.artifacts}
    import pymicroglia.pipelines.intervention.windows as intervention_windows, pymicroglia.pipelines.intervention.evidence as intervention_evidence, pymicroglia.pipelines.intervention.controls as intervention_controls, pymicroglia.pipelines.intervention.control_inputs as intervention_control_inputs

    def forbidden(*a, **k):
        raise AssertionError('Scientific work repeated during saved reuse')
    monkeypatch.setattr(intervention_windows, 'prepare', forbidden)
    monkeypatch.setattr(intervention_evidence, 'analyse', forbidden)
    with monkeypatch.context() as local:
        local.setattr(intervention_control_inputs, 'aggregate', forbidden)
        local.setattr(intervention_controls, 'compare_units', forbidden)
        local.setattr(intervention_controls, 'correct', forbidden)
        reopened = run_request(resolved, paths, root, only=['control-comparisons'], presentation={'report': {'title': 'Changed appearance'}})
        assert reopened.successful and reopened.results['control-comparisons'].outcome.status == 'reused'
    request['controls']['aggregation'] = 'median'
    changed = resolve(request, tables, paths)
    rerun = run_request(changed, paths, root, only=['control-comparisons'])
    assert rerun.successful
    assert rerun.results['aligned-windows'].outcome.status == 'reused' and rerun.results['response-evidence'].outcome.status == 'reused'
    assert rerun.results['control-comparisons'].outcome.scientific_id != saved.outcome.scientific_id
    assert all((file_hash(Path(path)) == value for path, value in {**originals, **before}.items()))
