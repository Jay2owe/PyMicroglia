"""Original within-cell pairing, complete categories and real sample replication."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from pymicroglia.pipelines import parse
import pymicroglia.pipelines.intervention.patterns as patterns
import pymicroglia.pipelines.intervention.pattern_statistics as native
from pymicroglia.pipelines.intervention.options import resolve_request, run_request
from pymicroglia.pipelines.intervention.windows import prepare
from pymicroglia.pipelines.intervention.evidence import analyse as evidence
from pymicroglia.pipelines._screening import file_hash

def evidence_settings():
    return {'method': 'independent_sample_permutation', 'resamples': 999, 'seed': 29, 'independence_justification': 'Original independent software sample identifiers', 'exchangeability_justification': 'Under the software null the paired sample effects are exchangeable within this one condition'}

def fixture(tmp_path):
    rows = []
    anchors = {}
    mapping = {}
    conditions = {}
    for movie, index, values in [(f'record-{i}', i, [float(i)]) for i in range(1, 9)] + [('repeat-1', 1, [9.0, 11.0])]:
        anchors[movie] = {'hours': 4.0, 'kind': 'intervention', 'label': 'Software event'}
        mapping[movie] = f'sample-{index}'
        conditions[movie] = 'software-condition'
        for cell, change in enumerate(values, 7):
            for frame in range(16):
                post = frame >= 8
                rows.append({'stem': movie, 'identity': cell, 'frame_index': frame, 'hours': frame * 0.5, 'first': 10.0 + post * change, 'second': 20.0 + post * 2 * change, 'opposed': 30.0 - post * 3 * change, 'constant': 1.0, 'missing': np.nan})
    frame = pd.DataFrame(rows)
    tables = {'cell_frame': frame, 'cell_summary': frame[['stem', 'identity']].drop_duplicates()}
    paths = {}
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        paths[name] = tmp_path / (name + '.csv')
        table.to_csv(paths[name], index=False)
    request = {'pipeline': 'intervention-response', 'measurements': ['first', 'second', 'opposed', 'constant', 'missing'], 'summary': 'mean', 'anchors': anchors, 'windows': [{'name': 'baseline', 'coordinate': 'relative_hours', 'start': -4.0, 'end': 0.0}, {'name': 'followup', 'coordinate': 'relative_hours', 'start': 0.0, 'end': 4.0, 'baseline': 'baseline'}], 'support': {'max_gap_hours': 0.75}, 'biological_samples': mapping, 'conditions': conditions, 'coordinated': {'enabled': True, 'pairs': {'mode': 'all'}, 'aggregation': 'mean', 'evidence': evidence_settings()}, 'inference': {'alpha': 0.05, 'multiple_testing': 'bh', 'correction_scope': 'all'}}
    return (request, tables, paths, resolve(request, tables, paths))

def resolve(request, tables, paths):
    return resolve_request(parse([request])[0], source_run='pattern-software', tables=tables, input_hashes={key: file_hash(path) for key, path in paths.items()})

def calculate(resolved, tables):
    prepared = prepare(resolved, tables)
    return patterns.analyse(prepared, evidence(prepared, resolved), resolved, patterns.policy(resolved.request.coordinated.as_dict(), [m.column for m in resolved.measurements], resolved.request.inference.as_dict()))

def test_native_pairing_permutation_matches_spearman_reference_and_keeps_ties():
    a = np.array([1.0, 1.0, 3.0, 4.0, 5.0])
    b = np.array([7.0, 4.0, 3.0, 2.0, 1.0])
    options = evidence_settings()
    actual = native.compare(a, b, list('abcde'), options)
    expected = stats.permutation_test((a,), lambda perm: stats.spearmanr(perm, b).statistic, permutation_type='pairings', alternative='two-sided', vectorized=False, n_resamples=999, random_state=np.random.default_rng(29))
    assert actual['native_permutations'] == 120 and actual['estimate'] == pytest.approx(stats.spearmanr(a, b).statistic)
    assert actual['p_value'] == pytest.approx(expected.pvalue) and np.allclose(actual['null_distribution'], expected.null_distribution)
    assert actual['unit_ids'] == list('abcde')
    with pytest.raises(ValueError, match='uniquely'):
        native.compare(a, b, list('aabcd'), options)
    assert native.compare([1] * 5, b, list('abcde'), options)['p_value'] is None

def test_joint_response_categories_do_not_treat_missing_or_nonsignificant_as_coordination():

    def row(p, supported=False, direction='increase'):
        return {'p_value': p, 'response_supported': supported, 'response_direction': direction}
    assert patterns.category(row(0.001, True), row(0.001, True)) == 'both_supported_same_direction'
    assert patterns.category(row(0.001, True), row(0.001, True, 'decrease')) == 'both_supported_opposite_directions'
    assert patterns.category(row(0.001, True), row(0.8)) == 'reference_supported_only'
    assert patterns.category(row(0.8), row(0.001, True)) == 'target_supported_only'
    assert patterns.category(row(0.8), row(0.8)) == 'neither_has_a_detected_response'
    assert patterns.category(row(None), row(0.8)) == 'reference_inconclusive'
    assert patterns.category(row(0.8), row(None)) == 'target_inconclusive'
    assert patterns.category(row(None), row(None)) == 'both_inconclusive'

def test_complete_pairs_and_equal_recording_samples_retain_missing_and_constant_metrics(tmp_path):
    _, tables, _, resolved = fixture(tmp_path)
    data = calculate(resolved, tables)
    cells = data['cell_pairs']
    samples = data['sample_pairs']
    associations = data['associations']
    assert len(cells) == 100 and cells.pair_id.is_unique and (len(samples) == 80)
    assert set(cells.joint_response) == {'both_inconclusive'} and (not cells.jointly_tested.any())
    both = samples.loc[samples.reference.eq('first') & samples.target.eq('second')].set_index('sample')
    assert len(both) == 8 and both.loc['sample-1', 'requested_recordings'] == 2 and (both.loc['sample-1', 'requested_cells'] == 3)
    assert both.loc['sample-1', 'reference_value'] == pytest.approx(5.5) and both.loc['sample-1', 'target_value'] == pytest.approx(11.0)
    assert sum((len(row) for row in both.cell_pair_ids)) == 10
    positive = associations.loc[associations.reference.eq('first') & associations.target.eq('second')].iloc[0]
    negative = associations.loc[associations.reference.eq('first') & associations.target.eq('opposed')].iloc[0]
    assert positive.estimate == pytest.approx(1.0) and positive.supported and (positive.eligible_samples == 8)
    assert negative.estimate == pytest.approx(-1.0) and negative.supported
    unavailable = associations.loc[associations.reference.isin(['constant', 'missing']) | associations.target.isin(['constant', 'missing'])]
    assert unavailable.p_value.isna().all() and len(associations) == 10 and (data['families'].iloc[0].requested == 10)
    assert set(cells.loc[cells.target.eq('missing'), 'joint_direction']) == {'unavailable'}
    assert samples.both_supported_fraction_of_jointly_tested.isna().all()

def test_pair_modes_use_freely_selected_columns_and_original_cell_keys(tmp_path):
    request, tables, paths, _ = fixture(tmp_path)
    request['coordinated']['pairs'] = {'mode': 'reference', 'reference': 'opposed'}
    result = calculate(resolve(request, tables, paths), tables)
    assert len(result['cell_pairs']) == 40 and set(result['cell_pairs'].reference) == {'opposed'}
    request['coordinated']['pairs'] = {'mode': 'explicit', 'pairs': [['second', 'first']]}
    result = calculate(resolve(request, tables, paths), tables)
    assert len(result['cell_pairs']) == 10 and set(result['cell_pairs'].reference) == {'second'}
    assert result['cell_pairs'].loc[result['cell_pairs'].identity.eq(7), 'movie'].nunique() == 9

def test_unconfirmed_recordings_keep_descriptive_inventory_without_becoming_samples(tmp_path):
    request, tables, paths, _ = fixture(tmp_path)
    request['biological_samples'] = {}
    data = calculate(resolve(request, tables, paths), tables)
    assert len(data['sample_pairs']) == 90 and (not data['sample_pairs'].formal_eligible.any())
    assert data['associations'].eligible_samples.eq(0).all() and data['associations'].p_value.isna().all()
    assert data['sample_pairs'].paired_available.any() and data['families'].iloc[0].requested == 10

def test_control_branch_is_optional_and_pattern_reopening_never_reanalyses(tmp_path, monkeypatch):
    request, tables, paths, resolved = fixture(tmp_path / 'source')
    root = tmp_path / 'pipeline'
    first = run_request(resolved, paths, root, only=['coordinated-responses'])
    assert first.successful
    assert 'control-comparisons' not in first.results
    saved = first.results['coordinated-responses']
    data = patterns.read_patterns(saved, first.results['response-evidence'].outcome.scientific_id)
    assert len(data['cell_pairs']) == 100
    originals = {str(path): file_hash(path) for path in paths.values()}
    artifacts = {str(item.artifact(ref.name)): file_hash(item.artifact(ref.name)) for item in first.results.values() for ref in item.outcome.artifacts}

    def forbidden(*a, **k):
        raise AssertionError('Saved coordinated response reopened scientific analysis')
    import pymicroglia.pipelines.intervention.windows as intervention_windows, pymicroglia.pipelines.intervention.evidence as intervention_evidence
    monkeypatch.setattr(intervention_windows, 'prepare', forbidden)
    monkeypatch.setattr(intervention_evidence, 'analyse', forbidden)
    with monkeypatch.context() as local:
        local.setattr(patterns, 'analyse', forbidden)
        local.setattr(native, 'compare', forbidden)
        second = run_request(resolved, paths, root, only=['coordinated-responses'], presentation={'report': {'title': 'Reopened patterns'}})
        assert second.successful and second.results['coordinated-responses'].outcome.status == 'reused'
    request['coordinated']['aggregation'] = 'median'
    changed = resolve(request, tables, paths)
    third = run_request(changed, paths, root, only=['coordinated-responses'])
    assert third.successful
    assert third.results['aligned-windows'].outcome.status == 'reused' and third.results['response-evidence'].outcome.status == 'reused'
    assert third.results['coordinated-responses'].outcome.scientific_id != saved.outcome.scientific_id
    assert all((file_hash(Path(path)) == digest for path, digest in {**originals, **artifacts}.items()))

def test_disabled_patterns_make_no_association_calls(tmp_path, monkeypatch):
    request, tables, paths, _ = fixture(tmp_path)
    request['coordinated'] = {'enabled': False}
    resolved = resolve(request, tables, paths)

    def forbidden(*a, **k):
        raise AssertionError('Disabled branch calculated associations')
    monkeypatch.setattr(patterns, 'analyse', forbidden)
    monkeypatch.setattr(native, 'compare', forbidden)
    result = run_request(resolved, paths, tmp_path / 'pipeline', only=['coordinated-responses'])
    assert result.successful and result.results['coordinated-responses'].outcome.status == 'skipped-empty'
