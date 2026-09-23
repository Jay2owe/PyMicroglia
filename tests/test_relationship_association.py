"""Reference, serial-dependence calibration and full-family relationship evidence."""
from pymicroglia._results import read_document
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import stats
import pymicroglia.measure.relationship_statistics as statistics
from pymicroglia.pipelines.relationships.options import resolve_request, run_request
from pymicroglia.pipelines._screening import file_hash, read_table
from tests.test_relationship_options import request
REFERENCE = read_document(Path(__file__).parent / 'test_data/relationship_tts_reference.json')

def question(**changes):
    return {'enabled': True, 'statistic': 'pearson', 'evidence': {'method': 'truncated_time_shift', 'radius_hours': 19.5, 'stationary_series': 'target', 'stationarity_justification': 'Controlled stationary autoregressive example'}, **changes}

@pytest.mark.parametrize('case', REFERENCE['cases'], ids=lambda case: case['name'])
def test_shift_counts_and_distribution_match_the_published_reference(case):
    result = statistics.truncated_time_shift(case['x'], case['y'], case['radius'])
    assert result['exceedances'] == case['exceedances']
    assert result['unclipped_bound'] == case['bound']
    assert result['p_value'] == min(1.0, case['bound'])
    reference = dict(zip(case['shifts'], case['statistics']))
    np.testing.assert_allclose(result['shift_statistics'], [reference[shift] for shift in result['shifts_observations']], atol=1e-12)
    assert result['tested_observations'] == len(case['x']) - 2 * case['radius']
    assert result['effect_interval'] is None

def test_negative_and_rank_associations_keep_direction_separate_from_evidence():
    x = np.random.default_rng(9).normal(size=160)
    positive = statistics.truncated_time_shift(x, np.exp(x), 39, 'spearman')
    negative = statistics.truncated_time_shift(x, -np.exp(x), 39, 'spearman')
    assert positive['effect'] == pytest.approx(1.0) and negative['effect'] == pytest.approx(-1.0)
    assert positive['p_value'] == negative['p_value'] == 0.025

def test_conservative_test_is_calibrated_on_independent_autocorrelated_processes():
    rng = np.random.default_rng(171)
    trials, alpha = (384, 0.05)
    limits = int(stats.binom.ppf(1 - 0.001 / 2, trials, alpha))
    observed = []
    for statistic in ('pearson', 'spearman'):
        count = 0
        for _ in range(trials):
            values = rng.normal(size=(360, 2))
            for i in range(1, len(values)):
                values[i] += [0.85, 0.65] * values[i - 1]
            x, y = values[200:].T
            count += statistics.truncated_time_shift(x, y, 39, statistic)['p_value'] <= alpha
        observed.append(count)
    assert all((count <= limits for count in observed)), {'false_positives': observed, 'limit': limits, 'trials': trials}

def inputs(tmp_path, req=None):
    cases = {case['name']: case for case in REFERENCE['cases']}
    rows = []
    for movie, identity, name in [('a', 1, 'positive'), ('a', 2, 'negative'), ('b', 1, 'independent'), ('b', 2, 'constant'), ('b', 3, 'gap')]:
        source = cases.get(name, cases['positive'])
        x, y = (np.array(source['x']), np.array(source['y']))
        if name == 'constant':
            y[:] = 3.0
        if name == 'gap':
            x[50] = np.nan
        rows.extend(({'stem': movie, 'identity': identity, 'frame_index': i, 'hours': 50 + i * 0.5, 'corrected_mean': a, 'area_px': b} for i, (a, b) in enumerate(zip(x, y))))
    frame = pd.DataFrame(rows)
    path = tmp_path / 'cell_frame.csv'
    frame.to_csv(path, index=False)
    req = req or request(within_cell=question(), inference={'alpha': 0.1, 'multiple_testing': 'bh', 'correction_scope': 'all'}, biological_samples={'a': 'one', 'b': 'two'})
    resolved = resolve_request(req, source_run='same-source', tables={'cell_frame': frame}, input_hashes={'cell_frame': file_hash(path)})
    return (resolved, {'cell_frame': path})

def test_results_keep_central_effect_missing_tests_and_complete_family(tmp_path, monkeypatch):
    from pymicroglia import workbench as circadian
    monkeypatch.setattr(circadian, 'estimate_one', lambda *a, **k: pytest.fail('Association fitted a rhythm'))
    resolved, paths = inputs(tmp_path)
    execution = run_request(resolved, paths, tmp_path / 'pipeline', only=('within-cell-association',))
    assert execution.successful, {name: saved.outcome.reason for name, saved in execution.results.items()}
    saved = execution.results['within-cell-association']
    rows = read_table(saved.artifact('results'))
    families = read_table(saved.artifact('families'))
    assert len(rows) == 5 and set(rows.family_requested) == {5} and (set(rows.family_tested) == {3})
    assert set(families.requested) == {5} and set(families.unavailable) == {2}
    by_cell = rows.set_index(['movie', 'identity'])
    assert by_cell.loc[('a', 1), 'status'] == 'positive-association'
    assert by_cell.loc[('a', 2), 'status'] == 'negative-association'
    assert by_cell.loc[('b', 1), 'status'] == 'no-detected-association'
    assert by_cell.loc[('b', 2), 'status'] == by_cell.loc[('b', 3), 'status'] == 'untestable'
    assert pd.isna(by_cell.loc[('b', 3), 'p_value']) and pd.isna(by_cell.loc[('b', 3), 'q_value'])
    assert by_cell.loc[('a', 1), 'q_value'] == pytest.approx(0.0625)
    assert by_cell.loc[('a', 1), 'tested_observations'] == 82
    assert by_cell.loc[('a', 1), 'tested_start_hours'] == 69.5
    assert rows.effect_interval.isna().all()
    selected = next((s for s in saved.outcome.selections if s.name == 'association-supported'))
    assert len(selected.members) == 2
    assert all((member['source_run'] == 'same-source' for member in selected.members))
    monkeypatch.setattr(statistics, 'same_time_evidence', lambda *a, **k: pytest.fail('Saved display request repeated a test'))
    monkeypatch.setattr(circadian, 'adjust_pvalues', lambda *a, **k: pytest.fail('Saved display request recorrected a subset'))
    reopened = run_request(resolved, paths, tmp_path / 'pipeline', only=('within-cell-association',), presentation={'columns': 1})
    assert reopened.successful and reopened.results['within-cell-association'].outcome.status == 'reused'
    assert reopened.results['within-cell-association'].outcome.selections == saved.outcome.selections

def test_descriptive_and_disabled_are_never_not_detected(tmp_path):
    resolved, paths = inputs(tmp_path, request())
    execution = run_request(resolved, paths, tmp_path / 'descriptive', only=('within-cell-association',))
    assert execution.successful
    rows = read_table(execution.results['within-cell-association'].artifact('results'))
    assert set(rows.status) == {'descriptive', 'untestable'}
    assert rows.p_value.isna().all()

@pytest.mark.parametrize('changes,reason', [({'evidence': {'method': 'unverified'}}, 'method'), ({'statistic': 'unverified'}, 'statistic'), ({'evidence': {**question()['evidence'], 'stationarity_justification': ''}}, 'justification')])
def test_unsupported_or_unspecified_methods_are_not_silently_selected(changes, reason):
    with pytest.raises(ValueError, match=reason):
        statistics.validate_question(question(**changes))
