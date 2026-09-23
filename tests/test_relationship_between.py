"""Between-cell summaries respect recording identity and independent samples."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from scipy import stats
import pymicroglia.measure.relationship_population_statistics as statistics
from pymicroglia.pipelines.relationships.options import resolve_request, run_request
from pymicroglia.pipelines._screening import file_hash, read_table
from tests.test_relationship_options import request

def question(**changes):
    return {'enabled': True, 'statistic': 'pearson', 'experimental_unit': 'biological_sample', 'aggregation': 'mean', 'evidence': {'method': 'sample_permutation', 'resamples': 1999, 'seed': 71, 'min_samples': 3, 'exchangeability_justification': 'Independent samples drawn from a common controlled population', 'interval': {'method': 'paired_bootstrap', 'confidence': 0.9, 'resamples': 1000, 'seed': 73}}, **changes}

def declaration(**changes):
    return request(**{'measurements': [{'column': 'corrected_mean', 'summary': 'mean'}, {'column': 'area_px', 'summary': 'mean'}], 'within_cell': {'enabled': False}, 'between_cells': question(), 'inference': {'alpha': 0.05, 'multiple_testing': 'bh', 'correction_scope': 'all'}, **changes})

def inputs(tmp_path, req=None, frame=None, samples=12):
    if frame is None:
        frame = pd.DataFrame([{'stem': f'm{s:02d}_{movie}', 'identity': cell, 'hours': 50.0 + i, 'frame_index': i, 'corrected_mean': float(s) + cell * 0.1, 'area_px': float(s) * 2 + cell * 0.1} for s in range(samples) for movie in range(2) for cell in (1, 2) for i in range(6)])
    path = tmp_path / 'cell_frame.csv'
    frame.to_csv(path, index=False)
    mapping = {movie: movie.split('_')[0] for movie in frame.stem.unique()}
    req = req or declaration(biological_samples=mapping)
    resolved = resolve_request(req, source_run='source-one', tables={'cell_frame': frame}, input_hashes={'cell_frame': file_hash(path)})
    return (resolved, {'cell_frame': path}, frame)

def execute(resolved, paths, output, only=('between-cell-association',)):
    result = run_request(resolved, paths, output, only=only)
    assert result.successful, {k: v.outcome.reason for k, v in result.results.items()}
    return result

@pytest.mark.parametrize('statistic', ['pearson', 'spearman'])
def test_native_pairing_permutation_and_paired_bootstrap_agree(statistic):
    rng = np.random.default_rng(41)
    x = rng.normal(size=12)
    y = 0.8 * x + 0.2 * rng.normal(size=12)
    q = question(statistic=statistic)
    result = statistics.sample_association(x, y, q)
    fn = (lambda a, b: stats.pearsonr(a, b).statistic) if statistic == 'pearson' else lambda a, b: stats.spearmanr(a, b).statistic
    native = stats.permutation_test((x,), lambda a: abs(fn(a, y)), permutation_type='pairings', alternative='greater', n_resamples=1999, vectorized=False, random_state=np.random.default_rng(71))
    np.testing.assert_allclose(result['null_distribution'], native.null_distribution, atol=1e-12)
    assert result['p_value'] == native.pvalue
    bootstrap = stats.bootstrap((x, y), fn, paired=True, vectorized=False, method='percentile', confidence_level=0.9, n_resamples=1000, random_state=np.random.default_rng(73))
    np.testing.assert_allclose(result['effect_interval'], bootstrap.confidence_interval, atol=1e-12)

def test_exhaustive_small_sample_test_preserves_undefined_interval():
    q = question()
    q['evidence']['resamples'] = 100
    x = np.arange(3.0)
    result = statistics.sample_association(x, x, q)
    assert result['exact'] and result['permutation_count'] == 6 and (result['p_value'] == pytest.approx(2 / 6))
    assert result['interval_status'] == 'unavailable' and result['effect_interval'] is None
    assert result['invalid_bootstrap_replicates'] > 0

def test_permutation_evidence_calibration_for_independent_biological_samples():
    rng, trials, alpha = (np.random.default_rng(20260913), 192, 0.05)
    q = question()
    q['evidence'].update(resamples=199, interval={'method': 'none'})
    rejected = 0
    for index in range(trials):
        x, y = rng.normal(size=(2, 12))
        q['evidence']['seed'] = index
        rejected += statistics.sample_association(x, y, q)['p_value'] <= alpha
    assert rejected <= stats.binom.ppf(0.999, trials, alpha), {'rejected': rejected, 'trials': trials}

def test_cell_means_and_within_cell_changes_answer_distinct_questions(tmp_path, monkeypatch):
    from pymicroglia import workbench as circadian
    monkeypatch.setattr(circadian, 'estimate_one', lambda *a, **k: pytest.fail('Unexpected rhythm fit'))
    req = declaration(within_cell={'enabled': True, 'statistic': 'pearson', 'evidence': {'method': 'none'}}, biological_samples={f'm{s:02d}_{movie}': f'sample-{s}' for s in range(12) for movie in range(2)})
    resolved, paths, _ = inputs(tmp_path, req)
    result = execute(resolved, paths, tmp_path / 'pipeline', only=('within-cell-association', 'between-cell-association'))
    within = read_table(result.results['within-cell-association'].artifact('results'))
    assert len(within) == 48 and within.status.eq('untestable').all()
    saved = result.results['between-cell-association']
    rows = read_table(saved.artifact('results'))
    units = read_table(saved.artifact('units'))
    pairs = read_table(saved.artifact('paired_scalars'))
    assert rows.iloc[0].effect == pytest.approx(1.0) and rows.iloc[0].status == 'positive-association'
    assert set(rows.reference) == set(pairs.reference) == set(units.reference) == {resolved.request.pairs[0].reference}
    assert all((member['reference'] == resolved.request.pairs[0].reference for selection in saved.outcome.selections for member in selection.members))
    assert rows.iloc[0].cells_requested == 48 and rows.iloc[0].recordings_requested == 24 and (rows.iloc[0].experimental_units == 12)
    assert len(units) == 12 and units.cells.eq(4).all() and (len(pairs) == 48)
    assert all((len(members) == 4 for members in units.members))
    monkeypatch.setattr(statistics, 'sample_association', lambda *a, **k: pytest.fail('Reopening retested an association'))
    reopened = run_request(resolved, paths, tmp_path / 'pipeline', only=('between-cell-association',), presentation={'report_cells': [1]})
    assert reopened.successful and reopened.results['between-cell-association'].outcome.status == 'reused'
    assert reopened.results['between-cell-association'].outcome.selections == saved.outcome.selections

@pytest.mark.parametrize('mode', ['unconfirmed', 'partial', 'cell_unit', 'few_samples', 'constant', 'missing'])
def test_unavailable_biological_evidence_keeps_scalar_population(tmp_path, mode):
    resolved, paths, frame = inputs(tmp_path)
    raw = resolved.request.declaration.as_dict()
    if mode == 'unconfirmed':
        raw['biological_samples'] = {}
    if mode == 'partial':
        raw['biological_samples'].pop('m00_0')
    if mode == 'cell_unit':
        raw['between_cells']['experimental_unit'] = 'cell'
        raw['between_cells'].pop('aggregation')
    if mode == 'few_samples':
        raw['biological_samples'] = {movie: 'same-sample' for movie in frame.stem.unique()}
    if mode == 'constant':
        frame['area_px'] = 3.0
    if mode == 'missing':
        frame['area_px'] = np.nan
    from pymicroglia.pipelines import parse
    resolved, paths, _ = inputs(tmp_path, parse([raw])[0], frame)
    result = execute(resolved, paths, tmp_path / 'pipeline')
    saved = result.results['between-cell-association']
    rows = read_table(saved.artifact('results'))
    pairs = read_table(saved.artifact('paired_scalars'))
    assert rows.iloc[0].status == 'untestable' and pd.isna(rows.iloc[0].p_value) and pd.isna(rows.iloc[0].q_value)
    assert len(pairs) == 48
    if mode not in {'constant', 'missing'}:
        assert rows.iloc[0].cell_effect > 0.99
    if mode == 'few_samples':
        assert rows.iloc[0].experimental_units == 1

def test_summary_and_range_changes_use_new_original_observations(tmp_path):
    resolved, paths, frame = inputs(tmp_path)
    frame.loc[frame.frame_index.eq(5), 'corrected_mean'] += 100
    resolved, paths, _ = inputs(tmp_path, frame=frame)
    original = execute(resolved, paths, tmp_path / 'pipeline').results['between-cell-association']
    from pymicroglia.pipelines import parse
    raw = resolved.request.declaration.as_dict()
    raw['time_range_hours'] = [50.0, 55.0]
    changed, paths, _ = inputs(tmp_path, parse([raw])[0], frame)
    truncated = execute(changed, paths, tmp_path / 'pipeline').results['between-cell-association']
    assert original.outcome.scientific_id != truncated.outcome.scientific_id
    a = read_table(original.artifact('cell_scalars'))
    b = read_table(truncated.artifact('cell_scalars'))
    a = a.loc[a.measurement.eq('corrected_mean')].iloc[0]
    b = b.loc[b.measurement.eq('corrected_mean')].iloc[0]
    assert a.value - b.value == pytest.approx(100 / 6) and a.summary_id != b.summary_id
    raw.pop('time_range_hours')
    raw['measurements'][0]['summary'] = 'median'
    median, paths, _ = inputs(tmp_path, parse([raw])[0], frame)
    assert median.scientific_id != resolved.scientific_id

def test_saved_scalar_and_trace_summaries_keep_their_distinct_grains(tmp_path):
    from pymicroglia.pipelines import parse
    frame = pd.DataFrame({'stem': ['a', 'b', 'c'], 'identity': [1, 1, 1], 'custom_scalar': [1.0, 2.0, 3.0], 'second_scalar': [2.0, 4.0, 6.0]})
    path = tmp_path / 'cell_summary.csv'
    frame.to_csv(path, index=False)
    req = request(measurements=['custom_scalar', 'second_scalar'], within_cell={'enabled': False}, between_cells=question(experimental_unit='cell', aggregation=None, evidence={'method': 'none'}))
    resolved = resolve_request(req, source_run='scalars', tables={'cell_summary': frame}, input_hashes={'cell_summary': file_hash(path)})
    result = execute(resolved, {'cell_summary': path}, tmp_path / 'pipeline')
    saved = result.results['between-cell-association']
    assert read_table(saved.artifact('cell_scalars')).kind.eq('scalar').all()
    assert read_table(saved.artifact('results')).iloc[0].effect == pytest.approx(1.0)
    raw = req.declaration.as_dict()
    raw['time_range_hours'] = [0, 2]
    with pytest.raises(ValueError, match='whole-recording scalar'):
        resolve_request(parse([raw])[0], source_run='scalars', tables={'cell_summary': frame}, input_hashes={'cell_summary': file_hash(path)})

def test_disabled_between_question_produces_complete_empty_outputs(tmp_path):
    resolved, paths, frame = inputs(tmp_path, request())
    result = execute(resolved, paths, tmp_path / 'pipeline')
    saved = result.results['between-cell-association']
    assert read_table(saved.artifact('results')).status.eq('disabled').all()
    assert read_table(saved.artifact('cell_scalars')).empty

def test_missing_pair_and_poorly_covered_cell_remain_in_requested_accounting(tmp_path):
    resolved, paths, frame = inputs(tmp_path)
    frame['custom_measurement'] = np.nan
    frame.loc[frame.stem.eq('m00_0') & frame.identity.eq(1) & frame.frame_index.gt(0), 'area_px'] = np.nan
    raw = resolved.request.declaration.as_dict()
    raw['measurements'].append({'column': 'custom_measurement', 'summary': 'mean'})
    from pymicroglia.pipelines import parse
    resolved, paths, _ = inputs(tmp_path, parse([raw])[0], frame)
    saved = execute(resolved, paths, tmp_path / 'pipeline').results['between-cell-association']
    rows, pairs, families = [read_table(saved.artifact(name)) for name in ('results', 'paired_scalars', 'families')]
    assert len(rows) == 3 and set(rows.family_requested) == {3} and (set(rows.family_tested) == {1})
    assert rows.p_value.isna().sum() == 2 and rows.q_value.isna().sum() == 2
    assert families.iloc[0].unavailable == 2 and len(pairs) == 144
    valid = rows.loc[rows.p_value.notna()].iloc[0]
    assert valid.cells_requested == 48 and valid.cells_eligible == 47 and (valid.q_value == pytest.approx(valid.p_value * 3))
