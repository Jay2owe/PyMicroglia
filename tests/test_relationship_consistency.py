"""Saved relationship heterogeneity, sample identity and exact sign evidence."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from scipy import stats
import pymicroglia.measure.relationship_consistency_statistics as statistics
import pymicroglia.pipelines.relationships.consistency as producer
from pymicroglia.pipelines._contracts import ArtifactRef, StepResult, content_id
from pymicroglia.pipelines._runner import SavedResult
from pymicroglia.pipelines._screening import file_hash, read_table, write_table
from tests.test_relationship_options import request, lag
from tests.test_relationship_between import inputs

def settings(**changes):
    return {'enabled': True, 'aggregation': 'mean', 'evidence': {'method': 'sample_sign_test', 'min_samples': 3, 'independence_justification': 'Controlled independent biological samples with fixed measurement procedures'}, **changes}

def context(tmp_path, changes=None, unconfirmed=False):
    movies = {f'm{s}_{i}': f's{s}' for s in range(6) for i in range(2)}
    if unconfirmed:
        movies.pop('m0_0')
    req = request(sample_summary=settings(), lag=lag(), biological_samples=movies, inference={'alpha': 0.05, 'multiple_testing': 'bh', 'correction_scope': 'all'})
    frame = pd.DataFrame([{'stem': f'm{s}_{j}', 'identity': i, 'frame_index': k, 'hours': 50.0 + k, 'corrected_mean': k + i, 'area_px': 2 * k + i} for s in range(6) for j in range(2) for i in (1, 2, 3) for k in range(6)])
    resolved, _, _ = inputs(tmp_path, req, frame)
    pair = req.pairs[0]
    within = []
    delayed = []
    profiles = []
    for cell in resolved.inputs.cells:
        i = cell.identity
        sample = next((s for s in resolved.inputs.samples if s.movie == cell.movie))
        key = {**cell.as_dict(), 'pair_id': pair.record_id, **pair.as_dict(), 'sample': sample.sample, 'sample_confirmed': sample.confirmed}
        effect = {1: 0.8, 2: -0.4, 3: 0.2}[i]
        base = {**key, 'effect': 0.99, 'full_overlap_effect': effect, 'support_status': 'eligible', 'status': 'positive-association' if i == 1 else 'negative-association' if i == 2 else 'no-detected-association', 'p_value': 0.01 if i != 3 else 0.8, 'q_value': 0.02 if i != 3 else 0.9, 'significant': i != 3}
        within.append(base)
        delayed.append({**key, 'effect': effect, 'status': 'positive-association' if i == 1 else 'no-detected-association', 'p_value': 0.01 if i == 1 else 0.8, 'q_value': 0.02 if i == 1 else 0.9, 'significant': i == 1, 'delay_supported': i == 1, 'delay_hours': -1.0 if i == 1 else None, 'delay_interval_hours': [-1.25, -0.75] if i == 1 else None, 'resolution_status': 'resolved' if i == 1 else 'broad', 'association_sign': 'positive' if effect > 0 else 'negative'})
        for delay in np.arange(-2, 2.1, 0.5):
            profiles.append({**key, 'lag_hours': delay, 'effect': effect if delay == -1 else effect / 10, 'status': 'descriptive'})
    if changes:
        changes(within, delayed, profiles)
    saved = {}
    for name, tables in {'within-cell-association': {'results': pd.DataFrame(within)}, 'lag-association': {'results': pd.DataFrame(delayed), 'profiles': pd.DataFrame(profiles)}}.items():
        folder = tmp_path / name
        folder.mkdir()
        refs = []
        sid = content_id(name)
        for label, table in tables.items():
            path = folder / (label + '.json')
            path = write_table(path, table)
            refs.append(ArtifactRef(label, path.name, file_hash(path), sid, columns=tuple(table.columns)))
        saved[name] = SavedResult(folder, StepResult(name, sid, 'completed', 'Controlled saved fixture', tuple(refs)))
    return SimpleNamespace(request=resolved, saved=lambda name: saved[name], scientific_id=content_id('summary'), output=tmp_path / 'summary', step=SimpleNamespace(name='sample-consistency'))

def run(ctx):
    outcome = producer.produce(ctx)
    saved = SavedResult(ctx.output, outcome)
    return (saved, {name: read_table(saved.artifact(name)) for name in ('members', 'summaries', 'units', 'families')})

def test_exact_sign_evidence_matches_public_scipy_reference_and_keeps_ties():
    values = [0.2, 0.8, 0.3, -0.4, 0.0, 0.0]
    result = statistics.sign_evidence(values, settings(), 0.05)
    native = stats.binomtest(3, 4, p=0.5, alternative='two-sided')
    ci = native.proportion_ci(0.95, method='exact')
    assert result['p_value'] == native.pvalue and result['positive_fraction_interval'] == [ci.low, ci.high]
    assert result['tied_samples'] == 2 and result['tested_samples'] == 4 and (result['positive_fraction'] == 0.75)
    missing = statistics.sign_evidence([0.0, 0.2], settings(), 0.05)
    assert missing['status'] == 'untestable' and missing['p_value'] is None

def test_all_eligible_effects_keep_heterogeneity_and_equal_sample_weight(tmp_path, monkeypatch):
    from pymicroglia import workbench as circadian
    import pymicroglia.measure.relationship_lag_statistics as relationship_lag_statistics
    import pymicroglia.measure.relationship_statistics as relationship_statistics
    monkeypatch.setattr(circadian, 'detrend_trace', lambda *a, **k: pytest.fail('Repeated preprocessing'))
    monkeypatch.setattr(circadian, 'estimate_one', lambda *a, **k: pytest.fail('Unexpected rhythm fit'))
    monkeypatch.setattr(relationship_lag_statistics, 'evaluate', lambda *a, **k: pytest.fail('Repeated delay search'))
    monkeypatch.setattr(relationship_statistics, 'same_time_evidence', lambda *a, **k: pytest.fail('Repeated association test'))
    saved, data = run(context(tmp_path))
    members = data['members']
    within = members.loc[members.question.eq('within_cell')]
    assert len(members) == 72 and len(within) == 36 and within.eligible.all()
    assert set(within.effect) == {0.8, -0.4, 0.2} and within.original_status.eq('no-detected-association').sum() == 12
    units = data['units'].loc[data['units'].question.eq('within_cell')]
    assert len(units) == 6 and units.requested_cells.eq(6).all() and units.requested_recordings.eq(2).all()
    np.testing.assert_allclose(units.effect, 0.2)
    summary = data['summaries'].loc[data['summaries'].question.eq('within_cell') & data['summaries'].level.eq('across_samples')].iloc[0]
    assert summary.effect == pytest.approx(0.2) and summary.mixed_directions and (summary.negative_cells == 12)
    assert summary.positive_fraction == 1.0 and summary.positive_fraction_interval[0] > 0.5
    assert summary.status == 'positive-consistency' and summary.tested_samples == 6
    assert summary.reference == within.reference.iloc[0] and summary.target == within.target.iloc[0]
    assert summary.family_requested == 2 and len(summary.members) == 36
    assert all((m['source_run'] == 'source-one' and m['source_scientific_id'] for m in summary.members))
    assert len(read_table(saved.artifact('members'))) == 72

def test_delays_exclude_unresolved_and_withhold_incompatible_consensus(tmp_path):

    def change(within, delayed, profiles):
        delayed[0].update(delay_hours=1.0, delay_interval_hours=[0.75, 1.25])
        delayed[1].update(significant=True, resolution_status='broad', delay_supported=False, delay_hours=None)
    _, data = run(context(tmp_path, change))
    across = data['summaries'].loc[data['summaries'].question.eq('lag') & data['summaries'].level.eq('across_samples')].iloc[0]
    assert across.delay_summary_status == 'incompatible' and pd.isna(across.delay_summary_hours)
    assert across.delay_supported_cells == 12 and across.delay_unresolved_cells == 1
    assert 0.0 not in across.delay_values_hours and len(across.delay_values_hours) == 12

def test_missing_mapping_preserves_recording_descriptions_and_withholds_sample_test(tmp_path):
    _, data = run(context(tmp_path, unconfirmed=True))
    across = data['summaries'].loc[data['summaries'].level.eq('across_samples')]
    assert across.status.eq('untestable').all() and across.p_value.isna().all()
    assert across.unconfirmed_eligible_cells.eq(3).all()
    assert len(data['summaries'].loc[data['summaries'].level.eq('recording')]) == 24

def test_inference_uses_sign_frequency_not_the_sign_of_the_mean_coefficient(tmp_path):

    def change(within, delayed, profiles):
        for row in within:
            row['full_overlap_effect'] = 0.8 if row['movie'].startswith('m0_') else -0.01
    _, data = run(context(tmp_path, change))
    row = data['summaries'].loc[data['summaries'].level.eq('across_samples') & data['summaries'].question.eq('within_cell')].iloc[0]
    assert row.effect > 0 and row.positive_fraction == pytest.approx(1 / 6)
    assert row.p_value == stats.binomtest(1, 6).pvalue
    assert row.status == 'no-detected-consistency'

def test_unavailable_effects_are_retained_and_disabled_summary_is_explicit(tmp_path):

    def change(within, delayed, profiles):
        within[0].update(full_overlap_effect=None, status='untestable', p_value=None, q_value=None, significant=False)
    ctx = context(tmp_path, change)
    _, data = run(ctx)
    row = data['members'].loc[data['members'].question.eq('within_cell')].iloc[0]
    assert not row.eligible and pd.isna(row.effect) and (row.original_status == 'untestable')
    from dataclasses import replace
    from pymicroglia.pipelines._contracts import Settings
    ctx.request = replace(ctx.request, request=replace(ctx.request.request, sample_summary=Settings({'enabled': False})))
    ctx.output = tmp_path / 'disabled'
    _, disabled = run(ctx)
    assert len(disabled['members']) == 72 and disabled['summaries'].empty

def test_invalid_consistency_models_are_rejected_before_analysis():
    bad = settings()
    bad['evidence']['method'] = 'pooled_frame_significance'
    with pytest.raises(ValueError, match='sample_sign_test'):
        statistics.validate(bad)
