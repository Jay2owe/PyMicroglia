"""Exact detection categories and independent-sample limits through the public engine."""
import numpy as np
import pandas as pd
import pytest
from pymicroglia.pipelines import parse
from pymicroglia.pipelines._contracts import Settings, StepSpec, content_id
from pymicroglia.pipelines.rhythm.discovery import resolve_request, run_request
from pymicroglia.pipelines._runner import ExecutionContext
from pymicroglia.pipelines._screening import file_hash, read_table
from tests.test_rhythm_reports import make_screen

def fixture(tmp_path, monkeypatch, *, scope='movie_measurement', mapping=True):

    def transform(frame):
        base = frame[frame.stem.eq('a') & frame.identity.eq(1)].copy()
        parts = []
        for offset, movie in enumerate(('a', 'b', 'c', 'd')):
            for identity, (first, second) in enumerate([(10, 80), (10, 30), (30, 10), (30, 30), (55, 10), (10, 55), (55, 55)] + [(10, 10)] * offset, 1):
                part = base.copy()
                part['stem'], part['identity'] = (movie, identity)
                part['signal'], part['other'] = (first + part.hours / 100, second + part.hours / 100)
                parts.append(part)
        return pd.concat(parts, ignore_index=True)
    original, paths, _ = make_screen(tmp_path / 'run', monkeypatch, transform=transform)
    block = {'pipeline': 'rhythm-discovery', 'test_measurements': ['signal', 'other'], 'analysis_options': original.request.analysis_options.as_dict(), 'correction_scope': scope, 'biological_samples': {'a': 's1', 'b': 's1', 'c': 's2', 'd': 's3'} if mapping else {}, 'detection_agreement': {'sample_method': 'spearman_permutation', 'permutations': 99}}
    resolved = resolve_request(parse([block])[0], source_run=original.inputs.source_run, tables={k: pd.read_csv(p) for k, p in paths.items()}, input_hashes={k: file_hash(p) for k, p in paths.items()})
    execution = run_request(resolved, paths, tmp_path / 'pipeline', only=('detection-agreement',))
    assert execution.successful, {k: v.outcome.reason for k, v in execution.results.items()}
    return (resolved, paths, execution)

def test_all_categories_and_denominators_match_original_calls(tmp_path, monkeypatch):
    _, _, execution = fixture(tmp_path, monkeypatch)
    result = execution.results['detection-agreement']
    pairs, summary, units = [read_table(result.artifact(name)) for name in ('pairs', 'summary', 'units')]
    assert len(pairs) == 34
    assert pairs.outcome.value_counts().to_dict() == {'both': 10, 'first_only': 4, 'second_only': 4, 'neither': 4, 'first_missing': 4, 'second_missing': 4, 'both_missing': 4}
    row = summary.iloc[0]
    assert row.requested == 34 and row.joint_tested == 22
    assert row.first_tested == row.second_tested == 26
    assert row.unit_total_units == row.unit_eligible_units == 3
    primary = units[units.scope.eq('sample_or_recording')]
    assert len(primary) == 3 and primary[primary.unit_label.eq('s1')].requested.iloc[0] == 15
    assert row.unit_status == 'available' and row.test_status == 'available' and (row.test_units == 3)
    assert row.test_statistic == pytest.approx(1) and row.q_value == pytest.approx(1 / 3)
    unresolved = pairs[pairs.identity.eq(1)]
    assert unresolved.outcome.eq('both').all() and unresolved.second_period_underdetermined.all()
    selections = {s.name.split(':')[0]: s for s in result.outcome.selections}
    assert sum((len(s.members) for s in selections.values())) == 34
    assert len(selections['both'].members) == 10

@pytest.mark.parametrize('scope,mapping,reason', [('all', True, 'correction families'), ('movie_measurement', False, 'sample mapping')])
def test_unconfirmed_or_coupled_samples_cannot_supply_independent_inference(tmp_path, monkeypatch, scope, mapping, reason):
    _, _, execution = fixture(tmp_path, monkeypatch, scope=scope, mapping=mapping)
    row = read_table(execution.results['detection-agreement'].artifact('summary')).iloc[0]
    assert not row.independent_units
    assert row.unit_status == row.test_status == 'unavailable'
    assert reason in row.unit_reason and pd.isna(row.q_value) and pd.isna(row.unit_lower)
    assert np.isfinite(row.kappa) and np.isfinite(row.unit_mean)

def test_empty_pair_choice_reuses_screen_and_completes_without_figures(tmp_path, monkeypatch):
    from pymicroglia import workbench as circadian
    resolved, paths, execution = fixture(tmp_path, monkeypatch)
    request = resolved.request.declaration.as_dict()
    request['pairs'] = {'mode': 'explicit', 'pairs': []}
    changed = resolve_request(parse([request])[0], source_run=resolved.inputs.source_run, tables={k: pd.read_csv(p) for k, p in paths.items()}, input_hashes={k: file_hash(p) for k, p in paths.items()})

    def forbidden(*a, **k):
        raise AssertionError('No new rhythm fits')
    monkeypatch.setattr(circadian, 'estimate_one', forbidden)
    result = run_request(changed, paths, tmp_path / 'pipeline', only=('detection-agreement-figures',))
    assert result.successful, {k: v.outcome.reason for k, v in result.results.items()}
    assert result.results['rhythm-screen'].outcome.status == 'reused'
    assert read_table(result.results['detection-agreement'].artifact('pairs')).empty
    assert not list(result.results['detection-agreement-figures'].root.glob('*.svg'))

def test_renderer_pages_all_units_without_statistics_and_reopens_saved_binding(tmp_path, monkeypatch):
    from pymicroglia import workbench as circadian
    from pymicroglia.pipelines.rhythm.agreement_figures import produce
    resolved, paths, execution = fixture(tmp_path, monkeypatch)
    import matplotlib.pyplot as plt

    def forbidden(*a, **k):
        raise AssertionError('Saved agreement rendering attempted analysis')
    for name in ('estimate_one', 'adjust_pvalues', 'rhythm_detection_agreement'):
        monkeypatch.setattr(circadian, name, forbidden)
    from tests.panel_helpers import capture_pages, reopen_page
    captured = capture_pages(monkeypatch)
    appearance = {'detection_agreement': {'overview_rows': 1}}
    context = ExecutionContext(StepSpec('detection-agreement-figures', 'detection-agreement-figures', ('detection-agreement',), kind='render'), resolved, Settings(), paths, {'detection-agreement': execution.results['detection-agreement']}, None, tmp_path / 'render/renders/agreement/id/inv', 'saved-agreement', Settings(appearance), content_id(appearance))
    assert produce(context).status == 'completed'
    assert len(captured) == 4
    assert sum((len(r.auxiliary.get('units.csv', [])) for ctx, r in captured)) == 3
    assert all((len(r.auxiliary['members.csv']) == 34 for ctx, r in captured))
    ctx, result = captured[-1]
    reopened = reopen_page(ctx)
    assert reopened.figure_data.equals(result.figure_data)

def test_gateway_forwards_the_complete_public_request(monkeypatch):
    from pymicroglia import workbench as circadian
    from types import SimpleNamespace
    captured = {}

    def run(records, **settings):
        captured.update(records=records, **settings)
        return SimpleNamespace(data={'units': []}, run_record={'proof': 'public call'})
    monkeypatch.setattr(circadian.cw, 'detection_agreement', run)
    result = circadian.rhythm_detection_agreement([], independent_units=True, confidence=0.9, sample_method='spearman_permutation', permutations=199, seed=4)
    assert captured == {'records': [], 'independent_units': True, 'confidence': 0.9, 'sample_method': 'spearman_permutation', 'permutations': 199, 'seed': 4}
    assert result['workbench_run_record']['proof'] == 'public call'
