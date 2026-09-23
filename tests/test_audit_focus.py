"""Focused pages preserve complete saved evidence and deterministic case identity."""
import copy
import json
import pandas as pd
import pytest
from pymicroglia import workbench as circadian
from pymicroglia.pipelines.audit.focus import DEFAULTS, select_pages, page_records
from tests.test_audit_benchmarks import fixture

def example(resolved):
    recipe = resolved.candidates[0].as_dict()
    alternatives = [dict(recipe, candidate_id='recipe-a'), dict(recipe, candidate_id='recipe-b')]
    results, traces = ([], [])
    for movie in ('first', 'second'):
        for cell in (1, 2, 3):
            case = f'{movie}/{cell}/signal'
            for candidate in ('recipe-a', 'recipe-b'):
                row = {'case_id': case, 'profile_id': case, 'candidate_id': candidate, 'family_id': candidate + '/family', 'measurement': 'signal', 'movie': movie, 'identity': cell, 'period_available': True, 'period_underdetermined': False, 'period_hours': 12.0 if candidate == 'recipe-a' else 13.0, 'estimate_status': 'ok', 'test_status': 'ok', 'p_value': 0.001, 'q_value': 0.006, 'status': 'significant', 'significant': True, 'reason': '', 'estimate_reason': '', 'observations': 3, 'input_observations': 3, 'estimate_result': {'diagnostics': {}, 'workbench_version': resolved.source.workbench_version}, 'components': []}
                if cell == 2 and candidate == 'recipe-a':
                    row.update(estimate_status='failed', period_available=False, period_underdetermined=True, period_hours=None, estimate_reason='Illustrative estimator failure')
                if cell == 3 and candidate == 'recipe-a':
                    row.update(test_status='insufficient', status='untestable', p_value=None, q_value=None, significant=False, reason='Illustrative insufficient data')
                results.append(row)
                traces.append({**{k: row[k] for k in ('case_id', 'candidate_id')}, 'hours': [0.0, 1.0, 2.0, 9.0], 'raw': [1.0, None, 2.0, 1.0], 'filtered': [1.0, None, 2.0, 1.0], 'processed_trace': {'hours': [0.0, 2.0, 9.0], 'values': [1.0, 2.0, 1.0]}, 'processing_source_method': 'lomb', 'processing_reason': 'Saved independent detrending diagnostic', 'native_series': {}})
    selection = {'selection_id': 'frozen', 'candidates': alternatives, 'decisions': [{'candidate_ids': ['recipe-b']}]}
    return (pd.DataFrame(results), pd.DataFrame(traces), selection)

def test_seeded_reasons_keep_movies_and_complete_candidate_continuations(fixture):
    results, traces, selection = example(fixture[0])
    options = {**DEFAULTS, 'audit_page_size': 1}
    first = select_pages(results, pd.DataFrame(), selection, options)
    second = select_pages(results.sample(frac=1, random_state=3), pd.DataFrame(), selection, options)
    assert first == second
    assert all((page['population_cases'] == 6 for page in first))
    assert first[0]['candidate_ids'] == ['recipe-b']
    assert {p['candidate_ids'][0] for p in first} == {'recipe-a', 'recipe-b'}
    manual = select_pages(results, pd.DataFrame(), selection, {**DEFAULTS, 'audit_case_ids': ['first/1/signal', 'second/1/signal']})
    assert len(manual) == 2 and {p['movie'] for p in manual} == {'first', 'second'}
    assert all(('manual inspection' in p['reasons'] for p in manual))

def test_focused_records_keep_failure_branches_and_missing_coordinates(fixture, monkeypatch):
    results, traces, selection = example(fixture[0])
    from pymicroglia.visualisation.panels import rhythm_audit
    for name in ('estimate_one', 'adjust_pvalues', 'generate_benchmark_cases', 'benchmark_score_interval', 'detrend_values'):
        if hasattr(circadian, name):
            monkeypatch.setattr(circadian, name, lambda *a, **k: pytest.fail('render called science'))
    options = {**DEFAULTS, 'audit_case_ids': ['first/2/signal', 'first/3/signal']}
    for page in select_pages(results, pd.DataFrame(), selection, options):
        points, evidence = page_records(page, results, traces, selection['candidates'])
        assert points.query("panel == 'input' and series == 'raw'").y.isna().sum() == 2
        assert set(points.query("panel == 'processed'").x) == {0, 2, 9}
        from pymicroglia.figure_tables.focused_display import prepare
        fig, axes = rhythm_audit.draw_saved(prepare(points,evidence), page, title='Illustrative states', footnote='No saved file', canvas=__import__('matplotlib.pyplot', fromlist=['figure']).figure())
        text = ' '.join((item.get_text() for ax in axes for item in ax.texts))
        assert 'independent test lomb' in text
        if page['identity'] == 2:
            assert 'estimate failed' in text and 'q=0.006' in text and ('detected' in text)
        else:
            assert 'insufficient' in text and '12 h' in text
        import matplotlib.pyplot as plt
        plt.close(fig)

def test_native_curve_is_only_a_saved_public_series(fixture):
    results, traces, selection = example(fixture[0])
    trace = traces.iloc[0].to_dict()
    trace['native_series'] = {'native_fit': {'x': [0.0, 2.0, 9.0], 'y': [1.0, 2.0, 1.0], 'x_unit': 'hours', 'y_unit': 'intensity'}}
    traces.at[0, 'native_series'] = trace['native_series']
    page = select_pages(results, pd.DataFrame(), selection, {**DEFAULTS, 'audit_case_ids': ['first/1/signal']})[0]
    points, evidence = page_records(page, results, traces, selection['candidates'])
    assert points.query("panel == 'native'").x.to_list() == [0, 2, 9]
    assert evidence.native_curve_count.sum() == 1
    with pytest.raises(ValueError, match='Unknown saved case'):
        select_pages(results, pd.DataFrame(), selection, {**DEFAULTS, 'audit_case_ids': ['absent']})

def test_canonical_annotation_retains_valid_test_after_estimator_failure():
    from pymicroglia.visualisation.panels import rhythm_audit
    row = pd.Series({'estimate_status': 'failed', 'estimated_period_hours': None, 'p_value': 0.001, 'q_value': 0.004, 'correction_method': 'fdr_bh', 'rhythm_status': 'significant', 'period_underdetermined': False})
    text = rhythm_audit._significance_text(row, {})
    assert 'estimate unavailable' in text and 'q=0.004' in text and ('significant' in text)

def test_producer_honours_manual_cells_candidate_chunks_and_page_numbers(fixture, tmp_path, monkeypatch):
    from pymicroglia.pipelines.audit.focus import produce_focused_pages
    from pymicroglia.pipelines._runner import SavedResult, ExecutionContext
    from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, result_to_dict, content_id
    from pymicroglia.pipelines._screening import _write_json, write_table, file_hash
    from pymicroglia.pipelines.audit.workflow import RECIPE
    results, traces, selection = example(fixture[0])
    saved = {}
    definitions = {'real-candidates': {'results': results, 'traces': traces}, 'real-stability': {'stability_pairs': pd.DataFrame()}, 'candidate-shortlist': {'frozen_selection': selection}, 'independent-confirmation': {'confirmation_record': {'selection_id': 'frozen', 'confirmation_id': 'illustrative'}}}
    for step, artifacts in definitions.items():
        root = tmp_path / 'sources' / step
        root.mkdir(parents=True)
        refs = []
        for name, value in artifacts.items():
            path = root / (name + '.json')
            path = (write_table if isinstance(value, pd.DataFrame) else _write_json)(path, value)
            refs.append(ArtifactRef(name, path.name, file_hash(path), 'illustrative'))
        outcome = StepResult(step, 'illustrative', 'completed', 'Illustrative display fixture', tuple(refs))
        _write_json(root / 'execution-result.json', {'result': result_to_dict(outcome), 'result_sha256':content_id(result_to_dict(outcome))})
        saved[step] = SavedResult(root, outcome)
    from pymicroglia.pipelines import _saved_figures
    captured = []

    def save(page, output, name, *, sources, settings, claim, **kwargs):
        captured.append((settings['audit_page'], json.loads(page.auxiliary['display.csv'].iloc[0].page_json)))
    monkeypatch.setattr(_saved_figures, 'save_page', save)
    for name in ('estimate_one', 'adjust_pvalues', 'generate_benchmark_cases', 'benchmark_score_interval', 'filter_rhythm_trace'):
        monkeypatch.setattr(circadian, name, lambda *a, **k: pytest.fail('producer recomputed science'))
    step = next((s for s in RECIPE.steps if s.name == 'focused-pages'))
    context = ExecutionContext(step, fixture[0], Settings(), {}, saved, None, tmp_path / 'output/renders/focused/id/inv', 'illustrative', Settings({'focus': {'audit_case_ids': ['first/1/signal', 'second/1/signal'], 'audit_page_size': 1}}), 'presentation')
    assert produce_focused_pages(context).status == 'completed'
    assert [p for p, _ in captured] == [1, 2, 3, 4]
    assert [p['movie'] for _, p in captured] == ['first', 'first', 'second', 'second']
    assert [p['candidate_ids'] for _, p in captured] == [['recipe-b'], ['recipe-a']] * 2
