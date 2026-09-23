from pymicroglia.figure_tables.coordination_display import prepare as prepare_coordination
"""Saved population and evidence meanings survive coordination rendering."""
from tests.panel_helpers import panel_canvas
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
import pytest
import pymicroglia.pipelines.coordination.overview_data as overview
from pymicroglia.pipelines.coordination.options import BRANCHES, run_request
from pymicroglia.pipelines._contracts import Settings
from tests.test_coordination_samples import fixture

def test_matrix_mirrors_only_declared_symmetric_pairs_and_keeps_unknown_distinct_from_zero():
    base = {'entry_id': 'source-row', 'group_id': 'definition', 'source_run': 'source', 'movie': 'a', 'effect_id': 'effect', 'reference_identity': 7, 'target_identity': 8, 'oriented': True, 'display_value': 0.0, 'supported': False, 'reference': 'movement', 'target': 'shape', 'decision': 'descriptive', 'decision_reason': 'Saved without inference'}
    frame = pd.DataFrame(overview.matrix([base], [7, 8], [7, 8]))
    lookup = frame.set_index(['row_identity', 'column_identity'])
    assert lookup.loc[(7, 8), 'display_value'] == 0.0 and pd.isna(lookup.loc[(8, 7), 'display_value'])
    assert lookup.loc[(8, 7), 'status'] == 'not_requested' and lookup.loc[(7, 7), 'status'] == 'same_cell_not_tested'
    frame = pd.DataFrame(overview.matrix([{**base, 'oriented': False, 'target': 'movement'}], [7, 8], [7, 8]))
    assert frame.display_value.notna().sum() == 2 and frame.mirrored_symmetric.sum() == 1
    assert frame.effect_id.dropna().eq('effect').all()
    with pytest.raises(ValueError, match='repeats'):
        overview.matrix([base, base], [7, 8], [7, 8])

def test_failed_science_and_disabled_optional_branches_still_have_complete_diagnostics():
    failed = SimpleNamespace(outcome=SimpleNamespace(status='failed', reason='Original coordinates unavailable', scientific_id='failed-input'))
    request = SimpleNamespace(questions={question: {'enabled': question == 'simultaneous'} for question in BRANCHES.values()}, sample_summary={'enabled': False})
    context = SimpleNamespace(request=SimpleNamespace(request=request), dependencies={'pair-inputs': failed}, saved=lambda name: failed)
    saved = overview.collect(context)
    settings, _ = overview.options(Settings())
    values, statistics, pages = overview.pages(saved, settings)
    assert len(pages) == 1 and len(values) == 7 and statistics.p_value.isna().all()
    assert values.loc[values.question.eq('simultaneous'), 'status'].eq('unavailable').all()
    assert values.loc[values.question.ne('simultaneous'), 'status'].eq('disabled').all()
    from pymicroglia.visualisation.figures import load as _figure_schema
    _figure_schema()
    from pymicroglia.visualisation.panels import coordination_overview
    import matplotlib.pyplot as plt
    fig, _ = coordination_overview.draw(prepare_coordination(values,pages[0]), pages[0], canvas=panel_canvas())
    plt.close(fig)

def test_complete_pair_and_sample_values_match_saved_science_without_computations(tmp_path):
    import pymicroglia.measure.sample_contrasts as sample_contrasts
    import pymicroglia.workbench as circadian
    resolved, paths = fixture(tmp_path)
    execution = run_request(resolved, paths, tmp_path / 'pipeline', only=['coordination-samples', 'lagged-coordination', 'changing-proximity'])
    assert execution.successful
    assert execution.results['lagged-coordination'].outcome.status == 'skipped-empty'
    assert execution.results['changing-proximity'].outcome.status == 'skipped-empty'
    context = SimpleNamespace(request=resolved, dependencies=execution.results, saved=execution.results.__getitem__)
    with patch.object(sample_contrasts, 'compare', side_effect=AssertionError('No comparison during rendering')), patch.object(circadian, 'adjust_pvalues', side_effect=AssertionError('No correction during rendering')):
        saved = overview.collect(context)
        settings, _ = overview.options(Settings())
        values, statistics, pages = overview.pages(saved, settings)
    assert values.loc[values.kind.eq('effect'), 'effect_id'].nunique() == 9
    coverage = values.loc[values.kind.eq('coverage')]
    assert len(coverage) == 7 and coverage.status.eq('disabled').sum() == 5
    matrices = [page for page in pages if page['view'] == 'matrices']
    assert len(matrices) == 9
    for page in matrices:
        rows = values.loc[values.entry_id.isin(page['entry_ids'])]
        assert rows.movie.nunique() == 1 and rows.display_value.notna().sum() == 2
    units = values.loc[values.kind.eq('sample')]
    assert len(units) == 8 and units.recordings.sum() == 9
    comparison = values.loc[values.kind.eq('comparison')].iloc[0]
    assert comparison.effect == pytest.approx(0.6) and comparison.q_value == pytest.approx(2 / 70)
    assert set(statistics.entry_id) <= set(values.entry_id)
    from pymicroglia.visualisation.figures import load as _figure_schema
    _figure_schema()
    from pymicroglia.visualisation.panels import coordination_overview
    import matplotlib.pyplot as plt
    for view in ['effects', 'matrices', 'samples', 'contrasts']:
        page = next((page for page in pages if page['view'] == view))
        rows = values.loc[values.entry_id.isin(page['entry_ids'])]
        fig, axes = coordination_overview.draw(prepare_coordination(rows,page), page, canvas=panel_canvas())
        if view == 'matrices':
            array = axes['evidence'].images[0].get_array()
            assert array.count() == 2
        plt.close(fig)

def test_disabled_temporal_branches_do_not_require_an_unrequested_preparation():
    from pymicroglia.pipelines.coordination.options import execution_recipe
    from tests.test_coordination_options import request
    req = request(target_measurements=['custom_signal'], questions={'characteristics': {'enabled': True, 'summary': 'mean', 'statistic': 'absolute_difference', 'evidence': {'method': 'none'}}})
    steps = {step.name: step for step in execution_recipe(req).steps}
    assert steps['lagged-coordination'].prerequisites == ('pair-inputs',)
    assert steps['changing-proximity'].prerequisites == ('pair-inputs',)

def test_large_native_model_payload_remains_in_source_but_is_not_a_plotted_value():
    payload = {'native_component_model': 'x' * 140000}
    effects = pd.DataFrame([{'effect_id': 'source-result', 'question': 'rhythm', 'evidence_level': 'pair', 'source_run': 'source', 'movie': 'a', 'effect': 2.0, 'period_hours': 8.0, 'offset_interval_hours': [1.9, 2.1], 'native_timing_evidence': payload}])
    rows = overview.effect_rows({'evidence': {'effects': effects}, 'prepared': None})
    assert 'native_timing_evidence' not in rows[0] and rows[0]['effect'] == 2.0 and (rows[0]['offset_interval_hours'] == [1.9, 2.1])
    assert effects.native_timing_evidence.iloc[0] == payload
