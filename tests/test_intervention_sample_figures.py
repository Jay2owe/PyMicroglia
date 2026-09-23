from pymicroglia.figure_tables.intervention_display import prepare as prepare_intervention_display
"""Original sample membership, paired categories and saved-only figures."""
from tests.panel_helpers import panel_canvas
from types import SimpleNamespace
import numpy as np
import pytest
from pymicroglia import workbench as circadian
import pymicroglia.measure.sample_contrasts as sample_contrasts
import pymicroglia.pipelines.intervention.sample_figures as figures, pymicroglia.pipelines.intervention.controls as intervention_controls, pymicroglia.pipelines.intervention.patterns as intervention_patterns
from pymicroglia.pipelines._contracts import Settings
from pymicroglia.pipelines.intervention.options import run_request

def context(resolved, execution):
    return SimpleNamespace(request=resolved, dependencies=execution.results, saved=execution.results.__getitem__)

def test_native_control_values_keep_original_samples_and_cell_counts(tmp_path, monkeypatch):
    from tests.test_intervention_controls import fixture
    _, _, paths, resolved = fixture(tmp_path, extra=True)
    execution = run_request(resolved, paths, tmp_path / 'pipeline', only=['control-comparisons'])
    assert execution.successful

    def forbidden(*a, **k):
        raise AssertionError('Saved sample figure recomputed evidence')
    monkeypatch.setattr(intervention_controls, 'compare_units', forbidden)
    monkeypatch.setattr(sample_contrasts, 'compare', forbidden)
    monkeypatch.setattr(circadian, 'adjust_pvalues', forbidden)
    data = figures.collect(context(resolved, execution))
    settings, _ = figures.options(Settings())
    values, statistics, pages = figures.pages(data, settings)
    units = values.loc[values.kind.eq('sample_change')]
    assert len(units) == 8
    first = units.loc[units['sample'].eq('reference-sample-0')].iloc[0]
    assert first.value == 6.0 and first.available_recordings == 2 and (first.available_cells == 4)
    contrast = statistics.loc[statistics.kind.eq('control_comparison')].iloc[0]
    original = data['controls']['comparisons'].iloc[0]
    assert contrast.effect == original.effect and contrast.q_value == original.q_value
    assert contrast.reference_samples == 4 and contrast.target_samples == 4 and (contrast.reference_cells == 7)
    from pymicroglia.visualisation.figures import load as _figure_schema
    _figure_schema()
    from pymicroglia.visualisation.panels import intervention_samples as panel
    import matplotlib.pyplot as plt
    for page in pages:
        selected = values.loc[values.entry_id.isin(page['entry_ids'])]
        fig, _ = panel.draw(prepare_intervention_display(selected,page,'intervention_samples'), page, canvas=panel_canvas())
        plt.close(fig)

def test_paired_population_and_missing_recurrence_denominator_remain_complete(tmp_path, monkeypatch):
    from tests.test_intervention_patterns import fixture, resolve
    request, tables, paths, _ = fixture(tmp_path)
    request['coordinated']['pairs'] = {'mode': 'explicit', 'pairs': [['first', 'second'], ['first', 'missing']]}
    resolved = resolve(request, tables, paths)
    execution = run_request(resolved, paths, tmp_path / 'pipeline', only=['coordinated-responses'])
    assert execution.successful

    def forbidden(*a, **k):
        raise AssertionError('Paired display retested sample association')
    monkeypatch.setattr(intervention_patterns, 'analyse', forbidden)
    monkeypatch.setattr(circadian, 'adjust_pvalues', forbidden)
    data = figures.collect(context(resolved, execution))
    settings, _ = figures.options(Settings())
    values, statistics, pages = figures.pages(data, settings)
    cells = values.loc[values.kind.eq('cell_pair')]
    samples = values.loc[values.kind.eq('sample_pair')]
    assert len(cells) == 20 and cells.pair_id.is_unique and (len(samples) == 16)
    assert cells.joint_response.eq('both_inconclusive').all() and samples.both_supported_fraction_of_jointly_tested.isna().all()
    first = samples.loc[samples['sample'].eq('sample-1') & samples.target.eq('second')].iloc[0]
    assert first.reference_value == pytest.approx(5.5) and first.target_value == pytest.approx(11.0) and (len(first.cell_pair_ids) == 3)
    original = data['patterns']['associations'].set_index('association_id')
    actual = statistics.loc[statistics.kind.eq('association')]
    assert np.allclose(actual.q_value, original.loc[actual.association_id, 'q_value'], equal_nan=True)
    settings['rows_per_page'] = 3
    again, _, more = figures.pages(data, settings)
    assert set(again.entry_id) == set(values.entry_id)
    recurrence = [p for p in more if p['view'] == 'recurrence']
    assert len(recurrence) == 6
    from pymicroglia.visualisation.figures import load as _figure_schema
    _figure_schema()
    from pymicroglia.visualisation.panels import intervention_samples as panel
    import matplotlib.pyplot as plt
    for view in ['cell_pairs', 'joint_outcomes', 'sample_pairs', 'recurrence']:
        page = next((p for p in pages if p['view'] == view))
        selected = values.loc[values.entry_id.isin(page['entry_ids'])]
        fig, axes = panel.draw(prepare_intervention_display(selected,page,'intervention_samples'), page, canvas=panel_canvas())
        if view == 'recurrence':
            assert not any((line.get_marker() == 'o' for line in axes['evidence'].lines))
        plt.close(fig)

def test_disabled_sample_questions_do_not_invent_units(tmp_path):
    from tests.test_intervention_evidence import fixture
    _, _, _, resolved = fixture(tmp_path)
    data = figures.collect(SimpleNamespace(request=resolved, dependencies={}))
    values, statistics, pages = figures.pages(data, figures.options(Settings())[0])
    assert values.status.eq('disabled').all() and len(pages) == 1 and (len(values) == 2)
