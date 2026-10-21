"""Cell selection, failure isolation and UI discovery for Workbench plot grids."""
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from pymicroglia import workbench
from pymicroglia.figure_tables.cell_analysis_grid import prepare
from pymicroglia.visualisation.figures import load
from pymicroglia.visualisation.panels.cell_analysis_grid import analyses


def frame():
    return pd.DataFrame([{'identity': identity, 'frame_index': i, 'hours': float(i),
                          'integrated_density_mean3': 10 + np.sin(2 * np.pi * i / 24),
                          'area_px': 20 + np.cos(2 * np.pi * i / 24)}
                         for identity in (3, 8) for i in range(96)])


def options(**overrides):
    spec = load()['cell_analysis_grid']
    return {**{option.name: option.default for option in spec.options}, **overrides}


def source(table):
    return SimpleNamespace(table=lambda _: table.copy())


def test_cell_selection_and_exact_inputs_are_retained():
    table = frame()
    table.loc[(table.identity == 3) & (table.frame_index == 20), 'integrated_density_mean3'] = np.nan
    prepared, evidence = prepare(source(table), options(cells='3,', analysis='autocorrelation/autocorrelation'))
    assert evidence.identity.tolist() == [3]
    assert evidence.status.tolist() == ['ok']
    inputs = prepared.auxiliary['input_traces.csv']
    assert inputs.hours.tolist() == list(np.arange(96.))
    assert inputs.loc[inputs.frame_index == 20, 'value'].isna().all()
    recorded = json.loads(evidence.run_records_json.iloc[0])
    assert recorded[0]['action'] == 'autocorrelation'
    assert prepared['analyses']['rows'] == prepared['analyses']['columns'] == 1


def test_invalid_cell_stays_visible_and_other_cells_complete(monkeypatch):
    native = workbench.analyze_cell_view
    def isolated(hours, values, **settings):
        if settings['name'].startswith('Cell 3 '):
            raise workbench.cw.WorkbenchInputError('insufficient observations')
        return native(hours, values, **settings)
    monkeypatch.setattr(workbench, 'analyze_cell_view', isolated)
    prepared, evidence = prepare(source(frame()), options(analysis='autocorrelation/autocorrelation'))
    assert evidence.status.tolist() == ['failed', 'ok']
    assert len(prepared['analyses']['panels']) == 2
    drawn = analyses(Figure(), prepared['analyses'])
    assert len(drawn.axes) == 2
    assert any('insufficient observations' in text.get_text() for text in drawn.axes[0].texts)


def test_multiple_metrics_are_separate_panels():
    prepared, evidence = prepare(source(frame()), options(analysis='autocorrelation/autocorrelation',
                                                         metrics=['integrated_density_mean3', 'area_px']))
    assert len(evidence) == 4
    assert {(x.identity, x.metric) for x in evidence.itertuples()} == {
        (identity, metric) for identity in (3, 8) for metric in ('integrated_density_mean3', 'area_px')}
    with pytest.raises(ValueError, match='hold all'):
        prepare(source(frame()), options(grid_rows=1, grid_columns=1))


def test_absent_cell_rows_use_known_native_clock_slots_without_interpolation():
    table = frame()
    table = table.loc[~((table.identity == 3) & table.frame_index.between(40, 43))]
    prepared, evidence = prepare(source(table), options(analysis='wavelet_spectrum/wavelet',
                                analysis_parameters={'wavelet_options': {'n_sim': 0}}))
    assert evidence.status.tolist() == ['ok', 'ok']
    inputs = prepared.auxiliary['input_traces.csv']
    missing = inputs.loc[(inputs.identity == 3) & inputs.frame_index.between(40, 43)]
    assert missing.hours.tolist() == [40., 41., 42., 43.]
    assert missing.value.isna().all()
    completed = json.loads(evidence.results_json.iloc[0])[0]['data']
    assert completed['hours'][40:44] == [40., 41., 42., 43.]
    assert not np.asarray(completed['valid_mask'])[:, 40:44].any()


def test_reference_column_is_required_for_paired_views():
    with pytest.raises(ValueError, match='reference_metric'):
        prepare(source(frame()), options(analysis='wavelet_coherence/wavelet_coherence'))


def test_old_workbench_does_not_break_catalogue_but_names_required_update(monkeypatch):
    monkeypatch.setattr(workbench, 'cw', SimpleNamespace())
    assert workbench.cell_analysis_catalogue() == {}
    with pytest.raises(RuntimeError, match='Update Circadian Workbench'):
        workbench.validate_cell_analysis('autocorrelation/autocorrelation')


def test_ui_discovers_labeled_analysis_choices_and_existing_grid_stays_default():
    from fastapi.testclient import TestClient
    from pymicroglia.ui.server import create_app
    client = TestClient(create_app())
    result = client.get('/api/actions/cell_analysis_grid')
    assert result.status_code == 200
    spec = result.json()
    assert spec['pending'] is False
    choices = next(row for row in spec['params'] if row['name'] == 'analysis')
    assert 'autocorrelation/autocorrelation' in choices['choices']
    assert choices['analysis_entries']['wavelet_coherence/wavelet_coherence']['paired']
    assert choices['choice_labels']['autocorrelation/autocorrelation'] == 'Lag correlation (autocorrelation)'
    old = {o.name: o.default for o in load()['all_cell_trace_grid'].options}
    assert old['fft_component_test'] is True and old['fit_method'] == 'fft_nlls'


def test_public_figure_action_exports_plotted_values_and_replay_evidence(monkeypatch):
    from pymicroglia.figure_tables import actions, saved
    from pymicroglia.visualisation import panels
    from pymicroglia.pipelines import _saved_figures

    supplied = source(frame())
    supplied.manifest, supplied.sources, supplied.stem = {}, [], 'synthetic'
    monkeypatch.setattr(saved, 'Tables', lambda *args: supplied)
    captured = {}

    def save(figure, path, **kwargs):
        captured.update(kwargs)
        captured['panels'] = len(figure.axes)
        return path

    monkeypatch.setattr(panels, 'save', save)
    monkeypatch.setattr(_saved_figures, 'save_auxiliary',
                        lambda tables, *args: captured.update(auxiliary=tables))
    result = actions.cell_analysis_grid('unused', analysis='autocorrelation/autocorrelation', theme='classic')
    assert result.name == 'synthetic'
    assert captured['panels'] == 2
    assert set(captured['table']['identity']) == {3, 8}
    assert captured['statistics_status'] == 'complete'
    assert all(json.loads(row['scripts_json']) for row in captured['statistics'])
    assert all(json.loads(row['definition_json'])['options']['display']['theme'] == 'classic'
               for row in captured['statistics'])
    assert {'statistics.csv', 'input_traces.csv', 'settings.csv'} <= set(captured['auxiliary'])
