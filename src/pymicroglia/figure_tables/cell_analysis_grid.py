"""Bind measured cells to Circadian Workbench's declared per-trace plots."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

from .. import workbench
from ..visualisation.labels import describe
from .all_cell_traces import grid_shape
from .cell_grid_inputs import identities as selected_cells, recording_clock, native_cell, recipe, process
from .prepared import PreparedViews


def prepare(source, options, *, progress=None):
    frame = source.table('cell_frame.csv')
    required = {'identity', 'hours', 'frame_index'}
    if required - set(frame):
        raise ValueError('cell_frame.csv needs identity, hours and frame_index')
    metrics = options.get('metrics')
    if isinstance(metrics, str):
        metrics = [metrics]
    if not isinstance(metrics, (list, tuple)) or not metrics or len(set(metrics)) != len(metrics):
        raise ValueError('metrics must name distinct measured columns')
    reference = options.get('reference_metric')
    for metric in [*metrics, *([reference] if reference else [])]:
        if metric not in frame or not pd.api.types.is_numeric_dtype(frame[metric]):
            raise ValueError(f'{metric}: choose a numeric column in cell_frame.csv')
    identities = selected_cells(frame, options.get('cells', 'all'))
    clock = recording_clock(frame)
    analysis = options.get('analysis', 'wavelet_spectrum/wavelet')
    catalogue = workbench.cell_analysis_catalogue()
    if not catalogue:
        workbench.validate_cell_analysis(analysis)
    if analysis not in catalogue:
        raise ValueError('Choose analysis from the installed Workbench cell-grid catalogue')
    analyses = options.get('analysis_views') or [analysis]
    if (not isinstance(analyses, (list, tuple)) or not analyses or
            len(set(analyses)) != len(analyses) or any(key not in catalogue for key in analyses)):
        raise ValueError('analysis_views must name distinct declared analysis/view keys')
    entry = catalogue[analysis]
    if any(catalogue[key]['paired'] for key in analyses) and not reference:
        raise ValueError('Choose reference_metric for this paired analysis')
    settings = options.get('analysis_settings', {})
    parameters = options.get('analysis_parameters', {})
    display = options.get('analysis_display', {})
    if any(not isinstance(value, dict) for value in (settings, parameters, display)):
        raise ValueError('analysis_settings, analysis_parameters and analysis_display must be mappings')
    display = {'theme': getattr(source, 'figure_theme', 'pyflash'), **display}
    preprocessing = recipe(options.get('preprocessing'))
    if preprocessing and preprocessing['mean_frames'] > 1 and any(metric.endswith('_mean3') for metric in [*metrics, *([reference] if reference else [])]):
        raise ValueError('Choose the raw measurement when applying a frame mean; this column is already mean3')
    for key in analyses:
        workbench.validate_cell_analysis(key, settings=settings, parameters=parameters, display=display)
    rows, columns = grid_shape(len(identities) * len(metrics) * len(analyses), options.get('grid_rows'), options.get('grid_columns'))
    panels, plotted, evidence, inputs = [], [], [], []
    component_evidence = []
    total = len(identities) * len(metrics) * len(analyses)
    with workbench.analysis_session():
        for identity in identities:
            # Other measured cells provide the native clock for an absent cell.
            # Join those clock slots as missing values, never interpolate photons.
            cell = native_cell(frame, identity, clock)
            hours = cell.hours.to_numpy(float)
            for metric in metrics:
                label = f'Cell {identity} | {describe(metric).label}'
                raw = cell[metric].to_numpy(float)
                reference_values = cell[reference].to_numpy(float) if reference else None
                reference_raw = reference_values.copy() if reference_values is not None else None
                reference_mean = reference_raw.copy() if reference_raw is not None else None
                mean, values, preprocessing_records = raw.copy(), raw.copy(), ()
                preprocessing_complete = False
                try:
                    mean, values, preprocessing_records = process(hours, raw, cell.frame_index, preprocessing, name=label)
                    if reference_values is not None:
                        reference_mean, reference_values, reference_records = process(hours, reference_values, cell.frame_index, preprocessing, name=label + ' reference')
                        preprocessing_records = (*preprocessing_records, *reference_records)
                    preprocessing_complete = True
                    arguments = dict(name=label,
                        reference_values=reference_values, settings=settings,
                        parameters=parameters, display=display,
                        value_label=describe(metric).label,
                        value_unit=describe(metric).unit_text(
                            workbench.sample_interval_minutes(hours) if len(hours) > 1 else None))
                    if preprocessing is not None and len(hours) > 1:
                        arguments['bin_minutes'] = workbench.sample_interval_minutes(hours)
                    if len(analyses) == 1:
                        key = analyses[0]
                        completed_views = {key: workbench.analyze_cell_view(hours, values, analysis=key, **arguments)}
                    else:
                        completed_views = workbench.analyze_cell_views(hours, values, analyses=analyses, **arguments)
                except workbench.cw.WorkbenchError as error:
                    if not preprocessing_complete:
                        mean = np.full(raw.shape, np.nan)
                        values = np.full(raw.shape, np.nan)
                        if reference_values is not None:
                            reference_values = np.full(raw.shape, np.nan)
                    completed_views = {key: error for key in analyses}
                for key in analyses:
                    completed = completed_views[key]
                    status, reason, records, scripts, warnings, results = 'ok', '', [], [], [], []
                    panel_label = label if len(analyses) == 1 else label + ' | ' + catalogue[key]['title']
                    if isinstance(completed, workbench.cw.WorkbenchError):
                        status, reason = 'failed', str(completed)
                        definition = workbench.empty_cell_view(panel_label, reason)
                    else:
                        definition = completed.definition
                        records, scripts, warnings, results = completed.records, completed.scripts, completed.warnings, completed.results
                        if definition.kind == 'empty':
                            status, reason = 'unavailable', ' '.join(definition.annotations)
                        elif (options.get('grid_presentation') or {}).get('fft_components') and catalogue[key].get('fft_components'):
                            from .fft_component_grid import _test_seed
                            tested = workbench.test_fft_component_view(definition,
                                alpha=settings.get('rhythmic_alpha', .05), seed=_test_seed(getattr(source, 'stem', None), identity, 20260923))
                            definition = workbench.style_fft_component_view(definition, tested['tests'], alpha=tested['settings']['alpha'])
                            component_evidence.append({'identity': identity, 'metric': metric, 'analysis': key,
                                'test_record_json': json.dumps(tested, allow_nan=False)})
                    definition = replace(definition, title=panel_label)
                    panels.append((panel_label, definition))
                    plotted.extend({'identity': identity, 'metric': metric, 'analysis': key, **row}
                                   for row in workbench.cell_view_data(definition))
                    evidence.append({'identity': identity, 'metric': metric, 'analysis': key,
                                 'status': status, 'reason': reason, 'reference_metric': reference,
                                 'workbench_version': workbench.WORKBENCH_VERSION,
                                 'preprocessing_json': json.dumps(preprocessing),
                                 'preprocessing_records_json': json.dumps(preprocessing_records),
                                 'run_records_json': json.dumps(records),
                                 'results_json': json.dumps(results, allow_nan=False),
                                 'scripts_json': json.dumps(scripts), 'warnings_json': json.dumps(warnings),
                                 'definition_json': json.dumps(definition.as_dict(), allow_nan=False)})
                    if progress:
                        progress({'completed': len(panels), 'total': total, 'identity': identity, 'metric': metric, 'analysis': key})
                inputs.extend({'identity': identity, 'metric': metric, 'frame_index': int(native_frame),
                               'hours': float(hour), 'raw_value': float(raw[index]), 'filtered_value': float(mean[index]), 'value': float(value),
                               'reference_raw_value': float(reference_raw[index]) if reference_raw is not None else np.nan,
                               'reference_filtered_value': float(reference_mean[index]) if reference_mean is not None else np.nan,
                               'reference_value': float(reference_values[index]) if reference_values is not None else np.nan}
                              for index, (native_frame, hour, value) in enumerate(zip(cell.frame_index, hours, values)))
    table, statistics = pd.DataFrame(plotted), pd.DataFrame(evidence)
    claim = 'Each cell is analysed separately with Circadian Workbench; original recording times and missing observations are retained.'
    resolved = {'analysis': analysis, 'analysis_settings': settings, 'analysis_parameters': parameters,
                'analysis_display': display, 'metrics': list(metrics), 'identities': identities,
                'reference_metric': reference, 'rows': rows, 'columns': columns,
                'preprocessing': preprocessing, 'analysis_views': list(analyses)}
    prepared = PreparedViews({'analyses': {'panels': panels, 'table': table, 'rows': rows, 'columns': columns}},
                             auxiliary={'statistics.csv': statistics, 'input_traces.csv': pd.DataFrame(inputs),
                                        'settings.csv': pd.DataFrame([{'settings_json': json.dumps(resolved)}])},
                             wording={'title': entry['title'] + ' by cell', 'claim': claim})
    if component_evidence:
        prepared.auxiliary['component_tests.csv'] = pd.DataFrame(component_evidence)
    return prepared, statistics
