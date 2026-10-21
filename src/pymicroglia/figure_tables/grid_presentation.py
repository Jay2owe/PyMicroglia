"""Registry-backed composition of completed cell-grid data."""
from copy import deepcopy
import math
import json

DEFAULTS = {'layout': 'default', 'trace_layers': ['processed'], 'show_fit': True,
            'legend': True, 'line_width': 1.15, 'alpha': 1.0, 'fft_components': False}


def contract():
    return {
        'fft_components': {'type': 'bool', 'default': False, 'fft_only': True,
                           'label': 'All tested fitted components',
                           'description': 'Show every Fourier transform plus nonlinear least squares fitted wave. Red means its conditional, unadjusted component test passed; grey means otherwise. Use Overlay or Stacked lanes. Native component views prepare 199 block surrogates per component before drawing.'},
        'layout': {'type': 'str', 'default': DEFAULTS['layout'],
                   'choices': ['default', 'overlay', 'stack', 'separate'],
                   'choice_labels': {'default': 'Existing layout', 'overlay': 'Overlay', 'stack': 'Stacked lanes', 'separate': 'Separate panels'},
                   'description': 'Default retains the existing layout. Overlay combines compatible curves; stack keeps independent labelled lanes within each cell; separate gives each series a grid slot.'},
        'trace_layers': {'type': 'list', 'default': DEFAULTS['trace_layers'],
                         'item_choices': ['processed', 'raw', 'mean', 'fit'], 'trace_only': True,
                         'choice_labels': {'processed': 'Processed', 'raw': 'Raw observations', 'mean': 'Frame mean', 'fit': 'Completed fit'},
                         'description': 'Choose processed traces, raw observations, frame means or completed fits. Raw and mean layers retain original units; use stack for mixed scales.'},
        'show_fit': {'type': 'bool', 'default': True, 'trace_only': True,
                     'description': 'Show fitted curves where available. Fitting completes during trace preparation, before drawing; display changes never rerun period tests.'},
        'legend': {'type': 'bool', 'default': True, 'trace_only': True,
                   'description': 'Show the trace and layer legend. Native plots use analysis_display legend settings.'},
        'line_width': {'type': 'float', 'default': DEFAULTS['line_width'], 'trace_only': True,
                       'description': 'Width of trace lines in points.'},
        'alpha': {'type': 'float', 'default': 1.0, 'trace_only': True, 'label': 'Opacity',
                  'description': 'Trace opacity from greater than zero through one.'},
    }


def resolve(value=None):
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError('grid_presentation accepts ' + ', '.join(DEFAULTS))
    result = {**deepcopy(DEFAULTS), **deepcopy(value)}
    if result['layout'] not in contract()['layout']['choices']:
        raise ValueError('Choose grid layout from default, overlay, stack or separate')
    layers = result['trace_layers']
    if not isinstance(layers, list) or not layers or not all(isinstance(x, str) for x in layers) or len(set(layers)) != len(layers) or set(layers) - set(contract()['trace_layers']['item_choices']):
        raise ValueError('trace_layers must select distinct processed, raw, mean or fit layers')
    for name in ('show_fit', 'legend', 'fft_components'):
        if not isinstance(result[name], bool):
            raise ValueError(name + ' must be true or false')
    for name in ('line_width', 'alpha'):
        number = result[name]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
            raise ValueError(name + ' must be a positive finite number')
    if result['alpha'] > 1:
        raise ValueError('alpha must be at most one')
    return result


def apply(prepared, options):
    from .. import workbench
    result = resolve(options.get('grid_presentation'))
    data = prepared['analyses']
    if 'panels' in data:
        if result['trace_layers'] != ['processed']:
            raise ValueError('Trace layers apply to Cell traces; native overlays use the selected measurements')
        layout = result['layout']
        if layout in {'overlay', 'stack'}:
            statistics = prepared.auxiliary['statistics.csv']
            groups, plotted, compositions = [], [], []
            for identity in statistics.identity.drop_duplicates():
                indices = statistics.index[statistics.identity.eq(identity)]
                if layout == 'stack':
                    lanes = []
                    for i in indices:
                        label, definition = data['panels'][i]
                        definitions = workbench.fft_component_lanes(definition) if result['fft_components'] else [definition]
                        lanes.extend((part.title if len(definitions) > 1 else label, part) for part in definitions)
                    groups.append(lanes)
                else:
                    for analysis in statistics.loc[indices, 'analysis'].drop_duplicates():
                        selected = [data['panels'][i] for i in indices if statistics.loc[i, 'analysis'] == analysis]
                        label = f'Cell {identity} | {analysis.split("/")[-1].replace("_", " ")}'
                        definition = workbench.overlay_cell_views(selected, title=label)
                        groups.append((label, definition))
                        plotted.extend({'identity': identity, 'analysis': analysis, **row}
                                       for row in workbench.cell_view_data(definition))
                        compositions.append({'identity': identity, 'analysis': analysis,
                                             'definition_json': json.dumps(definition.as_dict(), allow_nan=False)})
            data['panels'] = groups
            data['stacked'] = layout == 'stack'
            if layout == 'overlay':
                import pandas as pd
                prepared.auxiliary['source_plotted_data.csv'] = data['table'].copy()
                prepared.auxiliary['composed_definitions.csv'] = pd.DataFrame(compositions)
                data['table'] = pd.DataFrame(plotted)
            data['rows'], data['columns'] = workbench.cell_grid_shape(len(groups), options.get('grid_rows'), options.get('grid_columns'))
    else:
        _trace_layers(data, result)
        if 'plot_table' in data:
            prepared.auxiliary['plotted_layers.csv'] = data['plot_table'].copy()
        if result['layout'] == 'separate':
            data['settings']['rows'], data['settings']['columns'] = workbench.cell_grid_shape(
                sum(len(trace_specs(data, identity)) for identity in data['settings']['identities']), options.get('grid_rows'), options.get('grid_columns'))
    data['grid_presentation'] = result
    settings_table = prepared.auxiliary.get('settings.csv')
    if settings_table is not None and not settings_table.empty:
        recorded = json.loads(settings_table.iloc[0]['settings_json'])
        recorded['grid_presentation'] = result
        if 'panels' in data:
            recorded.update(rows=data['rows'], columns=data['columns'])
        else:
            recorded.update(rows=data['settings']['rows'], columns=data['settings']['columns'])
        settings_table.loc[settings_table.index[0], 'settings_json'] = json.dumps(recorded)
    # Save composition independently of the unchanged numerical evidence.
    import pandas as pd
    prepared.auxiliary['grid_presentation.csv'] = pd.DataFrame([
        {'schema_version': 1, 'settings_json': json.dumps(result)}])
    return prepared


def _trace_layers(data, presentation):
    import pandas as pd
    from ..visualisation.labels import describe
    settings, points = data['settings'], data['table']
    layout = presentation['layout']
    if layout == 'default' and presentation == DEFAULTS:
        return
    layout = 'overlay' if layout == 'default' else layout
    columns = {'processed': 'value', 'raw': 'raw_value', 'mean': 'filtered_value', 'fit': 'descriptive_fit'}
    frames, specifications = [], []
    for metric, label in zip(settings['metrics'], settings['labels']):
        for layer in presentation['trace_layers']:
            series = points.loc[points.metric.eq(metric)].copy()
            key = metric + ':' + layer
            unit = ('Self-normalized' if settings['trace_normalization'] != 'none' and layer in {'processed', 'fit'}
                    else describe(metric).unit_text(None))
            series['plot_value'] = series.get(columns[layer], float('nan'))
            series['plot_series'] = key
            series['trace_layer'] = layer
            specifications.append({'key': key, 'metric': metric, 'layer': layer, 'label': label + ' | ' + layer, 'unit': unit})
            frames.append(series)
        if presentation['fft_components']:
            component_frames, component_specs = _component_layers(data, metric, label)
            frames.extend(component_frames)
            specifications.extend(component_specs)
    if layout == 'overlay' and len({row['unit'] for row in specifications}) > 1:
        raise ValueError('Overlay layers have different units or normalization; use stack/separate, or set trace_normalization to none for compatible measurements')
    data['plot_table'] = pd.concat(frames, ignore_index=True)
    data['series_specs'] = specifications
    data['settings'] = {**settings, 'grid_presentation': {**presentation, 'layout': layout}}


def trace_specs(data, identity):
    """Do not reserve phantom component lanes for cells with fewer waves."""
    points = data['plot_table']
    keys = set(points.loc[points.identity.eq(identity), 'plot_series'])
    return [spec for spec in data['series_specs'] if spec['key'] in keys]


def _component_layers(data, metric, label):
    import numpy as np
    from .. import workbench
    from ..visualisation.labels import describe
    points, evidence, settings = data['table'], data['annotations'], data['settings']
    frames, specs = [], {}
    unit = 'Self-normalized contributions' if settings['trace_normalization'] != 'none' else describe(metric).unit_text(None)
    # Contributions share the processed trace's affine scale, but have no mesor.
    display_unit = 'Self-normalized' if settings['trace_normalization'] != 'none' else unit
    for identity in settings['identities']:
        segment = points.loc[points.identity.eq(identity) & points.metric.eq(metric)].copy()
        matched = evidence.loc[evidence.identity.eq(identity) & evidence.metric.eq(metric)]
        row = matched.iloc[0] if len(matched) == 1 else {}
        components = json.loads(row.get('estimate_components_json', '[]'))
        tests = {int(test['component']): test for test in json.loads(row.get('component_results_json', '[]'))}
        diagnostics = json.loads(row.get('estimate_diagnostics_json', '{}'))
        valid = np.isfinite(segment.processed_value) & np.isfinite(segment.value)
        waves, note = [], 'No completed components'
        if components and valid.sum() >= 2 and np.ptp(segment.loc[valid, 'processed_value']) > 0:
            design = np.column_stack([segment.loc[valid, 'processed_value'], np.ones(valid.sum())])
            slope, intercept = np.linalg.lstsq(design, segment.loc[valid, 'value'], rcond=None)[0]
            if np.allclose(design @ [slope, intercept], segment.loc[valid, 'value'], rtol=1e-8, atol=1e-10):
                try:
                    waves = workbench.fft_component_values(segment.hours.to_numpy(float),
                        {'method': 'fft_nlls', 'diagnostics': diagnostics, 'components': components})
                    waves = [np.where(valid, slope * wave, np.nan) for wave in waves]
                except ValueError as error:
                    note = str(error)
            else:
                note = 'Component display requires affine normalization'
        for index in range(max(1, len(components))):
            number = index + 1
            test = tests.get(number, {'status': 'not_tested'})
            period = components[index]['period_hours'] if components else None
            key = metric + ':component_' + str(number)
            series = segment.copy()
            series['plot_value'] = waves[index] if index < len(waves) else np.nan
            series['plot_series'], series['trace_layer'] = key, 'component'
            series['component'] = number if components else np.nan
            series['component_period_hours'] = period if period is not None else np.nan
            series['component_p_value'] = test.get('empirical_p_uncorrected') if test.get('empirical_p_uncorrected') is not None else np.nan
            series['component_status'] = test.get('status', 'not_tested')
            series['component_significant'] = workbench.fft_component_test_significant(test, row.get('alpha', .05))
            series['component_label'] = (f'Component {number}: ' + workbench.fft_component_test_label(test, period)) if period else note
            series['component_display_note'] = '' if index < len(waves) else note
            frames.append(series)
            specs[key] = {'key': key, 'metric': metric, 'layer': 'component',
                          'label': label + f' | Component {number}', 'unit': display_unit}
    return frames, list(specs.values())
