"""One public cell-grid family with registry-backed analysis adapters."""
from __future__ import annotations

from copy import deepcopy
from .. import workbench

DEFAULT_PREPROCESSING = {'mean_frames': 3, 'detrend': workbench.DEFAULT_DETREND_METHOD}
COMMON_DEFAULTS = {'metrics': ['integrated_density'], 'cells': 'all',
                   'preprocessing': DEFAULT_PREPROCESSING, 'grid_rows': None, 'grid_columns': None}
PRIMARY_OPTIONS = ('analysis', 'metrics', 'cells', 'grid_columns', 'reference_metric')


def keys():
    return ('trace', *workbench.cell_analysis_catalogue())


def catalogue():
    """Same metadata supplies Python discovery and local interface controls."""
    from ..visualisation.figures import load
    from ..visualisation.figures._vocabulary import meaning
    common = set(COMMON_DEFAULTS)
    trace = load()['all_cell_trace_grid']
    parameters = [{'name': option.name, **meaning(option.name), 'default': deepcopy(option.default),
                   'required': False, **({'choices': list(option.choices)} if option.choices else {})}
                  for option in trace.options if option.name not in common]
    return {'trace': {'key': 'trace', 'action': 'trace', 'view': 'traces', 'title': 'Cell traces',
                      'fft_components': True,
                      'paired': False, 'in_ui': True, 'parameters': parameters,
                      'summary': 'Measured traces, optional native fits and unadjusted period tests.'},
            **workbench.cell_analysis_catalogue()}


def prepare(source, options, *, progress=None, workers=1):
    validate_options(options)
    from .grid_presentation import apply, resolve
    if options.get('analysis') != 'trace':
        from .cell_analysis_grid import prepare
        numerical_options = dict(options)
        if (options.get('grid_presentation') or {}).get('layout') in {'overlay', 'stack'}:
            numerical_options.update(grid_rows=None, grid_columns=None)
        prepared, evidence = prepare(source, numerical_options, progress=progress)
        return apply(prepared, options), evidence
    from . import all_cell_traces
    from .cell_grid_inputs import recipe
    from ..visualisation.figures import load
    spec = load()['all_cell_trace_grid']
    settings = {option.name: deepcopy(option.default) for option in spec.options}
    params = options.get('analysis_parameters', {})
    allowed = {row['name'] for row in catalogue()['trace']['parameters']}
    if not isinstance(params, dict) or set(params) - allowed:
        raise ValueError('Unknown trace analysis_parameters; choose ' + ', '.join(sorted(allowed)))
    if options.get('analysis_settings') or options.get('analysis_views'):
        raise ValueError('Trace settings belong in analysis_parameters; trace has one view')
    if options.get('analysis_display'):
        raise ValueError('Trace appearance uses the shared figure theme and export controls, not Workbench analysis_display')
    settings.update(params)
    settings['grid_presentation'] = resolve(options['grid_presentation']) if options.get('grid_presentation') else {}
    settings.update({key: deepcopy(options[key]) for key in COMMON_DEFAULTS})
    resolved = recipe(settings['preprocessing'])
    if resolved:
        if any(metric.endswith('_mean3') for metric in settings['metrics']) and resolved['mean_frames'] > 1:
            raise ValueError('Choose the raw measurement when applying a frame mean; this column is already mean3')
        if params.get('trace_filter') is not None:
            raise ValueError('Choose preprocessing or trace_filter, not both')
        if 'detrend' in params and params['detrend'] != resolved['detrend']:
            raise ValueError('Trace detrending must agree with preprocessing')
        settings['detrend'] = resolved['detrend']
        settings['trace_view'] = 'raw' if resolved['detrend'] == 'none' else 'detrended'
        settings.update({'detrend_' + key: value for key, value in resolved['detrend_options'].items()})
    prepared, evidence = all_cell_traces.prepare(source, settings, progress=progress, workers=workers)
    prepared['analyses'] = prepared.pop('traces')
    return apply(prepared, options), evidence


def validate_options(options):
    """Validate the declared recipe before starting a job or reading data."""
    from .cell_grid_inputs import recipe
    from .grid_presentation import resolve
    presentation = resolve(options.get('grid_presentation'))
    selected = options.get('analysis', 'trace')
    entries = catalogue()
    if selected not in entries:
        raise ValueError('Choose analysis from ' + ', '.join(entries))
    recipe(options.get('preprocessing'))
    workbench.cell_grid_shape(1, options.get('grid_rows'), options.get('grid_columns'))
    params = options.get('analysis_parameters', {})
    if selected == 'trace':
        allowed = {row['name'] for row in entries['trace']['parameters']}
        if not isinstance(params, dict) or set(params) - allowed:
            raise ValueError('Unknown trace analysis_parameters; choose ' + ', '.join(sorted(allowed)))
        if options.get('analysis_settings') or options.get('analysis_views') or options.get('analysis_display'):
            raise ValueError('Trace options belong in analysis_parameters; appearance uses shared figure controls')
        if presentation['fft_components']:
            from ..visualisation.figures import load
            defaults = {option.name: option.default for option in load()['all_cell_trace_grid'].options}
            parameters = {**defaults, **params}
            if (parameters.get('fit_method') not in (None, 'fft_nlls') or
                    not parameters.get('period_testing') or not parameters.get('fft_component_test')):
                raise ValueError('All tested fitted components requires fit_method=fft_nlls, period_testing=true and fft_component_test=true')
            preprocessing = recipe(options.get('preprocessing'))
            if preprocessing is not None and preprocessing['detrend'] not in workbench.FFT_COMPONENT_DETRENDS:
                raise ValueError('Conditional component tests require asymmetric least-squares or robust linear detrending')
        if options.get('grid_presentation') and presentation['layout'] in {'default', 'overlay'}:
            from ..visualisation.labels import describe
            normalization = params.get('trace_normalization', 'minmax')
            units = {( 'Self-normalized' if normalization != 'none' and layer in {'processed', 'fit'}
                       else describe(metric).unit_text(None))
                     for metric in options.get('metrics', ['integrated_density'])
                     for layer in presentation['trace_layers']}
            if presentation['fft_components']:
                units.update('Self-normalized' if normalization != 'none' else describe(metric).unit_text(None)
                             for metric in options.get('metrics', ['integrated_density']))
            if len(units) > 1:
                raise ValueError('Overlay layers have different units or normalization; use stack/separate, or set trace_normalization to none for compatible measurements')
    else:
        if presentation['trace_layers'] != ['processed']:
            raise ValueError('Trace layers apply to Cell traces; native overlays use the selected measurements')
        from .grid_presentation import DEFAULTS
        if any(presentation[key] != DEFAULTS[key] for key in ('show_fit', 'legend', 'line_width', 'alpha')):
            raise ValueError('Trace appearance applies to Cell traces; native plots use analysis_display')
        views = options.get('analysis_views') or [selected]
        if presentation['fft_components'] and not any(entries.get(key, {}).get('fft_components') for key in views):
            raise ValueError('All tested fitted components applies to Cell traces or the Fourier transform plus nonlinear least squares component view')
        if not isinstance(views, (list, tuple)) or not all(isinstance(key, str) for key in views) or len(set(views)) != len(views):
            raise ValueError('analysis_views must name distinct declared analysis/view keys')
        for key in views:
            workbench.validate_cell_analysis(key, settings=options.get('analysis_settings'),
                parameters=params, display=options.get('analysis_display'))
            if entries[key]['paired'] and not options.get('reference_metric'):
                raise ValueError('Choose reference_metric for this paired analysis')
