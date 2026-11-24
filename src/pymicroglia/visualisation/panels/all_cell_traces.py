"""Portable, data-only small-multiple plot of independently scaled cell traces."""
from __future__ import annotations
from . import _layout
from ._layout import finish
from . import colour as house_colour
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.lines import Line2D
COLORS = (house_colour('blue'), house_colour('circadian_red'), house_colour('circadian_teal'), house_colour('circadian_amber'), house_colour('circadian_purple'), house_colour('dark'))
PERIOD_TEXT_COLOUR = house_colour('nan_text')

def _period_annotation(evidence, identity, metrics, labels, *, period_only=False):
    """Return compact per-metric period-test results for one cell."""
    if evidence is None or evidence.empty or 'identity' not in evidence:
        return ''
    rows = evidence[evidence.identity.eq(identity)]
    by_metric = {metric: label for metric, label in zip(metrics, labels)}
    annotations = []
    for metric, label in by_metric.items():
        selected = rows[rows.metric.eq(metric)]
        if selected.empty:
            continue
        result = selected.iloc[0]
        period = result.get('period_hours')
        if result.get('estimation_method') == 'fft_nlls':
            from ...figure_tables.all_cell_traces import strongest_display_component
            component = strongest_display_component(result)
            if component is not None:
                period = component['period_hours']
        q_value = result.get('q_value')
        p_value = result.get('p_value')
        try:
            period = float(period)
        except (TypeError, ValueError):
            period = math.nan
        try:
            q_value = float(q_value)
        except (TypeError, ValueError):
            q_value = math.nan
        try:
            p_value = float(p_value)
        except (TypeError, ValueError):
            p_value = math.nan
        if np.isfinite(period) and period > 0:
            text = f'{period:.3g} h'
            if not bool(result.get('supported_period', True)):
                text += ' (limited)'
            if bool(result.get('alternate_fft_component', False)):
                text += f" [component {int(result.get('period_source_component'))}]"
            if period_only:
                annotations.append(text)
                continue
            if np.isfinite(q_value):
                text += f', q={q_value:.2g}'
            elif np.isfinite(p_value):
                text += f', p={p_value:.2g}'
        elif result.get('estimate_status') in {'not_tested', 'not_requested'}:
            text = 'not tested'
        else:
            text = 'unresolved'
        annotations.append(f'{label}: {text}')
    return '\n'.join(annotations)

def draw(points, settings, evidence=None, *, canvas=None):
    """Draw original-clock traces; share the y-axis only within each row.

    Optional ``identity_label`` changes the title prefix (default ``Cell``).
    Optional ``colors`` supplies the metric palette, including significant
    traces and their fits in period style; other cells remain grey. Omitted
    settings preserve the established rendering. Neither option changes data
    or fit evidence.
    ``identity_annotations`` may supply a right-hand label per identity instead
    of the period annotation, for example an independently calculated ratio.
    ``grid_layout_inches`` optionally sets width, row_height, top and bottom
    in inches, so margins do not grow with a long grid's row count.
    """
    identities = settings['identities']
    columns, rows = (settings['columns'], settings['rows'])
    from ... import workbench
    figure, axes = workbench.create_cell_grid(canvas, count=len(identities),
        rows=rows, columns=columns, sharex=True, sharey='row')
    # Poster reference: compact stacked rows, not square analysis panels.
    layout = settings.get('grid_layout_inches') or dict(
        width=columns * 4., row_height=2.1, top=1.4, bottom=1.05)
    if layout is not None:
        if set(layout) != {'width', 'row_height', 'top', 'bottom'}:
            raise ValueError('grid_layout_inches needs width, row_height, top and bottom')
        if any(isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0
               for value in layout.values()):
            raise ValueError('Grid dimensions must be positive finite inches')
        width = float(layout['width'])
        height = rows * float(layout['row_height']) + float(layout['top']) + float(layout['bottom'])
        figure.set_size_inches(width, height)
        axes[0].get_subplotspec().get_gridspec().update(
            top=1-float(layout['top'])/height, bottom=float(layout['bottom'])/height,
            hspace=.22)
    width, height = figure.get_size_inches()
    metric_names = settings['metrics']
    colors = settings.get('colors', COLORS)
    if not colors:
        raise ValueError('colors must contain at least one trace colour')
    significant_colors = colors if 'colors' in settings else (house_colour('circadian_red'),)
    period_style = bool(settings.get('period_display_style', False))
    xlim = settings['xlim']
    for index, ax in enumerate(axes):
        row, column = divmod(index, columns)
        cell = points[points.identity.eq(identities[index])]
        significant = False
        for number, metric in enumerate(metric_names):
            series = cell[cell.metric.eq(metric)].sort_values('frame_index', kind='stable')
            if not series.empty:
                one_point = np.isfinite(series.value.to_numpy(float)).sum() == 1
                significant = bool(series.period_test_significant.iloc[0]) if period_style and 'period_test_significant' in series else False
                trace_color = (significant_colors[number % len(significant_colors)] if significant else house_colour('nan_text')) if period_style else colors[number % len(colors)]
                ax.plot(series.hours.to_numpy(float), series.value.to_numpy(float), color=trace_color, linewidth=1.25 if period_style else 1.15, marker='o' if one_point else None, markersize=3, label=settings['labels'][number])
                if period_style and 'descriptive_fit' in series and series.descriptive_fit.notna().any():
                    ax.plot(series.hours.to_numpy(float), series.descriptive_fit.to_numpy(float), color=trace_color, linestyle=':', linewidth=2.0, zorder=4)
        if cell.empty or not np.isfinite(cell.value.to_numpy(float)).any():
            ax.text(0.5, 0.5, 'No measured values', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{settings.get('identity_label', 'Cell')} {identities[index]}", loc='left', fontsize=20, pad=2)
        custom_annotations = settings.get('identity_annotations', {})
        custom_annotation = custom_annotations.get(identities[index], custom_annotations.get(str(identities[index])))
        if custom_annotation is not None:
            ax.set_title(str(custom_annotation), loc='right', fontsize=18, color=PERIOD_TEXT_COLOUR, pad=2)
        elif settings.get('period_testing', True):
            annotation = _period_annotation(evidence, identities[index], metric_names, settings['labels'], period_only=period_style)
            if annotation:
                if period_style and len(metric_names) == 1:
                    annotation = annotation.removeprefix(settings['labels'][0] + ': ')
                ax.set_title(annotation, loc='right', fontsize=18, color=PERIOD_TEXT_COLOUR, pad=2)
        ax.set_xlim(*xlim)
        from ..time_guides import time_guides
        for hour in time_guides(xlim, settings.get('time_guides_hours'), settings.get('time_guides_offset_hours', 0.0)):
            ax.axvline(hour, color=PERIOD_TEXT_COLOUR, linestyle='--', linewidth=0.65, alpha=0.6, zorder=0)
        tick_hours = float(settings.get('hour_ticks', 12.0))
        if not np.isfinite(tick_hours) or tick_hours <= 0:
            raise ValueError('hour_ticks must be a positive finite number')
        first_tick = tick_hours * math.ceil(xlim[0] / tick_hours)
        last_tick = tick_hours * math.floor(xlim[1] / tick_hours)
        if first_tick <= last_tick:
            ticks = np.arange(first_tick, last_tick + tick_hours / 2, tick_hours)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            # Keep every requested tick and guide; label fewer ticks when the
            # poster-sized horizontal text would otherwise merge together.
            stride = max(1, math.ceil(len(ticks) / 7))
            ax.xaxis.set_major_formatter(FixedFormatter([
                f'{hour:g}' if i % stride == 0 else '' for i, hour in enumerate(ticks)]))
            plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='both', labelsize=18, length=3)
        if period_style:
            ax.set_yticks([])
        if column:
            ax.tick_params(labelleft=False)
        if row != rows - 1 and index + columns < len(identities):
            ax.tick_params(labelbottom=False)
    if period_style:
        test_label = ('Conditional FFT component' if settings.get('fft_component_test') else 'Significant period test')
        handles = [Line2D([], [], color=significant_colors[i % len(significant_colors)], linewidth=2,
                          label=f"{label + ': ' if len(metric_names) > 1 else ''}{test_label} (uncorrected p < {settings.get('rhythmic_alpha', 0.05):g})")
                   for i, label in enumerate(settings['labels'])]
        handles.append(Line2D([], [], color=house_colour('nan_text'), linewidth=2, label='Other cells'))
        if 'descriptive_fit' in points and points.descriptive_fit.notna().any():
            handles.append(Line2D([], [], color=house_colour('dark'), linewidth=2, linestyle=':', label=settings.get('fit_label') or 'Descriptive cosinor fit'))
    else:
        handles = [plt.Line2D([], [], color=colors[i % len(colors)], linewidth=2, label=label) for i, label in enumerate(settings['labels'])]
    width, height = figure.get_size_inches()
    legend_columns = min(len(handles), 5)
    legend = figure.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1 - 0.48 / height), ncol=legend_columns, frameon=False, fontsize=18)
    if period_style and len(metric_names) > 1 and columns > 1:
        # Two reporter names need another legend row on narrower native grids.
        # Measure with the actual font; keep the poster-sized text unchanged.
        from matplotlib.backends.backend_agg import RendererAgg
        def measure():
            renderer = RendererAgg(int(figure.bbox.width), int(figure.bbox.height), figure.dpi)
            figure.draw(renderer)
            return renderer
        renderer = measure()
        while legend_columns > 1:
            bounds = legend.get_window_extent(renderer)
            if bounds.x0 >= 5 and bounds.x1 <= figure.bbox.width - 5:
                break
            legend.remove(); legend_columns -= 1
            legend = figure.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1 - 0.48 / height), ncol=legend_columns, frameon=False, fontsize=18)
            renderer = measure()
        title_top = max(max(ax._left_title.get_window_extent(renderer).y1,
                            ax._right_title.get_window_extent(renderer).y1) for ax in axes[:columns])
        clearance = title_top + 8 - legend.get_window_extent(renderer).y0
        if clearance > 0 and 'grid_layout_inches' not in settings:
            extra = clearance / figure.dpi + .05
            height += extra
            figure.set_size_inches(width, height)
            axes[0].get_subplotspec().get_gridspec().update(
                top=1-(float(layout['top'])+extra)/height,
                bottom=float(layout['bottom'])/height)
            legend.set_bbox_to_anchor((.5, 1-.48/height))
    figure.text(0.02, 1 - 0.14 / height, settings['title'], fontsize=24, fontweight='bold', va='top')
    figure.text(0.012, 0.5, settings['unit'], rotation=90, ha='center', va='center', fontsize=24)
    time_note = settings.get('time_note')
    figure.supxlabel(settings.get('time_label', 'Recording time (hours)'),
                     y=(0.24 if time_note else 0.16) / height, fontsize=24)
    if time_note:
        figure.text(0.02, 0.02 / height, time_note, fontsize=12, va='bottom')
    return (figure, [ax for ax in axes if ax.get_visible()])


def traces(figure, data, *, style=None, **options):
    from ._contract import Drawn
    settings={**data['settings'], 'title':'', 'claim':''}
    if 'plot_table' in data:
        figure, axes = draw_layers(data, settings, canvas=figure)
        return Drawn(data['plot_table'], axes)
    figure,axes=draw(data['table'],settings,data['annotations'],canvas=figure)
    return Drawn(data['table'],axes)


def draw_layers(data, settings, *, canvas):
    """Draw declared layers on shared axes or independent lanes; data only."""
    from ... import workbench
    specs, identities = data['series_specs'], settings['identities']
    presentation = settings['grid_presentation']
    mode = presentation['layout']
    from ...figure_tables.grid_presentation import trace_specs
    groups = ([(identity, [spec]) for identity in identities for spec in trace_specs(data, identity)]
              if mode == 'separate' else
              [(identity, trace_specs(data, identity)) for identity in identities])
    sizes = [len(selected) if mode == 'stack' else 1 for _, selected in groups]
    figure, lanes = workbench.create_cell_grid_lanes(canvas, sizes=sizes,
        rows=settings['rows'], columns=settings['columns'], sharex=True)
    from ..time_guides import time_guides
    for (identity, selected), group in zip(groups, lanes):
        for index, spec in enumerate(selected):
            axis = group[index] if mode == 'stack' else group[0]
            rows = data['plot_table'].loc[data['plot_table'].identity.eq(identity) & data['plot_table'].plot_series.eq(spec['key'])].sort_values('frame_index', kind='stable')
            color = COLORS[specs.index(spec) % len(COLORS)]
            component = spec['layer'] == 'component'
            label = spec['label']
            if presentation['fft_components']:
                color = house_colour('circadian_red') if component and not rows.empty and bool(rows.component_significant.iloc[0]) else house_colour('nan_text')
            if component and not rows.empty:
                label = rows.component_label.iloc[0]
                if len(settings['metrics']) > 1:
                    label = spec['label'].split(' | ')[0] + ' | ' + label
            values = rows.plot_value.to_numpy(float)
            axis.plot(rows.hours.to_numpy(float), values, color=color,
                linewidth=presentation['line_width'], alpha=presentation['alpha'],
                linestyle=':' if spec['layer'] == 'fit' else '--' if spec['layer'] == 'mean' else '-',
                marker='o' if np.isfinite(values).sum() == 1 else None, label=label)
            if presentation['show_fit'] and spec['layer'] == 'processed' and 'fit' not in presentation['trace_layers'] and 'descriptive_fit' in rows and rows.descriptive_fit.notna().any():
                axis.plot(rows.hours.to_numpy(float), rows.descriptive_fit.to_numpy(float),
                    color=color, linestyle=':', linewidth=presentation['line_width'], alpha=presentation['alpha'])
            if mode != 'overlay':
                axis.set_title(f'Cell {identity} | {spec["label"]}', loc='left', fontsize=10, pad=2)
                axis.set_ylabel(spec['unit'], fontsize=8)
                if not np.isfinite(values).any():
                    message = rows.component_display_note.iloc[0] if component and not rows.empty else 'No completed fit' if spec['layer'] == 'fit' else 'No measured values'
                    axis.text(.5, .5, message, transform=axis.transAxes, ha='center', va='center', fontsize=8)
                if component:
                    title = f'Cell {identity} | Component {int(rows.component.iloc[0])}' if not rows.empty and np.isfinite(rows.component.iloc[0]) else f'Cell {identity} | Components'
                    if len(settings['metrics']) > 1:
                        title += ' | ' + spec['label'].split(' | ')[0]
                    axis.set_title(title, loc='left', fontsize=9, pad=2)
                    axis.set_title(label.split(': ', 1)[-1], loc='right', fontsize=8, color=color, pad=2)
                if settings.get('period_testing') and spec['layer'] in {'processed', 'fit'}:
                    annotation = _period_annotation(data['annotations'], identity, [spec['metric']], [spec['label']])
                    annotation = annotation.removeprefix(spec['label'] + ': ')
                    axis.set_title(annotation, loc='right', fontsize=8, color=PERIOD_TEXT_COLOUR, pad=2)
        for axis in group:
            if mode == 'overlay':
                axis.set_title(f'Cell {identity}', loc='left', fontsize=10, pad=2)
                axis.set_ylabel(selected[0]['unit'], fontsize=8)
                if settings.get('period_testing'):
                    axis.set_title(_period_annotation(data['annotations'], identity, settings['metrics'], settings['labels']), loc='right', fontsize=8, color=PERIOD_TEXT_COLOUR, pad=2)
                if presentation['fft_components'] and presentation['legend']:
                    handles, labels = axis.get_legend_handles_labels()
                    pairs = [(handle, label) for handle, label in zip(handles, labels) if 'Component ' in label]
                    if pairs:
                        axis.legend([pair[0] for pair in pairs], [pair[1] for pair in pairs], loc='upper right', fontsize=7, frameon=False)
                if presentation['fft_components']:
                    notes = data['plot_table'].loc[data['plot_table'].identity.eq(identity) & data['plot_table'].trace_layer.eq('component'), 'component_display_note'].dropna()
                    notes = list(dict.fromkeys(note for note in notes if note))
                    if notes:
                        axis.text(.02, .02, '\n'.join(notes), transform=axis.transAxes, fontsize=8, va='bottom')
            axis.set_xlim(*settings['xlim'])
            for hour in time_guides(settings['xlim'], settings.get('time_guides_hours'), settings.get('time_guides_offset_hours', 0.0)):
                axis.axvline(hour, color=PERIOD_TEXT_COLOUR, linestyle='--', linewidth=.65, alpha=.6, zorder=0)
            interval = float(settings['hour_ticks'])
            first = interval * math.ceil(settings['xlim'][0] / interval)
            axis.xaxis.set_major_locator(FixedLocator(np.arange(first, settings['xlim'][1] + 1e-9, interval)))
            axis.spines[['top', 'right']].set_visible(False)
            axis.tick_params(axis='both', labelsize=8, length=3)
    if presentation['legend']:
        handles = [Line2D([], [], color=COLORS[index % len(COLORS)],
                    linestyle=':' if spec['layer'] == 'fit' else '--' if spec['layer'] == 'mean' else '-',
                    label=spec['label']) for index, spec in enumerate(specs)]
        if presentation['fft_components']:
            handles = [Line2D([], [], color=house_colour('circadian_red'), label=f"Significant component (unadjusted p < {settings['rhythmic_alpha']:g})"),
                       Line2D([], [], color=house_colour('nan_text'), label='Other components / observations')]
        if presentation['show_fit'] and 'processed' in presentation['trace_layers'] and 'fit' not in presentation['trace_layers'] and 'descriptive_fit' in data['plot_table'] and data['plot_table'].descriptive_fit.notna().any():
            handles.append(Line2D([], [], color=house_colour('dark'), linestyle=':',
                                  label=settings.get('fit_label') or 'Completed fit'))
        figure.legend(handles=handles, loc='upper center', frameon=False, ncol=min(len(handles), 4), fontsize=10)
    figure.supxlabel('Recording time (hours)', fontsize=10)
    return figure, [axis for group in lanes for axis in group]
