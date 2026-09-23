"""Portable, data-only small-multiple plot of independently scaled cell traces."""
from __future__ import annotations
from . import _layout
from ._layout import finish
from . import colour as house_colour
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator
from matplotlib.lines import Line2D
COLORS = (house_colour('blue'), house_colour('circadian_red'), house_colour('circadian_teal'), house_colour('circadian_amber'), house_colour('circadian_purple'), house_colour('dark'))
PERIOD_TEXT_COLOUR = house_colour('nan_text')

def _period_annotation(evidence, identity, metrics, labels):
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
    """Draw original-clock traces; share the y-axis only within each row."""
    identities = settings['identities']
    columns, rows = (settings['columns'], settings['rows'])
    width, height = (max(8.0, columns * 2.7), max(5.0, rows * 1.75 + 2.1))
    figure = _layout.figure(canvas, figsize=(width, height))
    top, bottom = (1.25 / height, 1.1 / height)
    grid = figure.add_gridspec(rows, columns, left=0.065, right=0.985, top=1 - top, bottom=bottom, wspace=0.055, hspace=0.28)
    axes = []
    metric_names = settings['metrics']
    period_style = bool(settings.get('period_display_style', False))
    xlim = settings['xlim']
    for index in range(rows * columns):
        row, column = divmod(index, columns)
        share = axes[row * columns] if column and row * columns < len(axes) else None
        ax = figure.add_subplot(grid[row, column], sharex=axes[0] if axes else None, sharey=share)
        axes.append(ax)
        if index >= len(identities):
            ax.set_visible(False)
            continue
        cell = points[points.identity.eq(identities[index])]
        significant = False
        for number, metric in enumerate(metric_names):
            series = cell[cell.metric.eq(metric)].sort_values('frame_index', kind='stable')
            if not series.empty:
                one_point = np.isfinite(series.value.to_numpy(float)).sum() == 1
                significant = bool(series.period_test_significant.iloc[0]) if period_style and 'period_test_significant' in series else False
                trace_color = (house_colour('circadian_red') if significant else house_colour('nan_text')) if period_style else COLORS[number % len(COLORS)]
                ax.plot(series.hours.to_numpy(float), series.value.to_numpy(float), color=trace_color, linewidth=1.25 if period_style else 1.15, marker='o' if one_point else None, markersize=3, label=settings['labels'][number])
                if period_style and 'descriptive_fit' in series and series.descriptive_fit.notna().any():
                    ax.plot(series.hours.to_numpy(float), series.descriptive_fit.to_numpy(float), color=trace_color, linestyle=':', linewidth=2.0, zorder=4)
        if cell.empty or not np.isfinite(cell.value.to_numpy(float)).any():
            ax.text(0.5, 0.5, 'No measured values', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Cell {identities[index]}', loc='left', fontsize=10, pad=2)
        if settings.get('period_testing', True):
            annotation = _period_annotation(evidence, identities[index], metric_names, settings['labels'])
            if annotation:
                if period_style and len(metric_names) == 1:
                    annotation = annotation.removeprefix(settings['labels'][0] + ': ')
                ax.set_title(annotation, loc='right', fontsize=8, color=house_colour('circadian_red') if period_style and significant else PERIOD_TEXT_COLOUR, pad=2)
        ax.set_xlim(*xlim)
        tick_hours = float(settings.get('hour_ticks', 12.0))
        if not np.isfinite(tick_hours) or tick_hours <= 0:
            raise ValueError('hour_ticks must be a positive finite number')
        first_tick = tick_hours * math.ceil(xlim[0] / tick_hours)
        last_tick = tick_hours * math.floor(xlim[1] / tick_hours)
        if first_tick <= last_tick:
            ax.xaxis.set_major_locator(FixedLocator(np.arange(first_tick, last_tick + tick_hours / 2, tick_hours)))
            if (last_tick - first_tick) / tick_hours + 1 >= 6:
                plt.setp(ax.get_xticklabels(), rotation=90, ha='center', va='top')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='both', labelsize=8, length=3)
        if column:
            ax.tick_params(labelleft=False)
        if row != rows - 1 and index + columns < len(identities):
            ax.tick_params(labelbottom=False)
    if period_style:
        handles = [Line2D([], [], color=house_colour('circadian_red'), linewidth=2, label=f"Significant period test (uncorrected p < {settings.get('rhythmic_alpha', 0.05):g})"), Line2D([], [], color=house_colour('nan_text'), linewidth=2, label='Other cells')]
        if 'descriptive_fit' in points and points.descriptive_fit.notna().any():
            handles.append(Line2D([], [], color=house_colour('dark'), linewidth=2, linestyle=':', label=settings.get('fit_label') or 'Descriptive cosinor fit'))
        grid.update(wspace=0.18, hspace=0.4)
        figure.set_size_inches(max(width, columns * 5.2), height + 0.35 * rows + 0.8)
    else:
        handles = [plt.Line2D([], [], color=COLORS[i % len(COLORS)], linewidth=2, label=label) for i, label in enumerate(settings['labels'])]
    width, height = figure.get_size_inches()
    figure.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1 - 0.68 / height), ncol=min(len(handles), 5), frameon=False, fontsize=10)
    figure.text(0.02, 1 - 0.14 / height, settings['title'], fontsize=15, fontweight='bold', va='top')
    figure.text(0.012, 0.5, settings['unit'], rotation=90, ha='center', va='center', fontsize=16)
    figure.supxlabel('Recording time (hours)', y=0.16 / height, fontsize=10)
    return (figure, [ax for ax in axes if ax.get_visible()])


def traces(figure, data, *, style=None, **options):
    from ._contract import Drawn
    settings={**data['settings'], 'title':'', 'claim':''}
    figure,axes=draw(data['table'],settings,data['annotations'],canvas=figure)
    return Drawn(data['table'],axes)
