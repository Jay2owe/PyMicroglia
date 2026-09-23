"""Portable rendering of frozen directed timing evidence; no scientific fits."""
from . import _layout
from ._layout import finish
from . import colour as house_colour
import math
from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
import numpy as np
from .rhythm_evidence import trace as saved_trace

def wrap(text, width=100):
    return '\n'.join((textwrap.fill(line, width) for line in str(text).splitlines()))

def number(value):
    try:
        return f'{float(value):.3g}' if math.isfinite(float(value)) else 'unavailable'
    except (ValueError, TypeError):
        return 'unavailable'

def _finish(figure, settings, height, *, extra=''):
    figure.text(0.02, 1 - 0.16 / height, wrap(settings['title'], 108), fontsize=14, fontweight='bold', va='top')
    figure.text(0.02, 1.1 / height, wrap(settings['evidence_note'] + ('\n' + extra if extra else ''), 150), fontsize=8, va='bottom')
    figure.text(0.02, 0.12 / height, wrap(settings['footnote'], 155), fontsize=8, va='bottom')

def draw(values, settings, *, selected_view=None, canvas=None):
    if settings['kind'] == 'matrix':
        return _matrix(values, settings, canvas=canvas)
    if settings['kind'] == 'pair-summary':
        return _summary(values, settings, selected_view=selected_view, canvas=canvas)
    return _traces(values, settings, selected_view=selected_view, canvas=canvas)

def _matrix(values, settings, *, canvas=None):
    rows, columns = (settings['rows'], settings['columns'])
    width, height = (max(9.5, 3.8 + 0.8 * len(columns)), max(7.8, 4.2 + 0.8 * len(rows)))
    figure, axis = _layout.subplots(canvas, figsize=(width, height))
    figure.subplots_adjust(left=2.1 / width, right=0.87, top=1 - 1.4 / height, bottom=2.55 / height)
    matrix,limit=values['matrix'],values['limit']
    offset=settings['level']=='offset'
    colour = plt.get_cmap('RdBu_r' if offset else 'YlGnBu').with_extremes(bad=house_colour('blank'))
    painted = axis.imshow(matrix, cmap=colour, vmin=-limit if offset else 0, vmax=limit, interpolation='none', aspect='equal')
    for row in values['rows']:
        if np.isfinite(row['value']):
            count = '{} assessed cells'.format(int(row['within_cell_assessed'])) if settings['level']=='within-cell' else '{} multi-cell units'.format(int(row['within_sample_units'])) if settings['level']=='within-sample' else '{} cells / {} units'.format(int(row['cells']),int(row['units']))
            label = number(row['value']) + (' h' if offset else '') + '\n' + count
            if offset:
                label += '\nperiod ' + number(row['period_hours']) + ' h'
            axis.text(row['column_index'], row['row_index'], label, ha='center', va='center', fontsize=7, color='white' if abs(row['value']) / limit > 0.7 else 'black')
        elif row['requested']:
            axis.text(row['column_index'], row['row_index'], 'Unavailable', ha='center', va='center', fontsize=7)
    axis.set_xticks(range(len(columns)), [wrap(value, 24) for value in columns], rotation=45, ha='right', fontsize=9)
    axis.set_yticks(range(len(rows)), [wrap(value, 25) for value in rows], fontsize=9)
    axis.set_xlabel('Target measurement', fontsize=10)
    axis.set_ylabel('Reference measurement', fontsize=10)
    bar = figure.colorbar(painted, ax=axis, fraction=0.045, pad=0.055)
    bar.set_label('Offset (h)' if offset else 'Saved fraction / concentration', fontsize=9)
    bar.ax.tick_params(labelsize=8)
    _finish(figure, settings, height, extra=settings['unit'])
    return (figure, {'matrix': axis})

def _offset_rows(axis, values, *, title, comparable, limit):
    for index, row in enumerate(values):
        if np.isfinite(row['value']):
            if np.isfinite(row['lower']) and np.isfinite(row['upper']):
                axis.plot([row['lower'], row['upper']], [index, index], color=house_colour('okabe_bluish_green'), linewidth=1.3)
            axis.scatter([row['value']], [index], s=22, color=house_colour('okabe_bluish_green'), zorder=3)
        else:
            axis.text(0.03, index, 'Not comparable / unidentified', transform=axis.get_yaxis_transform(), fontsize=7, va='center')
    axis.set_yticks(range(len(values)), [wrap(value, 23) for value in [row['label'] for row in values]], fontsize=8)
    axis.set_ylim(max(0.5, len(values) - 0.5), -0.5)
    axis.set_xlim(-limit, limit)
    axis.axvline(0, color=house_colour('raw'), linewidth=1, zorder=0)
    axis.set_xlabel('Signed offset (h); positive target follows', fontsize=8)
    axis.set_title(wrap(title, 42), fontsize=10)
    if not values:
        axis.set_yticks([])
        axis.text(0.5, 0.5, 'No rows on this page', transform=axis.transAxes, ha='center', fontsize=9)
    if not comparable:
        axis.set_xlabel('Combined phase comparison unavailable', fontsize=8)
    axis.spines[['top', 'right']].set_visible(False)
    axis.tick_params(labelsize=8)

def _summary(values, settings, *, selected_view=None, canvas=None):
    cells,units,detection=[values['groups'][role] for role in ('cell','unit','detection')]
    names=['detection'] if selected_view=='detection' else ['cells','units'] if selected_view=='offsets' else ['detection','cells','units']
    row_count = max(len(cells), len(units))
    longest = max([1, *[len(wrap(label, 23).splitlines()) for label in values['labels']]])
    width, height = (15.0, max(8.8, 4.6 + row_count * max(0.28, 0.16 * longest)))
    figure = _layout.figure(canvas, figsize=(width, height))
    grid = figure.add_gridspec(1, len(names), left=0.07, right=0.98, bottom=2.55 / height, top=1 - 1.6 / height, width_ratios=[{'detection':.9,'cells':1.25,'units':1.15}[name] for name in names], wspace=0.78)
    axes = {name: figure.add_subplot(grid[0, i]) for i, name in enumerate(names)}
    if 'detection' in axes:
        axis = axes['detection']
        axis.barh(range(len(detection)), [row['value'] for row in detection], color=[house_colour('okabe_bluish_green'), house_colour('circadian_amber'), house_colour('blue'), house_colour('actogram_separator'), house_colour('shade'), house_colour('shade'), house_colour('shade')])
        axis.set_yticks(range(len(detection)), [row['label'] for row in detection], fontsize=8)
        axis.invert_yaxis()
        axis.set_xlim(0, max(1.0, float(max(row['value'] for row in detection))) * 1.2)
        for index, value in enumerate([row['value'] for row in detection]):
            axis.text(value, index, f' {int(value)}', va='center', fontsize=8)
        axis.set_title('Complete pair detection context', fontsize=10)
        axis.set_xlabel('Requested cells (all period strata)', fontsize=8)
        axis.spines[['top', 'right']].set_visible(False)
    limit=values['limit']
    if 'cells' in axes:_offset_rows(axes['cells'],cells,title='Comparable cell offsets; native bounds',comparable=settings['comparable'],limit=limit)
    if 'units' in axes:_offset_rows(axes['units'],units,title='Named-unit offsets; propagated input bounds',comparable=settings['comparable'],limit=limit)
    population = settings['population']
    extra = 'Observed named-unit mean: ' + number(population.get('mean_offset_hours')) + ' h; concentration ' + number(population.get('resultant')) + '.'
    region = population.get('sampling_region')
    if isinstance(region, dict):
        interval = region.get('direction_interval_hours')
        extra += ' Independent-sample direction: ' + (f'[{number(interval[0])}, {number(interval[1])}] h' if interval else 'unidentified')
        extra += f"; concentration bounds [{number(region.get('resultant_lower'))}, {number(region.get('resultant_upper'))}], confidence {number(population.get('confidence'))}."
    else:
        extra += ' Independent-sample bounds unavailable.'
    _finish(figure, settings, height, extra=extra)
    return (figure, axes)

def _traces(values, settings, *, selected_view=None, canvas=None):
    cells = values['cells']
    names=['timing'] if selected_view=='timing' else ['reference','target'] if selected_view=='traces' else ['reference','target','timing']
    width, height = (16.0, 3.7 + 2.9 * len(cells))
    figure = _layout.figure(canvas, figsize=(width, height))
    grid = figure.add_gridspec(len(cells), len(names), left=0.065, right=0.98, bottom=2.15 / height, top=1 - 1.5 / height, hspace=0.95, wspace=0.45)
    axes = {}
    for index,prepared in enumerate(cells):
        cell,bounds,timing=prepared['cell'],prepared['bounds'],prepared['timing']
        for column,role in enumerate(name for name in names if name!='timing'):
            axis=figure.add_subplot(grid[index,column])
            found=prepared['traces'][role]['evidence']
            saved_trace(axis,prepared['traces'][role]['trace'],settings['trace_view'],bounds,unavailable=found.get('processing_reason') if settings['trace_view']=='detrended' else 'No saved observations')
            axis.set_title(wrap(f"{cell['movie']} / cell {cell['identity']} | {role}: {found['measurement_label']}\n{found['display_state']}; adjusted p {number(found.get('q_value'))}", 57), fontsize=9)
            axes[f'{index}-{role}'] = axis
        if 'timing' not in names:continue
        axis=figure.add_subplot(grid[index,names.index('timing')])
        if prepared['has_timing']:
            for x,y in prepared['segments']:
                axis.plot(x,y,color=house_colour('okabe_bluish_green'),linewidth=1.,marker='.',markersize=1.5)
        else:
            axis.text(.5,.5,wrap(timing.get('reason') or 'No saved temporal phase',54),transform=axis.transAxes,ha='center',va='center',fontsize=8)
        if bounds and bounds['max'] > bounds['min']:
            axis.set_xlim(bounds['min'], bounds['max'])
        axis.set_xlabel('Recording time (h)', fontsize=8)
        axis.set_ylabel('Target follows reference (h)', fontsize=8)
        axis.set_title(wrap('Observed temporal offset | ' + str(timing.get('temporal_status') or 'unavailable') + '\nSupported reporting period: ' + number(timing.get('period_hours')) + ' h', 55), fontsize=9)
        axis.axhline(0, color=house_colour('raw'), linewidth=0.8, zorder=0)
        axis.tick_params(labelsize=8)
        axis.spines[['top', 'right']].set_visible(False)
        axes[f'{index}-timing'] = axis
    _finish(figure, settings, height)
    return (figure, axes)
