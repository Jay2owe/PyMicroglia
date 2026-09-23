"""Portable original trace cards and complete selected-cell grids."""
from . import _layout
from ._layout import finish
from . import colour as house_colour
from pathlib import Path
import math, textwrap
import numpy as np
import matplotlib.pyplot as plt

def finite(value):
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def number(value):
    return f'{float(value):.3g}' if finite(value) else 'unavailable'

def probability(value):
    return 'below numeric precision' if finite(value) and float(value) == 0 else number(value)

def truth(value):
    return value is True or (isinstance(value, (np.bool_, str)) and str(value).lower() == 'true')

def wrap(value, width=115):
    return '\n'.join((textwrap.fill(line, width) for line in str(value).splitlines()))

def trace(axis,prepared,coordinate,anchor,unit):
    for x,y in prepared['segments']:axis.plot(x,y,color=house_colour('circadian_teal'),lw=1.,marker='.',ms=2.8)
    for window in prepared['windows']:axis.axvspan(window['low'],window['high'],color=house_colour('blue') if window['baseline'] else house_colour('circadian_amber'),alpha=.075,zorder=-5)
    axis.axvline(0 if coordinate=='relative_hours' else anchor['hours'],color=house_colour('dark'),ls='--',lw=1.)
    if not prepared['segments']:axis.text(.5,.5,'Original trace unavailable' if not prepared['counts']['original_points'] else 'No valid original-clock values',transform=axis.transAxes,ha='center',va='center',fontsize=10)
    axis.set_xlabel('Hours from '+('control anchor' if anchor['kind']=='control' else 'intervention') if coordinate=='relative_hours' else 'Original recording hours',fontsize=12)
    axis.set_ylabel(wrap(unit or 'Original units',20),fontsize=12);axis.tick_params(labelsize=10);finish(axis)
    return prepared['counts']

def evidence_lines(rows, settings):
    lines = []
    for row in rows:
        kind = row['kind']
        if kind == 'window':
            bounds = number(row['start']) + ' to ' + number(row['end']) + ' frames' if row['coordinate'] == 'frames' else number(row['start_hours']) + ' to ' + number(row['end_hours']) + ' original hours'
            lines.append('Window ' + str(row['window']) + ' [' + bounds + '; end excluded]: ' + str(row['status']).replace('_', ' ') + '; original valid observations ' + number(row.get('valid_observations')) + '/' + number(row.get('observations')) + '; elapsed coverage ' + number(row.get('coverage_fraction')) + '; gaps ' + number(row.get('gap_count')) + '.')
        elif kind == 'effect':
            lines.extend(['Measurement change | ' + str(row['outcome']).replace('_', ' ') + ': observed ' + number(row.get('absolute_change')) + '; model ' + number(row.get('estimate')) + ' [' + number(row.get('interval_low')) + ', ' + number(row.get('interval_high')) + ']; q ' + probability(row.get('q_value')) + '.', '  Saved method ' + str(row.get('method')) + '; correction ' + str(row.get('correction')) + '; family ' + number(row.get('family_tested')) + '/' + number(row.get('family_requested')) + ' tested. ' + str(row.get('reason', ''))])
        elif kind == 'timing':
            lines.append('Observed timing | ' + str(row.get('status')).replace('_', ' ') + ': first qualifying response ' + number(row.get('response_delay_hours')) + ' h [' + number(row.get('response_lower_hours')) + ', ' + number(row.get('response_upper_hours')) + ']; sustained return ' + number(row.get('recovery_delay_hours')) + ' h. ' + str(row.get('recovery_status')).replace('_', ' ') + '. Sampling brackets are not confidence intervals.')
        elif kind == 'rhythm_window':
            lines.append('Window rhythm ' + str(row['window']) + ' | period ' + number(row.get('period_hours')) + ' h; estimate supported ' + str(truth(row.get('period_supported'))) + '; ' + str(row.get('detection_outcome')) + '; q ' + probability(row.get('q_value')) + '; estimator ' + str(row.get('method')) + ', test ' + str(row.get('significance_method')) + '. ' + str(row.get('window_status')).replace('_', ' ') + '.')
        elif kind == 'rhythm_change':
            lines.append('Direct rhythm ' + str(row['property']) + ' change | ' + str(row.get('outcome')).replace('_', ' ') + ': ' + number(row.get('effect')) + ' ' + str(row.get('unit')) + ' [' + number(row.get('interval_low')) + ', ' + number(row.get('interval_high')) + ']; q ' + probability(row.get('q_value')) + '. ' + str(row.get('reason', '')))
    for step, label in [('response-timing', 'Response timing'), ('rhythm-changes', 'Window rhythms')]:
        branch = settings['branches'][step]
        if branch['status'] != 'available':
            lines.append(label + ': ' + branch['status'] + '. ' + branch['reason'])
    return [line for text in lines for line in wrap(text, 140).splitlines()]

def draw(prepared, settings, *, selected_view=None, canvas=None):
    axes = {}
    view = settings['view']
    if view == 'cell_report':
        lines = prepared['lines']
        height = max(10.5, 6.8 + 0.18 * len(lines))
        figure = _layout.figure(canvas, figsize=(14, height))
        anchor = settings['anchor']
        clocks=[] if selected_view=='evidence' else ['hours'] if selected_view=='original' else ['relative_hours'] if selected_view=='anchored' else ['hours','relative_hours']
        counts=prepared['traces']['hours']['counts']
        for index, coordinate in enumerate(clocks):
            axis = figure.add_axes([0.085 + index * 0.49, 1 - 4.65 / height, 0.395 if len(clocks)==2 else .83, 2.85 / height])
            axes[coordinate] = axis
            counts = trace(axis, prepared['traces'][coordinate], coordinate, anchor, settings.get('unit'))
            axis.set_title('Original clock' if index == 0 else 'Relative to the declared anchor', fontsize=12, pad=12)
        description = anchor['label'] + ' (' + anchor['kind'] + ') at original hour ' + number(anchor['hours']) + '. ' + str(counts['original_points']) + ' original rows; ' + str(counts['unplotted_invalid_points']) + ' invalid values/clocks retained in the data.'
        figure.text(0.035, 1 - 1.0 / height, wrap(description, 140), fontsize=10, va='top')
        if selected_view in {None,'evidence'}:figure.text(0.035, 1 - (1.8 if selected_view=='evidence' else 5.55) / height, '\n'.join(lines), fontsize=10, linespacing=1.25, va='top')
    else:
        ids = settings['cell_ids']
        nrows = math.ceil(len(ids) / 2)
        height = 2.8 + 3.35 * nrows
        figure, grid = _layout.subplots(canvas, nrows, 2, figsize=(14, height), squeeze=False, sharey=settings['shared_y'])
        figure.subplots_adjust(left=0.07, right=0.98, top=1 - 1.05 / height, bottom=1.5 / height, hspace=0.8, wspace=0.28)
        for index, axis in enumerate(grid.ravel()):
            if index >= len(ids):
                axis.axis('off')
                continue
            key = ids[index]
            member = prepared['cells'][key]
            clock='hours' if selected_view=='original' else 'relative_hours' if selected_view=='anchored' else settings['grid_clock']
            axes[key] = axis
            counts = trace(axis, member['traces'][clock], clock, settings['anchors'][key], settings.get('unit'))
            if settings.get('y_limits') is not None:
                axis.set_ylim(settings['y_limits'])
            supported,changed=member['supported'],member['changed']
            axis.set_title(wrap(member['label'], 55) + '\n' + str(supported) + ' measurement / ' + str(changed) + ' rhythm contrasts supported; ' + str(counts['unplotted_invalid_points']) + ' invalid rows', fontsize=10, pad=10)
    figure.text(0.035, 1 - 0.25 / height, wrap(settings['title'], 110), va='top', fontsize=15, weight='bold')
    figure.text(0.035, 0.58 / height, wrap(settings['footnote'], 160), va='top', fontsize=8.5)
    return (figure, axes)
