"""Portable saved timing brackets, window estimates and direct contrasts."""
from . import _layout
from ._layout import finish
from . import colour as house_colour
from pathlib import Path
import math, textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

def wrap(value, width=125):
    return '\n'.join((textwrap.fill(line, width) for line in str(value).splitlines()))

def reason(value):
    return wrap(textwrap.shorten(str(value).replace('_', ' '), width=118, placeholder='...'), 45)

def draw(prepared, settings, *, canvas=None):
    rows = prepared['rows']
    view = settings['view']
    height = max(7.8, 3.9 + 1.0 * len(rows))
    figure, axis = _layout.subplots(canvas, figsize=(14, height))
    figure.subplots_adjust(left=0.22, right=0.67, top=1 - 1.55 / height, bottom=1.4 / height)
    if view == 'coverage':
        axis.axis('off')
        axis.set_xlim(0, 1)
        axis.set_ylim(len(rows), -0.5)
        figure.subplots_adjust(left=0.06, right=0.96)
        for y, row in enumerate(rows):
            axis.text(0, y, row['label'], fontsize=12, va='center', weight='bold')
            axis.text(0.45, y, row['status'] + '\n' + wrap(row['reason'], 63), fontsize=10, va='center')
            axis.text(0.45, y + 0.45, 'Original cells ' + number(row['original_cells']) + '; windows ' + number(row.get('window_rows')) + '; comparisons ' + number(row.get('comparison_rows')) + '; direct contrasts ' + number(row.get('direct_rows')), fontsize=9)
    else:
        axis.set_ylim(len(rows) - 0.5, -0.6)
        axis.set_yticks(range(len(rows)), [wrap(row['cell_label'] + (' / ' + str(row['window']) if view.startswith('window_') else ''), 32) for row in rows], fontsize=9)
        for y, row in enumerate(rows):
            if view in {'response_delay', 'recovery'}:
                prefix = 'response' if view == 'response_delay' else 'recovery'
                observed = truth(row.get(prefix + '_observed'))
                value = row.get(prefix + '_delay_hours')
                low, high = (row.get(prefix + '_lower_hours'), row.get(prefix + '_upper_hours'))
                if observed and finite(value):
                    color = house_colour('circadian_teal') if prefix == 'response' else house_colour('blue')
                    if finite(low) and finite(high):
                        axis.plot([low, high], [y, y], color=color, lw=2.0)
                    axis.plot(value, y, 'o', color=color, ms=6, label='Observed event')
                else:
                    endpoint = row.get('observed_endpoint_relative_hours')
                    if finite(endpoint):
                        axis.plot(endpoint, y, marker='|', color=house_colour('nan_text'), ms=12, mew=2, label='Last observation')
                    axis.text(0.015, y + 0.22, 'Event not observed' if row.get(prefix + '_status') != 'not_requested' else 'Not requested', transform=axis.get_yaxis_transform(), fontsize=8, color=house_colour('nan_text'))
                status = row.get('response_timing_status') if prefix == 'response' else row.get('recovery_status')
                annotation = reason(status) + '\n' + ('event ' + number(value) + ' h; bracket ' + number(low) + ' to ' + number(high) + ' h' if observed else 'last observation ' + number(row.get('observed_endpoint_relative_hours')) + ' h')
                annotation += '\n' + reason(row.get('observation_status'))
                if prefix == 'response':
                    annotation += '\nthreshold ' + number(row.get('response_threshold')) + ' ' + str(row.get('unit', '')) + '; persistence ' + number(row.get('response_persistence_hours')) + ' h'
                else:
                    annotation += '\ntolerance ' + number(row.get('recovery_tolerance')) + ' ' + str(row.get('unit', '')) + '; persistence ' + number(row.get('recovery_persistence_hours')) + ' h'
            elif view.startswith('window_'):
                field = 'period_hours' if view == 'window_period' else 'amplitude'
                value = row.get(field)
                supported = truth(row.get('period_supported'))
                color = house_colour('circadian_teal') if supported else house_colour('grade_grey')
                if finite(value):
                    axis.plot(value, y, 'o', color=color, mfc=color if supported else 'white', ms=6)
                else:
                    axis.text(0.02, y, 'Estimate unavailable', transform=axis.get_yaxis_transform(), fontsize=9, color=house_colour('circadian_amber'))
                annotation = ('Period estimate supported' if supported else 'Period unresolved / insufficient') + '; ' + str(row.get('detection_outcome')) + '\nq ' + probability(row.get('q_value')) + '; cycles ' + number(row.get('cycles_observed')) + ' / min ' + number(row.get('min_cycles'))
                annotation += '\nestimator ' + str(row.get('method')) + '; test ' + str(row.get('significance_method')) + '\n' + reason(row.get('window_status'))
            else:
                value = row.get('effect')
                low, high = (row.get('interval_low'), row.get('interval_high'))
                color = house_colour('circadian_teal') if truth(row.get('change_supported')) else house_colour('nan_text')
                if finite(value):
                    if finite(low) and finite(high):
                        axis.plot([low, high], [y, y], color=color, lw=2)
                    axis.plot(value, y, 'o', color=color, ms=6)
                else:
                    axis.text(0.02, y, 'Untestable', transform=axis.get_yaxis_transform(), fontsize=9, color=house_colour('circadian_amber'))
                annotation = reason(row.get('outcome')) + '\nq ' + probability(row.get('q_value')) + '; family ' + number(row.get('family_tested')) + '/' + number(row.get('family_requested')) + ' tested'
                annotation += '\ncomponent band ' + str(row.get('component_period_band')) + ' h'
                if row.get('property') == 'phase':
                    annotation += '; reference ' + number(row.get('phase_reference_hours')) + ' h'
                if not finite(row.get('p_value')):
                    annotation += '\n' + reason(row.get('reason'))
            axis.text(1.035, y, annotation, transform=axis.get_yaxis_transform(), fontsize=8.5, va='center')
            axis.axhline(y + 0.49, color=house_colour('blank'), lw=0.5)
        if view in {'response_delay', 'recovery'}:
            axis.set_xlabel("Hours from each recording's declared intervention or control anchor", fontsize=12)
            handles = [Line2D([], [], color=house_colour('circadian_teal') if view == 'response_delay' else house_colour('blue'), marker='o', lw=2, label='Observed event / sampling bracket'), Line2D([], [], color=house_colour('nan_text'), marker='|', ls='', ms=10, label='Last observation, event unobserved')]
            figure.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.22, 1 - 1.12 / height), ncol=2, fontsize=9)
        elif view.startswith('window_'):
            axis.set_xlabel('Saved period estimate (hours)' if view == 'window_period' else 'Saved fitted amplitude (original measurement units)', fontsize=12)
        else:
            units = rows[0].get('unit') or 'original units'
            axis.set_xlabel('Follow-up minus baseline ' + settings['property'] + ' (' + str(units) + ')', fontsize=12)
            axis.axvline(0, color=house_colour('muted'), ls='--', lw=1)
        finish(axis)
        axis.set_ylabel('Original cell / window' if view.startswith('window_') else 'Original cell', fontsize=12)
    figure.text(0.035, 1 - 0.3 / height, wrap(settings['title'], 115), va='top', weight='bold', fontsize=15)
    if view not in {'coverage', 'response_delay', 'recovery'}:
        methods = sorted({str(row.get('method')) for row in rows})
        versions = sorted({str(row.get('workbench_version')) for row in rows})
        correction = sorted({str(row.get('correction')) for row in rows})
        figure.text(0.035, 1 - 1.0 / height, wrap('Saved methods ' + ', '.join(methods) + '; correction ' + ', '.join(correction) + '; Circadian Workbench ' + ', '.join(versions) + '. Original requested rows ' + str(settings['total_requested']) + '.', 155), fontsize=9, va='top')
    figure.text(0.035, 0.68 / height, wrap(settings['footnote'], 155), fontsize=8.5, va='top')
    return (figure, {'evidence': axis})
