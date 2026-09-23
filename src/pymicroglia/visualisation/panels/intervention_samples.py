"""Portable sample summaries and paired-response displays; no analysis calls."""
from . import _layout
from ._layout import finish
from . import colour as house_colour
from pathlib import Path
import math, textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
COLOURS = {'both_supported_same_direction': house_colour('circadian_teal'), 'both_supported_opposite_directions': house_colour('blue'), 'reference_supported_only': house_colour('circadian_red'), 'target_supported_only': house_colour('circadian_amber'), 'neither_has_a_detected_response': house_colour('nan_text'), 'reference_inconclusive': house_colour('actogram_dark'), 'target_inconclusive': house_colour('muted'), 'both_inconclusive': house_colour('raw')}
OUTCOMES = {'increase': (house_colour('circadian_red'), '+'), 'decrease': (house_colour('blue'), '-'), 'no_detected_change': (house_colour('shade'), 'nd'), 'inconclusive': (house_colour('actogram_dark'), '?'), 'detected_below_meaningful_threshold': (house_colour('actogram_dark'), 'below')}

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

def short(value, width=95):
    return wrap(textwrap.shorten(str(value).replace('_', ' '), width=width, placeholder='...'), 43)

def sample_label(row):
    name = row.get('sample')
    name = name if isinstance(name, str) and name else row.get('movies') or row.get('recordings')
    return ('Sample ' if truth(row.get('sample_confirmed')) else 'Unconfirmed recording ') + str(name)

def draw(prepared, settings, *, canvas=None):
    all_rows,rows = prepared['all_rows'],prepared['rows']
    view = settings['view']
    height = max(8.0, 3.8 + 0.85 * len(rows))
    figure, axis = _layout.subplots(canvas, figsize=(14, height))
    figure.subplots_adjust(left=0.23, right=0.66, top=1 - 1.6 / height, bottom=1.45 / height)
    if view == 'coverage':
        axis.axis('off')
        axis.set_xlim(0, 1)
        axis.set_ylim(len(rows), -0.5)
        figure.subplots_adjust(left=0.06, right=0.96)
        for y, row in enumerate(rows):
            axis.text(0, y, row['label'], fontsize=12, weight='bold', va='center')
            detail = row['status'] + '\n' + wrap(row['reason'], 63) + '\nOriginal requested cells ' + number(row['requested_cells'])
            if finite(row.get('saved_unit_rows')):
                detail += '; original units ' + number(row['saved_unit_rows'])
            if finite(row.get('saved_cell_pairs')):
                detail += '; cell pairs ' + number(row['saved_cell_pairs']) + '; sample pairs ' + number(row['saved_sample_pairs'])
            axis.text(0.45, y, detail, fontsize=9, va='center')
    elif view in {'cell_pairs', 'sample_pairs'}:
        for index, row in enumerate(rows):
            x, y = (row.get('reference_value'), row.get('target_value'))
            available = truth(row.get('paired_available'))
            colour = COLOURS.get(row.get('joint_response'), house_colour('circadian_teal') if truth(row.get('formal_eligible')) else house_colour('nan_text'))
            if finite(x) and finite(y):
                axis.plot(x, y, marker='o' if available else 'x', ls='', color=colour, ms=6)
                axis.annotate(str(settings['page_start'] + index), (x, y), xytext=(4, 4), textcoords='offset points', fontsize=8)
            label = row['cell_label'] if view == 'cell_pairs' else sample_label(row)
            status = str(row['joint_response']).replace('_', ' ') if view == 'cell_pairs' else 'paired cells ' + number(row.get('paired_available_cells')) + '/' + number(row.get('requested_cells')) + '; recordings ' + number(row.get('included_recordings')) + '/' + number(row.get('requested_recordings'))
            if not finite(x) or not finite(y):
                status += '; missing paired value'
            elif not available:
                status += '; original support unavailable'
            axis.text(1.07, 1 - (index + 0.5) / len(rows), str(settings['page_start'] + index) + '. ' + wrap(label, 43) + '\n' + wrap(status, 43), transform=axis.transAxes, va='center', fontsize=8.5)
        definition = settings['definition']
        axis.set_xlabel(definition['reference'] + ' ' + definition['quantity'].replace('_', ' ') + ' (' + (definition.get('reference_unit') or 'original units') + ')', fontsize=12)
        axis.set_ylabel(definition['target'] + ' ' + definition['quantity'].replace('_', ' ') + ' (' + (definition.get('target_unit') or 'original units') + ')', fontsize=12)
        axis.axhline(0, color=house_colour('shade'), ls='--', lw=0.8)
        axis.axvline(0, color=house_colour('shade'), ls='--', lw=0.8)
        finish(axis)
        association = next((row for row in all_rows if row['kind'] == 'association'), None)
        if association:
            note = 'Saved all-sample association: ' + str(association.get('statistic')) + ' ' + number(association.get('estimate')) + '; q ' + probability(association.get('q_value')) + '; eligible original samples ' + number(association.get('eligible_samples')) + '/' + number(association.get('confirmed_samples'))
            note += '\nMethod ' + str(association.get('method')) + '; correction ' + str(association.get('correction')) + '. ' + short(association.get('reason'), 100)
        else:
            note = 'Descriptive original-cell pairs; this display performs no association test. Missing pairs remain listed; x marks indicate unavailable original support.'
        figure.text(0.035, 1 - 0.92 / height, wrap(note, 160), fontsize=8.5, va='top')
    elif view == 'joint_outcomes':
        keys = list(OUTCOMES)
        matrix=prepared['matrix']
        axis.imshow(matrix, cmap=ListedColormap([OUTCOMES[key][0] for key in keys]), vmin=-0.5, vmax=len(keys) - 0.5, aspect='auto', interpolation='nearest')
        axis.set_yticks(range(len(rows)), [wrap(row['cell_label'], 32) for row in rows], fontsize=9)
        axis.set_xticks([0, 1], [wrap(settings['definition'][key], 22) for key in ['reference', 'target']], fontsize=10)
        for y, row in enumerate(rows):
            for x, role in enumerate(['reference', 'target']):
                key = row[role + '_outcome']
                axis.text(x, y, OUTCOMES[key][1], ha='center', va='center', fontsize=11, color='white' if key in {'increase', 'decrease'} else house_colour('dark'))
            axis.text(1.04, y, short(row['joint_response']) + '\nsaved q ' + probability(row.get('reference_q_value')) + ' / ' + probability(row.get('target_q_value')), transform=axis.get_yaxis_transform(), fontsize=8.5, va='center')
        axis.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
        axis.grid(which='minor', color='white', lw=1.5)
        axis.tick_params(which='minor', bottom=False, left=False)
        figure.text(0.035, 1 - 1.0 / height, '+ supported increase; - supported decrease; nd no detected change; ? inconclusive; below detected but below declared magnitude.', fontsize=9, va='top')
    else:
        labels = [short(row['name'] + ' | ' + row['reference_condition'] + ' to ' + row['target_condition'], 75) if view == 'control_effects' else wrap(sample_label(row), 34) for row in rows]
        axis.set_ylim(len(rows) - 0.5, -0.6)
        axis.set_yticks(range(len(rows)), labels, fontsize=9)
        for y, row in enumerate(rows):
            value = row.get('effect') if view == 'control_effects' else row.get('value') if view == 'sample_changes' else row.get('both_supported_fraction_of_jointly_tested')
            confirmed = truth(row.get('sample_confirmed')) if view != 'control_effects' else True
            color = house_colour('circadian_teal') if confirmed and (view != 'control_effects' or truth(row.get('supported'))) else house_colour('nan_text')
            if finite(value):
                if view == 'control_effects' and finite(row.get('interval_low')) and finite(row.get('interval_high')):
                    axis.plot([row['interval_low'], row['interval_high']], [y, y], color=color, lw=2)
                axis.plot(value, y, marker='o' if confirmed else 'x', color=color, ms=6, ls='')
            else:
                axis.text(0.02, y, 'Unavailable', transform=axis.get_yaxis_transform(), color=house_colour('circadian_amber'), fontsize=9, va='center')
            if view == 'control_effects':
                note = 'q ' + probability(row.get('q_value')) + '; ' + str(row.get('design')) + '\nsamples ' + number(row.get('reference_samples')) + '/' + number(row.get('target_samples')) + '; cells ' + number(row.get('reference_cells')) + '/' + number(row.get('target_cells'))
                if row.get('design') == 'matched':
                    note += '; complete matches ' + number(row.get('complete_matches'))
                note += '\n' + short(row.get('reason'))
            elif view == 'sample_changes':
                note = str(row.get('condition') or 'No declared condition') + '\ncells ' + number(row.get('available_cells')) + '/' + number(row.get('requested_cells')) + '; recordings ' + number(row.get('available_recordings')) + '/' + number(row.get('requested_recordings')) + '\n' + short(row.get('status'))
            else:
                note = 'both supported ' + number(row.get('both_supported_cells')) + ' / jointly tested ' + number(row.get('jointly_tested_cells')) + '\npaired available ' + number(row.get('paired_available_cells')) + ' / requested cells ' + number(row.get('requested_cells')) + '\n' + str(row.get('experimental_unit')).replace('_', ' ')
            axis.text(1.04, y, note, transform=axis.get_yaxis_transform(), va='center', fontsize=8.5)
            axis.axhline(y + 0.49, color=house_colour('blank'), lw=0.5)
        if view == 'recurrence':
            axis.set_xlim(-0.03, 1.03)
            axis.set_xlabel('Saved fraction among jointly tested original cells', fontsize=12)
        else:
            axis.axvline(0, color=house_colour('muted'), ls='--', lw=1)
            axis.set_xlabel('Target minus reference sample change' if view == 'control_effects' else settings['definition']['quantity'].replace('_', ' ') + ' (' + str(settings['unit'] or 'original units') + ')', fontsize=12)
        finish(axis)
    figure.text(0.035, 1 - 0.25 / height, wrap(settings['title'], 120), va='top', weight='bold', fontsize=15)
    if view not in {'coverage', 'cell_pairs', 'sample_pairs', 'joint_outcomes'}:
        figure.text(0.035, 1 - 1.0 / height, str(len(rows)) + ' displayed / ' + str(settings['total_requested']) + ' original rows. Original sample membership and all unavailable outcomes are retained.', fontsize=9, va='top')
    figure.text(0.035, 0.68 / height, wrap(settings['footnote'], 155), va='top', fontsize=8.5)
    return (figure, {'evidence': axis})
