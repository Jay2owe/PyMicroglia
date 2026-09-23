"""Portable drawing of saved time quantities; no inference or aggregation."""
from . import _layout
from ._layout import finish
from . import colour as house_colour
import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from ._format import numeric, missing, present, number as display_number
from .behaviour_profiles import state_colour, unpack, number, wrap
KEYS = ['source_run', 'movie', 'identity']

def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def cell(frame, key):
    mask = np.ones(len(frame), dtype=bool)
    for name in KEYS:
        mask &= frame[name].eq(key[name]).to_numpy()
    return frame.loc[mask]

def names(settings):
    return {row['state_id']: settings['state_display_names'].get(row['state_id'], row['label']) for row in settings['state_definitions']}

def label(key):
    return wrap(str(key['movie']) + ' | cell ' + str(key['identity']), 31)

def tidy(axis):
    for side in ['top', 'right']:
        axis.spines[side].set_visible(False)
    axis.tick_params(labelsize=8, width=0.8)

def _matrix(axis, values, labels, rows, heading, texts=None, probability=True):
    palette = plt.get_cmap('Blues').copy()
    palette.set_bad(house_colour('blank'))
    axis.imshow(values['values'], aspect='auto', interpolation='none', cmap=palette, vmin=0, vmax=values['maximum'])
    axis.set_xticks(range(len(labels)), [wrap(item, 16) for item in labels], fontsize=8)
    axis.set_yticks(range(len(rows)), rows, fontsize=8)
    axis.set_title(heading, fontsize=10, pad=10)
    for y in range(len(rows)):
        for x in range(len(labels)):
            axis.text(x, y, values['labels'][y][x], ha='center', va='center', fontsize=8, color='white' if values['bright'][y,x] else house_colour('dark'))
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0)

def _cell_figure(settings, columns=1, top=1.2, ratios=None, *, canvas=None):
    height = max(5.5, 0.43 * len(settings['cells']) + top + 1.5)
    figure, array = _layout.subplots(canvas, 1, columns, figsize=(15, height), squeeze=False, gridspec_kw={'width_ratios': ratios} if ratios else None)
    figure.subplots_adjust(left=0.25, right=0.965, top=1 - top / height, bottom=1.15 / height, wspace=0.24)
    return (figure, array[0])

def occupancy(values, settings, *, canvas=None):
    states = settings['states']
    state_names = names(settings)
    figure, axes = _cell_figure(settings, 3, ratios=[len(states), len(states), 1.25], canvas=canvas)
    arrays, unknown, labels = values['arrays'], values['unknown'], values['labels']
    _matrix(axes[0], arrays[0], [state_names[state] for state in states], labels, 'Fraction of all observed time')
    _matrix(axes[1], arrays[1], [state_names[state] for state in states], [''] * len(labels), 'Fraction of assigned time')
    _matrix(axes[2], unknown, ['Unknown'], [''] * len(labels), 'Unknown / observed')
    return (figure, {'observed': axes[0], 'assigned': axes[1], 'unknown': axes[2]})

def switch_rates(values, settings, *, canvas=None):
    figure, array = _cell_figure(settings, canvas=canvas)
    axis = array[0]
    labels = []
    for index, item in enumerate(values):
        row = item['row']
        labels.append(item['label'])
        if finite(row.value):
            axis.scatter([row.value], [index], color=house_colour('blue'), s=24)
        else:
            axis.text(0.02, index, 'Unavailable', transform=axis.get_yaxis_transform(), va='center', color=house_colour('nan_text'), fontsize=8)
        axis.text(1.02, index, number(row.numerator) + ' switches / ' + number(row.denominator) + ' h', transform=axis.get_yaxis_transform(), va='center', fontsize=8)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_ylim(len(labels) - 0.5, -0.5)
    axis.set_xlim(left=-0.02)
    axis.set_xlabel('Switches per eligible transition hour', fontsize=10)
    axis.set_title('Observation spacing: ' + number(settings['spacing']) + ' h', fontsize=10)
    figure.subplots_adjust(right=0.78)
    tidy(axis)
    return (figure, {'switch_rate': axis})

def transitions(values, settings, *, canvas=None):
    figure, axes = _cell_figure(settings, 2, canvas=canvas)
    state_names = names(settings)
    counts, probabilities, shape = values['counts'], values['probabilities'], values['shape']
    columns = [state_names[pair['source']] + ' to\n' + state_names[pair['target']] for pair in settings['pairs']]
    spacing = '\nObservation spacing: ' + number(settings['spacing']) + ' h'
    _matrix(axes[0], counts, columns, [label(key) for key in settings['cells']], 'Count / source opportunities' + spacing)
    _matrix(axes[1], probabilities, columns, [''] * shape[0], 'Next-state probability' + spacing)
    return (figure, {'counts': axes[0], 'probabilities': axes[1]})

def bout_marker(row):
    onset, ending = (row['onset_observed'] == True, row['ending_observed'] == True)
    return 'o' if onset and ending else '<' if ending else '>' if onset else 'x'

def bouts(values, settings, *, canvas=None):
    definitions = {row['state_id']: row for row in settings['state_definitions']}
    handles = [Line2D([], [], marker='o', color=state_colour(row['component']), linestyle='none', label=names(settings)[row['state_id']]) for row in settings['state_definitions']]
    top = 1.2 + 0.22 * math.ceil(len(handles) / 4)
    figure, array = _cell_figure(settings, top=top, canvas=canvas)
    axis = array[0]
    labels = []
    for index, item in enumerate(values):
        labels.append(item['label'])
        for mark in item['marks']:
            axis.scatter([mark['x']], [mark['y']], marker=mark['marker'], color=state_colour(definitions[mark['state_id']]['component']), s=22, alpha=0.8)
        if item['unavailable']:
            axis.text(1.02, index, str(item['unavailable']) + ' without duration', transform=axis.get_yaxis_transform(), va='center', fontsize=8)
        if item['empty']:
            axis.text(0.02, index, 'No observed bouts', transform=axis.get_yaxis_transform(), va='center', color=house_colour('nan_text'), fontsize=8)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_ylim(len(labels) - 0.6, -0.6)
    axis.set_xlim(left=-0.02)
    axis.set_xlabel('Observed bout allocation (h)', fontsize=10)
    tidy(axis)
    figure.subplots_adjust(right=0.84)
    figure.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.245, 1 - 0.58 / figure.get_figheight()), ncol=4, fontsize=8, frameon=False)
    return (figure, {'bouts': axis})

def quantity(row, settings):
    state_names = names(settings)
    metric = row['metric']
    text = {'occupancy_observed': 'State time / observed time', 'occupancy_assigned': 'State time / assigned time', 'unknown_fraction': 'Unknown time / observed time', 'switch_rate': 'Switches / eligible hour', 'transition_probability': 'Next-state probability'}[metric]
    if row.get('state_id') in state_names:
        text += ': ' + state_names[row['state_id']]
    if row.get('target_state_id') in state_names:
        text += ' to ' + state_names[row['target_state_id']]
    if finite(row.get('interval_hours')):
        text += ' (' + number(row['interval_hours']) + ' h spacing)'
    return text

def samples(values, settings, *, canvas=None):
    figure, array = _layout.subplots(canvas, len(settings['questions']), 1, figsize=(15, 2.9 * len(settings['questions']) + 2.3), squeeze=False)
    figure.subplots_adjust(left=0.11, right=0.965, top=1 - 1.0 / figure.get_figheight(), bottom=1.5 / figure.get_figheight(), hspace=0.85)
    axes = {}
    for index, question in enumerate(settings['questions']):
        axis = array[index, 0]
        axes[question] = axis
        item = values[index]
        positions, labels = item['positions'], item['labels']
        for ordinal, row in enumerate(item['rows']):
            marker = 'o' if row['sample_confirmed'] == True else 'x'
            colour = house_colour('circadian_teal') if row['independent_of_state_choice'] == True else house_colour('blue') if row['sample_confirmed'] == True else house_colour('nan_text')
            if finite(row['value']):
                axis.scatter([positions[ordinal]], [row['value']], marker=marker, edgecolors=colour if marker == 'o' else None, facecolors=colour if row['independent_of_state_choice'] == True or marker == 'x' else 'none', s=22)
            else:
                axis.plot([positions[ordinal]], [0.015], '|', color=house_colour('grade_grey'), transform=axis.get_xaxis_transform(), markersize=6)
        axis.set_xticks(range(len(labels)), labels)
        axis.set_xlim(-0.55, len(labels) - 0.45)
        axis.set_title(item['title'], fontsize=9)
        axis.set_ylabel('Saved unit value', fontsize=9)
        axis.set_xlabel('One point per experimental unit; horizontal spread is for visibility; grey ticks mark unavailable values', fontsize=8)
        if item['metric'] != 'switch_rate':
            axis.set_ylim(-0.06, 1.06)
        tidy(axis)
    return (figure, axes)

def contrasts(values, settings, *, canvas=None):
    if not values:
        figure, axis = _layout.subplots(canvas, figsize=(15, 5))
        axis.axis('off')
        axis.text(0.5, 0.5, 'No condition contrasts were declared or available', ha='center', va='center', transform=axis.transAxes)
        return (figure, {'contrast_status': axis})
    height = 1.8 + 1.45 * len(settings['comparisons'])
    figure, array = _layout.subplots(canvas, len(settings['comparisons']), 1, figsize=(15, height), squeeze=False)
    figure.subplots_adjust(left=0.08, right=0.43, top=1 - 0.9 / height, bottom=1.0 / height, hspace=1.4)
    axes = {}
    for index, comparison in enumerate(settings['comparisons']):
        axis = array[index, 0]
        axes[comparison] = axis
        item = values[index]
        row, interval, interval_label = item['row'], item['interval'], item['interval_label']
        if finite(row.effect):
            axis.scatter([row.effect], [0], color=house_colour('blue'), s=24)
            if interval is not None:
                axis.plot(interval, [0, 0], color=house_colour('blue'), lw=2)
        else:
            axis.text(0.5, 0.5, 'Formal effect unavailable', ha='center', va='center', transform=axis.transAxes, fontsize=9)
        if finite(row.descriptive_effect):
            axis.scatter([row.descriptive_effect], [-0.55], marker='D', facecolors='none', edgecolors=house_colour('nan_text'), s=22)
        axis.set_xlim(*item['bounds'])
        axis.set_ylim(-0.85, 0.45)
        axis.axvline(0, color=house_colour('raw'), lw=0.6)
        axis.set_yticks([0, -0.55], ['Formal', 'All samples'], fontsize=8)
        axis.set_title(wrap(str(row.target_condition) + ' minus ' + str(row.reference_condition) + ' | ' + quantity(row, settings), 71), fontsize=9, loc='left')
        text = str(row.status).replace('_', ' ') + ' | formal samples ' + str(int(row.reference_samples)) + ' / ' + str(int(row.target_samples))
        text += '\np=' + number(row.p_value) + '; corrected p=' + number(row.q_value) + ' (' + str(row.multiple_testing) + ', ' + str(int(row.family_requested)) + ' requested hypotheses)'
        text += '\nInterval: ' + str(row.interval_status).replace('_', ' ') + interval_label
        text += '\n' + wrap(row.reason, 79)
        axis.text(1.12, 0.45, text, transform=axis.transAxes, va='center', fontsize=8)
        tidy(axis)
    return (figure, axes)

def draw(values, settings, *, canvas=None):
    renderer = {'occupancy': occupancy, 'switch_rates': switch_rates, 'transitions': transitions, 'bouts': bouts, 'samples': samples, 'contrasts': contrasts}[settings['view']]
    figure, axes = renderer(values, settings, canvas=canvas)
    height = figure.get_figheight()
    figure.suptitle(settings['title'], x=0.035, ha='left', y=1 - 0.18 / height, fontsize=15, weight='bold')
    note = settings['footnote']
    if settings['view'] == 'samples':
        note += ' Weighting: ' + ('equal eligible cell means' if settings['sample_provenance']['aggregation'] == 'mean' else 'pooled eligible time/opportunities') + '.'
    figure.text(0.035, 0.3 / height, wrap(note, 161), fontsize=8, va='center')
    return (figure, axes)
