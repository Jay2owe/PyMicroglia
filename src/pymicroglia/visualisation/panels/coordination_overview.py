"""Portable drawing of complete saved coordination results, with no estimation."""
from . import _layout
from ._layout import finish
from . import colour as house_colour
from pathlib import Path
import math, json, ast, textwrap
import numpy as np
from ._format import numeric, missing, present, number as display_number
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
BLUE = house_colour('blue')
RED = house_colour('circadian_red')
GREY = house_colour('nan_text')
TEAL = house_colour('circadian_teal')
ORANGE = house_colour('circadian_amber')
NAMES = {'characteristics': 'Spatial characteristics', 'simultaneous': 'Simultaneous changes', 'delay': 'Delayed relationships', 'proximity': 'Changing proximity', 'rhythm': 'Rhythm timing', 'states': 'Accepted cell states', 'samples': 'Biological samples'}

def unpack(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
    return value

def finite(value):
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def number(value):
    return f'{float(value):.3g}' if finite(value) else 'unavailable'

def wrap(value, width=100):
    return '\n'.join((textwrap.fill(line, width) for line in str(value).splitlines()))

def descriptor(settings):
    info = settings.get('definition', {})
    parts = []
    if info.get('question'):
        parts.append(NAMES.get(info['question'], info['question']))
    if info.get('reference'):
        parts.append(info['reference'] + ' → ' + str(info.get('target', '')))
    parts += [str(info[key]).replace('_', ' ') for key in ['representation', 'adjustment', 'statistic', 'evidence_kind'] if info.get(key)]
    if info.get('evidence_level'):
        parts.append('Evidence: ' + info['evidence_level'].replace('_', ' '))
    if info.get('model_id'):
        parts.append('Accepted model ' + info['model_id'][:12])
    return ' | '.join(parts)

def interval(value):
    value = unpack(value)
    if isinstance(value, dict):
        if value.get('wraps_cycle_boundary'):
            return None
        value = value.get('direction_interval_hours', value.get('bounds'))
    if isinstance(value, (list, tuple)) and len(value) == 2 and all((finite(item) for item in value)) and (value[0] <= value[1]):
        return value
    return None

def draw(prepared, settings, *, canvas=None):
    view = settings['view']
    height = max(7.5, 3.8 + 0.35 * prepared['count']) if view in {'coverage', 'recordings', 'timing', 'contrasts'} else 8.0
    figure, axis = _layout.subplots(canvas, figsize=(12, height))
    axes = {'evidence': axis}
    figure.subplots_adjust(left=0.24, right=0.77, top=1 - 1.7 / height, bottom=1.45 / height)
    if view == 'coverage':
        axis.axis('off')
        axis.set_xlim(0, 1)
        axis.set_ylim(prepared['count'] + 0.7, -0.8)
        headings = [(0.0, 'Saved question'), (0.43, 'Outcome'), (0.7, 'Effects'), (0.81, 'Tested'), (0.92, 'Supported')]
        for x, label in headings:
            axis.text(x, -0.5, label, fontsize=10, weight='bold')
        for y, row in enumerate(prepared['rows']):
            colour = TEAL if row['status'] == 'available' else GREY if row['status'] == 'disabled' else ORANGE
            axis.text(0, y, NAMES.get(row['question'], row['question']) + '\n' + row['evidence_level'].replace('_', ' '), va='center', fontsize=9)
            axis.text(0.43, y, row['status'].replace('_', ' '), va='center', fontsize=9, color=colour)
            for x, key in [(0.7, 'effect_rows'), (0.81, 'tested'), (0.92, 'supported')]:
                axis.text(x, y, str(int(row[key])), va='center', fontsize=10)
            axis.axhline(y + 0.43, color=house_colour('shade'), lw=0.5)
        figure.subplots_adjust(left=0.07, right=0.96)
    elif view == 'matrices':
        matrix=prepared['matrix']
        colour = plt.get_cmap('RdBu_r').copy()
        colour.set_bad(house_colour('blank'))
        limit=prepared['limit']
        handle = axis.imshow(matrix, cmap=colour, vmin=-limit, vmax=limit, aspect='equal', interpolation='nearest')
        axis.set_xticks(range(len(settings['column_ids'])), [str(item) for item in settings['column_ids']])
        axis.set_yticks(range(len(settings['row_ids'])), [str(item) for item in settings['row_ids']])
        axis.set_xlabel('Target cell')
        axis.set_ylabel('Reference cell')
        for row in prepared['rows']:
            x, y = (int(row['column_index']), int(row['row_index']))
            value = row.get('display_value')
            if finite(value):
                axis.text(x, y, number(value), ha='center', va='center', fontsize=9, color='white' if abs(value) > 0.65 * limit else house_colour('dark'))
            elif row.get('effect_id'):
                axis.text(x, y, '?', ha='center', va='center', color=GREY, fontsize=11)
            if row.get('supported') is True:
                axis.plot([x + 0.31], [y - 0.3], 'o', ms=3, color='black')
        axis.set_xticks(np.arange(-0.5, len(settings['column_ids']), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(settings['row_ids']), 1), minor=True)
        axis.grid(which='minor', color='white', lw=1)
        axis.tick_params(which='minor', bottom=False, left=False)
        figure.colorbar(handle, ax=axis, pad=0.04, fraction=0.045, label='Saved effect')
    elif view in {'effects', 'proximity'}:
        for point,colour,label in zip(prepared['points'],(GREY,BLUE),('Saved descriptive / unsupported effect','Corrected pair support')):
            if len(point['x']):axis.scatter(point['x'],point['y'],s=32,c=colour,alpha=.8,label=label)
        if not prepared['finite']:axis.text(.5,.5,'No finite effects with this saved geometry',transform=axis.transAxes,ha='center',color=GREY)
        unit=prepared['unit']
        axis.set_xlabel(('Window distance' if view == 'proximity' else 'Distance between median positions') + ' (' + unit + ')')
        axis.set_ylabel('Saved window coordination' if view == 'proximity' else 'Saved effect')
        finish(axis)
        if prepared['finite']:
            axis.legend(frameon=False, fontsize=8, loc='best')
        figure.text(0.24, 1 - 1.38 / height, '{} / {} saved rows have both coordinates; {} unavailable'.format(prepared['finite'],prepared['count'],prepared['count']-prepared['finite']), fontsize=9, color=GREY)
    elif view == 'sample_timing':
        axis.axis('off')
        row = prepared['rows'][0]
        lines = ['Measurement roles: ' + row['reference'] + ' → ' + row['target'], 'Scope: ' + row['scope'] + ' | stratum: ' + row['stratum'], 'Native outcome: ' + str(row['status']).replace('_', ' '), str(row['reason']), 'Compatible reference period: ' + number(row.get('period_hours')) + ' h', 'Mean timing offset: ' + number(row.get('mean_offset_hours')) + ' h', 'Circular concentration: ' + number(row.get('resultant')), 'Eligible / requested units: ' + number(row.get('eligible_units')) + ' / ' + number(row.get('requested_units')), 'Population sampling region: ' + ('available in the accompanying native result' if isinstance(unpack(row.get('sampling_region')), dict) else 'unavailable'), 'Repetition: ' + str(row.get('repetition_status', 'unresolved')).replace('_', ' ')]
        axis.text(0, 1, '\n\n'.join((wrap(line, 90) for line in lines)), transform=axis.transAxes, va='top', fontsize=10)
        figure.subplots_adjust(left=0.08, right=0.94)
    else:
        rows = prepared['rows']
        labels = []
        for y, row in enumerate(rows):
            value = row.get('value' if view == 'samples' else 'display_value' if view in {'recordings', 'timing'} else 'effect')
            if view == 'samples':
                label = str(row.get('sample') or (row.get('movies') or ['unconfirmed'])[0]) + ' | ' + str(row.get('condition') or 'no condition')
            elif view == 'contrasts':
                label = row['reference_condition'] + ' → ' + row['target_condition']
            elif view == 'timing':
                label = row['movie'] + ' | ' + str(int(row['reference_identity'])) + ' / ' + str(int(row['target_identity']))
            else:
                label = row['movie']
            labels.append(wrap(label, 29))
            permitted = row.get('formal_eligible') is True if view == 'samples' else row.get('supported') is True
            if finite(value):
                bounds = interval(row.get('offset_interval_hours') if view == 'timing' else row.get('effect_interval'))
                if bounds is not None:
                    axis.plot(bounds, [y, y], lw=2, color=BLUE)
                axis.scatter([value], [y], s=44, facecolors=BLUE if permitted else 'white', edgecolors=BLUE, zorder=3)
            else:
                axis.text(0.02, y, 'Unavailable', transform=axis.get_yaxis_transform(), fontsize=9, color=GREY, va='center')
            if view == 'timing':
                note = 'Period ' + number(row.get('period_hours')) + ' h; ' + str(row.get('status', '')).replace('_', ' ')
            elif view == 'samples':
                note = str(int(row['recordings_eligible'])) + ' / ' + str(int(row['recordings'])) + ' recordings; ' + ('formal eligible' if permitted else 'descriptive only')
            else:
                note = 'q ' + number(row.get('q_value')) + '; ' + str(row.get('status', '')).replace('_', ' ')
            axis.text(1.04, y, wrap(note, 31), transform=axis.get_yaxis_transform(), va='center', fontsize=8)
        if not rows:
            axis.text(0.5, 0.5, 'No finite sample summaries', transform=axis.transAxes, ha='center', color=GREY)
        axis.set_yticks(range(len(labels)), labels)
        axis.set_ylim(max(len(labels) - 0.5, 0.5), -0.5)
        axis.tick_params(labelsize=9)
        axis.set_xlabel('Saved offset (h)' if view == 'timing' else 'Saved sample aggregate' if view == 'samples' else 'Saved target − reference effect' if view == 'contrasts' else 'Saved recording effect')
        finish(axis)
        if view == 'samples':
            figure.text(0.24, 1 - 1.38 / height, str(settings['available_units']) + ' / ' + str(settings['total_units']) + ' requested units have finite summaries', fontsize=9, color=GREY)
    figure.suptitle(settings['title'], x=0.05, y=1 - 0.24 / height, ha='left', fontsize=16, weight='bold')
    description = descriptor(settings)
    if description:
        figure.text(0.05, 1 - 0.8 / height, wrap(description, 125), va='top', fontsize=9, color=house_colour('dark'))
    figure.text(0.05, 0.45 / height, wrap(settings['footnote'], 145), fontsize=8, va='center')
    return (figure, axes)
