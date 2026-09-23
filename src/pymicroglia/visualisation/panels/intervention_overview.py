"""Portable rendering of saved original effects; no scientific model imports."""
from . import _layout
from ._layout import finish
from . import colour as house_colour
from pathlib import Path
import math, textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
RED = house_colour('circadian_red')
BLUE = house_colour('blue')
GREY = house_colour('nan_text')
ORANGE = house_colour('circadian_amber')
TEAL = house_colour('circadian_teal')
STATUS = {'increase': ('Supported increase', RED), 'decrease': ('Supported decrease', BLUE), 'no_detected_change': ('No detected change', house_colour('shade')), 'inconclusive': ('Inconclusive', house_colour('actogram_dark')), 'detected_below_meaningful_threshold': ('Below declared magnitude', house_colour('actogram_dark'))}

def finite(value):
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (ValueError, TypeError):
        return False

def number(value):
    return f'{float(value):.3g}' if finite(value) else 'unavailable'

def probability(value):
    return 'below numeric precision' if finite(value) and float(value) == 0 else number(value)

def wrap(value, width=110):
    return '\n'.join((textwrap.fill(line, width) for line in str(value).splitlines()))

def truth(value):
    return value is True or (isinstance(value, (np.bool_, str)) and str(value).lower() == 'true')

def colour(row):
    return STATUS.get(row.get('outcome'), ('Unavailable', GREY))[1]

def draw(prepared, settings, *, canvas=None):
    rows = prepared['rows']
    view = settings['view']
    count = len(settings.get('cell_ids', rows))
    height = max(7.8, 4.0 + (0.8 if view == 'controls' else 0.68 if view == 'model_effects' else 0.55) * count)
    figure, axis = _layout.subplots(canvas, figsize=(14, height))
    axes = {'evidence': axis}
    figure.subplots_adjust(left=0.25, right=0.73, top=1 - 1.65 / height, bottom=1.7 / height)
    if view == 'coverage':
        axis.axis('off')
        axis.set_xlim(0, 1)
        axis.set_ylim(len(rows) + 0.2, -1.0)
        for x, label in [(0.0, 'Question'), (0.42, 'Saved outcome'), (0.68, 'Original requested population')]:
            axis.text(x, -0.7, label, weight='bold', fontsize=10)
        for y, row in enumerate(rows):
            axis.text(0, y, row['label'], va='center', fontsize=10)
            axis.text(0.42, y, str(row['status']).replace('-', ' '), va='center', fontsize=10, color=TEAL if row['status'] == 'available' else GREY if row['status'] == 'disabled' else ORANGE)
            axis.text(0.68, y, str(int(row['requested_cells'])) + ' cells; ' + str(int(row['requested_measurements'])) + ' measurements', va='center', fontsize=10)
            axis.axhline(y + 0.43, color=house_colour('shade'), lw=0.6)
        figure.subplots_adjust(left=0.07, right=0.97)
    elif view == 'status_matrix':
        ids = settings['cell_ids']
        metrics = settings['measurements']
        keys = list(STATUS)
        matrix=prepared['matrix']
        cmap = ListedColormap([STATUS[key][1] for key in keys])
        cmap.set_bad('white')
        axis.imshow(matrix, cmap=cmap, vmin=-0.5, vmax=len(keys) - 0.5, aspect='auto', interpolation='nearest')
        axis.set_xticks(range(len(metrics)), [wrap(metric, 20) for metric in metrics], fontsize=10)
        axis.set_yticks(range(len(ids)), [wrap(label, 33) for label in settings['cell_labels']], fontsize=9)
        for row in rows:
            y, x = (ids.index(row['cell_id']), metrics.index(row['measurement']))
            outcome = row['outcome']
            label = '+' if outcome == 'increase' else '-' if outcome == 'decrease' else '?' if outcome == 'inconclusive' else 'below' if outcome == 'detected_below_meaningful_threshold' else 'nd'
            axis.text(x, y, label, ha='center', va='center', fontsize=10, color='white' if outcome in {'increase', 'decrease'} else house_colour('dark'))
        axis.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(ids), 1), minor=True)
        axis.grid(which='minor', color='white', lw=1.5)
        axis.tick_params(which='minor', bottom=False, left=False)
        axis.set_xlabel('Requested measurement')
        axis.set_ylabel('Original cell')
        handles = [Line2D([], [], marker='s', ls='', mfc=color, mec=house_colour('muted'), label=label) for label, color in STATUS.values()]
        axis.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    else:
        axis.set_ylim(len(rows) - 0.5, -0.8)
        axis.set_yticks(range(len(rows)))
        if view == 'controls':
            labels = [wrap(str(row['target_condition']) + ' - ' + str(row['reference_condition']), 28) + '\n' + str(row['baseline']) + ' to ' + str(row['target_window']) for row in rows]
        else:
            labels = [wrap(row['cell_label'], 32) for row in rows]
        axis.set_yticklabels(labels, fontsize=9)
        axis.set_ylabel('Biological-sample comparison' if view == 'controls' else 'Original cell')
        for y, row in enumerate(rows):
            color = TEAL if view == 'controls' and truth(row.get('supported')) else GREY if view == 'controls' else colour(row)
            if view == 'before_after':
                before, after = (row.get('baseline_value'), row.get('target_value'))
                if finite(before) and finite(after):
                    axis.plot([before, after], [y, y], color=color, lw=1.6)
                if finite(before):
                    axis.plot(before, y, 'o', mfc='white', mec=house_colour('dark'), ms=6)
                if finite(after):
                    axis.plot(after, y, 'o', color=color, ms=6)
                annotation = 'baseline ' + number(before) + '; follow-up ' + number(after)
                if not finite(before) or not finite(after):
                    annotation += '\nmissing original summary'
            else:
                value = row.get('effect') if view == 'controls' else row.get('estimate') if view == 'model_effects' else row.get('absolute_change')
                low, high = (row.get('interval_low'), row.get('interval_high'))
                if finite(value):
                    if view in {'controls', 'model_effects'} and finite(low) and finite(high) and (low <= high):
                        axis.plot([low, high], [y, y], color=color, lw=2)
                    axis.plot(value, y, 'o', color=color if color not in {house_colour('shade'), house_colour('actogram_dark')} else GREY, ms=6)
                else:
                    axis.text(0.02, y, 'unavailable', transform=axis.get_yaxis_transform(), color=ORANGE, va='center', fontsize=9)
                annotation = 'effect ' + number(value) + '; q ' + probability(row.get('q_value')) if view == 'controls' else 'saved model q ' + probability(row.get('q_value'))
                if view == 'controls':
                    annotation += '\nsamples ' + number(row.get('reference_samples')) + '/' + number(row.get('target_samples')) + '; cells ' + number(row.get('reference_cells')) + '/' + number(row.get('target_cells'))
                    if row.get('design') == 'matched':
                        annotation += '; matches ' + number(row.get('complete_matches'))
                elif view == 'model_effects':
                    annotation += '\nobservations ' + number(row.get('baseline_observations')) + '/' + number(row.get('target_observations'))
                if not finite(row.get('q_value')) and view != 'before_after':
                    annotation += '\n' + wrap(textwrap.shorten(str(row.get('reason', 'Evidence unavailable')), width=76, placeholder='...'), 38)
            axis.text(1.04, y, annotation, transform=axis.get_yaxis_transform(), va='center', fontsize=8.5, color=house_colour('dark'))
            axis.axhline(y + 0.48, color=house_colour('blank'), lw=0.5)
        if view != 'before_after':
            axis.axvline(0, color=house_colour('muted'), ls='--', lw=1)
        unit = settings.get('unit') or (rows[0].get('unit') if rows else None) or 'original measurement units'
        axis.set_xlabel(('Original window summary' if view == 'before_after' else 'Observed follow-up minus baseline' if view == 'original_changes' else 'Conditional model contrast' if view == 'model_effects' else 'Target minus reference sample change') + ' (' + str(unit) + ')')
        finish(axis)
        if view == 'before_after':
            figure.legend(handles=[Line2D([], [], marker='o', ls='', mfc='white', mec=house_colour('dark'), label='Baseline'), Line2D([], [], marker='o', ls='', color=GREY, label='Follow-up')], loc='upper left', bbox_to_anchor=(0.25, 1 - 1.28 / height), ncol=2, fontsize=9)
    title = settings.get('title', 'Saved intervention responses')
    figure.text(0.035, 1 - 0.3 / height, wrap(title, 100), va='top', weight='bold', fontsize=15)
    if view not in {'coverage', 'status_matrix'}:
        methods = sorted({str(row['method']) for row in rows if isinstance(row.get('method'), str)})
        correction = sorted({str(row['correction']) for row in rows if isinstance(row.get('correction'), str)})
        tested = sum((finite(row.get('p_value')) for row in rows))
        figure.text(0.035, 1 - 1.0 / height, wrap(str(len(rows)) + ' requested rows; ' + str(tested) + ' with saved tests. Method: ' + (', '.join(methods) or 'unavailable') + '. Correction: ' + (', '.join(correction) or 'unavailable'), 135), fontsize=9, va='top')
    figure.text(0.035, 0.8 / height, wrap(settings.get('footnote', 'Saved original results only.'), 142), fontsize=9, va='top')
    return (figure, axes)
