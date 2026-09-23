"""Portable paired observations and frozen diagnostics, without scientific fits."""
from . import _layout
from ._layout import finish
from . import colour as house_colour
from pathlib import Path
import math, json, ast, textwrap
import numpy as np
from ._format import numeric, missing, present, number as display_number
import matplotlib.pyplot as plt
BLUE = house_colour('blue')
ORANGE = house_colour('circadian_amber')
GREY = house_colour('nan_text')
TEAL = house_colour('circadian_teal')
RED = house_colour('circadian_red')

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

def wrap(text, width=100):
    return '\n'.join((textwrap.fill(line, width) for line in str(text).splitlines()))

def observed_line(axis,chunks,*,colour,label=None):
    for index,chunk in enumerate(chunks):
        a,b=zip(*chunk)
        axis.plot(a,b,color=colour,lw=1.3,marker='.' if len(chunk)==1 else None,ms=3,label=label if index==0 else None)
    return sum(map(len,chunks))

def table(axis, frame):
    axis.axis('off')
    if not frame:
        axis.text(0, 0.85, 'No additional scientific result for this display panel', fontsize=10)
        return
    rows = frame
    axis.set_xlim(0, 1)
    axis.set_ylim(len(rows) + 0.2, -0.8)
    for x, label in [(0, 'Saved question'), (0.34, 'Effect'), (0.47, 'Corrected q'), (0.62, 'Outcome / evidence meaning')]:
        axis.text(x, -0.55, label, fontsize=9, weight='bold')
    for i, row in enumerate(rows):
        name = str(row.get('question') or row.get('evidence_kind') or 'saved evidence').replace('_', ' ')
        representation = row.get('representation')
        adjustment = row.get('adjustment')
        if isinstance(representation, str):
            name += '\n' + representation + ' | ' + (adjustment if isinstance(adjustment, str) else 'none')
        if row.get('evidence_kind'):
            name += '\n' + str(row['evidence_kind']).replace('_', ' ')
        state = row.get('reference_state_id')
        if isinstance(state, str):
            name += '\n' + state[:8] + ' / ' + str(row.get('target_state_id'))[:8]
        status = str(row.get('decision') or row.get('status') or 'unavailable').replace('_', ' ')
        meaning = str(row.get('effect_population') or row.get('decision_reason') or row.get('reason') or '')
        axis.text(0, i, wrap(name, 32), va='center', fontsize=8)
        axis.text(0.34, i, number(row.get('effect')), va='center', fontsize=9)
        axis.text(0.47, i, number(row.get('q_value')), va='center', fontsize=9)
        axis.text(0.62, i, wrap(status + '; ' + meaning, 69), va='center', fontsize=7.5, color=TEAL if row.get('supported') is True else GREY)
        axis.axhline(i + 0.46, color=house_colour('shade'), lw=0.5)

def draw(prepared, settings, *, canvas=None):
    view = settings['view']
    pair = settings['pair']
    effects = prepared['effects']
    rows = {'core': 4, 'representation': 5, 'lag': 3, 'proximity': 3, 'rhythm': 3, 'states': 4}[view]
    table_height = max(1.8, 0.58 * len(effects))
    heights = [1.5] * (rows - 1) + [table_height]
    height = sum(heights) + 3.8
    figure, array = _layout.subplots(canvas, rows, 1, figsize=(13, height), gridspec_kw={'height_ratios': heights})
    axes = {str(index): axis for index, axis in enumerate(array)}
    figure.subplots_adjust(left=0.13, right=0.94, top=1 - 1.6 / height, bottom=0.8 / height, hspace=0.95)
    gap = settings['max_gap_hours']
    if view in {'core', 'representation'}:
        for index, role in enumerate(['reference', 'target']):
            endpoint=prepared['endpoints'][role]
            measure=pair['endpoint_definitions'][role]['measurement']
            colour=BLUE if index==0 else ORANGE
            count=observed_line(array[index],endpoint['segments'],colour=colour)
            array[index].set_title(role.capitalize() + ' cell ' + str(pair[role + '_identity']) + ' | ' + measure.get('label', measure['column']), loc='left', fontsize=11)
            array[index].set_ylabel(str(measure.get('unit') or 'Measured value'), fontsize=10)
            if not count:
                note='Saved scalar: '+number(endpoint['scalar'])+'; no time series' if endpoint['has_scalar'] else 'No observed trace for this endpoint'
                array[index].text(0.5, 0.5, note, transform=array[index].transAxes, ha='center', color=GREY, fontsize=10)
            spans=prepared['spans']
            if spans:
                array[index].broken_barh(spans, (0, 1), transform=array[index].get_xaxis_transform(), facecolors=TEAL, alpha=0.07, zorder=0)
        if view == 'core':
            count=observed_line(array[2],prepared['distance'],colour=GREY)
            array[2].set_ylabel('Distance (' + pair['distance_unit'] + ')', fontsize=10)
            array[2].set_title('Original paired-position observations', loc='left', fontsize=11)
            if not count:
                array[2].text(0.5, 0.5, 'No matched observed positions; no distance history invented', transform=array[2].transAxes, ha='center', fontsize=10, color=GREY)
        else:
            refs = prepared['references']
            if len(refs):
                for role, colour in [('reference', BLUE), ('target', ORANGE)]:
                    selected=refs[role]
                    observed_line(array[2], selected['segments'], colour=colour, label=role + ' shared reference')
                    array[3].scatter(selected['hours'],selected['members'], s=10, c=colour, label=role)
                array[2].legend(frameon=False, fontsize=8)
                array[3].legend(frameon=False, fontsize=8)
            else:
                array[2].text(0.5, 0.5, 'This saved representation has no shared-reference subtraction', transform=array[2].transAxes, ha='center', fontsize=10, color=GREY)
            array[2].set_ylabel('Saved reference', fontsize=10)
            array[2].set_title(settings['temporal_view']['representation'] + ' | ' + settings['temporal_view']['adjustment'], loc='left', fontsize=11)
            array[3].set_ylabel('Reference contributors', fontsize=10)
            array[3].set_title('Actual recorded reference coverage; missing reference is not zero adjustment', loc='left', fontsize=10)
        if prepared['bounds'] is not None:
            for axis in array[:-1]:axis.set_xlim(*prepared['bounds'])
    elif view == 'lag':
        observed_line(array[0], prepared['overlap'], colour=GREY, label='Original per-lag overlap')
        observed_line(array[0], prepared['tested'], colour=BLUE, label='Fixed common test window')
        for row in prepared['bands']:
            if finite(row.get('coefficient_lower')) and finite(row.get('coefficient_upper')):
                array[0].plot([row['lag_hours']] * 2, [row['coefficient_lower'], row['coefficient_upper']], color=BLUE, alpha=0.35, lw=2)
        array[0].legend(frameon=False, fontsize=8)
        array[0].set_ylabel('Saved coefficient', fontsize=10)
        array[0].set_title('Complete saved lag grid and native curve intervals', loc='left', fontsize=11)
        array[1].plot(prepared['hours'],prepared['pairs'], 'o-', color=GREY, lw=1, ms=3)
        array[1].set_ylabel('Paired observations', fontsize=10)
        for axis in array[:-1]:
            axis.set_xlabel('Lag (h); negative means reference leads target', fontsize=10)
    elif view == 'proximity':
        for index, column, colour in [(0, 'distance', GREY), (1, 'coordination', BLUE)]:
            array[index].scatter(*prepared['points'][column], s=20, c=colour)
            array[index].set_ylabel('Saved window ' + column, fontsize=10)
        array[0].set_title('Each point is an original, potentially overlapping window', loc='left', fontsize=11)
    elif view == 'rhythm':
        endpoints = prepared['endpoints']
        array[0].axis('off')
        lines = []
        for row in endpoints:
            lines.append(str(row['endpoint_role']) + ' ' + str(row['measurement']) + ': ' + str(row['screen_status']) + '; period ' + number(row.get('period_hours')) + ' h; trace q ' + number(row.get('trace_q_value')))
            lines.append('Estimator ' + str(row.get('estimator')) + '; separate significance method ' + str(row.get('significance_method')))
        array[0].text(0, 1, '\n\n'.join((wrap(line, 120) for line in lines)), va='top', fontsize=9)
        points=prepared['segments']
        if len(points):
            for segment,x,y in points:
                array[1].scatter(x,y, s=15, label='Original segment ' + str(int(segment)))
            array[1].legend(frameon=False, fontsize=8)
        else:
            array[1].text(0.5, 0.5, 'No native comparable timing timecourse', transform=array[1].transAxes, ha='center', fontsize=10, color=GREY)
        array[1].set_ylabel('Native timing offset (h)', fontsize=10)
        array[1].set_title('Original native reference and segment limits; no phase wrapping or refitting', loc='left', fontsize=10)
    else:
        exposure=prepared['exposure']
        for row in exposure:
            colour = TEAL if row.get('jointly_assigned') is True else ORANGE if row.get('jointly_observed') is True else house_colour('raw')
            array[0].broken_barh([(row['start_hours'], row['duration_hours'])], (0, 0.8), facecolors=colour, edgecolors='none')
        array[0].set_yticks([])
        array[0].set_ylim(-0.1, 1.0)
        array[0].set_title('Joint time: assigned (teal), observed but unknown (orange), unobserved (grey)', loc='left', fontsize=10)
        occupancy=prepared['occupancy']
        labels = []
        for index, row in enumerate(occupancy):
            labels.append(str(row['reference_state_id'])[:8] + ' / ' + str(row['target_state_id'])[:8])
            for shift, key, colour in [(-0.13, 'fraction_observed', BLUE), (0.13, 'fraction_assigned', ORANGE)]:
                if finite(row.get(key)):
                    array[1].scatter([row[key]], [index + shift], c=colour, s=25)
        array[1].set_yticks(range(len(labels)), labels)
        array[1].set_xlabel('Fraction of observed (blue) / assigned (orange) joint time', fontsize=9)
        array[1].set_ylabel('Accepted state-ID pairs', fontsize=9)
        for row in prepared['switches']:
            y = 0 if row['endpoint_role'] == 'reference' else 1
            colour = BLUE if y == 0 else ORANGE
            array[2].plot([row['earliest_hours'], row['latest_hours']], [y, y], color=colour, lw=1)
            array[2].scatter([row['event_hours']], [y], s=12, c=colour, marker='|' if row.get('eligible_for_coincidence') is True else 'x')
        array[2].set_yticks([0, 1], ['Reference switches', 'Target switches'])
        array[2].set_ylim(-0.5, 1.5)
        array[2].set_title('Original switch-time bounds and midpoint convention; not exact event times', loc='left', fontsize=10)
    table(array[-1], effects)
    for index, axis in enumerate(array[:-1]):
        if axis.axison:
            if view != 'lag' and (not (view == 'states' and index == 1)):
                axis.set_xlabel('Original recording time (h)', fontsize=10)
            axis.tick_params(labelsize=9)
            finish(axis)
    figure.suptitle(settings['title'], x=0.06, y=1 - 0.22 / height, ha='left', fontsize=15, weight='bold')
    subtitle = pair['reference'] + ' (reference) / ' + pair['target'] + ' (target) | ' + view.replace('_', ' ')
    figure.text(0.06, 1 - 0.72 / height, wrap(subtitle, 130), va='top', fontsize=10)
    note = settings['footnote']
    if view in {'core', 'representation'}:
        note += ' Pale shading marks saved valid joint observation intervals (core) or matched representation support. Time axes share the same original range.'
    if view == 'representation':
        note += ' ' + settings['adjustment_meaning']
    if view == 'rhythm':
        note += ' Trace significance and native pair timing are separate; different or unsupported periods have no shared phase.'
    if view == 'states':
        note += ' State IDs belong to the recorded accepted model. Observed and assigned denominators remain separate.'
    figure.text(0.06, 0.35 / height, wrap(note, 150), fontsize=8, va='center')
    return (figure, axes)
