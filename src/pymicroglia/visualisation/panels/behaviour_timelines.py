"""Portable categorical intervals on independent original recording clocks."""
from . import _layout
from ._layout import finish
from . import colour as house_colour
import math
from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
from ._format import numeric, missing, present, number as display_number
from .behaviour_profiles import state_colour
KEYS = ['source_run', 'movie', 'identity']
UNKNOWN = {'missing_features': ('Missing measurements', '..', house_colour('shade')), 'ambiguous': ('Ambiguous assignment', '//', house_colour('shade')), 'outside_training_distribution': ('Outside learned distribution', 'xx', house_colour('shade')), 'invalid_prediction': ('Invalid model output', '++', house_colour('shade'))}
MARKERS = {'missing_features': 'x', 'ambiguous': 'o', 'outside_training_distribution': 's', 'invalid_prediction': 'D'}

def _wrap(value, width=40):
    return '\n'.join((textwrap.fill(line, width) for line in str(value).splitlines()))

def _hatch(component):
    return '' if component < 6 else ['/', '\\', '|', '-', '+', 'x', 'o', '.'][(component - 6) % 8]

def draw(prepared, settings, *, selected_view=None, canvas=None):
    if selected_view not in {None,'intervals','observations','membership'}:raise ValueError('Unknown timeline view')
    states = prepared['states']
    names = {key: settings['state_display_names'].get(key, states[key]['label']) for key in settings['states']}
    handles = [Patch(facecolor=state_colour(states[key]['component']), edgecolor=house_colour('slate_tick'), linewidth=0.5, hatch=_hatch(states[key]['component']), label=names[key]) for key in settings['states']]
    present_statuses=prepared['present_statuses']
    for status, (label, hatch, colour) in UNKNOWN.items():
        if status in present_statuses:
            handles.append(Line2D([], [], marker=MARKERS[status], color=house_colour('slate_tick'), markerfacecolor='white', linestyle='none', label=label))
    handles.append(Patch(facecolor='white', edgecolor=house_colour('grade_grey'), linestyle=':', label='Unobserved interval'))
    legend_rows = math.ceil(len(handles) / 3)
    top_inches = 1.2 + 0.23 * legend_rows
    cell_height = 1.65 if settings['show_membership'] else 1.2
    height = top_inches + 0.95 + cell_height * len(settings['cells'])
    figure, array = _layout.subplots(canvas, len(settings['cells']), 1, figsize=(14, height), squeeze=False)
    figure.subplots_adjust(left=0.27, right=0.96, top=1 - top_inches / height, bottom=0.95 / height, hspace=0.95)
    axes = {}
    for index,cell in enumerate(prepared['cells']):
        key=cell['key']
        axis = array[index, 0]
        axes['cell_' + str(index)] = axis
        inventory=cell['inventory']
        intervals,shown=cell['intervals'],cell['observations']
        origin = str(key['movie']) + ' | cell ' + str(key['identity'])
        label = _wrap(origin, 34)
        counts = f"{int(inventory['assigned_observations'])}/{int(inventory['observations'])} assigned observations"
        axis.text(-0.05, 0.6, label + '\n' + counts, transform=axis.transAxes, ha='right', va='center', fontsize=8.5)
        if inventory['status'] == 'invalid_clock_order':
            axis.text(0.5, 0.5, 'Nonincreasing recording clock: timeline unavailable', transform=axis.transAxes, ha='center', va='center', color=house_colour('circadian_red'), fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            continue
        for row in intervals if selected_view not in {'observations','membership'} else []:
            start,end=row['start'],row['start']+row['width']
            if row['support'] == 'assigned':
                state = states[row['state_id']]
                colour = state_colour(state['component'])
                hatch = _hatch(state['component'])
                style = '-'
            elif row['support'] == 'unknown':
                if row['assignment_status'] not in UNKNOWN:
                    raise ValueError('Unknown assignment type has no declared categorical encoding')
                _, hatch, colour = UNKNOWN[row['assignment_status']]
                style = '-'
            elif row['support'] == 'unobserved':
                colour, hatch, style = ('white', '', ':')
            else:
                raise ValueError('Unrecognised saved physical-time support')
            axis.add_patch(Rectangle((start, 0.4), end - start, 0.4, facecolor=colour, edgecolor=house_colour('grade_grey'), linewidth=0.45, hatch=hatch, linestyle=style))
        for row in shown if selected_view not in {'intervals','membership'} else []:
            colour = state_colour(states[row['state_id']]['component']) if row['status'] == 'assigned' else house_colour('slate_tick')
            axis.plot([row['hours'], row['hours']], [0.38, 0.82], color=colour, linewidth=0.65, alpha=0.85)
            if row['status'] in MARKERS:
                axis.plot([row['hours']], [0.6], marker=MARKERS[row['status']], markersize=4, markerfacecolor='white', color=house_colour('slate_tick'), linestyle='none')
        axis.set_xlim(*cell['bounds'])
        if cell['message']:
            if not shown:axis.set_xticks([])
            axis.text(.5,.98 if shown else .6,cell['message'],transform=axis.transAxes,ha='center',fontsize=8 if shown else 9)
        if settings['show_membership'] and selected_view in {None,'membership'}:
            axis.scatter(*cell['membership'],color=house_colour('slate_tick'),s=5,zorder=3)
            axis.axhline(-0.35, color=house_colour('shade'), lw=0.5)
            axis.axhline(0.15, color=house_colour('shade'), lw=0.5)
            axis.text(1.01, -0.35, '0', transform=axis.get_yaxis_transform(), fontsize=7, va='center')
            axis.text(1.01, 0.15, '1', transform=axis.get_yaxis_transform(), fontsize=7, va='center')
            axis.set_ylim(-0.45, 1.05)
        else:
            axis.set_ylim(0.2, 1.05)
        axis.set_yticks([])
        for name in ['left', 'right', 'top']:
            axis.spines[name].set_visible(False)
        if len(shown) or len(intervals):
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        axis.xaxis.set_major_formatter(formatter)
        axis.tick_params(axis='x', labelsize=8, length=3, width=1)
        axis.set_xlabel('Own recording time (h)', fontsize=8)
    figure.suptitle(settings['title'], x=0.04, ha='left', y=1 - 0.2 / height, fontsize=15, weight='bold')
    figure.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.035, 1 - 0.53 / height), ncol=3, fontsize=8, frameon=False)
    figure.text(0.04, 0.35 / height, _wrap(settings['footnote'], 155), fontsize=8, va='center')
    return (figure, axes)
