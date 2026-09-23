"""Portable display of saved cells, biological samples and unresolved delays."""
from . import _layout
from ._layout import finish
from . import colour as house_colour
import json
import math
from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
import numpy as np
from ._format import numeric, missing, present, number as display_number

def wrap(value, width=105):
    return '\n'.join((textwrap.fill(line, width) for line in str(value).splitlines()))

def number(value):
    try:
        return f'{float(value):.3g}' if math.isfinite(float(value)) else 'unavailable'
    except (ValueError, TypeError):
        return 'unavailable'

def group_label(value):
    prefix, name = value.split(':', 1)
    return ('Biological sample ' if prefix == 'sample' else 'Recording (sample unconfirmed) ') + name

def draw(prepared, settings, *, selected_view=None, canvas=None):
    if selected_view not in {None,'cells','samples'}:raise ValueError('Unknown population view')
    groups = settings['groups']
    view = settings['view']
    between = view == 'between_cells'
    delay = view == 'delay'
    height = max(7.0, 3.5 + 0.55 * len(groups))
    figure = _layout.figure(canvas, figsize=(14, height))
    layout = figure.add_gridspec(2, 2 if selected_view is None else 1, height_ratios=[4, 1.8])
    figure.subplots_adjust(left=0.22 if not between else 0.08, right=0.97, top=1 - 1.1 / height, bottom=0.6 / height, wspace=0.65, hspace=0.55)
    left=figure.add_subplot(layout[0,0]) if selected_view!='samples' else None
    right=figure.add_subplot(layout[0,1 if selected_view is None else 0]) if selected_view!='cells' else None
    axes={name:axis for name,axis in [('cells',left),('samples',right)] if axis is not None}
    caption = figure.add_subplot(layout[1, :])
    caption.axis('off')
    for axis in axes.values():
        axis.tick_params(labelsize=8)
    colours = {group: plt.get_cmap('tab20')(index % 20) for index, group in enumerate(settings['all_groups'])}
    request = settings['request']
    measurements = settings['measurements']
    pair = settings['pair']

    def scalar_label(name):
        m = measurements[name]
        unit = m.get('unit') or 'unit not recorded'
        return wrap((m.get('summary') or 'saved') + ' ' + m['label'] + f' ({unit})', 35)
    note = prepared['notes']
    if between:
        if left is not None:
            for group in prepared['groups']:
                if len(group['x']):left.scatter(group['x'],group['y'],s=20,color=colours[group['group']],alpha=.65,label=group_label(group['group']),edgecolors='none')
        if right is not None:right.scatter(prepared['samples']['x'],prepared['samples']['y'],s=42,color=house_colour('circadian_teal'),marker='D')
        for axis in axes.values():
            axis.set_xlabel(scalar_label(pair['reference']), fontsize=9)
            axis.set_ylabel(scalar_label(pair['target']), fontsize=9)
        if left is not None:
            left.set_title('One saved summary pair per eligible cell',fontsize=10)
            if left.get_legend_handles_labels()[0]:left.legend(fontsize=7,loc='best')
            else:
                left.text(.5,.5,'No eligible cell-summary pairs',transform=left.transAxes,ha='center',fontsize=9)
                left.set_xticks([]);left.set_yticks([])
        if right is not None:
            right.set_title('One saved aggregate per biological sample',fontsize=10)
            if not len(prepared['samples']['x']):
                right.text(.5,.5,'Biological-sample units were not saved',transform=right.transAxes,ha='center',fontsize=9)
                right.set_xticks([]);right.set_yticks([])
    else:
        for index,group in enumerate(prepared['groups']):
            if left is not None:
                for mark in group['marks']:left.scatter(mark['x'],mark['y'],color=colours[group['group']],marker=mark['marker'],s=22,alpha=.7)
            if right is not None:
                if group['sample_available']:right.scatter([group['sample']],[index],color=colours[group['group']],s=50,marker='D')
                else:right.text(.02,index,wrap(group['message'],32),transform=right.get_yaxis_transform(),fontsize=8,va='center')
        for axis in axes.values():
            axis.set_yticks(range(len(groups)))
            axis.set_ylim(len(groups) - 0.55, -0.55)
            axis.set_xlabel('Saved delay (hours)' if delay else 'Saved signed association coefficient', fontsize=9)
            axis.axvline(0, color=house_colour('raw'), lw=0.8)
            if not delay:
                axis.set_xlim(-1.08, 1.08)
            elif request['lag']['enabled']:
                axis.set_xlim(request['lag']['range_hours'])
        if left is not None:
            left.set_yticklabels([wrap(group_label(group),28) for group in groups],fontsize=8)
            left.set_title('Supported resolved cell delays' if delay else 'Every eligible cell, regardless of detection',fontsize=10)
        if right is not None:
            right.set_yticklabels([wrap(group_label(group),28) for group in groups] if selected_view=='samples' else [])
            right.set_title('Saved compatible sample descriptions' if delay else 'Saved biological-sample aggregates',fontsize=10)
    figure.text(0.02, 1 - 0.1 / height, wrap(settings['title'] + f" | page {settings['page_number']}", 125), fontsize=14, fontweight='bold', va='top')
    figure.text(0.02, 1 - 0.52 / height, wrap('Reference: ' + measurements[pair['reference']]['label'] + ' | Target: ' + measurements[pair['target']]['label'], 140), fontsize=10, va='top')
    caption.text(0, 1, wrap('\n'.join(note), 120), fontsize=8, va='top')
    figure.text(0.02, 0.1 / height, wrap(settings['footnote'], 155), fontsize=8, va='bottom')
    return figure,axes
