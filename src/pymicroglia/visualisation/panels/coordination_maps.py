"""Portable measured-geometry drawing; no scientific analysis or edge selection."""
from . import _layout
from ._layout import finish
from . import colour as house_colour
from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from ._format import numeric, missing, present, number as display_number
from . import spatial
SOURCES = [spatial.__file__]

def draw(prepared, settings, *, selected_view=None, canvas=None):
    names=[selected_view] if selected_view else ['map','evidence']
    if any(name not in ('map','evidence') for name in names):raise ValueError('Unknown connection view')
    figure,array=_layout.subplots(canvas,1,len(names),figsize=(13 if len(names)==2 else 8,9),squeeze=False)
    figure.subplots_adjust(left=.08,right=.97,bottom=.25,top=.81,wspace=.3)
    axes={name:array[0,index] for index,name in enumerate(names)}
    axis,notes=axes.get('map'),axes.get('evidence')
    if notes is not None:notes.axis('off')
    if not settings['supported_effects']:
        if axis is not None:
            axis.axis('off')
            axis.text(0, 0.7, 'No supported pair connections', fontsize=14, weight='bold')
        if notes is not None:notes.text(0, 0.8, 'Whole-recording evidence does not\nestablish individual supported edges.', fontsize=11)
    else:
        located,finite,lines=prepared['located'],prepared['finite'],prepared['lines']
        if axis is not None:
            spatial.connection_paths(axis, finite, curvature=0.18, cmap='RdBu_r', vmin=-1, vmax=1, linewidth=2)
            axis.scatter(located.x, located.y, s=110, color='white', edgecolors=house_colour('dark'), lw=1.3, zorder=4)
            for row in located.itertuples():
                axis.annotate(str(int(row.identity)), (row.x, row.y), xytext=(5, 6), textcoords='offset points', fontsize=9)
            axis.set_aspect('equal', adjustable='datalim')
            axis.margins(0.15)
            axis.set_xlabel('Recorded x (' + settings['unit'] + ')')
            axis.set_ylabel('Recorded y (' + settings['unit'] + ')')
            finish(axis)
        if notes is not None:notes.text(0, 1, '\n\n'.join((textwrap.fill(line, 43) if '\n' not in line else line for line in lines)), va='top', fontsize=9)
        handles = [Line2D([], [], color=house_colour('circadian_red'), lw=2, label='Positive saved effect'), Line2D([], [], color=house_colour('blue'), lw=2, label='Opposing saved effect'), Line2D([], [], color=house_colour('nan_text'), lw=2, label='Effect sign unresolved')]
        figure.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.075, 0.13), ncol=3, frameon=False, fontsize=9)
    figure.suptitle(settings['title'], x=0.06, y=0.96, ha='left', fontsize=16, weight='bold')
    figure.text(0.06, 0.055, textwrap.fill(settings['footnote'], 155), fontsize=8, va='center')
    return (figure, axes)
