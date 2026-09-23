"""Portable group distributions, retained sample pairs and observation context."""
from . import _layout
from ._layout import finish
from . import colour as house_colour
from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
import numpy as np
GROUPS = ('significant', 'not-significant')
LABELS = ('Detected', 'Not detected')
COLOURS = (house_colour('okabe_bluish_green'), house_colour('nan_text'))

def wrap(value, width=105):
    return '\n'.join((textwrap.fill(line, width) for line in str(value).splitlines()))

def points(axis, rows, label):
    """Every finite cell is one point; horizontal offsets only separate marks."""
    for index, group in enumerate(rows['groups']):
        axis.scatter(group['x'], group['y'], s=20, alpha=0.65, color=COLOURS[index], edgecolors='none')
        axis.text(index, 1.02, f"{group['finite']}/{group['total']} finite", transform=axis.get_xaxis_transform(), ha='center', fontsize=9)
    axis.set_xticks([0, 1], LABELS, fontsize=10)
    axis.set_xlim(-0.45, 1.45)
    axis.set_ylabel(wrap(label, 30), fontsize=10)
    axis.spines[['right', 'top']].set_visible(False)
    axis.tick_params(labelsize=9)
    if not rows['available']:
        axis.set_yticks([])
        axis.text(0.5, 0.5, 'No finite comparison values', transform=axis.transAxes, ha='center', fontsize=10)

def draw(prepared, settings, *, selected_view=None, canvas=None):
    notes = settings['test_notes']
    footnote = wrap(settings['footnote'], 150)
    note_lines = sum((wrap(note, 145).count('\n') + 1 for note in notes))
    names = [selected_view] if selected_view else ['cells', 'samples', 'duration', 'missingness']
    if any(name not in ('cells','samples','duration','missingness') for name in names):
        raise ValueError('Unknown group view')
    height = (9.4 if selected_view is None else 6.4) + 0.15 * max(0, note_lines - 3)
    figure = _layout.figure(canvas, figsize=(13, height))
    figure.text(0.025, 1 - 0.18 / height, wrap(settings['title'], 105), va='top', fontsize=15, fontweight='bold')
    figure.text(0.025, 1 - 0.91 / height, wrap(settings['membership_note'], 150), va='top', fontsize=9)
    bottom = 1.4 + 0.17 * note_lines
    grid = figure.add_gridspec(2 if selected_view is None else 1, 2 if selected_view is None else 1, left=0.085, right=0.97, top=1 - 1.7 / height, bottom=bottom / height, hspace=0.63, wspace=0.45)
    axes = {name: figure.add_subplot(grid[index // 2, index % 2]) for index, name in enumerate(names)}
    label = settings['value_label']
    if 'cells' in axes:
        points(axes['cells'], prepared['cells'], label)
        axes['cells'].set_title('Individual cells', fontsize=11, pad=26)
    if 'samples' in axes:
        axis = axes['samples']
        if not prepared['samples']:
            axis.set_axis_off()
            axis.text(0.5, 0.5, wrap(settings['unit_reason'], 48), ha='center', va='center', transform=axis.transAxes, fontsize=10)
        else:
            for row in prepared['samples']:
                if row['paired']:
                    axis.plot([0, 1], row['values'], color=house_colour('raw'), linewidth=1, zorder=1)
                for x, value in row['points']:
                    axis.scatter(x, value, color=COLOURS[x], s=26, zorder=2)
            axis.set_xticks([0, 1], LABELS, fontsize=10)
            axis.set_xlim(-0.4, 1.4)
            axis.set_ylabel(wrap(label, 30), fontsize=10)
            axis.spines[['right', 'top']].set_visible(False)
            axis.tick_params(labelsize=9)
            if not prepared['sample_finite']:
                axis.set_yticks([])
            axis.set_title(prepared['sample_title'], fontsize=10, pad=14)
    for name, metric, title, ylabel in (('duration', 'screen_duration_hours', 'Recording span', 'Observed span (h)'), ('missingness', 'screen_invalid_fraction', 'Unusable supplied observations', 'Invalid / supplied observations')):
        if name not in axes: continue
        axis = axes[name]
        rows = prepared[name]
        if settings['grouping_kind'] == 'union':
            axis.set_axis_off()
            axis.text(0.5, 0.5, 'Context is measurement-specific.\nSee the individual grouping pages.', ha='center', va='center', transform=axis.transAxes, fontsize=10)
        else:
            points(axis, rows, ylabel)
            axis.set_title(title, fontsize=11, pad=26)
            if name == 'missingness':
                axis.set_ylim(-0.03, 1.03)
    figure.text(0.025, 1.03 / height, '\n'.join((wrap(note, 145) for note in notes)), va='bottom', fontsize=8)
    figure.text(0.025, 0.12 / height, footnote, va='bottom', fontsize=8)
    return (figure, axes)
