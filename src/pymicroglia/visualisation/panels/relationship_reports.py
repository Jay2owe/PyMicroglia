"""Portable selected-pair evidence and original-clock observation panels."""
from . import _layout
from ._layout import finish
from . import colour as house_colour
import math
from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
import numpy as np
from ._format import numeric, missing, present, number as display_number

def wrap(value, width=60):
    return '\n'.join((textwrap.fill(line, width) for line in str(value).splitlines()))

def number(value):
    try:
        return f'{float(value):.3g}' if math.isfinite(float(value)) else 'unavailable'
    except (ValueError, TypeError):
        return 'unavailable'

def draw(prepared, settings, *, selected_view=None, canvas=None):
    rows = len(settings['members'])
    cell_view = settings['view'] == 'cells'
    height = 2.0 + (4.4 if cell_view else 3.2) * rows
    indices=([0,1] if selected_view=='traces' else [2] if selected_view=='scatter' else [3] if selected_view=='lags' else [0,1,2,3]) if cell_view else [0,1]
    columns=len(indices)
    figure = _layout.figure(canvas, figsize=(16 if cell_view else 12, height))
    layout = figure.add_gridspec(rows * 2, columns, height_ratios=([3, 1.1] if cell_view else [1.8, 1.1]) * rows)
    grid = {(i,j):figure.add_subplot(layout[2*i,column]) for i in range(rows) for column,j in enumerate(indices)}
    captions = {(i,j):figure.add_subplot(layout[2*i+1,column]) for i in range(rows) for column,j in enumerate(indices)}
    for axis in captions.values():
        axis.axis('off')
    for axis in grid.values():
        axis.tick_params(labelsize=8)
    figure.subplots_adjust(left=0.06, right=0.98, top=1 - 1.1 / height, bottom=0.65 / height, hspace=0.8, wspace=0.65)
    measurements = settings['measurements']
    axes = {}
    for i, member in enumerate(settings['members']):
        data=prepared[i]
        within,lag=data['within'],data['lag']
        cell_label = f"{member['movie']} / cell {member['identity']}"

        def label(name):
            m = measurements[name]
            unit = m.get('unit')
            return m['label'] + (f' ({unit})' if unit else ' (unit not recorded)')

        def evidence(row):
            return str(row.status).replace('-', ' ') + f'; p={number(row.p_value)}, q={number(row.q_value)}'
        if not cell_view:
            for j, (row, effect, title) in enumerate(((within, within.get('full_overlap_effect'), 'Same-time full overlap'), (lag, lag.get('effect'), 'Complete lag-search test window'))):
                axis = grid[i, j]
                axis.axvline(0, color=house_colour('raw'), lw=0.8)
                if number(effect) != 'unavailable':
                    axis.plot([effect], [0], 'o', color=house_colour('circadian_teal'), ms=8)
                else:
                    axis.text(0.5, 0.5, 'Coefficient unavailable', transform=axis.transAxes, ha='center')
                axis.set_xlim(-1.08, 1.08)
                axis.set_yticks([])
                axis.set_ylim(-1, 1)
                axis.set_xlabel(f"Saved {row.statistic or 'association'} coefficient", fontsize=9)
                axis.set_title(wrap(cell_label + ' | ' + title, 60), fontsize=11)
                text = evidence(row)
                if j == 0:
                    text += f"\nTest-window coefficient: {number(row.get('effect'))}; full-overlap: {number(effect)}"
                else:
                    text += f"\nDelay: {number(row.delay_hours)} h; {str(row.resolution_status).replace('-', ' ')}"
                    text += '\n' + str(row.resolution_reason)
                captions[i, j].text(0, 1, wrap(text, 72), fontsize=9, va='top')
                axes[f'cell_{i}_{j}'] = axis
            continue
        for j, name in enumerate((member['reference'], member['target'])):
            if j not in indices:continue
            axis = grid[i, j]
            frame = data['traces'][name]
            for column, colour, style, legend in (('raw_value', house_colour('actogram_day_separator'), '-', 'Raw'), ('processed_value', house_colour('circadian_teal'), '-', 'Processed')):
                if column == 'processed_value' and settings['representation'] == 'raw':
                    continue
                x,y=frame['series'][column]
                axis.plot(x, y, color=colour, ls=style, lw=0.85, label=legend)
            if frame['empty']:
                axis.text(0.5, 0.5, 'No trace observations', transform=axis.transAxes, ha='center')
            axis.set_xlabel('Recording time (hours)', fontsize=9)
            axis.set_ylabel(wrap(label(name), 25), fontsize=9)
            axis.set_title(wrap(cell_label + ' | ' + ('reference' if j == 0 else 'target'), 38), fontsize=10)
            axis.legend(fontsize=8, loc='upper right')
            axes[f'cell_{i}_{j}'] = axis
        if 2 in indices:
            scatter=data['scatter']
            axis = grid[i, 2]
            axis.scatter(*scatter, s=8, alpha=0.5, color=house_colour('circadian_teal'), edgecolors='none')
            axis.set_xlabel(wrap(label(member['reference']), 28), fontsize=9)
            axis.set_ylabel(wrap(label(member['target']), 28), fontsize=9)
            axis.set_title(f'Saved same-time pairs (n={len(scatter[0])})', fontsize=10)
            captions[i, 2].text(0, 1, wrap(evidence(within), 40), fontsize=8, va='top')
            axes[f'cell_{i}_2'] = axis
        if 3 in indices:
            profile=data['profile']
            axis = grid[i, 3]
            if len(profile['x']):
                axis.plot(profile['x'],profile['effect'], color=house_colour('grade_grey'), marker='.', label='Per-lag overlap')
                if profile['native'] is not None:
                    axis.plot(profile['x'],profile['native'], color=house_colour('circadian_teal'), marker='.', label='Fixed test window')
                axis.legend(fontsize=7)
            else:
                axis.text(0.5, 0.5, 'Lag question disabled', transform=axis.transAxes, ha='center')
            axis.set_ylim(-1.08, 1.08)
            axis.axhline(0, color=house_colour('raw'), lw=0.7)
            axis.set_xlabel('Saved lag (hours)', fontsize=9)
            axis.set_ylabel('Saved coefficient', fontsize=9)
            axis.set_title('Full saved lag profile', fontsize=10)
            text = evidence(lag) + f"\nDelay: {number(lag.delay_hours)} h; {str(lag.resolution_status).replace('-', ' ')}\n" + str(lag.resolution_reason)
            captions[i, 3].text(0, 1, wrap(text, 42), fontsize=8, va='top')
            axes[f'cell_{i}_3'] = axis
    figure.text(0.02, 1 - 0.1 / height, wrap(settings['title'] + f" | page {settings['page_number']}", 120), fontsize=14, fontweight='bold', va='top')
    pair = settings['members'][0]
    figure.text(0.02, 1 - 0.52 / height, wrap('Reference: ' + measurements[pair['reference']]['label'] + ' | Target: ' + measurements[pair['target']]['label'], 135), fontsize=10, va='top')
    figure.text(0.02, 0.1 / height, wrap(settings['footnote'], 155 if cell_view else 120), fontsize=8, va='bottom')
    return (figure, axes)
