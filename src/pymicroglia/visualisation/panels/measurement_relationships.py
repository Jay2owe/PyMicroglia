"""Portable drawing of saved relationship values and explicit missing states."""
from . import _layout
from ._layout import finish
from . import colour as house_colour
import math
from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
import numpy as np

def wrap(value, width=95):
    return '\n'.join((textwrap.fill(line, width) for line in str(value).splitlines()))

def number(value):
    try:
        return f'{float(value):.3g}' if math.isfinite(float(value)) else 'Unavailable'
    except (ValueError, TypeError):
        return 'Unavailable'

def draw(prepared, settings, *, canvas=None):
    rows, columns = (settings['rows'], settings['columns'])
    width, height = (max(10.0, 4.0 + 1.5 * len(columns)), max(8.0, 4.5 + 1.2 * len(rows)))
    figure, axis = _layout.subplots(canvas, figsize=(width, height))
    figure.subplots_adjust(left=2.2 / width, right=0.86, top=1 - 1.5 / height, bottom=2.55 / height)
    matrix=prepared['matrix']
    delay = settings['view'] == 'delay'
    limit = settings.get('lag_colour_limit_hours', 1.0) if delay else 1.0
    image = axis.imshow(matrix, cmap=plt.get_cmap('RdBu_r').with_extremes(bad=house_colour('blank')), vmin=-limit, vmax=limit, interpolation='none', aspect='equal')
    for row in prepared['rows']:
        if not row.requested:
            continue
        label = number(row.value) + (' h' if delay and np.isfinite(row.value) else '')
        status = str(row.status).replace('-', ' ')
        if label.lower() != status.lower():
            label += '\n' + wrap(status, 19)
        label += f'\n{int(row.eligible_cells)}/{int(row.requested_cells)} cells'
        if row.samples is not None and np.isfinite(row.samples):
            label += f'; {int(row.samples)} samples'
        if delay and row.delay_unresolved_cells is not None and np.isfinite(row.delay_unresolved_cells):
            label += f'\n{int(row.delay_supported_cells)} resolved; {int(row.delay_unresolved_cells)} unresolved'
        axis.text(row.column_index, row.row_index, label, ha='center', va='center', fontsize=9, color='white' if np.isfinite(row.value) and abs(row.value) / limit > 0.7 else house_colour('circadian_ink'))
    labels = settings['labels']
    axis.set_xticks(range(len(columns)), [wrap(labels.get(name, name), 24) for name in columns], rotation=35, ha='right', fontsize=9)
    axis.set_yticks(range(len(rows)), [wrap(labels.get(name, name), 24) for name in rows], fontsize=9)
    axis.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    axis.grid(which='minor', color='white', linewidth=1.5)
    axis.tick_params(which='minor', bottom=False, left=False)
    axis.set_xlabel('Target measurement', fontsize=10)
    axis.set_ylabel('Reference measurement', fontsize=10)
    bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.07)
    bar.set_label('Saved delay (hours)' if delay else 'Saved association coefficient', fontsize=9)
    bar.ax.tick_params(labelsize=8)
    figure.text(0.02, 1 - 0.15 / height, wrap(settings['title'], 100), fontsize=14, fontweight='bold', va='top')
    figure.text(0.02, 1.06 / height, wrap(settings['evidence_note'], 145), fontsize=8, va='bottom')
    figure.text(0.02, 0.12 / height, wrap(settings['footnote'], 145), fontsize=8, va='bottom')
    return (figure, {'matrix': axis})
