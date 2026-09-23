"""Portable display of saved detection agreement; no statistical calculations."""
from . import _layout
from ._layout import finish
from . import colour as house_colour
from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
import numpy as np

def wrap(text, width=105):
    return '\n'.join((textwrap.fill(line, width) for line in str(text).splitlines()))

def draw(prepared, settings, *, selected_view=None, canvas=None):
    names=['matrix'] if settings['kind']=='matrix' else ['counts','unit_agreement','fractions']
    if selected_view is not None:
        if selected_view not in names:raise ValueError('This saved page has no '+selected_view+' view')
        names=[selected_view]
    if settings['kind'] == 'matrix':
        rows, columns = (settings['rows'], settings['columns'])
        width, height = (max(9.0, 3.8 + 0.62 * len(columns)), max(7.0, 3.9 + 0.6 * len(rows)))
        figure, axis = _layout.subplots(canvas, figsize=(width, height))
        figure.subplots_adjust(left=2.25 / width, right=0.9, top=1 - 1.25 / height, bottom=2.45 / height)
        matrix = prepared['matrix']
        colour = plt.get_cmap('RdBu_r').with_extremes(bad=house_colour('blank'))
        drawn = axis.imshow(matrix, cmap=colour, vmin=-1, vmax=1, aspect='equal', interpolation='none')
        for row in prepared['records']:
            if row['label']:
                axis.text(row['x'],row['y'],row['label'],ha='center',va='center',fontsize=7,color='white' if row['bright'] else 'black')
        axis.set_xticks(range(len(columns)), [wrap(x, 24) for x in columns], rotation=50, ha='right', fontsize=9)
        axis.set_yticks(range(len(rows)), [wrap(x, 25) for x in rows], fontsize=9)
        colourbar = figure.colorbar(drawn, ax=axis, fraction=0.045, pad=0.045)
        colourbar.set_label('Mean within-unit kappa', fontsize=9)
        colourbar.ax.tick_params(labelsize=8)
        axes = {'matrix': axis}
    else:
        units = prepared['units']
        label_lines = max([1, *[len(wrap(str(row['unit_label']), 24).splitlines()) for row in units]])
        width, height = (14.0, max(7.8, 3.6 + max(0.23, 0.14 * label_lines) * len(units)))
        figure = _layout.figure(canvas, figsize=(width, height))
        grid = figure.add_gridspec(1, len(names), left=0.055, right=0.985, bottom=2.2 / height, top=1 - 1.55 / height, width_ratios=[{'counts':1.1,'unit_agreement':1.5,'fractions':1.05}[name] for name in names], wspace=0.65)
        axes = {name: figure.add_subplot(grid[0, i]) for i, name in enumerate(names)}
        categories, counts = (settings['categories'], settings['counts'])
        if 'counts' in axes:
            axis = axes['counts']
            axis.barh(range(len(counts)), counts, color=[house_colour('okabe_bluish_green'), house_colour('circadian_amber'), house_colour('blue'), house_colour('actogram_separator'), house_colour('shade'), house_colour('shade'), house_colour('shade')])
            axis.set_yticks(range(len(counts)), categories, fontsize=8)
            axis.invert_yaxis()
            axis.set_xlabel('Cells in complete pair population', fontsize=9)
            axis.set_title(f"All requested cells: {settings['requested_cells']}", fontsize=10)
            for y, count in enumerate(counts):
                axis.text(count, y, f' {count}', va='center', fontsize=8)
            axis.set_xlim(0, max([1, *counts]) * 1.2)
        if 'unit_agreement' in axes:
            axis = axes['unit_agreement']
            for index, row in enumerate(units):
                if row['valid_kappa']:
                    axis.scatter(row['kappa'], index, color=house_colour('okabe_bluish_green'), s=25)
                else:
                    axis.text(0.03, index, 'Constant/absent margins', transform=axis.get_yaxis_transform(), fontsize=7, va='center')
            axis.set_yticks(range(len(units)), [wrap(str(row['unit_label']), 24) for row in units], fontsize=8)
            axis.set_ylim(len(units) - 0.5, -0.5)
            axis.set_xlim(-1.08, 1.08)
            axis.axvline(0, color=house_colour('raw'), linewidth=1, zorder=0)
            axis.set_xlabel('Observed within-unit kappa', fontsize=9)
            axis.set_title(wrap(settings['unit_title'], 38), fontsize=10)
            if not units:
                axis.set_yticks([])
                axis.text(0.5, 0.5, 'No contributing units', transform=axis.transAxes, ha='center', fontsize=9)
        if 'fractions' in axes:
            axis = axes['fractions']
            axis.scatter(prepared['fraction_x'],prepared['fraction_y'],color=house_colour('okabe_bluish_green'),s=28)
            for point in prepared['annotations']:
                x,y=point['x'],point['y']
                axis.annotate(wrap(point['label'],26),(x,y),xytext=(-4 if x>.85 else 4,4),ha='right' if x>.85 else 'left',textcoords='offset points',fontsize=6)
            axis.plot([0, 1], [0, 1], color=house_colour('actogram_dark'), linestyle='--', linewidth=1)
            axis.set_xlim(-0.07, 1.13)
            axis.set_ylim(-0.07, 1.13)
            axis.set_xlabel('Reference detected / jointly tested', fontsize=9)
            axis.set_ylabel('Target detected / jointly tested', fontsize=9)
            axis.set_title('One paired fraction per named unit', fontsize=10)
        for axis in axes.values():
            axis.spines[['top', 'right']].set_visible(False)
            axis.tick_params(labelsize=8)
    figure.text(0.02, 1 - 0.15 / height, wrap(settings['title'], 110), fontsize=14, fontweight='bold', va='top')
    figure.text(0.02, 1.0 / height, wrap(settings['evidence_note'], 155), fontsize=8, va='bottom')
    figure.text(0.02, 0.12 / height, wrap(settings['footnote'], 155), fontsize=8, va='bottom')
    return (figure, axes)
