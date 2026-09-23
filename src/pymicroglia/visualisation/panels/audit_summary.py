"""Draw saved audit records only; table sizing follows plot-that's matrix recipe."""
from . import _layout
from ._layout import finish
from . import colour as house_colour
import math
from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib.patches import Rectangle
from ._format import numeric, missing, present, number as display_number
COLOURS = {'admissible': house_colour('grid'), 'confirmed_choice': house_colour('grid'), 'provisional_choice': house_colour('grid'), 'tie': house_colour('grid'), 'significant': house_colour('grid'), 'not-significant': house_colour('blank'), 'inadmissible': house_colour('blank'), 'no_acceptable_candidate': house_colour('blank'), 'insufficient': house_colour('blank'), 'insufficient_evidence': house_colour('blank'), 'untested': house_colour('shade'), 'unavailable': house_colour('shade'), 'unavailable-tests': house_colour('shade'), 'untestable': house_colour('shade')}

def style():
    pass

def wrap(text, width):
    return '\n'.join((line for paragraph in str(text).splitlines() or [''] for line in textwrap.wrap(paragraph, width, break_long_words=False, break_on_hyphens=False) or ['']))

def _matrix(data, title, footnote, *, canvas=None):
    specs = data['specs']
    width = max([9.0] + [2.8 + 3.4 * len(spec[3]) for spec in specs])
    footer = wrap(footnote, max(65, int(width * 11)))
    bottom = 0.35 + 0.18 * (footer.count('\n') + 1)
    height = 1.1 + sum((spec[-1] for spec in specs)) + bottom
    figure = _layout.figure(canvas, figsize=(width, height))
    figure.text(0.02, 1 - 0.15 / height, wrap(title, int(width * 6)), fontsize=18, fontweight='bold', va='top')
    top = height - 1.0
    axes = []
    for name, records, rows, columns, lookup, units, panel_height in specs:
        ax = figure.add_axes([0.02, (top - panel_height + 0.1) / height, 0.96, (panel_height - 0.15) / height])
        ax.set_axis_off()
        caption = str(name)
        if 'facet_label' in records:
            caption += '\n' + wrap(records.facet_label.iloc[0], int(width * 11))
        ax.text(0, 1, caption, va='top', fontsize=11, fontweight='bold', transform=ax.transAxes)
        table = Table(ax, bbox=[0, 0, 1, max(0.3, 1 - 0.65 / panel_height)])
        widths = [2.8 / width] + [(width - 2.8) / width / max(1, len(columns))] * len(columns)
        total = sum(units)
        for col, label in enumerate(['Detrending / cell'] + columns):
            cell = table.add_cell(0, col, widths[col], units[0] / total, text=wrap(label, 30), loc='center', facecolor=house_colour('dark'), edgecolor='white')
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
        for i, label in enumerate(rows, start=1):
            table.add_cell(i, 0, widths[0], units[i] / total, text=wrap(label, 26), loc='left', facecolor=house_colour('actogram_light'), edgecolor=house_colour('raw'))
            for j, column in enumerate(columns, start=1):
                record = lookup.get((label, column), {'state': 'untested', 'display_value': 'Untested'})
                table.add_cell(i, j, widths[j], units[i] / total, text=record['display_value'], loc='center', facecolor=COLOURS.get(record['state'], 'white'), edgecolor=house_colour('raw'))
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        for cell in table.get_celld().values():
            cell.PAD = 0.04
            cell.set_linewidth(0.8)
        ax.add_table(table)
        axes.append(ax)
        top -= panel_height
    figure.text(0.02, 0.12 / height, footer, fontsize=9, color=house_colour('slate_tick'), va='bottom')
    return (figure, axes)

def _periods(data, title, footnote, *, selected_view=None, canvas=None):
    panels=data['panels']
    heights=[panel['height'] for panel in panels]
    height = 1.3 + sum(heights) + 0.8
    figure = _layout.figure(canvas, figsize=(15, height))
    figure.text(0.02, 1 - 0.15 / height, wrap(title, 100), fontsize=18, fontweight='bold', va='top')
    top, axes = (height - 1.0, [])
    for panel,panel_height in zip(panels,heights):
        name,recovery=panel['name'],panel['recovery']
        if selected_view!='errors':
            ax = figure.add_axes([0.3, (top - panel_height + 0.7) / height, (.28 if selected_view is None else .65), (panel_height - 1.1) / height])
            for i, row in enumerate(recovery):
                if present(row['x']):
                    ax.plot([row['lower'], row['upper']], [i, i], color=house_colour('blue'), linewidth=2)
                    ax.plot(row['x'], i, 'o', color=house_colour('blue'), markersize=5)
                else:
                    ax.text(0.02, i, 'unavailable', color=house_colour('nan_text'), va='center', fontsize=9)
            labels = [f"{r['stratum']} | {r['candidate_id'][:10]} (n={r['n']})" for r in recovery]
            ax.set_yticks(range(len(labels)), labels=labels, fontsize=9)
            ax.set_ylim(max(0.5, len(labels) - 0.5), -0.5)
            ax.set_xlim(-0.02, 1.02)
            ax.set_xlabel('Correct recovery fraction (saved interval)', fontsize=11)
            ax.set_title(str(name) + ' — ' + (str(recovery[0]['facet']) if len(recovery) else 'no eligible strata'), fontsize=11, loc='left')
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            axes.append(ax)
        if selected_view!='recovery':
            other = figure.add_axes([.71 if selected_view is None else .2,(top-panel_height+.7)/height,.26 if selected_view is None else .7,(panel_height-1.1)/height])
            distributions=panel['errors']
            for i,row in enumerate(distributions):
                other.scatter(row['values'],[i]*len(row['values']),s=14,alpha=.5,color=house_colour('circadian_teal'))
            other.axvline(0, color=house_colour('nan_text'), linewidth=1)
            other.set_yticks(range(len(distributions)), labels=[str(row['candidate'])[:10] for row in distributions], fontsize=9)
            other.set_xlabel('Selected period − known target (h)', fontsize=11)
            other.set_title(f"Individual case errors\n{panel['missing']} without a comparable single target/estimate", fontsize=10, loc='left')
            for side in ('top', 'right'):
                other.spines[side].set_visible(False)
            axes.append(other)
        top -= panel_height
    figure.text(0.02, 0.12 / height, wrap(footnote, 150), fontsize=9, color=house_colour('slate_tick'), va='bottom')
    return (figure, axes)

def _cards(data, title, footnote, *, canvas=None):
    cards = data['cards']
    columns = min(2, max(1, len(cards)))
    groups = [cards[i:i + columns] for i in range(0, len(cards), columns)]
    heights = [max([2.0] + [0.21 * (wrap(row['display_value'], 65).count('\n') + 2) + 0.75 for row in group]) for group in groups]
    height = 1.2 + sum(heights) + 0.9
    width = 8.0 * columns
    figure = _layout.figure(canvas, figsize=(width, height))
    figure.text(0.02, 1 - 0.15 / height, wrap(title, int(width * 7)), fontsize=18, fontweight='bold', va='top')
    top, axes = (height - 1.0, [])
    for group, panel_height in zip(groups, heights):
        for index, row in enumerate(group):
            ax = figure.add_axes([0.02 + index / columns, (top - panel_height + 0.1) / height, 1 / columns - 0.04, (panel_height - 0.2) / height])
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=COLOURS.get(row['state'], house_colour('blank')), edgecolor='none', zorder=0))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(house_colour('raw'))
                spine.set_linewidth(1)
            ax.text(0.035, 0.97, wrap(row['panel'], 60), va='top', fontsize=12, fontweight='bold', transform=ax.transAxes)
            ax.text(0.035, 1 - 0.48 / panel_height, wrap(row['display_value'], 65), va='top', fontsize=10, transform=ax.transAxes)
            axes.append(ax)
        top -= panel_height
    figure.text(0.02, 0.12 / height, wrap(footnote, int(width * 10)), fontsize=9, color=house_colour('slate_tick'), va='bottom')
    return (figure, axes)

def draw(data, *, kind, title, footnote, selected_view=None, canvas=None):
    style()
    if data['empty']:
        figure, ax = _layout.subplots(canvas, figsize=(9, 3))
        ax.set_axis_off()
        ax.text(0.02, 0.8, title, fontsize=16, fontweight='bold', transform=ax.transAxes)
        ax.text(0.02, 0.5, 'No saved rows for this display selection', transform=ax.transAxes)
        figure.text(0.03, 0.05, wrap(footnote, 100), fontsize=9)
        return (figure, [ax])
    if kind in {'performance', 'disagreement'}:
        return _matrix(data, title, footnote, canvas=canvas)
    if kind == 'periods':
        return _periods(data, title, footnote, selected_view=selected_view, canvas=canvas)
    if kind == 'decisions':
        return _cards(data, title, footnote, canvas=canvas)
    raise ValueError('Unknown saved audit view: ' + str(kind))
