"""Portable drawing of saved cell evidence on original recording time."""
from . import _layout
from ._layout import finish
from . import colour as house_colour
import math
from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
import numpy as np

def wrap(value, width=80):
    return '\n'.join((textwrap.fill(line, width, break_long_words=False, break_on_hyphens=False) for line in str(value).splitlines()))

def number(value):
    try:
        return f'{float(value):.3g}' if math.isfinite(float(value)) else 'unavailable'
    except (ValueError, TypeError):
        return 'unavailable'

def evidence_label(row):
    period = number(row['period_hours']) + ' h' if row['supported_period'] else 'unresolved'
    return f"Adjusted p = {number(row['q_value'])}; period {period}; {number(row['observations'])} observations / {number(row['span_hours'])} h"

def trace(ax, points, view, bounds, *, unavailable='No saved trace available'):
    if not points['available']:
        ax.text(.5,.5,wrap(unavailable,40),transform=ax.transAxes,va='center',ha='center',fontsize=9)
    else:
        for series in points['series']:
            ax.plot(series['x'],series['y'],'-' if view=='native' else '.',markersize=3,linewidth=1.,label=series['name'],color=house_colour('blue'))
        ax.set_ylabel(wrap(points['units'],20),fontsize=8)
        if points['legend']:ax.legend(fontsize=7,frameon=False)
    if bounds and bounds['max'] > bounds['min']:
        ax.set_xlim(bounds['min'], bounds['max'])
    ax.set_xlabel('Recording time (h)', fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines[['top', 'right']].set_visible(False)

def draw(prepared, *, settings, selected_view=None, canvas=None):
    allowed=('raw','detrended','native') if settings['kind']=='grids' else ('raw','detrended','native','images')
    if selected_view is not None and selected_view not in allowed:raise ValueError('Unknown rhythm trace view')
    if settings['kind']=='grids':return draw_grid(prepared,settings,selected_view=selected_view,canvas=canvas)
    return draw_report(prepared,settings,selected_view=selected_view,canvas=canvas)


def draw_report(prepared, settings, *, selected_view=None, canvas=None):
    image_record=prepared['image_record']
    entries=[] if selected_view=='images' else prepared['entries']
    show_images=selected_view in {None,'images'}
    status_lines = [wrap(f"{r['measurement_label']}: {r['display_state'].replace('-', ' ')}", 110) for r in prepared['status']]
    status_height = 0.3 + 0.2 * sum((line.count('\n') + 1 for line in status_lines))
    images = show_images and image_record.get('status') == 'available'
    image_rows = math.ceil(len(image_record.get('tiles', [])) / 6)
    image_height = (2.6 * image_rows + .3 if images else .65) if show_images else 0
    width = 16.0
    header = 1.05 + status_height + image_height
    row_height, footer = (3.3, 1.05)
    height = header + len(entries) * row_height + footer
    figure = _layout.figure(canvas, figsize=(width, height))
    figure.text(0.02, 1 - 0.15 / height, wrap(settings['title'], 110), fontsize=16, fontweight='bold', va='top')
    figure.text(0.02, 1 - 1.05 / height, '\n'.join(status_lines), fontsize=9, va='top')
    axes = {}
    image_top = 1.05 + status_height
    if images:
        count = len(image_record['tiles'])
        tile_width = min(2.45, (width - 0.8) / min(count, 6))
        for i, saved in enumerate(prepared['tiles']):
            tile=saved['record']
            ax = figure.add_axes([(0.35 + i % 6 * tile_width) / width, (height - image_top - 2.4 - i // 6 * 2.6) / height, (tile_width - 0.08) / width, 2.05 / height])
            ax.imshow(saved['pixels'], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            if saved['outline'] is not None:
                ax.contour(saved['outline'], levels=[0.5], colors=[house_colour('cyan')], linewidths=0.6)
            ax.set_title(f"{number(tile['hours'])} h" + ('; cell absent' if not tile['cell_present'] else ''), fontsize=9)
            ax.set_axis_off()
            axes[f'image-{i}'] = ax
        figure.text(0.025, (height - image_top - image_height + 0.2) / height, 'Recorded snapshots; constant crop and display scale within this strip. Display settings do not alter measurements.', fontsize=8)
    elif show_images:
        figure.text(0.02, (height - image_top - 0.28) / height, wrap('Images: ' + image_record.get('reason', 'No saved image snapshots'), 140), fontsize=9, va='top')
    for i, entry in enumerate(entries):
        row=entry['evidence']
        top = height - header - i * row_height
        detail = f"{row['measurement_label']} | {evidence_label(row)}\nEstimator: {row['method']}; test: {row['significance_method']}; cycles observed: {number(row['cycles_observed'])}; estimate status: {row['estimate_status']}"
        figure.text(0.025, (top - 0.05) / height, wrap(detail, 140), fontsize=9, va='top')
        panels=(('raw', 'Original observations', 'No original observations'), ('detrended', 'Saved detrending diagnostic' + (f" ({row['processing_source_method']})" if row['processing_available'] else ''), row['processing_reason']), ('native', "Selected estimator's supplied waveform", row['native_reason']))
        panels=[item for item in panels if selected_view is None or item[0]==selected_view]
        stride=15.45/len(panels)
        for j,(view,label,reason) in enumerate(panels):
            ax = figure.add_axes([(0.7 + j * stride) / width, (top - 2.7) / height, (stride-.6) / width, 1.65 / height])
            trace(ax, entry['traces'][view], view, settings['bounds'], unavailable=reason)
            ax.set_title(label, fontsize=9)
            axes[f'{i}-{view}'] = ax
    figure.text(0.02, 0.13 / height, wrap(settings['footnote'], 170), fontsize=8, va='bottom')
    return (figure, axes)

def draw_grid(prepared, settings, *, selected_view=None, canvas=None):
    entries=prepared['entries']
    view=selected_view or settings['trace_view']
    columns = min(settings['grid_columns'], max(1,len(entries)))
    rows = max(1,math.ceil(len(entries) / columns))
    width, height = (max(8.0, columns * 4.9), 2.55 + rows * 3.05)
    figure = _layout.figure(canvas, figsize=(width, height))
    figure.text(0.02, 1 - 0.15 / height, wrap(settings['title'], int(width * 7)), fontsize=16, fontweight='bold', va='top')
    axes = {}
    for i, entry in enumerate(entries):
        row=entry['evidence']
        x, y = (i % columns, i // columns)
        ax = figure.add_axes([(x * width / columns + 0.72) / width, (height - 1.15 - (y + 1) * 3.05 + 0.55) / height, (width / columns - 1.05) / width, 1.9 / height])
        trace(ax, entry['traces'][view], view, settings['bounds'], unavailable=row['processing_reason'])
        ax.set_title(f"{row['cell_label']}\n" + wrap(evidence_label(row), 50), fontsize=9)
        axes[str(i)] = ax
    figure.text(0.02, 0.12 / height, wrap(settings['footnote'], int(width * 11)), fontsize=8, va='bottom')
    return (figure, axes)
