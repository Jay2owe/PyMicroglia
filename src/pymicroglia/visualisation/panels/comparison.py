"""Native group views, shared by browser previews, Python and figure export."""
import numpy as np

from .rhythm_groups import points
from .aligned_clocks import wall
from ._contract import Drawn


def draw(figure, data, *, style=None, **options):
    import matplotlib as mpl
    font_size=mpl.rcParams['font.size']
    view = data['config']['view']
    if view == 'effects':
        from .contrast import effects
        return effects(figure, data['effects'], style=style)
    axis = figure.add_subplot(111)
    labels, colours = data['labels'], data['colours']
    rows = data['cells'] if view == 'cells' else data['samples']
    # The UI keeps local paths for click-through navigation. Published plotted
    # data uses the stable recording_id, which also distinguishes multiple runs.
    exported_rows = rows.drop(columns=['run'], errors='ignore')
    if not labels:
        axis.text(.5, .5, 'No included finite measurements', ha='center', transform=axis.transAxes)
        axis.set_axis_off()
        return Drawn(exported_rows, axis)
    if view in ('cells', 'samples', 'paired'):
        points(axis, data['positioned'], data['ylabel'], labels=labels, colours=colours, font_size=font_size)
        if view == 'paired':
            if data['config']['unit'] == 'cell':
                raise ValueError('Paired samples require recording or subject summaries')
            pairs = rows[rows.pair.fillna('').ne('')].groupby(['pair', 'group']).value.agg(data['config']['aggregate']).unstack('group').reindex(columns=data['groups'])
            if pairs.empty:
                raise ValueError('Assign pairing identifiers under Advanced to show paired samples')
            for _, row in pairs.iterrows():
                axis.plot(range(len(labels)), row.to_numpy(float), color=colours[0], alpha=.35, linewidth=1, zorder=0)
    elif view in ('box', 'violin'):
        for index, group in enumerate(data['groups']):
            values = rows[rows.group.eq(group)].value.dropna().to_numpy(float)
            if not len(values):
                continue
            if view == 'box':
                box = axis.boxplot([values], positions=[index], widths=.45, patch_artist=True)
                box['boxes'][0].set_facecolor(colours[index])
                box['boxes'][0].set_alpha(.25)
            elif len(values) > 1 and np.ptp(values) > 0:
                violin = axis.violinplot([values], positions=[index], widths=.7, showmedians=True, bw_method=data['config']['bandwidth'])
                for body in violin['bodies']:
                    body.set_facecolor(colours[index])
        points(axis, data['positioned'], data['ylabel'], labels=labels, colours=colours, font_size=font_size)
    elif view == 'histogram':
        values = [rows[rows.group.eq(group)].value.dropna().to_numpy(float) for group in data['groups']]
        finite = np.concatenate(values)
        edges = np.histogram_bin_edges(finite, bins=data['config']['bins']) if len(finite) else data['config']['bins']
        for value, label, hue in zip(values, labels, colours):
            axis.hist(value, bins=edges, histtype='step', label=label, color=hue)
        axis.set(xlabel=data['ylabel'], ylabel='Cells' if data['config']['unit'] == 'cell' else 'Samples')
        axis.legend()
    elif view == 'time_course':
        for series, label, hue in zip(data['timeline'], labels, colours):
            wall(axis, dict(table=series['table'], metric=data['config']['metric']), colour=hue, label=label)
        axis.set_ylabel(data['ylabel'])
        axis.legend()
    axis.spines[['right', 'top']].set_visible(False)
    plotted = exported_rows
    if view == 'time_course':
        import pandas as pd
        plotted = pd.concat([row['table'].assign(group=row['group'])
                             for row in data['timeline']], ignore_index=True)
    return Drawn(plotted, axis)
