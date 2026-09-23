"""Draw prepared cell tiles and independently selectable measurement traces."""
import numpy as np
from matplotlib.colors import to_rgba
from . import colour,resolve_colour
from ._contract import Drawn
from ..labels import semantic_label
from ..roles import ROLES


def tiles(figure,data,*,style=None,**options):
    from auto_organotypic.render.luts import colormap
    crops=data['crops'];axes=[]
    grid=figure.add_gridspec(1,max(1,len(crops)))
    for i,(crop,boundary) in enumerate(zip(crops,data['boundaries'])):
        ax=figure.add_subplot(grid[i]);axes.append(ax);ax.imshow(crop,cmap=colormap(data['lut']),vmin=0,vmax=1,interpolation='nearest')
        overlay=np.zeros((*crop.shape,4));overlay[boundary]=to_rgba(resolve_colour(ROLES.get(data['outline'],data['outline'])));ax.imshow(overlay,interpolation='nearest')
        ax.set_title(f"{data['table'].hours.iloc[i]:.0f} h");ax.set_axis_off()
    if not crops:
        ax=figure.add_subplot(grid[0]);ax.text(.5,.5,'No image tiles requested',ha='center',transform=ax.transAxes);ax.set_axis_off();axes.append(ax)
    return Drawn(data['table'],axes)


def traces(figure,data,*,style=None,**options):
    t=data['table'];overlay=data['layout']=='overlay';grid=figure.add_gridspec(1 if overlay else len(data['metrics']),1);axes=[]
    if overlay:axes.append(figure.add_subplot(grid[0]))
    for i,metric in enumerate(data['metrics']):
        ax=axes[0] if overlay else figure.add_subplot(grid[i])
        if not overlay:axes.append(ax)
        rows=t.loc[t.metric.eq(metric)];name=semantic_label(metric)
        requested=data['trace_luts'][i] if i<len(data['trace_luts']) else ['teal','orange','blue','plum'][i%4]
        try:hue=resolve_colour(requested)
        except (KeyError,ValueError):
            from matplotlib import colormaps
            hue=colormaps[requested](.75)
        ax.plot(rows.hours,rows.plotted_value,color=hue,label=name)
        if rows.fitted_value.notna().any():ax.plot(rows.hours,rows.fitted_value,color=hue,linestyle=(0,(6,3)))
        if not overlay:
            ax.set_ylabel(name)
            if metric in data['fit_notes']:
                ax.text(1.,1.04,data['fit_notes'][metric],transform=ax.transAxes,ha='right',va='bottom')
        if i==len(data['metrics'])-1:ax.set_xlabel('Hours from recording start')
        if options.get('hour_ticks') is not None:
            from matplotlib.ticker import MultipleLocator
            ax.xaxis.set_major_locator(MultipleLocator(float(options['hour_ticks'])))
    if overlay and data['fit_notes']:
        import textwrap
        axes[0].text(1.,1.04,textwrap.fill(' | '.join(data['fit_notes'].values()),110),transform=axes[0].transAxes,ha='right',va='bottom')
    if overlay:axes[0].set_ylabel('Within-trace standard deviations');axes[0].legend()
    return Drawn(t,axes)
