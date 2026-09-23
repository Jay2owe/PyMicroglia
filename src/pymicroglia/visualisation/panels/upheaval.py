"""Draw frozen event rows, exposure-normalized rates and matched image pairs."""
import numpy as np
from matplotlib.colors import to_rgba
from . import colour
from ._contract import Drawn
from .exchange import _hours


def raster(ax,data,*,style=None,**options):
    t=data['table']
    for row in data['spans'].itertuples():ax.plot([row.min,row.max],[row.row_order]*2,color=colour('raw'),linewidth=.7)
    ax.scatter(t.hours,t.row_order,color=colour('orange'),marker='|')
    ax.set(xlabel='Hours from recording start',ylabel='Cell rank (highest event rate at top)',ylim=(data['count']-.5,-.5))
    _hours(ax,options)
    return Drawn(t,ax)


def rates(ax,data,*,style=None,**options):
    t=data['table'];ax.plot(t['rank'],t.rate_percent,color=colour('orange'),marker='o')
    ax.axhline(100*data['reference'],color=colour('muted'),linestyle='--',label='All measurable transitions')
    ax.set(xlabel='Cell rank',ylabel='Detected transitions (%)');ax.legend()
    return Drawn(t,ax)


def tiles(figure,data,*,style=None,**options):
    items=data['tiles'];axes=[]
    if not items:
        ax=figure.add_subplot(111);ax.text(.5,.5,'No detected transitions to show',ha='center',transform=ax.transAxes);ax.set_axis_off();axes.append(ax)
    else:
        grid=figure.add_gridspec(2,len(items),hspace=.2,wspace=.15)
        for i,tile in enumerate(items):
            for row,state in enumerate(['before','after']):
                ax=figure.add_subplot(grid[row,i]);axes.append(ax)
                ax.imshow(tile[state],cmap='gray',vmin=data['low'],vmax=data['high'],interpolation='nearest')
                border=np.zeros((*tile[state].shape,4))
                if state=='after':border[tile['after_border']]=to_rgba(colour('teal'))
                border[tile['before_border']]=to_rgba(colour('orange'))
                ax.imshow(border,interpolation='nearest');ax.set_title(tile['titles'][row]);ax.set_axis_off()
    return Drawn(data['table'],axes)
