"""Draw original presence values and categorical tracker annotations."""
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from . import colour
from ._contract import Drawn
from .exchange import _hours
from ..labels import semantic_label


def counts(ax,data,*,style=None,**options):
    t=data['table'];ax.plot(t.hours,t.expected,color=colour('muted'),linestyle='--',label='Inside observed lifespan');ax.plot(t.hours,t.named,color=colour('dark'),label='Name on screen')
    ax.fill_between(t.hours,t.named,t.expected,color=colour('orange'),alpha=.28,label='Temporarily off screen')
    ax.set(xlabel='Hours from recording start',ylabel='Cell identities');ax.legend()
    _hours(ax,options)
    return Drawn(t,ax)


def foreground(ax,data,*,style=None,**options):
    t=data['table']
    if t.empty:ax.text(.5,.5,'Unclaimed foreground was not recorded',ha='center',transform=ax.transAxes)
    else:
        ax.fill_between(t.hours,0,t.claimed_px,color=colour('dark'),alpha=.18,label='Claimed foreground')
        ax.fill_between(t.hours,t.claimed_px,t.foreground_px,color=colour('orange'),alpha=.72,label='Unclaimed foreground');ax.legend()
    ax.set(xlabel='Hours from recording start',ylabel='Detected foreground (pixels)')
    _hours(ax,options)
    return Drawn(t,ax)


def persistence(ax,data,*,style=None,**options):
    t=data['table'];ax.pcolormesh(data['hours'],np.arange(len(data['identities'])),data['matrix'],cmap=ListedColormap([colour(c) for c in ['raw','dark','orange']]),vmin=0,vmax=2,shading='nearest')
    for label,rows in data['events'].groupby('event_label',sort=False):
        marker=rows.marker.iloc[0]
        kwargs=dict(color=colour('dark')) if marker=='x' else dict(facecolors=colour('page'),edgecolors=colour('dark'))
        ax.scatter(rows.hours,rows.row_position,marker=marker,label=label,**kwargs)
    ax.set(xlabel='Hours from recording start',ylabel='Cell, ordered by first appearance');ax.invert_yaxis()
    handles=[Patch(facecolor=colour(c),label=n) for n,c in zip(['Outside lifespan','Named','Temporary gap'],['raw','dark','orange'])]
    existing,labels=ax.get_legend_handles_labels();ax.legend(handles=[*handles,*existing],loc='upper left',bbox_to_anchor=(1.01,1))
    _hours(ax,options)
    return Drawn(t,ax)


def lifespan(ax,data,*,style=None,**options):
    t=data['table'];ax.barh(t.row_order,t.last_hour-t.first_hour+data['step'],left=t.first_hour,height=.68,color=colour('dark'))
    for row in data['cuts'].itertuples():ax.barh(row.row_order,data['step'],left=row.hours,height=.68,color=colour(row.gap_colour))
    marked=t.loc[t.marked];ax.scatter(marked.last_hour+data['step'],marked.row_order,marker='x',color=colour('red'))
    ax.set(xlabel='Hours from recording start',ylabel='Cell rank');ax.invert_yaxis()
    ax.legend(handles=[Patch(facecolor=colour(c),label=label.replace('_',' ')) for label,c in data['palette'].items()],loc='upper left',bbox_to_anchor=(1.01,1))
    _hours(ax,options)
    return Drawn(t,ax)


def distribution(ax,data,*,style=None,**options):
    t=data['table'];ax.bar(t.bin_left,t['count'],width=t.bin_right-t.bin_left,align='edge',color=colour('teal'))
    ax.set(xlabel=data['label'],ylabel='Cells')
    return Drawn(t,ax)
