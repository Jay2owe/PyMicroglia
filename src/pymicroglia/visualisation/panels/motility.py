"""Draw prepared movement curves and bands; retain gaps in the measured data."""
import numpy as np
from . import colour
from ._contract import Drawn


def histogram(ax,data,*,style,**options):
    table = data['table']
    name = options.get('trace_luts') or 'orange'
    from . import resolve_colour
    ax.bar(table[data['x']],table[data['y']],width=table[data['right']]-table[data['x']],
           align='edge',color=resolve_colour(name),edgecolor=colour('white'),linewidth=.3)
    if data['log']:
        ax.set_xscale('log')
    import textwrap
    ax.set(xlabel=textwrap.fill(data['xlabel'],45),ylabel=data['ylabel'])
    for index,(value,label) in enumerate(data['marks']):
        if np.isfinite(value):
            ax.axvline(value,color=colour('circadian_ink' if index==0 else 'red'),
                       linestyle='-' if index==0 else '--',label=f'{label}: {value:.2f}')
    if data['marks']:
        ax.legend(frameon=False)
    return Drawn(table,ax)


def curves(ax,data,*,style,**options):
    for index,(_,group) in enumerate(data['table'].groupby('identity',sort=True)):
        group = group.sort_values('lag_hours')
        ax.plot(group.lag_hours,group.msd,color=colour('orange'),alpha=.18,
                label='Individual cells' if index==0 else None)
    aggregate = data['aggregate']
    if not aggregate.empty:
        ax.fill_between(aggregate.lag_hours,aggregate.population_q25_msd,aggregate.population_q75_msd,
                        color=colour('orange'),alpha=.18,label='Middle 50% of cells')
        ax.plot(aggregate.lag_hours,aggregate.population_median_msd,color=colour('orange'),label='Median across cells')
    if len(data['reference']):
        ax.plot(data['reference'][:,0],data['reference'][:,1],color=colour('muted'),linestyle='--',label='Random-walk-like growth')
    ax.set(xscale='log',yscale='log',xlabel='Time between compared frames (hours)',ylabel='Average squared centroid shift')
    from matplotlib.ticker import LogLocator,FuncFormatter
    ax.xaxis.set_major_locator(LogLocator(base=2))
    ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
    for axis in (ax.xaxis,ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda value,position:f'{value:g}'))
    ax.legend(frameon=False)
    return Drawn(data['table'],ax)
