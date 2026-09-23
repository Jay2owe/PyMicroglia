"""Show recurrence distances and the saved matched-noise comparison."""
import numpy as np
from . import colour
from ._contract import Drawn
from .exchange import _hours


def wall(figure,data,*,style=None,**options):
    count=len(data['walls']);columns=min(4,int(np.ceil(np.sqrt(count))))
    layout=figure.add_gridspec(int(np.ceil(count/columns)),columns)
    axes=[]
    for index,(identity,(distance,hours,threshold)) in enumerate(data['walls'].items()):
        ax=figure.add_subplot(layout[index//columns,index%columns]);axes.append(ax)
        image=ax.pcolormesh(hours,hours,distance,shading='nearest',cmap='viridis')
        if distance.min()<threshold<distance.max():
            ax.contour(hours,hours,distance,levels=[threshold],colors=[colour('dark')],linewidths=.5)
        ax.set(title=f'Cell {identity}',xlabel='First observation (h)',ylabel='Second observation (h)')
        ax.set_aspect('equal')
    figure.colorbar(image,ax=axes,label='State difference',shrink=.65)
    return Drawn(data['table'],axes)


def rate(ax,data,*,style=None,**options):
    for _,rows in data['table'].groupby('identity',sort=False):
        ax.plot(rows.time_separation_hours,rows.same_state_pair_fraction,color=colour('teal'),alpha=.2)
    rows=data['aggregate'];x=rows.time_separation_hours
    ax.fill_between(x,rows.matched_noise_lower_95,rows.matched_noise_upper_95,color=colour('raw'),alpha=.4,label='Matched noise: 95% interval')
    ax.plot(x,rows.same_state_pair_fraction,color=colour('teal'),label='Median across cells')
    ax.set(xlabel='Time between observations (h)',ylabel='Same-state pair fraction');ax.legend();_hours(ax,options)
    return Drawn(data['table'],ax)


def quantified(ax,data,*,style=None,**options):
    rows=data['table'];y=np.arange(len(rows))
    left=rows.matched_noise_repeated_sequence_fraction;right=rows.repeated_sequence_fraction
    ax.hlines(y,left,right,color=colour('raw'))
    ax.scatter(left,y,facecolors='none',edgecolors=colour('dark'),label='Matched noise')
    ax.scatter(right,y,color=colour('teal'),label='Observed cell')
    ax.set(yticks=y,yticklabels=rows.identity.astype(str),xlabel='Repeated-sequence fraction',ylabel='Cell identity');ax.legend()
    return Drawn(rows,ax)
