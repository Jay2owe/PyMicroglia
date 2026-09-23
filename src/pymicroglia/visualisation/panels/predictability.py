"""Draw the saved mismatch summary and its original permutation interval."""
import numpy as np
from . import colour
from ._contract import Drawn
from .exchange import _hours


def error(ax,data,*,style=None,**options):
    t=data['table'];ax.fill_between(t.cycle_time_hours,t.surrogate_lower_95,t.surrogate_upper_95,color=colour('raw'),alpha=.4,label='Cycle-label permutation 95% interval')
    ax.plot(t.cycle_time_hours,t.surrogate_mean,color=colour('muted'))
    ax.fill_between(t.cycle_time_hours,t.lower_quartile,t.upper_quartile,color=colour('teal'),alpha=.18,label='Cell interquartile range')
    ax.plot(t.cycle_time_hours,t.mismatch_fraction,color=colour('teal'),label='Mean across cells')
    ax.set(xlabel=f"Time within declared {data['period']:g} h cycle",ylabel='Next-state mismatch fraction');ax.legend()
    _hours(ax,options)
    return Drawn(t,ax)


def by_regime(figure,data,*,style=None,**options):
    groups=list(data['table'].groupby('regime',sort=False));grid=figure.add_gridspec(max(1,len(groups)),1);axes=[]
    for i,(_,rows) in enumerate(groups):
        ax=figure.add_subplot(grid[i]);axes.append(ax);ax.fill_between(rows.centre,0,rows.height,color=colour('teal'),alpha=.65)
        ax.set(xlim=(0,data['period']),yticks=[])
        ax.set_title(rows.label.iloc[0],loc='left',fontsize=10)
        _hours(ax,options)
        if i==len(groups)-1:ax.set_xlabel('Hours within the declared cycle')
    return Drawn(data['table'],axes)


def dial(ax,data,*,style=None,**options):
    t=data['table'];factor=2*np.pi/data['period']
    ax.bar((t.bin_left+t.bin_right)/2*factor,t['count'],width=(t.bin_right-t.bin_left)*factor,color=colour('teal'))
    ax.set_theta_zero_location('N');ax.set_theta_direction(-1)
    step=options.get('hour_ticks') or data['period']/4
    ticks=np.arange(0.,data['period'],step)
    ax.set_xticks(ticks*factor,[f'{value:g} h' for value in ticks])
    vector=data['vector']
    if vector['direction_radians'] is not None:
        ax.annotate('',xy=(vector['direction_radians'],vector['radius']),xytext=(0,0),arrowprops=dict(arrowstyle='-|>',color=colour('dark'),linewidth=1.5))
    ax.set_title('Mismatched transitions',pad=30,fontsize=12)
    return Drawn(t,ax)
