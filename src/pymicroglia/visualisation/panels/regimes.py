"""Draw prepared categorical states, occupancy and transition evidence."""
import numpy as np
from . import colour
from ._contract import Drawn
from .exchange import _hours

PALETTE=('blue','teal','circadian_green','orange','red','circadian_purple','plum','muted','dark')


def state_colour(state):
    return colour(PALETTE[int(state)%len(PALETTE)])


def ribbon(ax,data,*,style,**options):
    from matplotlib.colors import to_rgba
    values=data['matrix'];margins=data['margins']
    rgba=np.zeros((*values.shape,4))
    for state in np.unique(values[np.isfinite(values)]).astype(int):
        selected=values==state
        rgba[selected,:3]=to_rgba(state_colour(state))[:3]
        rgba[selected,3]=.18+.82*np.clip(margins[selected],0,1)
    rgba[~np.isfinite(values)]=to_rgba(colour('raw'))
    hours=data['hours']
    ax.imshow(rgba,aspect='auto',interpolation='nearest',extent=(hours[0],hours[-1],values.shape[0]-.5,-.5))
    ax.set(xlabel='Hours from start of recording',ylabel='Cell, in declared row order')
    _hours(ax,options)
    return Drawn(data['table'],ax)


def occupancy(ax,data,*,style,**options):
    ax.stackplot(data['hours'],data['values'].T,
                 labels=[data['names'].get(state,f'State {state}') for state in data['regimes']],
                 colors=[state_colour(state) for state in data['regimes']])
    ax.set(xlabel='Hours from start of recording',ylabel='Population share',ylim=(0,1))
    ax.legend(frameon=False,fontsize=10,ncol=2)
    _hours(ax,options)
    return Drawn(data['table'],ax)


def matrix(ax,data,*,style,**options):
    values=np.array(data['matrix'],copy=True)
    if not values.size:
        ax.text(.5,.5,'No recorded transitions',ha='center',transform=ax.transAxes)
        ax.set_axis_off()
        return Drawn(data['table'],ax)
    if data.get('mask_diagonal',False):np.fill_diagonal(values,np.nan)
    handle=ax.imshow(np.ma.masked_invalid(values),aspect='auto',interpolation='nearest',vmin=data.get('vmin'),vmax=data.get('vmax'),cmap='Blues' if 'Transitions' in data['label'] or 'transitions' in data['label'] else 'RdBu_r')
    ax.set(yticks=range(len(data['rows'])),yticklabels=data['rows'],xticks=range(len(data['columns'])),xticklabels=data['columns'])
    ax.tick_params(axis='both',labelsize=11)
    if len(data['columns'])>10:
        for label in ax.get_xticklabels():label.set_rotation(75);label.set_ha('right')
    ax.figure.colorbar(handle,ax=ax,label=data['label'],shrink=.7)
    return Drawn(data['table'],ax)


def dwell(ax,data,*,style,**options):
    for state,group in data['table'].groupby('regime',sort=True):
        ax.step((group.bin_left_frames+group.bin_right_frames)/2,group.runs,where='mid',
                color=state_colour(state),label=data['names'].get(state,f'State {state}'))
    ax.set(xlabel='Dwell (frames; first bin is at least one frame)',ylabel='Runs')
    scale=data['interval']/60
    ax.secondary_xaxis('top',functions=(lambda value:value*scale,lambda value:value/scale)).set_xlabel('Dwell (hours)')
    ax.legend(frameon=False,fontsize=10,ncol=2)
    return Drawn(data['table'],ax)


def tree(ax,data,*,style,**options):
    for _,rows in data['table'].groupby('branch',sort=False):
        ax.plot(rows.distance,rows.lane,color=colour('teal'))
    ax.set(xlabel='Different aligned states (fraction)',ylabel='Cells',xlim=(0,None))
    return Drawn(data['table'],ax)
