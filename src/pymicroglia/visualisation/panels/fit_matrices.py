"""Draw prepared period estimates, test outcomes and observation sufficiency."""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle, Patch
from . import colour
from ._contract import Drawn
from .exchange import _hours


def status(ax,data,*,cmap=None,rotation=0):
    periods,states=data['periods'],data['statuses']
    normal=Normalize(data['low'],data['high'])
    palette=plt.get_cmap(cmap or 'viridis')
    for y,x in np.ndindex(states.shape):
        state=states[y,x]
        fill=palette(normal(periods[y,x])) if state=='rhythmic' and np.isfinite(periods[y,x]) else colour('blank' if state=='not rhythmic' else 'raw')
        uncertain=bool(data['uncertain'][y,x]) and state=='rhythmic'
        ax.add_patch(Rectangle((x-.5,y-.5),1,1,facecolor=fill,
            edgecolor=colour('dark') if uncertain else 'white',linewidth=1.2 if uncertain else .4,
            hatch='///' if state=='period unavailable' else None))
    ax.set(xlim=(-.5,states.shape[1]-.5),ylim=(states.shape[0]-.5,-.5),ylabel='Cell identity')
    ax.set_xticks(range(len(data['labels'])),data['labels'],rotation=rotation)
    ax.set_yticks(range(len(data['order'])),list(map(str,data['order'])))
    ax.tick_params(length=0)
    return plt.cm.ScalarMappable(norm=normal,cmap=palette)


def legend(figure):
    figure.legend(handles=[Patch(facecolor=colour('blank'),label='Tested; not significant'),
        Patch(facecolor=colour('raw'),label='Not tested'),
        Patch(facecolor='white',edgecolor=colour('dark'),label='Fewer than required cycles'),
        Patch(facecolor='white',hatch='///',label='Significant; period unavailable')],
        loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.03))


def matrix(figure,data,*,style=None,**options):
    ax=figure.add_subplot(111)
    handle=status(ax,data,cmap=options.get('matrix_lut'),rotation=options.get('column_label_rotation',0))
    figure.colorbar(handle,ax=ax,label='Estimated period (h)',shrink=.7,pad=.05)
    legend(figure)
    return Drawn(data['table'],[ax])


def radial(figure,data,*,style=None,**options):
    layout=figure.add_gridspec(1,2,width_ratios=[8,1],wspace=.4)
    ax=figure.add_subplot(layout[0,0]);evidence=figure.add_subplot(layout[0,1])
    raw=data['display']=='raw'
    image=ax.pcolormesh(data['hours'],np.arange(len(data['order'])),data['values'],shading='nearest',
        cmap=options.get('matrix_lut') or ('viridis' if raw else 'RdBu_r'),
        vmin=0 if raw else -data['limit'],vmax=1 if raw else data['limit'])
    ax.set(xlabel='Hours from recording start',ylabel='Cell identity')
    ax.set_yticks(range(len(data['order'])),list(map(str,data['order'])))
    ax.invert_yaxis();_hours(ax,options)
    figure.colorbar(image,ax=ax,label='Occupied fraction' if raw else 'Detrended occupancy (SD)',shrink=.7,pad=.02)
    handle=status(evidence,data);evidence.set_yticks([]);evidence.set_ylabel('')
    figure.colorbar(handle,ax=evidence,label='Estimated period (h)',shrink=.7,pad=.2)
    legend(figure)
    return Drawn(data['table'],[ax,evidence])
