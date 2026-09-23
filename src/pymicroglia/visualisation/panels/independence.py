"""Draw prepared spatial phase evidence without fitting or binning."""
from matplotlib.collections import PolyCollection
from . import colour
from ._contract import Drawn
from ..labels import semantic_label


def distance(ax,data,*,style=None,**options):
    handle=PolyCollection(data['polygons'],array=data['counts'],cmap='viridis');ax.add_collection(handle)
    ax.set(xlim=data['limits'][0],ylim=data['limits'][1],xlabel=f"Cell separation ({data['unit']})",ylabel='Peak difference (fraction of common period)')
    ax.axhline(data['median'],color=colour('muted'),linestyle='--',label='Observed median');ax.legend()
    ax.figure.colorbar(handle,ax=ax,label='Cell pairs per hexagon')
    return Drawn(data['table'],ax)


def field(ax,data,*,style=None,**options):
    t=data['table'];handle=ax.scatter(t.centroid_x,t.centroid_y,c=t.best_phase_fraction,cmap=data['cmap'],vmin=0,vmax=1)
    for row in t.itertuples():ax.annotate(str(row.identity),(row.centroid_x,row.centroid_y),xytext=(3,3),textcoords='offset points')
    field=data['field'];ax.set(xlim=(0,field['width']),ylim=(field['height'],0),aspect='equal',xlabel=f"X position ({data['unit']})",ylabel=f"Y position ({data['unit']})")
    ax.figure.colorbar(handle,ax=ax,label=f"Peak within common {data['period']:g} h period",ticks=[0,.25,.5,.75,1])
    return Drawn(t,ax)


def measures(ax,data,*,style=None,**options):
    t=data['table']
    for i,(metric,rows) in enumerate(t.groupby('metric')):ax.scatter(rows.distance,rows.phase_difference_fraction,color=colour(['teal','orange','blue','plum'][i%4]),alpha=.5,label=semantic_label(metric))
    ax.set(xlabel=f"Cell separation ({data['unit']})",ylabel='Peak difference (fraction of common period)');ax.legend()
    return Drawn(t,ax)
