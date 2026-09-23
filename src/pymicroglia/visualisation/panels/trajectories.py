"""Draw prepared path segments without bridging missing frames."""
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from . import colour
from ._contract import Drawn


def paths(ax,data,*,style=None,**options):
    lines=LineCollection(data['segments'],array=data['values'],cmap=data['cmap'],norm=Normalize(0,data['vmax']),linewidths=data['widths'],linestyles=data['dashes'])
    ax.add_collection(lines)
    starts=data['starts']
    ax.scatter([s['x'] for s in starts],[s['y'] for s in starts],c=[s['value'] for s in starts],cmap=data['cmap'],vmin=0,vmax=data['vmax'],s=[s['marker_area'] for s in starts])
    ax.set(xlim=(0,data['field']['width']),ylim=(data['field']['height'],0),aspect='equal',xlabel='X position in field (px)',ylabel='Y position in field (px)')
    ax.figure.colorbar(lines,ax=ax,label=data['colour_label'])
    handles=[];labels=[]
    if data['width_label']:
        for name,value,width in data['width_legend']:
            handles.append(Line2D([],[],color=colour('dark'),linewidth=width));labels.append(f"{data['width_label']}: {name}, {value:.3g}")
    if data['dash_label']:
        for name,pattern in data['dash_styles'].items():
            handles.append(Line2D([],[],color=colour('dark'),linestyle=pattern));labels.append(name)
    if handles:ax.legend(handles,labels,title=data['dash_label'] or None,loc='upper left',bbox_to_anchor=(1.25,1))
    return Drawn(data['table'],ax)
