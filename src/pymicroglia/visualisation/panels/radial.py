"""Draw prepared cell-by-cell radial measurements in physical time."""
from ._contract import Drawn
from .exchange import _hours
from ..labels import semantic_label


def kymograph(figure,data,*,style,**options):
    grids=data['grids'];columns=min(4,len(grids));rows=(len(grids)+columns-1)//columns
    layout=figure.add_gridspec(rows,columns);axes=[]
    for index,grid in enumerate(grids):
        ax=figure.add_subplot(layout[index//columns,index%columns]);axes.append(ax)
        image=ax.pcolormesh(grid['hours'],grid['radii'],grid['values'],shading='nearest',vmin=0,vmax=data['maximum'],cmap='viridis')
        ax.set(title=f"Cell {grid['identity']}",xlabel='Hours from recording start',ylabel=data['ylabel'])
        _hours(ax,options)
    figure.colorbar(image,ax=axes,label=semantic_label(data['metric']),shrink=.7)
    return Drawn(data['table'],axes)
