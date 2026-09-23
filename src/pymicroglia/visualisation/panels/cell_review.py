"""Display per-cell distribution and equal-area reference plots."""
from . import colour
from ._contract import Drawn


def density(ax,data,*,style,**options):
    table=data['table'];row=table.iloc[0]
    handle=ax.hexbin(table.x_value,table.y_value,gridsize=options['bins'],cmap=options['density_lut'],xscale=row.x_scale,yscale=row.y_scale,mincnt=1)
    ax.figure.colorbar(handle,ax=ax,label='Cells per hexagon')
    if not data['means'].empty:ax.plot(data['means'].x,data['means']['mean'],color=colour('dark'))
    ax.axvline(data['median_x'],color=colour('muted'),linestyle=':');ax.axhline(data['median_y'],color=colour('muted'),linestyle=':')
    ax.set(xlabel=data['xlabel'],ylabel=data['ylabel'])
    return Drawn(table,ax)


def anchoring(ax,data,*,style,**options):
    table=data['table']
    ax.scatter(table.median_footprint_area,table.centroid_path_area,color=colour('teal'))
    ax.plot([0,data['maximum']],[0,data['maximum']],color=colour('muted'),linestyle=':',label='Equal areas')
    ax.set_xscale('symlog',linthresh=1);ax.set_yscale('symlog',linthresh=1)
    ax.set(xlabel=f"Typical footprint area ({data['unit']})",ylabel=f"Cell-centre path area ({data['unit']})",xlim=(0,data['maximum']),ylim=(0,data['maximum']))
    ax.legend(frameon=False)
    return Drawn(table,ax)
