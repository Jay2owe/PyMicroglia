"""Draw prepared territory maps, curves and distribution bins."""
import math
from . import colour
from ._contract import Drawn
from .exchange import _hours


def revisit(figure,data,*,style=None,**options):
    crops=data['crops'];columns=min(8,max(1,len(crops)));rows=math.ceil(len(crops)/columns)
    grid=figure.add_gridspec(rows,columns);axes=[]
    for index,crop in enumerate(crops):
        ax=figure.add_subplot(grid[index//columns,index%columns]);axes.append(ax)
        handle=ax.imshow(crop['image'],cmap='viridis',vmin=0,vmax=1,extent=(0,data['side'],data['side'],0),interpolation='nearest')
        ax.contour(crop['image'],levels=[data['contour']],colors=[colour('dark')],origin='upper',extent=(0,data['side'],data['side'],0))
        if crop['soma'] is not None:ax.scatter(*crop['soma'],marker='+',color=colour('orange'))
        ax.set(title=f"Cell {crop['identity']}",xticks=[],yticks=[])
    if axes:figure.colorbar(handle,ax=axes,label='Occupancy fraction of observed frames')
    return Drawn(data['table'],axes)


def coverage(ax,data,*,style=None,**options):
    rows=data['summary']
    ax.fill_between(rows.hours,rows.permutation_lo,rows.permutation_hi,color=colour('muted'),alpha=.22,label='95% frame-order permutation interval')
    ax.plot(rows.hours,rows.permutation_median,color=colour('muted'),label='Permutation median')
    for curve in data['curves']:ax.plot(data['hours'],curve,color=colour('teal'),alpha=.18)
    ax.plot(rows.hours,rows.observed_median,color=colour('teal'),label='Median observed cell')
    ax.text(.02,.97,f"Two-sided permutation p = {data['statistics']['p_value']:.3g}",transform=ax.transAxes,va='top')
    ax.set(xlabel='Hours from start of recording',ylabel='Share of final footprint touched',ylim=(0,1.03))
    ax.legend()
    _hours(ax,options)
    return Drawn(data['table'],ax)


def core(ax,data,*,style=None,**options):
    t=data['table'];ax.bar(t.bin_left,t['count'],width=t.bin_right-t.bin_left,align='edge',color=colour('circadian_green'))
    ax.set(xlabel=data['label'],ylabel='Cells')
    return Drawn(t,ax)
