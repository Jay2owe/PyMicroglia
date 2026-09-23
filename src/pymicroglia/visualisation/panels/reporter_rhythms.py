"""Draw cell traces and period distributions from prepared rhythm evidence."""
import numpy as np
from . import colour
from ._contract import Drawn
from .exchange import _hours


def raster(figure,data,*,style=None,**options):
    layout=figure.add_gridspec(1,2,width_ratios=[40,1],wspace=.04)
    ax=figure.add_subplot(layout[0,0]);strip=figure.add_subplot(layout[0,1])
    image=ax.pcolormesh(data['hours'],np.arange(len(data['order'])),data['values'],
        shading='nearest',cmap=options.get('trace_luts') or 'RdBu_r',vmin=-2.5,vmax=2.5)
    ax.invert_yaxis();ax.set(xlabel='Hours from recording start',ylabel='Cell display row');_hours(ax,options)
    cursor=0
    for count,hue,label in zip(data['blocks'],['teal','blank','raw'],['Significant','Not significant','Not tested']):
        if count:
            strip.barh(cursor+(count-1)/2,1,height=count,color=colour(hue))
            strip.text(1.1,cursor+(count-1)/2,label,rotation=90,va='center',fontsize=9)
        cursor+=count
    strip.set(ylim=(len(data['order'])-.5,-.5));strip.set_axis_off()
    figure.colorbar(image,ax=ax,label=f"Detrended {data['label']} (SD)",shrink=.6,pad=.04)
    return Drawn(data['table'],[ax,strip])


def peaks(ax,data,*,style=None,**options):
    image=ax.pcolormesh(data['phase_edges']*100,data['period_edges'],data['counts'],shading='flat',cmap='viridis',vmin=0)
    ax.set(xlabel="Peak position within each cell's own cycle (%)",ylabel='Detected period (h)')
    ax.figure.colorbar(image,ax=ax,label='Significant cells')
    return Drawn(data['table'],ax)


def histogram(ax,data,*,style=None,**options):
    rows=data['table']
    ax.bar(rows.period_start_hour,rows.significant_cell_count,width=rows.period_end_hour-rows.period_start_hour,align='edge',color=colour('teal'))
    ax.set(xlabel='Estimated period (h)',ylabel='Significant cells')
    return Drawn(rows,ax)
