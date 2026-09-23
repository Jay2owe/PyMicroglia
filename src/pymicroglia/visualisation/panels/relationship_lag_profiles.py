"""Portable physical lag grids with separately labelled saved uncertainty."""
from . import _layout
from ._layout import finish
from . import colour as house_colour
import math
from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
import numpy as np
from ._format import numeric, missing, present, number as display_number

def wrap(value, width=85):
    return '\n'.join((textwrap.fill(line, width) for line in str(value).splitlines()))

def number(value):
    try:
        return f'{float(value):.3g}' if math.isfinite(float(value)) else 'unavailable'
    except (ValueError, TypeError):
        return 'unavailable'

def draw(prepared,settings,*,selected_view=None,canvas=None):
    if selected_view not in {None,'coefficients','support'}:raise ValueError('Unknown lag view')
    names=[selected_view] if selected_view else ['coefficients','support']
    rows=len(prepared);height=1.7+5.2*rows
    figure=_layout.figure(canvas,figsize=(13,height))
    layout=figure.add_gridspec(rows*3,len(names),height_ratios=[3,.45,1.5]*rows,width_ratios=[1.3,1] if len(names)==2 else [1])
    figure.subplots_adjust(left=.08,right=.9,top=1-.7/height,bottom=.65/height,hspace=.45,wspace=.5)
    axes={}
    for i,row in enumerate(prepared):
        member=row['member'];panels={name:figure.add_subplot(layout[3*i,j]) for j,name in enumerate(names)}
        legend=figure.add_subplot(layout[3*i+1,:]);legend.axis('off')
        caption=figure.add_subplot(layout[3*i+2,:]);caption.axis('off')
        if 'coefficients' in panels:
            axis=panels['coefficients'];axes[f'profile_{i}']=axis
            title=f"{member['movie']} / cell {member['identity']} | {member['reference']} -> {member['target']}"
            axis.set_title(wrap(title,65),fontsize=10);axis.set_ylim(-1.2,1.08)
            axis.axhline(0,color=house_colour('raw'),lw=.7);axis.set_ylabel('Saved association coefficient',fontsize=9)
            if row['available']:
                axis.plot(row['x'],row['effect'],color=house_colour('actogram_day_separator'),marker='.',label='Per-lag overlap')
                if row['native'] is not None:axis.plot(row['x'],row['native'],color=house_colour('circadian_teal'),marker='.',label='Fixed test window')
                if row['bounded'] is not None:axis.fill_between(row['x'],row['lower'],row['upper'],where=row['bounded'],color=house_colour('circadian_teal'),alpha=.18,label='Within-curve coefficient confidence band')
                if row['candidates']:axis.scatter(row['candidates'],[-1.13]*len(row['candidates']),marker='|',s=80,color=house_colour('grade_green'),label='Compatible delay grid values')
                for x,y in row['peaks']:axis.scatter([x],[y],marker='*',color=house_colour('circadian_red'),s=65,zorder=5)
                legend.legend(*axis.get_legend_handles_labels(),loc='upper left',ncol=2,fontsize=7)
            else:axis.text(.5,.5,wrap(member['reason'],45),transform=axis.transAxes,ha='center',va='center',fontsize=10)
        if 'support' in panels:
            support=panels['support'];axes[f'support_{i}']=support
            support.set_title('Original per-lag observation support',fontsize=10);support.set_ylabel('Paired observations',fontsize=9)
            if row['available']:
                x=row['x'];support.plot(x,row['pairs'],color=house_colour('circadian_teal'),marker='.',label='Observation pairs')
                span=support.twinx();span.tick_params(labelsize=8);axes[f'span_{i}']=span
                span.plot(x,row['span'],color=house_colour('circadian_red'),ls='--',marker='.',label='Overlap duration')
                span.set_ylabel('Overlap duration (hours)',fontsize=9)
                if row['failed'].any():support.scatter(x[row['failed']],row['pairs'][row['failed']],marker='x',color=house_colour('circadian_red'),s=45,label='Insufficient support')
                support.legend(loc='upper left',fontsize=7);span.legend(loc='upper right',fontsize=7)
            else:support.text(.5,.5,'No lag observation grid',transform=support.transAxes,ha='center',fontsize=10)
        for axis in panels.values():
            axis.tick_params(labelsize=8);axis.set_xlabel('Lag (hours)',fontsize=9)
            if row['available']:
                axis.set_xlim(row['display'])
                for boundary in row['declared']:axis.axvline(boundary,color=house_colour('grade_grey'),ls=':',lw=.8)
        caption.text(0,1,wrap(row['caption'],145),fontsize=8,va='top')
    figure.text(.02,1-.1/height,wrap(settings['title']+f" | page {settings['page_number']}",115),fontsize=14,fontweight='bold',va='top')
    figure.text(.02,.1/height,wrap(settings['footnote'],145),fontsize=8,va='bottom')
    return figure,axes
