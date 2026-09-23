"""Draw prepared observation tiles and independently selectable support status."""
import textwrap
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import _layout
from . import colour
STATES={'significant-supported':('S',colour('okabe_bluish_green'),'Significant; supported period'),
        'significant-unresolved':('U',colour('circadian_amber'),'Significant; unresolved period'),
        'not-significant':('NS',colour('actogram_dark'),'Valid test; not significant'),
        'untestable':('?',colour('muted'),'No valid test')}


def wrap(text,width):
    return '\n'.join(textwrap.fill(line,width,break_long_words=True,break_on_hyphens=False) for line in str(text).splitlines())


def draw(prepared,settings,*,selected_view=None,canvas=None):
    if selected_view not in {None,'matrix','status'}:raise ValueError('Unknown time-matrix view')
    rows=prepared['rows'];labels=[wrap(row['cell_label'],30) for row in rows]
    label_lines=max([1]+[label.count('\n')+1 for label in labels])
    height=max(5.8,3.2+max(.42,.1+.18*label_lines)*len(rows));width=8. if selected_view=='status' else 13.
    figure=_layout.figure(canvas,figsize=(width,height));axes={}
    bottom,plot_height=1.65,height-2.9
    if selected_view!='status':
        ax=figure.add_axes([2.55/width,bottom/height,8.95/width,plot_height/height]);axes['matrix']=ax
        cmap=plt.get_cmap('RdBu_r' if settings['scale'] in {'zscore','robust_zscore','robust_scale'} else 'viridis')
        norm=Normalize(*prepared['colour_bounds']) if prepared['colour_bounds'] is not None else None
        if len(prepared['polygons']) and norm is not None:
            ax.add_collection(PolyCollection(prepared['polygons'],array=prepared['colours'],cmap=cmap,norm=norm,edgecolors='none'))
        for row in rows:
            if row['missing_message']:ax.text(.015,row['panel_row'],row['missing_message'],transform=ax.get_yaxis_transform(),color=colour('slate_tick'),va='center',fontsize=8)
        ax.set_yticks([r['panel_row'] for r in rows],labels,fontsize=9);ax.tick_params(axis='y',length=0,pad=30 if selected_view is None else 8)
        if prepared['time_bounds'] is not None:ax.set_xlim(*prepared['time_bounds'])
        else:ax.set_xticks([])
        ax.set_xlabel('Recording time (h)' if prepared['time_bounds'] is not None else 'Recording time unavailable')
        if norm is not None:
            cax=figure.add_axes([11.85/width,bottom/height,.16/width,plot_height/height])
            bar=figure.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax)
            bar.set_label(wrap(settings['scale_label'],35),fontsize=9)
            if settings['colour_min']==settings['colour_max']:bar.set_ticks([settings['colour_min']],[f"{settings['colour_min']:.3g} (constant)"])
        else:figure.text(.88,(bottom+plot_height/2)/height,'No available\ndisplay values',fontsize=9)
    if selected_view!='matrix':
        state=figure.add_axes([.35 if selected_view=='status' else 2.23/width,bottom/height,.5 if selected_view=='status' else .2/width,plot_height/height]);axes['status']=state
        for row in rows:
            abbreviation,face,_=STATES[row['display_state']]
            state.barh(row['panel_row'],1,height=.8,color=face,edgecolor='white')
            state.text(.5,row['panel_row'],abbreviation,ha='center',va='center',fontsize=9 if selected_view=='status' else 7)
        state.set_xlim(0,1)
        if selected_view=='status':
            state.set_yticks([r['panel_row'] for r in rows],labels,fontsize=9);state.set_xticks([])
        else:state.set_axis_off()
        handles=[Patch(facecolor=face,label=f'{abbr}: {label}') for abbr,face,label in STATES.values()]
        figure.legend(handles=handles,loc='lower left',bbox_to_anchor=(.02,.76/height),fontsize=8,ncol=2,frameon=False)
    for axis in axes.values():
        axis.set_ylim(len(rows)-.5,-.5);axis.spines[['top','right','left']].set_visible(False)
    figure.text(.02,1-.15/height,wrap(settings['title'],98),fontsize=16,fontweight='bold',va='top')
    figure.text(.02,.1/height,wrap(settings['footnote'],160),va='bottom',fontsize=8)
    return figure,axes
