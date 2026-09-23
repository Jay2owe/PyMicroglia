"""Independent views of prepared screening periods, status and denominators."""
import textwrap
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle
import numpy as np
from . import _layout
from . import colour


def wrap(value,width=80):
    return '\n'.join(textwrap.fill(line,width,break_long_words=False,break_on_hyphens=False) for line in str(value).splitlines())


def matrix_view(ax,prepared,view):
    colours={'not-significant':colour('blank'),'untestable':colour('raw'),
             'significant-unresolved':colour('circadian_amber'),'significant-supported':colour('circadian_teal')}
    labels={'not-significant':'NS','untestable':'?','significant-unresolved':'U','significant-supported':'S'}
    norm=Normalize(*prepared['bounds']);cmap=plt.get_cmap('viridis')
    if not prepared['marks']:
        ax.text(.5,.5,'No requested cells in the saved screen',transform=ax.transAxes,ha='center')
        ax.set_axis_off();return
    for row in prepared['marks']:
        state=row['display_state'];supported=state=='significant-supported';value=row['plotted_period_hours']
        fill=cmap(norm(value)) if view=='periods' and supported else colours[state]
        ax.add_patch(Rectangle((row['x']-.5,row['y']-.5),1,1,facecolor=fill,edgecolor='white',linewidth=.7,
                               hatch='///' if state=='significant-unresolved' else None))
        label=f'{value:.2g}' if view=='periods' and supported else labels[state]
        ink='white' if view=='periods' and supported and norm(value)<.55 else colour('dark')
        ax.text(row['x'],row['y'],label,color=ink,ha='center',va='center',fontsize=9)
    ax.set_xlim(-.5,len(prepared['columns'])-.5);ax.set_ylim(len(prepared['rows'])-.5,-.5)
    ax.set_xticks(range(len(prepared['columns'])),[wrap(c['measurement_label'].replace('_',' '),17) for c in prepared['columns']])
    ax.set_yticks(range(len(prepared['rows'])),[wrap(r['cell_label'].replace('_',' '),25) for r in prepared['rows']],fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_title('Supported estimated periods' if view=='periods' else 'Test and period-support status')
    if view=='periods':
        bar=ax.figure.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,pad=.03,fraction=.045)
        bar.set_label('Supported estimated period (h)',fontsize=9)
    else:
        handles=[Patch(facecolor=colours[s],label=label,hatch='///' if s=='significant-unresolved' else None)
                 for s,label in [('significant-supported','S: significant, supported'),('significant-unresolved','U: significant, unresolved'),
                                 ('not-significant','NS: valid, not significant'),('untestable','?: untestable')]]
        ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(0,-.12),fontsize=8,frameon=False)


def summary_view(ax,row,view):
    if view=='fraction':
        fraction=row['significant_fraction']
        if fraction is not None and np.isfinite(fraction):
            ax.scatter([fraction],[.5],s=65,color=colour('circadian_teal'))
            title=f"{int(row['significant'])}/{int(row['tested'])} valid tests ({fraction:.0%})"
        else:title='Unavailable: 0 valid tests'
        ax.set_xlim(-.03,1.03);ax.set_ylim(0,1);ax.set_yticks([])
        ax.set_xticks([0,.5,1],['0%','50%','100%']);ax.set_xlabel('Significant / valid tests')
    else:
        histogram=row['histogram']
        if row['period_count']:
            ax.bar(histogram['left'],histogram['counts'],width=histogram['width'],align='edge',color=colour('blue'),edgecolor='white')
        else:ax.text(.5,.5,'No supported\ndetected periods',ha='center',transform=ax.transAxes)
        ax.set_xlim(*row['bounds']);ax.set_xlabel('Estimated period (h)');ax.set_ylabel('Cells')
        ax.yaxis.get_major_locator().set_params(integer=True);title='Supported detected periods'
    ax.set_title(wrap(row['measurement_label'].replace('_',' '),35)+'\n'+title,fontsize=10,loc='left')
    ax.spines[['top','right']].set_visible(False)


def draw(prepared, *, kind, title, footnote, selected_view=None, canvas=None):
    views=(selected_view,) if selected_view else (('periods','status') if kind=='matrix' else ('fraction','periods'))
    allowed={'periods','status'} if kind=='matrix' else {'fraction','periods'}
    if not set(views)<=allowed:raise ValueError('Unknown saved screening view')
    axes={}
    if kind=='matrix':
        width=max(9.,3.5+1.6*len(prepared['columns']))*len(views)
        height=max(6.,3.7+.52*len(prepared['rows']))
        figure=_layout.figure(canvas,figsize=(width,height))
        grid=figure.add_gridspec(1,len(views),left=.12,right=.95,bottom=2.0/height,top=1-1.1/height,wspace=.45)
        for i,view in enumerate(views):
            ax=figure.add_subplot(grid[0,i]);matrix_view(ax,prepared,view);axes[view]=ax
    else:
        width=14. if len(views)>1 else 10.
        captions=[]
        for row in prepared:
            note=f"{int(row['significant_supported'])} supported detected periods; {int(row['significant_unresolved'])} unresolved; {int(row['untestable'])} untestable"
            for key,label in [('exclusions','Unavailable tests'),('unresolved','Unresolved estimates')]:
                if row[key]:note+='\n'+label+': '+'; '.join(f'{k or "reason unavailable"} ({v})' for k,v in row[key].items())
            captions.append(wrap(note,42))
        sizes=[max(2.6,.2*(note.count('\n')+1)+.95) for note in captions]
        height=2.2+sum(sizes);figure=_layout.figure(canvas,figsize=(width,height));top=height-1.0
        for row,note,size in zip(prepared,captions,sizes):
            for i,view in enumerate(views):
                left=.1+i*.28 if len(views)>1 else .12
                ax=figure.add_axes([left,(top-size+.68)/height,.22 if len(views)>1 else .36,(size-1.45)/height])
                summary_view(ax,row,view);axes[row['measurement']+':'+view]=ax
            figure.text(.7 if len(views)>1 else .58,(top-.08)/height,note,fontsize=9,va='top');top-=size
    figure.text(.02,1-.15/height,wrap(title,int(width*7)),fontsize=16,fontweight='bold',va='top')
    figure.text(.02,.12/height,wrap(footnote,int(width*12)),va='bottom',fontsize=8)
    return figure,axes
