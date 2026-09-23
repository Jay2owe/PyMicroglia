"""Draw recorded comparisons, preserving separate effect scales and refusals."""
import textwrap
import numpy as np
from . import colour
from ._contract import Drawn


def _empty(ax,data):
    ax.text(.5,.5,data['message'],ha='center',va='center',transform=ax.transAxes)
    ax.set_axis_off()
    return Drawn(data['table'],ax)


def effects(figure,data,*,style,**options):
    if 'message' in data:return _empty(figure.add_subplot(111),data)
    groups=list(data['rows'].groupby('scale',sort=False))
    layout=figure.add_gridspec(len(groups),1,height_ratios=[max(2,len(rows)) for _,rows in groups])
    axes=[]
    for index,(scale,rows) in enumerate(groups):
        ax=figure.add_subplot(layout[index]);axes.append(ax)
        for y,row in enumerate(rows.itertuples(index=False)):
            if row.is_refusal:
                ax.text(.02,y,textwrap.fill(row.refusal,65),transform=ax.get_yaxis_transform(),va='center',fontsize=10)
                continue
            color=colour('teal' if row.significant else 'muted')
            if np.isfinite(row.effect_lo) and np.isfinite(row.effect_hi):
                ax.plot([row.effect_lo,row.effect_hi],[y,y],color=color)
            if np.isfinite(row.effect):ax.scatter(row.effect,y,color=color,s=45)
        ax.axvline(0,color=colour('muted'),linestyle=':',alpha=.6)
        ax.set(yticks=range(len(rows)),yticklabels=[textwrap.fill(v,38) for v in rows.label],xlabel=scale,ylim=(len(rows)-.5,-.5))
        ax.tick_params(labelsize=11)
    return Drawn(data['table'],axes)


def correction(ax,data,*,style,**options):
    if 'message' in data:return _empty(ax,data)
    rows=data['table']
    ax.plot([0,1],[0,1],color=colour('muted'),linestyle=':')
    for row in rows.itertuples(index=False):
        ax.scatter(row.p_value,row.p_corrected,color=colour('teal' if row.significant else 'muted'),s=45)
    ax.set(xlabel='Recorded p-value',ylabel='Recorded corrected p-value',xlim=(0,1),ylim=(0,1))
    return Drawn(rows,ax)


def design(ax,data,*,style,**options):
    if 'message' in data:return _empty(ax,data)
    table=data['table'];y=np.arange(len(table))
    ax.hlines(y,0,table.units,color=colour('muted'))
    ax.scatter(table.units,y,color=colour('teal'))
    ax.set(yticks=y,yticklabels=[textwrap.fill(v,38) for v in table.label],xlabel='Units in the smaller comparison group ('+data['unit']+')')
    ax.tick_params(labelsize=11)
    return Drawn(table,ax)
