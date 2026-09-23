"""Draw saved window changes and prepared distributions."""
import numpy as np
from . import colour
from ._contract import Drawn
from ..labels import semantic_label


def ledger(ax,data,*,style,**options):
    table=data['ranked'];y=np.arange(len(table))
    ax.hlines(y,0,table.median_scaled_change,color=colour('muted'))
    ax.scatter(table.median_scaled_change,y,color=colour('teal'))
    ax.set(yticks=y,yticklabels=[semantic_label(v) for v in table.metric],xlabel='Median scaled change across cells')
    ax.tick_params(labelsize=10);ax.axvline(0,color=colour('muted'),linestyle=':')
    return Drawn(data['table'],ax)


def paired(ax,data,*,style,**options):
    for row in data['table'].itertuples(index=False):
        ax.plot([0,1],[row.baseline_value,row.value],color=colour('teal' if row.change>0 else 'orange'),alpha=.5,marker='o')
    ax.set(xticks=[0,1],xticklabels=[data['baseline'].replace('_',' '),data['comparison'].replace('_',' ')],ylabel=data['label'])
    return Drawn(data['table'],ax)


def spread(ax,data,*,style,**options):
    names=[]
    for index,(name,rows) in enumerate(data['table'].groupby('metric',sort=False)):
        names.append(semantic_label(name))
        x=(rows.bin_left+rows.bin_right)/2
        ax.fill_between(x,index,index+rows.height*.8,color=colour('teal'),alpha=.55)
    ax.set(yticks=range(len(names)),yticklabels=names,xlabel='Scaled change for each cell')
    ax.tick_params(labelsize=10);ax.axvline(0,color=colour('muted'),linestyle=':')
    return Drawn(data['table'],ax)
