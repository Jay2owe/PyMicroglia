"""Draw prepared contact differences and the original permutation intervals."""
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import TwoSlopeNorm
from . import colour
from ._contract import Drawn


def pairs(ax,data,*,style=None,**options):
    matrix=data['matrix'];cmap=colormaps['RdBu_r'].with_extremes(bad=colour('raw'))
    handle=ax.imshow(np.ma.masked_invalid(matrix),aspect='auto',interpolation='nearest',cmap=cmap,norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=2))
    ax.set(xticks=np.arange(len(data['columns'])),xticklabels=data['columns'],yticks=np.arange(len(data['rows'])),yticklabels=data['rows'],xlabel='Cell measurement',ylabel='Cell pair | contact hours')
    ax.tick_params(axis='x',labelrotation=28)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value=matrix[row,column]
            if np.isfinite(value):ax.text(column,row,f'{value:.1f}',ha='center',va='center',color=colour('white' if value<=.32 or value>=1.64 else 'ink'))
    ax.figure.colorbar(handle,ax=ax,label='Pair difference / shuffled mean',ticks=[0,1,2])
    return Drawn(data['table'],ax)


def forest(ax,data,*,style=None,**options):
    table=data['table']
    for i,(_,row) in enumerate(table.iterrows()):
        hue=colour('plum' if row[data['significant']] else 'muted')
        estimate=row[data['estimate']];low=row[data['low']];high=row[data['high']]
        if np.isfinite(estimate):
            ax.plot([low,high],[i,i],color=hue);ax.scatter([estimate],[i],color=hue)
        else:ax.text(.02,i,'Too few eligible pairs',transform=ax.get_yaxis_transform(),va='center')
    ax.axvline(data['baseline'],color=colour('muted'),linestyle='--')
    ax.set(yticks=np.arange(len(table)),yticklabels=data['labels']);ax.invert_yaxis()
    if data['baseline']==1:ax.set(xlim=(0,None),xlabel='Mean pair difference / shuffled mean')
    else:ax.set(xlim=(-1,1),xlabel='Contact hours versus pair difference: Spearman correlation')
    return Drawn(table,ax)
