"""Draw saved daily-profile values without fitting or folding traces."""
import numpy as np
from . import colour
from ._contract import Drawn
from ..labels import semantic_label


def strength(figure,data,*,style=None,**options):
    grid=figure.add_gridspec(len(data['metrics']),1,hspace=.15);axes=[]
    for i,metric in enumerate(data['metrics']):
        ax=figure.add_subplot(grid[i]);axes.append(ax);rows=data['bins'].loc[data['bins'].metric.eq(metric)]
        ax.fill_between(rows.centre,0,rows.density,color=colour('teal'),alpha=.65)
        ax.set(ylabel=semantic_label(metric),yticks=[])
        if i==len(data['metrics'])-1:ax.set_xlabel('Daily relative amplitude (busiest 10 h versus quietest 5 h)')
    return Drawn(data['table'],axes)


def stability(ax,data,*,style=None,**options):
    table=data['table']
    for i,metric in enumerate(data['metrics']):
        rows=table.loc[table.metric.eq(metric)];hue=colour(['teal','orange','blue','plum'][i%4])
        alpha=data['alpha'].loc[rows.index].to_numpy()
        ax.scatter(rows.interdaily_stability,rows.intradaily_variability,color=hue,alpha=alpha,label=semantic_label(metric))
    ax.set(xlabel='Interdaily stability',ylabel='Intradaily variability');ax.legend()
    return Drawn(table,ax)


def ranking(ax,data,*,style=None,**options):
    t=data['table'];y=np.arange(len(t));ax.hlines(y,0,t['median'],color=colour('teal'));ax.scatter(t['median'],y,color=colour('teal'))
    ax.set(yticks=y,yticklabels=[semantic_label(m) for m in t.metric],xlabel='Median daily relative amplitude');ax.invert_yaxis()
    for i,row in enumerate(t.itertuples()):ax.annotate(f'mean {row.mean:.3f}, {row.cells} cells',(row.median,i),xytext=(7,0),textcoords='offset points')
    return Drawn(t,ax)


def spans(ax,data,*,style=None,**options):
    for segment in data['segments']:ax.hlines(segment['row'],segment['start'],segment['end'],color=colour('teal'))
    ax.set(xlim=(0,24),xlabel='Hour in explicitly requested daily profile',ylabel='Cell and signal row');ax.invert_yaxis()
    return Drawn(data['table'],ax)


def agreement(ax,data,*,style=None,**options):
    t=data['table'];ax.bar(t.bin_left,t['count'],width=t.bin_right-t.bin_left,align='edge',color=colour('dark'))
    ax.set(xlabel='Within-cell daily onset spread (h)',ylabel='Cells')
    return Drawn(t,ax)


def rest(ax,data,*,style=None,**options):
    t=data['table'];y=np.arange(len(t));ax.hlines(y,t.left,t.right,color=colour('muted'))
    ax.scatter(t.left,y,facecolors='none',edgecolors=colour('dark'),label='Quietest 5 h starts');ax.scatter(t.right,y,color=colour('orange'),marker='D',label='Busiest 10 h starts')
    ax.set(yticks=y,yticklabels=[semantic_label(m) for m in t.metric],xlabel='Hour in explicitly requested daily profile');ax.legend()
    return Drawn(t,ax)
