"""Display prepared channel traces and their recorded model amplitudes."""
import numpy as np
from . import colour
from ._contract import Drawn
from .exchange import _hours
from ..labels import semantic_label


def channels(ax,data,*,style,**options):
    palette=('teal','orange','blue','circadian_purple','red','dark')
    for index,(name,rows) in enumerate(data['table'].groupby('channel',sort=False)):
        ax.plot(rows.hours,rows.value,color=colour(palette[index%len(palette)]),label=semantic_label(name))
    ax.set(xlabel='Hours from recording start',ylabel='Within-signal standard deviations')
    ax.legend(frameon=False,fontsize=10);_hours(ax,options)
    return Drawn(data['table'],ax)


def amplitudes(ax,data,*,style,**options):
    table=data['table'];y=np.arange(len(table))
    inks=[colour('orange' if kind=='tracker' else 'teal') for kind in table.kind]
    ax.hlines(y,table.surrogate_amplitude,table.relative_fit_amplitude,color=inks)
    ax.scatter(table.surrogate_amplitude,y,facecolors='none',edgecolors=inks,label='Matched noise')
    ax.scatter(table.relative_fit_amplitude,y,color=inks,label='Observed signal')
    ax.set(yticks=y,yticklabels=[semantic_label(v) for v in table.channel],xlabel='Model amplitude / mean absolute signal')
    ax.legend(frameon=False,fontsize=11)
    return Drawn(table,ax)


def against_floor(ax,data,*,style,**options):
    table=data['table'];y=np.arange(len(table))
    ax.hlines(y,table.surrogate_amplitude,table.relative_fit_amplitude,color=colour('muted'))
    ax.scatter(table.surrogate_amplitude,y,facecolors='none',edgecolors=colour('muted'),label='Matched noise')
    ax.scatter(table.relative_fit_amplitude,y,color=[colour('teal' if a>b else 'muted') for a,b in zip(table.relative_fit_amplitude,table.surrogate_amplitude)],label='Observed signal')
    ax.set(yticks=y,yticklabels=[semantic_label(v) for v in table.channel],xlabel='Model amplitude / mean absolute signal')
    ax.legend(frameon=False,fontsize=11)
    return Drawn(table,ax)
