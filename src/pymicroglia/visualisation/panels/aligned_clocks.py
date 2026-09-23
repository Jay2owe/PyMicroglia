"""Draw prepared recording-time and peak-aligned summaries."""
from . import colour
from ._contract import Drawn
from .exchange import _hours
from ..labels import semantic_label


def wall(ax,data,*,style=None,**options):
    t=data['table'];ax.plot(t.hours,t['mean'],color=colour('teal'));ax.fill_between(t.hours,t.lo,t.hi,color=colour('teal'),alpha=.22)
    ax.set(xlabel='Hours from recording start',ylabel=semantic_label(data['metric']))
    _hours(ax,options)
    return Drawn(t,ax)


def aligned(ax,data,*,style=None,**options):
    t=data['table']
    for i,(metric,rows) in enumerate(t.groupby('metric',sort=False)):
        hue=colour(['teal','orange','blue','plum'][i%4]);ax.plot(rows.phase_hour,rows.mean_z,color=hue,label=semantic_label(metric));ax.fill_between(rows.phase_hour,rows.lo,rows.hi,color=hue,alpha=.22)
    ax.axvline(0,color=colour('muted'));ax.set(xlim=(-data['period']/2,data['period']/2),xlabel="Hours from each cell's defining peak",ylabel='Within-cell change (standard deviations)')
    if not t.empty:ax.legend()
    else:ax.text(.5,.5,'No carried measurements selected',transform=ax.transAxes,ha='center')
    _hours(ax,options)
    return Drawn(t,ax)
