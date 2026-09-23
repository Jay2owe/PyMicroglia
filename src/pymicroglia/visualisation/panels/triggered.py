"""Draw prepared responses without selecting events or computing intervals."""
from . import colour
from ._contract import Drawn
from ..labels import semantic_label


def responses(ax,data,*,style=None,**options):
    rows=data['table']
    if rows.empty:
        ax.text(.5,.5,'No ranked frames available',ha='center',transform=ax.transAxes)
    else:
        for index,(metric,group) in enumerate(rows.groupby('metric',sort=False)):
            hue=colour(['teal','orange','blue','plum','circadian_green'][index%5])
            ax.fill_between(group.offset_hours,group.lo,group.hi,color=hue,alpha=.25)
            ax.plot(group.offset_hours,group['mean'],color=hue,label=semantic_label(metric))
        ax.legend()
    ax.axvline(0,color=colour('muted'))
    ax.set(xlabel='Hours from selected frame',ylabel='Within-cell standardized level (z-score)')
    return Drawn(rows,ax)


def alignment(ax,data,*,style=None,**options):
    rows=data['table']
    ax.scatter(rows.event_hours,rows.identity,color=colour('orange'),marker='|')
    ax.set(xlabel='Selected frame time from recording start (h)',ylabel='Cell identity')
    return Drawn(rows,ax)
