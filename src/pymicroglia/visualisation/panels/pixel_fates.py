"""Draw prepared pixel-count flows without reclassifying any pixel."""
from matplotlib.path import Path
from matplotlib.patches import PathPatch,Rectangle,Patch
from . import colour
from ._contract import Drawn
from .exchange import _hours


def flow(ax,data,*,style=None,**options):
    for bar in data['bars']:ax.add_patch(Rectangle((bar['stage']-.025,bar['y']),.05,bar['height'],facecolor=colour(bar['colour']),edgecolor='none'))
    codes=[Path.MOVETO,Path.CURVE4,Path.CURVE4,Path.CURVE4,Path.LINETO,Path.CURVE4,Path.CURVE4,Path.CURVE4,Path.CLOSEPOLY]
    for path in data['paths']:ax.add_patch(PathPatch(Path(path['vertices'],codes),facecolor=colour(path['colour']),edgecolor='none',alpha=.42))
    ax.set(xlim=(-.1,len(data['labels'])-.9),ylim=(0,max(1,data['observations'])))
    ax.set_axis_off()
    for i,label in enumerate(data['labels']):ax.text(i,data['observations']*1.02,label,ha='center',va='bottom')
    ax.legend(handles=[Patch(facecolor=colour(c),label=n) for n,c in zip(['Core','Fringe','Transient','Vacant','Transferred'],['circadian_green','orange','teal','raw','plum'])],loc='upper center',bbox_to_anchor=(.5,-.03),ncol=5)
    return Drawn(data['table'],ax)


def shares(ax,data,*,style=None,**options):
    t=data['table'];ax.stackplot(t.hours,*[t[c] for c in data['classes']],colors=[colour(c) for c in data['colours']],labels=data['classes'])
    ax.set(xlabel='Hours from recording start',ylabel='Pooled pixels');ax.legend()
    _hours(ax,options)
    return Drawn(t,ax)
