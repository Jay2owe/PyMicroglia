"""Draw prepared occupancy classes and field fractions."""
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
from . import colour
from ._contract import Drawn
from .exchange import _hours


def maps(figure,data,*,style,**options):
    count=len(data['frames']);columns=min(count,3);rows=(count+columns-1)//columns
    layout=figure.add_gridspec(rows,columns);axes=[]
    cmap=ListedColormap([colour(row['colour']) for row in data['classes']])
    norm=BoundaryNorm(np.arange(len(data['classes'])+1)-.5,cmap.N)
    for index,(values,hours) in enumerate(zip(data['images'],data['hours'])):
        ax=figure.add_subplot(layout[index//columns,index%columns]);axes.append(ax)
        if data['backgrounds'] is not None:ax.imshow(data['backgrounds'][index],cmap='gray')
        ax.imshow(values,cmap=cmap,norm=norm,interpolation='nearest',alpha=.5 if data['backgrounds'] is not None else 1)
        ax.set(title=f'{hours:g} hours',xlabel='X (pixels)',ylabel='Y (pixels)')
    axes[-1].legend(handles=[Patch(facecolor=colour(row['colour']),label=row['occupancy_class']) for row in data['classes']],loc='upper left',bbox_to_anchor=(1.02,1),frameon=False,fontsize=10)
    return Drawn(data['table'],axes)


def composition(ax,data,*,style,**options):
    ax.stackplot(data['hours'],data['shares'],colors=[colour(row['colour']) for row in data['classes']],labels=[row['occupancy_class'] for row in data['classes']])
    ax.plot(data['hours'],data['total'],color=colour('dark'),label='Total occupied')
    ax.set(xlabel='Hours from recording start',ylabel='Share of field',ylim=(0,1))
    ax.legend(frameon=False,fontsize=11);_hours(ax,options)
    return Drawn(data['table'],ax)
