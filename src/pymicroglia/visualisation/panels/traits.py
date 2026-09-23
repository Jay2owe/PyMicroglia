"""Draw the prepared condition-specific variability ratios."""
import numpy as np
from . import colour
from ._contract import Drawn


def ranking(figure,data,*,style=None,**options):
    groups=list(data['table'].groupby('condition',sort=False))
    layout=figure.add_gridspec(1,len(groups));axes=[]
    for index,(_,rows) in enumerate(groups):
        ax=figure.add_subplot(layout[index]);axes.append(ax)
        hue=colour(rows.condition_hue.iloc[0]);values=rows.within_over_between.to_numpy(float)
        y=np.arange(len(rows));valid=np.isfinite(values)
        ax.axvspan(0,.5,color=colour('teal'),alpha=.055)
        ax.axvspan(.7,data['upper'],color=colour('orange'),alpha=.035)
        ax.axvline(1,color=colour('muted'),linestyle=':')
        ax.hlines(y[valid],0,values[valid],color=hue);ax.scatter(values[valid],y[valid],color=hue)
        for height,value in zip(y,values):
            ax.text(value+.02 if np.isfinite(value) else .02,height,f'{value:.2f}' if np.isfinite(value) else 'Not estimable',va='center',fontsize=10)
        ax.set(yticks=y,yticklabels=rows.label,xlim=(0,data['upper']),
               xlabel='Within-cell / between-cell interquartile range',title=rows.condition_label.iloc[0])
    return Drawn(data['table'],axes)
