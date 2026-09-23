"""Draw recorded within-cell correlation profiles."""
from . import colour
from ._contract import Drawn
from ..labels import semantic_label


def profile(ax,data,*,style,**options):
    colors=('teal','orange','circadian_purple','blue')
    for index,((first,second),rows) in enumerate(data['rows'].groupby(['metric_a','metric_b'],sort=False)):
        ink=colour(colors[index%len(colors)])
        ax.plot(rows.lag_hours,rows.mean_correlation,color=ink,label=f'{semantic_label(first)} → {semantic_label(second)}')
        ax.fill_between(rows.lag_hours,rows.lo,rows.hi,color=ink,alpha=.16)
        ax.fill_between(rows.lag_hours,rows.surrogate_lo,rows.surrogate_hi,color=colour('muted'),alpha=.1)
        ax.plot(rows.lag_hours,rows.surrogate_mean,color=colour('muted'),linestyle=':')
    ax.axvline(0,color=colour('muted'),linestyle=':')
    ax.set(xlabel='Lag (hours); negative = first series leads',ylabel='Mean correlation')
    ax.legend(frameon=False,fontsize=11)
    return Drawn(data['table'],ax)


def peaks(ax,data,*,style,**options):
    table=data['table']
    if not table.empty:ax.scatter(table.peak_lag_hours,table.peak_correlation,color=colour('teal'))
    ax.axvline(0,color=colour('muted'),linestyle=':');ax.axhline(0,color=colour('muted'),linestyle=':')
    ax.set(xlabel="Each cell's peak lag (hours)",ylabel='Correlation at the selected lag')
    return Drawn(table,ax)
