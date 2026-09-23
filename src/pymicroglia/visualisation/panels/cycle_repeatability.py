"""Draw within-cell cycle comparisons prepared from saved rhythm results."""
from . import colour
from ._contract import Drawn


def profiles(ax,data,*,style=None,**options):
    for cycle,rows in data['table'].groupby('cycle',sort=False):
        hue=colour('teal' if cycle==1 else 'orange');x=rows.position_from_cycle_1_peak_percent
        ax.fill_between(x,rows.lower_quartile_signal_sd,rows.upper_quartile_signal_sd,color=hue,alpha=.2)
        ax.plot(x,rows.median_signal_sd,color=hue,label=f'Cycle {cycle}')
    ax.axvline(0,color=colour('muted'),linestyle=':')
    ax.set(xlim=(-50,50),xlabel="Position from this cell's first peak (% of its period)",ylabel='Detrended signal (SD)');ax.legend()
    return Drawn(data['table'],ax)


def agreement(ax,data,*,style=None,**options):
    rows=data['table']
    image=ax.scatter(rows.detected_period_hours,rows.signed_peak_shift_percent,c=rows.waveform_correlation,cmap='RdBu_r',vmin=-1,vmax=1)
    ax.figure.colorbar(image,ax=ax,label='Within-cell waveform correlation')
    ax.axhline(0,color=colour('muted'),linestyle=':')
    ax.set(ylim=(-50,50),xlabel='Detected period (h)',ylabel='Peak shift to next cycle (% of own period)')
    return Drawn(rows,ax)


def shift(ax,data,*,style=None,**options):
    rows=data['table']
    ax.bar(rows.left,rows.cells,width=rows.right-rows.left,align='edge',color=colour('teal'))
    ax.axvline(data['median'],color=colour('dark'),linestyle=':',label=f"Median {data['median']:.0f}%")
    ax.set(xlim=(0,50),xlabel='Absolute peak shift (% of own period)',ylabel='Cells');ax.legend()
    return Drawn(rows,ax)
