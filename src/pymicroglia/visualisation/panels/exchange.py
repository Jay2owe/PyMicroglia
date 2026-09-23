"""Draw prepared footprint exchange and saved rhythm-test comparisons."""
import numpy as np
from . import colour
from ._contract import Drawn


def _hours(ax,options):
    if options.get('hour_ticks') is not None:
        from matplotlib.ticker import MultipleLocator
        spacing=float(options['hour_ticks'])
        if not np.isfinite(spacing) or spacing<=0:raise ValueError('hour_ticks must be finite and positive')
        ax.xaxis.set_major_locator(MultipleLocator(spacing))


def floor(ax,data,*,style,**options):
    table=data['table'];y=np.arange(len(table))
    for row,index in zip(table.itertuples(index=False),y):
        ax.plot([row.false_positive_rate_primary,row.observed_rate_primary],[index,index],color=colour('muted'))
        ax.annotate(f'{100*row.excess_over_null_primary:+.1f} pp',(max(row.false_positive_rate_primary,row.observed_rate_primary),index),xytext=(8,0),textcoords='offset points',va='center',fontsize=12)
    ax.scatter(table.false_positive_rate_primary,y,facecolors='none',edgecolors=colour('muted'),s=100,
               label=f"Drift-matched noise ({data['per_cell']} surrogates per cell)")
    ax.scatter(table.observed_rate_primary,y,color=colour('circadian_purple'),s=100,label='Observed cells')
    ax.set(yticks=y,yticklabels=table.label,xlim=(0,1.15),xlabel='Share called rhythmic by the saved primary test')
    from matplotlib.ticker import PercentFormatter
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False,fontsize=12)
    return Drawn(table,ax)


def ledger(ax,data,*,style,**options):
    for _,group in data['cells'].groupby('identity',sort=True):
        for column,name in [('gross_px','teal'),('abs_net_px','orange')]:
            ax.plot(group.hours,group[column],color=colour(name),alpha=.1)
    median=data['median']
    for column,name,label in [('gross_px','teal','Gained + lost area'),('abs_net_px','orange','Absolute net size change')]:
        ax.fill_between(median.index,data['low'][column],data['high'][column],color=colour(name),alpha=.16)
        ax.plot(median.index,median[column],color=colour(name),label=label)
    ax.set(xlabel='Hours from start of recording',ylabel=f"Area exchanged ({data['unit']})")
    ax.legend(frameon=False,fontsize=12)
    _hours(ax,options)
    return Drawn(data['table'],ax)


def cancelled(ax,data,*,style,**options):
    for _,group in data['table'].groupby('identity',sort=True):
        ax.plot(group.hours,group.cancelled_fraction,color=colour('teal'),alpha=.1)
    median=data['median']
    ax.plot(median.index,median.cancelled_fraction,color=colour('teal'))
    ax.set(ylim=(0,1),xlabel='Hours from start of recording',ylabel='Share of exchange cancelled\nby opposing edge motion')
    _hours(ax,options)
    return Drawn(data['table'],ax)


def against_size(figure,data,*,style,**options):
    layout=figure.add_gridspec(2,2,width_ratios=[4,1],height_ratios=[1,4])
    ax=figure.add_subplot(layout[1,0]);top=figure.add_subplot(layout[0,0],sharex=ax);right=figure.add_subplot(layout[1,1],sharey=ax)
    table=data['table'];xbins=data['xbins'];ybins=data['ybins']
    ax.scatter(table['size'],table.gross_px,color=colour('teal'),alpha=.55)
    top.bar(xbins.bin_left,xbins['count'],width=xbins.bin_right-xbins.bin_left,align='edge',color=colour('teal'),alpha=.65)
    right.barh(ybins.bin_left,ybins['count'],height=ybins.bin_right-ybins.bin_left,align='edge',color=colour('teal'),alpha=.65)
    top.tick_params(labelbottom=False);right.tick_params(labelleft=False)
    ax.set(xlabel=data['xlabel'],ylabel=f"Gross exchange ({data['unit']})")
    return Drawn(table,[ax,top,right])


def breath(ax,data,*,style,**options):
    table=data['median']
    ax.plot(table.hours,table.held_px,color=colour('muted'),alpha=.65,label='Retained area')
    ax.fill_between(table.hours,0,table.gained_px,color=colour('teal'),alpha=.45,label='Gained area')
    ax.fill_between(table.hours,-table.lost_px,0,color=colour('orange'),alpha=.45,label='Lost area')
    ax.plot(table.hours,table.net_px,color=colour('circadian_ink'),label='Net area change')
    ax.axhline(0,color=colour('muted'))
    ax.set(xlabel='Hours from start of recording',ylabel=f"Gained / lost area ({data['unit']})")
    ax.legend(frameon=False,fontsize=12)
    _hours(ax,options)
    return Drawn(data['table'],ax)


def small_multiples(ax,data,*,style,**options):
    for _,group in data['table'].groupby('identity',sort=False):
        ax.plot(group.hours,group.gained_px-group.lost_px,color=colour('teal'),alpha=.28)
    ax.axhline(0,color=colour('muted'))
    ax.set(xlabel='Hours from start of recording',ylabel=f"Gained − lost ({data['unit']})")
    _hours(ax,options)
    return Drawn(data['table'],ax)
