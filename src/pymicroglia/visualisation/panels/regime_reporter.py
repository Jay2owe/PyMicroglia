"""Draw prepared within-cell contrasts and transition-aligned evidence."""
from . import colour
from ._contract import Drawn
from ..labels import semantic_label


def paired(ax,data,*,style=None,**options):
    t=data['table'];shown=t.loc[t.metric.eq(data['metric'])]
    for row in shown.itertuples():ax.plot([0,1],[row.out_regime_mean,row.in_regime_mean],color=colour('teal'),alpha=.5,marker='o')
    if shown.empty:ax.text(.5,.5,'No cells observed both inside and outside the selected state',ha='center',wrap=True,transform=ax.transAxes)
    ax.set(xticks=[0,1],xticklabels=['Other states',data['state']],ylabel=semantic_label(data['metric']))
    return Drawn(t,ax)


def triggered(ax,data,*,style=None,**options):
    t=data['table']
    if not t.empty:
        for i,(metric,rows) in enumerate(t.groupby('metric',sort=False)):
            hue=colour(['teal','orange','blue','plum'][i%4])
            ax.plot(rows.offset,rows['mean'],color=hue,label=semantic_label(metric))
            ax.fill_between(rows.offset,rows.lo,rows.hi,color=hue,alpha=.22)
            if 'null_lo' in rows:ax.fill_between(rows.offset,rows.null_lo,rows.null_hi,color=colour('muted'),alpha=.16)
        ax.legend()
    else:ax.text(.5,.5,'No complete transition windows',ha='center',transform=ax.transAxes)
    ax.axvline(0,color=colour('muted'))
    ax.set(xlabel='Frames from a regime change',ylabel='Within-window standardized signal')
    return Drawn(t,ax)


def plane(figure,data,*,style=None,**options):
    grid=figure.add_gridspec(4,4,hspace=.05,wspace=.05)
    ax=figure.add_subplot(grid[1:,:3]);top=figure.add_subplot(grid[0,:3],sharex=ax);right=figure.add_subplot(grid[1:,3],sharey=ax)
    t=data['table'];ax.scatter(t.level,t.texture,color=colour('teal'),alpha=.4,s=12)
    x=data['xbins'];y=data['ybins']
    top.bar(x.bin_left,x['count'],width=x.bin_right-x.bin_left,align='edge',color=colour('teal'))
    right.barh(y.bin_left,y['count'],height=y.bin_right-y.bin_left,align='edge',color=colour('teal'))
    top.tick_params(labelbottom=False);right.tick_params(labelleft=False)
    ax.set(xlabel=semantic_label(data['x']),ylabel=semantic_label(data['y']))
    return Drawn(t,[ax,top,right])
