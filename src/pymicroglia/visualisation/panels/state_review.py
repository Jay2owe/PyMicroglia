"""Draw saved state histories, rates, period evidence and persistence controls."""
import numpy as np
from . import colour, colours, overflow_colours
from ._contract import Drawn


def _lanes(ax,data):
    cells = data['cells']
    ax.set(ylim=(len(cells)-.5,-.5),yticks=cells.display_row,yticklabels=cells.cell_label)


def history(ax,data,*,style,**options):
    from matplotlib.lines import Line2D
    count = len(data['states'])
    palette = colours(n=count) if count <= len(colours()) else overflow_colours(count)
    colors = dict(zip(data['states'],palette))
    frame = data['table']
    for state,group in frame.groupby('state'):
        ax.scatter(group.hours,group.display_row,marker='s',s=16,
                   color=colors.get(state,colour('muted')),linewidths=0)
    _lanes(ax,data)
    ax.set(xlabel='Recorded time (hours)',title='Snapshot state at each observed frame')
    handles = [Line2D([],[],color=colors[state],marker='s',linestyle='none',label=f'State {state}') for state in data['states']]
    if frame.state.lt(0).any():
        handles.append(Line2D([],[],color=colour('muted'),marker='s',linestyle='none',label='Uncertain / excluded'))
    if handles:
        ax.legend(handles=handles,frameon=False,loc='upper left',bbox_to_anchor=(1.02,1))
    return Drawn(frame,ax)


def period(ax,data,*,style,**options):
    frame = data['table']
    _lanes(ax,data)
    ax.set(xlabel='Estimated period (hours)',title='Saved period estimates')
    if frame.empty:
        ax.text(.5,.5,'Rhythm analysis disabled',transform=ax.transAxes,ha='center')
    else:
        for kind,offset,name,marker in [('state_probability',-.14,'blue','o'),('original_measurement',.14,'orange','^')]:
            for row in frame.loc[frame.trace_kind.eq(kind)].itertuples():
                if np.isfinite(row.period_hours):
                    ax.scatter(row.period_hours,row.display_row+offset,marker=marker,s=70,
                               facecolor=colour(name) if row.supported_period_for_grouping else 'none',edgecolor=colour(name))
                else:
                    ax.text(0,row.display_row+offset,'not tested',fontsize=9,color=colour(name))
        ax.set_xlim(0,data['maximum'])
    return Drawn(frame,ax)


def switching(ax,data,*,style,**options):
    frame,cells = data['table'],data['cells']
    ax.scatter(frame.display_row,frame.switches_per_hour,s=65,color=colour('dark'))
    rates = frame.switches_per_hour.to_numpy(float)
    maximum = float(np.max(rates[np.isfinite(rates)],initial=0))
    ax.set(title='Switching between confident states',ylabel='Switches / observed hour',
           xticks=cells.display_row,xticklabels=cells.cell_label,ylim=(0,max(.05,maximum*1.1)))
    return Drawn(frame,ax)


def persistence(ax,data,*,style,**options):
    frame,cells = data['table'],data['cells']
    ax.set(title='Persistence control',ylim=(-.03,1.23),yticks=[0,.5,1],
           ylabel='Fraction of null traces called rhythmic',xticks=cells.display_row,xticklabels=cells.cell_label)
    if frame.empty:
        ax.text(.5,.5,'No persistence simulations',transform=ax.transAxes,ha='center')
    else:
        ax.scatter(frame.display_row,frame.significant_fraction_of_tested,s=65,color=colour('blue'))
        for row in frame.itertuples():
            # Counts above the threshold stay clear of its reference line;
            # the actual fractions remain at their unchanged scatter positions.
            label_y = max(row.significant_fraction_of_tested,data['alpha'] or 0)
            ax.annotate(f'{row.significant_simulations}/{row.tested}',(row.display_row,label_y),
                        xytext=(0,9),textcoords='offset points',ha='center')
        if data['alpha'] is not None:
            ax.axhline(data['alpha'],color=colour('red'),linestyle=':')
    return Drawn(frame,ax)
