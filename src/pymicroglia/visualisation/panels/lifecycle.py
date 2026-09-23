"""Draw saved track events with distinct confirmed and candidate marks."""
import numpy as np
from . import colour
from ._contract import Drawn
from .exchange import _hours
from ...measure.modules.lifecycle import EVENTS,STARTS,ENDS

SETTLED={'present_at_start','present_at_end','entered_field','left_field'}
COLORS=dict(zip(EVENTS,('muted','teal','orange','teal','muted','red','red','circadian_purple','raw','raw','orange')))
MARKERS=dict(zip(EVENTS,('|','>','o','^','|','<','X','s','v','v','D')))


def _empty(ax,data):
    if not data['table'].empty:return False
    ax.text(.5,.5,'No events in this selection',ha='center',va='center',transform=ax.transAxes)
    ax.set_axis_off()
    return True


def lanes(ax,data,*,style,**options):
    if _empty(ax,data):return Drawn(data['table'],ax)
    bars=data['bars']
    ax.hlines(bars.lane,bars.start_hours,bars.end_hours,color=colour('teal'),linewidth=1.5)
    for edge in data['connections']:
        ax.plot([edge['hours']]*2,[edge['first'],edge['second']],color=colour('orange' if edge['kind']=='division' else 'circadian_purple'),alpha=.6)
    for row in bars.itertuples(index=False):
        for event,hour in [(row.start_event,row.start_hours),(row.end_event,row.end_hours)]:
            color=colour(COLORS.get(event,'muted'))
            ax.plot(hour,row.lane,marker=MARKERS.get(event,'o'),markerfacecolor=color if event in SETTLED else 'none',markeredgecolor=color,linestyle='none')
    if 'event' in data['table']:
        during=data['table'].loc[data['table'].event.eq('divided')]
        ax.scatter(during.hours,during.lane,marker='D',facecolors='none',edgecolors=colour('orange'))
    ax.set(xlabel='Hours from start of recording',ylabel='Tracked cell')
    ax.invert_yaxis();_hours(ax,options)
    return Drawn(data['table'],ax)


def ledger(ax,data,*,style,**options):
    table=data['table']
    if _empty(ax,data):return Drawn(table,ax)
    names=[name for name in EVENTS if name in set(table.event)]
    positions={name:index for index,name in enumerate(names)}
    for row in table.itertuples(index=False):
        settled=row.event_status in {'observed','censored'}
        ax.barh(positions[row.event]+(-.18 if settled else .18),row.count,height=.33,color=colour(COLORS[row.event]),hatch=None if settled else '///',alpha=1 if settled else .4)
    ax.set(yticks=range(len(names)),yticklabels=[name.replace('_',' ') for name in names],xlabel='Recorded events')
    return Drawn(table,ax)


def timing(ax,data,*,style,**options):
    table=data['table']
    if _empty(ax,data):return Drawn(table,ax)
    running='cumulative' in table
    for name,rows in table.groupby('event',sort=False):
        x=rows.hours if running else rows.bin_start_hours
        y=rows.cumulative if running else rows['count']
        ax.step(x,y,where='post',color=colour(COLORS[name]),linestyle='-' if name in SETTLED else '--',label=name.replace('_',' '))
    ax.set(xlabel='Hours from start of recording',ylabel='Events accumulated' if running else f"Events per {options['bin_hours']:g} hours",ylim=(0,None))
    ax.legend(frameon=False,fontsize=10);_hours(ax,options)
    return Drawn(table,ax)


def fates(ax,data,*,style,**options):
    table=data['table']
    if _empty(ax,data):return Drawn(table,ax)
    starts=[name for name in STARTS if name in set(table.start_event)]
    ends=[name for name in ENDS if name in set(table.end_event)]
    left={name:float(table.loc[table.start_event.eq(name),'cells'].sum()) for name in starts}
    right={name:float(table.loc[table.end_event.eq(name),'cells'].sum()) for name in ends}
    total=sum(left.values());offsets=[]
    for counts in (left,right):
        cursor=0.;positions={}
        for name,count in counts.items():positions[name]=cursor;cursor+=count
        offsets.append(positions)
    x=np.linspace(0,1,80);ease=(1-np.cos(np.pi*x))/2
    for row in table.itertuples(index=False):
        first=offsets[0][row.start_event];last=offsets[1][row.end_event]
        y=first+ease*(last-first)
        ax.fill_between(x,y,y+row.cells,color=colour(COLORS[row.end_event]),alpha=.45,hatch=None if row.end_event in SETTLED else '//')
        offsets[0][row.start_event]+=row.cells;offsets[1][row.end_event]+=row.cells
    for xpos,counts,align in [(-.02,left,'right'),(1.02,right,'left')]:
        cursor=0
        for name,count in counts.items():
            ax.text(xpos,cursor+count/2,f"{name.replace('_',' ')} ({count:g})",ha=align,va='center',fontsize=10)
            cursor+=count
    ax.set(xlim=(-.5,1.5),ylim=(0,total));ax.set_axis_off()
    return Drawn(table,ax)
