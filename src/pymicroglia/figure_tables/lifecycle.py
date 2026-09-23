"""Original saved lifecycle summaries, prepared independently of rendering."""
from __future__ import annotations
from typing import Any,Sequence
import numpy as np
import pandas as pd
from .prepared import PreparedViews
from ..measure.modules.lifecycle import EVENTS,STARTS,ENDS
EVENT_ROLES=dict.fromkeys(EVENTS)
STATUS_FILL={'observed':True,'censored':True,'candidate':False,'unproven':False}
def _filled(event: str):
    return event in ('present_at_start', 'present_at_end', 'entered_field', 'left_field')

def event_timelines(cells: Any, *, events: Any=None, order: str='start', hour_ticks: float | None=None, y_label: str='Cell', lineage: bool=True, legend: bool=True, max_cells: int | None=None):
    frame = pd.DataFrame(cells).copy()
    required = {'identity', 'start_hours', 'end_hours', 'start_event', 'end_event'}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"event_timelines needs {', '.join(sorted(missing))}")
    censored = frame['start_event'].eq('present_at_start') & frame['end_event'].eq('present_at_end')
    frame['_has_event'] = ~censored
    if order == 'start':
        frame = frame.sort_values(['start_hours', 'identity'])
    elif order == 'event':
        frame = frame.sort_values(['_has_event', 'start_hours', 'identity'], ascending=[False, True, True])
    elif order == 'identity':
        frame = frame.sort_values('identity')
    else:
        raise ValueError('order must be start, event or identity')
    left_out = 0
    if max_cells is not None and len(frame) > max_cells:
        keep = pd.concat([frame[frame['_has_event']], frame[~frame['_has_event']]])
        left_out = len(frame) - max_cells
        frame = keep.head(max_cells).sort_values(['start_hours', 'identity'])
    lane = {int(identity): position for position, identity in enumerate(frame['identity'].to_numpy())}
    event_frame = pd.DataFrame(events).copy() if events is not None else pd.DataFrame()
    if lineage and (not event_frame.empty) and ('partner_identity' in event_frame):
        births = event_frame[event_frame['event'].eq('born')]
        for _, row in births.iterrows():
            child, parent = (int(row['identity']), row['partner_identity'])
            if pd.isna(parent) or child not in lane or int(parent) not in lane:
                continue
    rows = []
    for end, column, hours_column in (('start', 'start_event', 'start_hours'), ('end', 'end_event', 'end_hours')):
        for event, group in frame.groupby(column):
            positions = [lane[int(identity)] for identity in group['identity']]
            rows.extend(({'identity': int(identity), 'lane': lane[int(identity)], 'hours': float(hour), 'event': str(event), 'end': end} for identity, hour in zip(group['identity'], group[hours_column])))
    if not event_frame.empty:
        during = event_frame[event_frame['event'].eq('divided')]
        during = during[during['identity'].isin(lane)]
        if not during.empty:
            rows.extend(({'identity': int(row['identity']), 'lane': lane[int(row['identity'])], 'hours': float(row['hours']), 'event': 'divided', 'end': 'during'} for _, row in during.iterrows()))
    return dict(data=pd.DataFrame(rows, columns=['identity', 'lane', 'hours', 'event', 'end']), extra={'cells_drawn': len(lane), 'cells_left_out': left_out})

def event_ledger(events: Any, *, order: Sequence[str] | None=None, show_censored: bool=True, legend: bool=True):
    frame = pd.DataFrame(events).copy()
    if frame.empty:
        return dict(data=pd.DataFrame(columns=['event', 'event_status', 'count']))
    if not show_censored:
        frame = frame[~frame['event_status'].eq('censored')]
    counts = frame.groupby(['event', 'event_status']).size().rename('count').reset_index()
    names = list(order) if order else [name for name in EVENT_ROLES if name in set(counts['event'])]
    names = [name for name in names if name in set(counts['event'])]
    position = {name: index for index, name in enumerate(reversed(names))}
    for _, row in counts.iterrows():
        name = str(row['event'])
        if name not in position:
            continue
        settled = STATUS_FILL.get(str(row['event_status']), False)
    return dict(data=counts)

def event_clock(events: Any, *, show: Sequence[str]=('born', 'died'), bin_hours: float=6.0, hour_ticks: float | None=None, legend: bool=True):
    frame = pd.DataFrame(events).copy()
    frame = frame[frame['event'].isin(list(show))]
    if frame.empty:
        return dict(data=pd.DataFrame(columns=['event', 'bin_start_hours', 'count']))
    span = float(frame['hours'].max())
    edges = np.arange(0.0, span + bin_hours, bin_hours)
    rows = []
    for name in show:
        hours = frame.loc[frame['event'].eq(name), 'hours'].to_numpy(float)
        counts, _ = np.histogram(hours, bins=edges)
        rows.extend(({'event': str(name), 'bin_start_hours': float(start), 'count': int(count)} for start, count in zip(edges[:-1], counts)))
    return dict(data=pd.DataFrame(rows))

def _spread(centres: Sequence[float], minimum: float):
    if not centres:
        return []
    out = list(centres)
    for index in range(1, len(out)):
        out[index] = min(out[index], out[index - 1] - minimum)
    for index in range(len(out) - 2, -1, -1):
        out[index] = max(out[index], out[index + 1] + minimum)
    return out

def fate_flow(cells: Any, *, legend: bool=False, minimum: int=1):
    frame = pd.DataFrame(cells).copy()
    missing = {'start_event', 'end_event'} - set(frame)
    if missing:
        raise ValueError(f"fate_flow needs {', '.join(sorted(missing))}")
    if frame.empty:
        return dict(data=pd.DataFrame(columns=['start_event', 'end_event', 'cells']))
    pairs = frame.groupby(['start_event', 'end_event']).size().rename('cells').reset_index()
    pairs = pairs[pairs['cells'] >= minimum]
    if pairs.empty:
        return dict(data=pairs)
    starts = [event for event in STARTS if event in set(pairs['start_event'])]
    ends = [event for event in ENDS if event in set(pairs['end_event'])]
    return dict(data=pairs.reset_index(drop=True),extra={'cells':int(pairs.cells.sum())})


def event_accumulation(events: Any, *, show: Sequence[str] | None=None, hour_ticks: float | None=None, legend: bool=True):
    frame = pd.DataFrame(events).copy()
    if frame.empty or 'hours' not in frame:
        return dict(data=pd.DataFrame(columns=['event', 'hours', 'cumulative']))
    wanted = [str(name) for name in show] if show is not None else [event for event in EVENTS if event in set(frame['event']) and event not in ('present_at_start', 'present_at_end')]
    frame = frame[frame['event'].isin(wanted)]
    if frame.empty:
        return dict(data=pd.DataFrame(columns=['event', 'hours', 'cumulative']))
    first, last = (float(frame['hours'].min()), float(frame['hours'].max()))
    rows = []
    for event in [name for name in EVENTS if name in set(frame['event'])]:
        hours = np.sort(frame.loc[frame['event'].eq(event), 'hours'].to_numpy(float))
        running = np.arange(1, hours.size + 1)
        rows.extend(({'event': event, 'hours': float(hour), 'cumulative': int(count)} for hour, count in zip(hours, running)))
    if legend:
        pass
    return dict(data=pd.DataFrame(rows, columns=['event', 'hours', 'cumulative']))

def lineage_trees(cells: Any, *, hour_ticks: float | None=None, max_families: int | None=None, legend: bool=True):
    frame = pd.DataFrame(cells).copy()
    missing = {'identity', 'start_hours', 'end_hours', 'start_event', 'end_event'} - set(frame)
    if missing:
        raise ValueError(f"lineage_trees needs {', '.join(sorted(missing))}")
    for column in ('parent_identity', 'absorbed_into_identity'):
        if column not in frame:
            frame[column] = pd.NA
    links = [(int(row.identity), int(row.parent_identity), 'division') for row in frame.itertuples() if not pd.isna(row.parent_identity)]
    links += [(int(row.identity), int(row.absorbed_into_identity), 'fusion') for row in frame.itertuples() if not pd.isna(row.absorbed_into_identity)]
    if not links:
        return dict(data=pd.DataFrame(columns=['identity', 'lane', 'family', 'relation']))
    home: dict[int, int] = {}

    def _root(value: int) -> int:
        while home.get(value, value) != value:
            value = home[value]
        return value
    for one, other, _ in links:
        home.setdefault(one, one)
        home.setdefault(other, other)
        first_root, second_root = (_root(one), _root(other))
        if first_root != second_root:
            home[second_root] = first_root
    known = frame.set_index('identity')
    families: dict[int, list[int]] = {}
    for identity in sorted(home):
        if identity in known.index:
            families.setdefault(_root(identity), []).append(identity)
    ordered = sorted((group for group in families.values() if group), key=lambda group: min((float(known.loc[member, 'start_hours']) for member in group)))
    if max_families is not None:
        ordered = ordered[:max_families]
    lane: dict[int, int] = {}
    labels: list[str] = []
    rows: list[dict] = []
    for number, group in enumerate(ordered):
        eldest = min(group, key=lambda value: float(known.loc[value, 'start_hours']))
        for member in sorted(group, key=lambda value: float(known.loc[value, 'start_hours'])):
            lane[member] = len(labels)
            labels.append(f'cell {member}')
            rows.append({'identity': member, 'lane': lane[member], 'family': number, 'relation': 'eldest' if member == eldest else 'relative'})
        labels.append('')
    if labels and labels[-1] == '':
        labels.pop()
    return dict(data=pd.DataFrame(rows, columns=['identity', 'lane', 'family', 'relation']), extra={'families': len(ordered), 'cells': len(lane)})


NOTE='Open marks and hatched bars identify candidate events or unexplained outcomes. They are cases for review, not confirmed biological events. Filled marks are settled by the field or recording boundaries.'


def _connections(cells,events,lanes):
    edges=[]
    for row in events.itertuples(index=False):
        parent=getattr(row,'partner_identity',None)
        if row.event=='born' and pd.notna(parent) and int(parent) in lanes and int(row.identity) in lanes:
            edges.append(dict(hours=float(row.hours),first=lanes[int(parent)],second=lanes[int(row.identity)],kind='division'))
    if 'absorbed_into_identity' in cells:
        for row in cells.itertuples(index=False):
            if pd.notna(row.absorbed_into_identity) and int(row.absorbed_into_identity) in lanes and int(row.identity) in lanes:
                edges.append(dict(hours=float(row.end_hours),first=lanes[int(row.identity)],second=lanes[int(row.absorbed_into_identity)],kind='fusion'))
    return edges


def _lanes(table,cells,events,*,lineage=True):
    lanes=dict(zip(table.identity.astype(int),table.lane.astype(int)))
    bars=cells.loc[cells.identity.isin(lanes)].copy()
    bars['lane']=bars.identity.map(lanes)
    return dict(table=table,bars=bars,connections=_connections(cells,events,lanes) if lineage else [])


def event_views(source,options):
    cells=source.table('lifecycle_cells');events=source.table('lifecycle_events')
    maximum=int(options['max_cells'])
    if maximum<0:raise ValueError('max_cells must be non-negative')
    width=float(options['bin_hours'])
    if not np.isfinite(width) or width<=0:raise ValueError('bin_hours must be finite and positive')
    timeline=event_timelines(cells,events=events,order=options['order'],max_cells=maximum or None)['data']
    ledger=event_ledger(events,show_censored=options['show_censored'])['data']
    timing=event_clock(events,show=options['show'],bin_hours=width)['data']
    return PreparedViews({'timelines':_lanes(timeline,cells,events,lineage=options['lineage']),
        'ledger':dict(table=ledger),'timing':dict(table=timing)},
        auxiliary={'lifecycle_cells.csv':cells,'lifecycle_events.csv':events},
        wording=dict(subtitle=f'{len(cells)} tracked cells; {len(events)} recorded events.',footnote=NOTE)),None


def summary_views(source,options):
    cells=source.table('lifecycle_cells');events=source.table('lifecycle_events')
    maximum=int(options['max_families'])
    if maximum<0:raise ValueError('max_families must be non-negative')
    fates=fate_flow(cells)['data']
    accumulation=event_accumulation(events,show=options['show'])['data']
    families=lineage_trees(cells,max_families=maximum or None)['data']
    return PreparedViews({'fates':dict(table=fates),'accumulation':dict(table=accumulation),
        'families':_lanes(families,cells,events)},auxiliary={'lifecycle_cells.csv':cells,'lifecycle_events.csv':events},
        wording=dict(subtitle=f'{len(cells)} tracks, each counted once at its beginning and once at its ending.',footnote=NOTE)),None
