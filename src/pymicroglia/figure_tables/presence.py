"""Prepare recorded presence states and tracker explanations without changing them."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from .distributions import histogram
from .prepared import PreparedViews

STATE_CODES: dict[str, int] = {'outside_lifespan': 0, 'named': 1, 'unclaimed': 2}

STATE_LABELS: dict[str, str] = {'named': 'present on screen', 'unclaimed': 'temporary gap', 'outside_lifespan': 'before first appearance / after last'}

MECHANISM_LABELS: dict[str, str] = {'blue_red_motion_link_candidate': 'motion-linked identity candidate', 'distant_alias_risk': 'distant identity alias risk', 'local_raw_supported_dropout': 'raw signal supports a missed outline', 'long_unproven_gap': 'long gap without supporting evidence', 'same_host_merge_hiding': 'cell hidden inside the same merged object', 'unresolved': 'tracker could not classify the gap'}

def mechanism_label(value: Any) -> str:
    """A tracker's compact mechanism code in words suitable for a legend."""
    name = str(value)
    return MECHANISM_LABELS.get(name, name.replace('_', ' '))

def persistence_values(frame: Any) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[int]]:
    """Prepare one three-state persistence raster from long-form presence rows.

    Returns the exact plotted long table, matrix, frame hours and identity order.
    Keeping this preparation beside the renderer makes the standalone raster and
    any composite page use the same row ordering and state codes.
    """
    presence = pd.DataFrame(frame).copy()
    required = {'identity', 'frame_index', 'hours', 'state', 'first_frame_index', 'last_frame_index'}
    missing = required - set(presence)
    if missing:
        raise ValueError('persistence_values needs ' + ', '.join(sorted(missing)))
    unknown = set(presence['state'].dropna().astype(str)) - set(STATE_CODES)
    if unknown:
        raise ValueError('unknown persistence states: ' + ', '.join(sorted(unknown)))
    order = presence.groupby('identity')[['first_frame_index', 'last_frame_index']].first().sort_values(['first_frame_index', 'last_frame_index']).index.tolist()
    row_of = {identity: row for row, identity in enumerate(order)}
    presence['row_position'] = presence['identity'].map(row_of).astype(int)
    presence['state_code'] = presence['state'].map(STATE_CODES).astype(int)
    presence = presence.sort_values(['row_position', 'frame_index'])
    matrix = presence.pivot(index='row_position', columns='frame_index', values='state_code').sort_index().to_numpy(float)
    hours = presence.groupby('frame_index')['hours'].first().sort_index().to_numpy(float)
    return (presence, matrix, hours, [int(identity) for identity in order])

def persistence_events(presence: Any, gap_frames: Any=None, lifespans: Any=None) -> pd.DataFrame:
    """One marker per temporary-absence interval and silent ending.

    Every orange run gets one marker at its temporal midpoint. Where the
    tracker's Gantt evidence names a mechanism, that exact code is retained in
    ``mechanism`` and translated only for ``event_label``. Silent mid-field
    endings receive the cross used by the lifespan chart.
    """
    frame = pd.DataFrame(presence).copy()
    required = {'identity', 'frame_index', 'hours', 'state', 'row_position'}
    missing = required - set(frame)
    if missing:
        raise ValueError('persistence_events needs ' + ', '.join(sorted(missing)))
    history = pd.DataFrame(gap_frames).copy() if gap_frames is not None else pd.DataFrame()
    if not history.empty:
        if 'still_missing_in_accepted_labels' in history:
            history = history[history['still_missing_in_accepted_labels'].astype(bool)]
        keep = [column for column in ('identity', 'frame_index', 'mechanism') if column in history]
        if {'identity', 'frame_index', 'mechanism'}.issubset(keep):
            frame = frame.merge(history[keep].drop_duplicates(['identity', 'frame_index']), on=['identity', 'frame_index'], how='left')
    if 'mechanism' not in frame:
        frame['mechanism'] = None
    gaps = frame[frame['state'] == 'unclaimed'].copy()
    rows: list[dict[str, Any]] = []
    if not gaps.empty:
        gaps = gaps.sort_values(['identity', 'frame_index'])
        new_run = gaps.groupby('identity')['frame_index'].diff().ne(1) | gaps['identity'].ne(gaps['identity'].shift())
        gaps['event_run'] = new_run.groupby(gaps['identity']).cumsum().astype(int)
        for (identity, event_run), group in gaps.groupby(['identity', 'event_run']):
            named = group['mechanism'].dropna().astype(str)
            mechanism = named.mode().iat[0] if not named.empty else None
            label = mechanism_label(mechanism) if mechanism is not None else 'temporary absence, no recorded cause'
            rows.append({'identity': int(identity), 'row_position': int(group['row_position'].iat[0]), 'event_class': 'temporary_absence', 'event_label': label, 'mechanism': mechanism, 'start_frame_index': int(group['frame_index'].min()), 'end_frame_index': int(group['frame_index'].max()), 'start_hour': float(group['hours'].min()), 'end_hour': float(group['hours'].max()), 'hours': float(group['hours'].mean())})
    life = pd.DataFrame(lifespans).copy() if lifespans is not None else pd.DataFrame()
    if not life.empty and {'identity', 'silent_nonborder_ending'}.issubset(life):
        silent = set(life.loc[life['silent_nonborder_ending'].astype(bool), 'identity'])
        endings = frame[frame['identity'].isin(silent) & (frame['state'] == 'named')].sort_values('frame_index').groupby('identity').tail(1)
        for _, ending in endings.iterrows():
            rows.append({'identity': int(ending['identity']), 'row_position': int(ending['row_position']), 'event_class': 'silent_ending', 'event_label': 'silent ending away from the field edge', 'mechanism': None, 'start_frame_index': int(ending['frame_index']), 'end_frame_index': int(ending['frame_index']), 'start_hour': float(ending['hours']), 'end_hour': float(ending['hours']), 'hours': float(ending['hours'])})
    columns = ['identity', 'row_position', 'event_class', 'event_label', 'mechanism', 'start_frame_index', 'end_frame_index', 'start_hour', 'end_hour', 'hours']
    return pd.DataFrame(rows, columns=columns)

def _lifespan_inputs(presence: pd.DataFrame, summary: pd.DataFrame, history_gaps: pd.DataFrame | None, history_lifespans: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame, list[str], set[int], int]:
    """Prepare the cell spans and accepted gap explanations used by the timeline."""
    longest: dict[int, int] = {}
    for identity, group in presence[presence['state'] == 'unclaimed'].groupby('identity'):
        frames = np.sort(group['frame_index'].to_numpy(int))
        runs = np.split(frames, np.flatnonzero(np.diff(frames) > 1) + 1)
        longest[int(identity)] = max((len(run) for run in runs), default=0)
    required = {'identity', 'first_hour', 'last_hour', 'observed_frames', 'gap_frames', 'coverage'}
    missing = required - set(summary)
    if missing:
        raise SystemExit('the lifespan panels need cell_summary.csv columns: ' + ', '.join(sorted(missing)))
    data = summary[list(required)].copy()
    data['span_hours'] = data['last_hour'] - data['first_hour']
    data['longest_gap_frames'] = data['identity'].map(longest).fillna(0).astype(int)
    data['ever_touches_border'] = summary.get('ever_touches_border', False)
    ordered = ['identity', 'first_hour', 'last_hour', 'span_hours', 'observed_frames', 'gap_frames', 'coverage', 'longest_gap_frames', 'ever_touches_border']
    data = data[ordered]
    gaps = presence.copy()
    gaps['named'] = gaps['state'] == 'named'
    if history_gaps is not None and 'mechanism' in history_gaps:
        accepted = history_gaps.copy()
        if 'still_missing_in_accepted_labels' in accepted:
            accepted = accepted[accepted['still_missing_in_accepted_labels'].astype(bool)]
        gaps = gaps.merge(accepted[['identity', 'frame_index', 'mechanism']].drop_duplicates(['identity', 'frame_index']), on=['identity', 'frame_index'], how='left')
    if 'mechanism' not in gaps:
        gaps['mechanism'] = None
    mechanisms = sorted({str(value) for value in gaps['mechanism'].dropna().unique() if str(value) not in ('', 'nan')})
    unexplained = int(((gaps['state'] == 'unclaimed') & gaps['mechanism'].isna()).sum())
    silent: set[int] = set()
    if history_lifespans is not None and 'silent_nonborder_ending' in history_lifespans:
        silent = {int(identity) for identity in history_lifespans.loc[history_lifespans['silent_nonborder_ending'].astype(bool), 'identity']}
        keep = [column for column in ('identity', 'silent_nonborder_ending', 'last_mask_touches_border', 'termination_mechanism') if column in history_lifespans]
        data = data.merge(history_lifespans[keep], on='identity', how='left')
        data['silent_nonborder_ending'] = data['silent_nonborder_ending'].astype('object').where(data['silent_nonborder_ending'].notna(), False).astype(bool)
    return (data, gaps, mechanisms, silent, unexplained)

def prepare(source,options):
    frames=source.table('presence_frame');presence=source.table('presence');summary=source.table('cell_summary')
    history=source.table('history_gap_frames',optional=True);lifetimes=source.table('history_lifespans',optional=True)
    lifespans,gaps,mechanisms,silent,unexplained=_lifespan_inputs(presence,summary,history,lifetimes)
    values,matrix,hours,identities=persistence_values(presence)
    events=persistence_events(values,history,lifetimes)
    ordinary=sorted(events.loc[events.event_class.ne('silent_ending'),'event_label'].astype(str).unique())
    marker_cycle=('o','s','^','D','v','P','h','*')
    styles={label:marker_cycle[i%len(marker_cycle)] for i,label in enumerate(ordinary)}
    styles['silent ending away from the field edge']='x'
    events['marker']=events.event_label.map(styles)
    counts=pd.DataFrame(dict(hours=frames.hours.to_numpy(float),named=frames.identities_named.to_numpy(float),expected=frames.identities_expected.to_numpy(float),unnamed=(frames.identities_expected-frames.identities_named).to_numpy(float)))
    foreground=pd.DataFrame(dict(hours=frames.hours.to_numpy(float),unclaimed_px=frames.unclaimed_px.to_numpy(float),claimed_px=frames.assigned_px.to_numpy(float),foreground_px=(frames.assigned_px+frames.unclaimed_px).to_numpy(float))) if 'unclaimed_px' in frames else pd.DataFrame()
    raster=values[['identity','row_position','frame_index','hours','state','state_code']].copy()
    raster['state_colour']=raster.state.map({'outside_lifespan':'raw','named':'dark','unclaimed':'orange'})
    order=options['order']
    if order=='first_appearance':ordered=lifespans.sort_values(['first_hour','identity'])
    elif order=='span':ordered=lifespans.sort_values(['span_hours','identity'],ascending=[False,True])
    elif order=='coverage':ordered=lifespans.sort_values(['coverage','identity'],ascending=[False,True])
    else:raise ValueError('order must be first_appearance, span or coverage')
    ordered=ordered.copy();ordered['row_order']=np.arange(len(ordered));ordered['marked']=ordered.identity.isin(silent);ordered['present_colour']='dark';ordered['end_marker']=np.where(ordered.marked,'x','')
    step=float(np.nanmedian(np.diff(np.sort(gaps.hours.unique())))) if not gaps.empty and gaps.hours.nunique()>1 else 1.
    palette={m:['teal','blue','plum','olive','red','cyan'][i%6] for i,m in enumerate(mechanisms)}
    cuts=gaps.loc[gaps.state.eq('unclaimed'),['identity','frame_index','hours','mechanism']].copy()
    cuts['mechanism_label']=cuts.mechanism.map(lambda m:mechanism_label(m) if pd.notna(m) else 'Gap with no recorded explanation')
    cuts['gap_colour']=cuts.mechanism.map(palette).fillna('orange')
    cuts['row_order']=cuts.identity.map(ordered.set_index('identity').row_order)
    arrivals=histogram(lifespans.first_hour,int(options['bins']));coverage=histogram(lifespans.coverage,int(options['bins']))
    return PreparedViews({'counts':dict(table=counts),'foreground':dict(table=foreground),
        'persistence':dict(table=raster,matrix=matrix,hours=hours,identities=identities,events=events),
        'lifespan':dict(table=ordered,cuts=cuts,step=step,palette=palette),
        'arrivals':dict(table=arrivals,label='First appearance (hours)'),
        'coverage':dict(table=coverage,label='Share of own lifespan observed')},
        auxiliary={'foreground.csv':foreground,'persistence.csv':raster,'lifespan.csv':ordered,'arrivals.csv':arrivals,'coverage.csv':coverage,'persistence_events.csv':events,'lifespan_summary.csv':lifespans,'gap_mechanisms.csv':cuts},
        wording=dict(title='Cells on screen: presence, persistence and lifespan',footnote='Persistence retains three states: named, temporarily missing, and outside the observed lifespan. Tracker explanations annotate the saved gaps; they do not change their state. Silent endings away from the field edge are marked with crosses.')),None
