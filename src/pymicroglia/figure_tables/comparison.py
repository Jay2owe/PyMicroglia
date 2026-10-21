"""Full saved measurements prepared for native, configurable group figures.

The browser consumes this preparation; it never computes its own statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .group_display import points
from ..measure.summarise import _STATISTICS

VIEWS = {
    'cells': 'Individual cells', 'samples': 'Recording / sample points',
    'paired': 'Paired samples', 'box': 'Box plot', 'violin': 'Violin plot',
    'histogram': 'Histogram', 'time_course': 'Condition time course',
    'effects': 'Effect sizes with intervals',
}
DEFAULTS = dict(view='samples', metric='integrated_density', group_by='condition',
                unit='movie', summary='mean', aggregate='mean', normalization='none',
                detrend='none', trace_filter={'method': 'none'}, start=None, end=None,
                frame=None, groups=[], band='iqr', bins=20, bandwidth=None,
                normalization_config={}, detrend_config={})


def settings(raw):
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError('Unknown comparison settings: ' + ', '.join(sorted(unknown)))
    result = {**DEFAULTS, **raw}
    choices = {'view': VIEWS, 'unit': ('cell', 'movie', 'subject'),
               'summary': _STATISTICS, 'aggregate': ('mean', 'median'),
               'band': ('iqr', 'sd', 'sem', 'none')}
    for key, values in choices.items():
        if result[key] not in values:
            raise ValueError(f'Unknown {key}: {result[key]}')
    for key in ('start', 'end', 'frame', 'bandwidth'):
        if result[key] is not None:
            result[key] = float(result[key])
            if not np.isfinite(result[key]):
                raise ValueError(f'{key} must be finite')
    if result['start'] is not None and result['end'] is not None and result['start'] > result['end']:
        raise ValueError('The time window ends before it starts')
    result['bins'] = int(result['bins'])
    if not 2 <= result['bins'] <= 200:
        raise ValueError('Histogram bins must be between 2 and 200')
    if result['bandwidth'] is not None and result['bandwidth'] <= 0:
        raise ValueError('Density bandwidth must be positive')
    if not isinstance(result['groups'], list):
        raise ValueError('Included groups must be a list')
    return result


def _process(frame, config):
    """Preserve gaps and delegate optional trace processing to Workbench."""
    from .. import workbench
    metric = config['metric']
    frame = frame.copy()
    frame['value'] = pd.to_numeric(frame[metric], errors='coerce').replace([np.inf, -np.inf], np.nan)
    processing = (config['normalization'] != 'none' or config['detrend'] != 'none'
                  or config['trace_filter'].get('method', 'none') != 'none')
    if not processing:
        return frame
    for _, block in frame.groupby(['recording_id', 'identity'], sort=False):
        block = block.sort_values('hours')
        # Fill absent frames with NaN, never interpolate across absences.
        frames = np.arange(int(block.frame_index.min()), int(block.frame_index.max()) + 1)
        observed = block.set_index('frame_index')
        supplied = observed.reindex(frames)
        known = block[['frame_index', 'hours']].drop_duplicates().to_numpy(float)
        if len(known) < 2:
            raise ValueError('Trace preprocessing needs at least two timed observations')
        interval = (known[-1, 1] - known[0, 1]) / (known[-1, 0] - known[0, 0])
        hours = known[0, 1] + (frames-known[0, 0])*interval
        if interval <= 0 or not np.allclose(known[:, 1], known[0, 1]+(known[:, 0]-known[0, 0])*interval):
            raise ValueError('Trace preprocessing requires a regular saved time axis')
        values = supplied.value.to_numpy(float)
        values = np.asarray(workbench.filter_rhythm_trace(hours, values, config['trace_filter'])['values'], dtype=float)
        if config['detrend'] != 'none':
            completed = workbench.detrend_trace(hours, values, config['detrend_config'], method=config['detrend'])
            values = np.asarray(completed['processed_trace']['values'], dtype=float)
        if config['normalization'] != 'none':
            completed = workbench.normalize_trace(hours, values, config['normalization'],
                detrended=config['detrend'] != 'none', **config['normalization_config'])
            values = np.asarray(completed['processed_trace']['values'], dtype=float)
        frame.loc[block.index, 'value'] = values[block.frame_index.to_numpy(int)-frames[0]]
    return frame


def prepare_frames(frame, config, *, conditions=(), metric_label=None):
    config = settings(config)
    clocks=[]
    for _, block in frame.groupby('recording_id', sort=False):
        clock=block[['frame_index','hours']].drop_duplicates().sort_values('frame_index').to_numpy(float)
        if len(clock)>1:
            interval=(clock[-1,1]-clock[0,1])/(clock[-1,0]-clock[0,0])
            if interval>0 and np.allclose(clock[:,1],clock[0,1]+(clock[:,0]-clock[0,0])*interval):
                clocks.extend((clock[0,1]+np.arange(int(clock[-1,0]-clock[0,0])+1)*interval).tolist())
            else:
                clocks.extend(clock[:,1].tolist())
        elif len(clock):
            clocks.append(float(clock[0,1]))
    if config['metric'] not in frame:
        raise ValueError(f"No saved measurements for {config['metric']}")
    frame = _process(frame, config)
    for key, op in [('start', 'ge'), ('end', 'le')]:
        if config[key] is not None:
            frame = frame[getattr(frame.hours, op)(config[key])]
    if config['frame'] is not None:
        frame = frame[frame.frame_index.eq(config['frame'])]
    group_by = config['group_by']
    if group_by == 'all':
        frame = frame.assign(group='All cells')
    else:
        if group_by not in frame:
            raise ValueError(f'No assignments for grouping by {group_by}')
        frame = frame.assign(group=frame[group_by].fillna('Unassigned').astype(str).replace('', 'Unassigned'))
    if config['groups']:
        frame = frame[frame.group.isin(config['groups'])]
    declared = {row['id']: row for row in conditions}
    present = list(dict.fromkeys(frame.group))
    order = ([row['id'] for row in conditions if row['id'] in present]
             if group_by == 'condition' else [])
    groups = order + [group for group in present if group not in order]
    labels = [declared.get(group, {}).get('label', group) if group_by == 'condition' else group for group in groups]
    from ..visualisation.panels import colour
    palette = [colour(key) for key in ('teal', 'orange', 'blue', 'plum', 'okabe_bluish_green', 'muted')]
    colours = [declared.get(group, {}).get('colour', palette[i % len(palette)]) for i, group in enumerate(groups)]
    keys = ['recording_id', 'identity']
    if frame.empty:
        cells = pd.DataFrame(columns=[*keys, 'group', 'value', 'run', 'stem', 'subject', 'pair'])
        cells['value'] = pd.Series(dtype=float)
    else:
        cells = _STATISTICS[config['summary']](frame.groupby(keys, sort=False), 'value').rename('value').reset_index()
        metadata = frame.groupby(keys, sort=False).first().reset_index()[[*keys, 'group', 'run', 'stem', 'subject', 'pair']]
        cells = cells.merge(metadata, on=keys, validate='one_to_one')
    samples = cells.copy()
    if config['unit'] != 'cell' and not cells.empty:
        unit_key = 'recording_id' if config['unit'] == 'movie' else 'subject'
        if unit_key == 'subject' and cells.subject.eq('').any():
            raise ValueError('Assign a subject to every included recording to use subject summaries')
        samples = cells.groupby(['group', unit_key], sort=False).agg(value=('value', config['aggregate']), cells=('identity', 'count')).reset_index()
        if unit_key == 'recording_id':
            samples = samples.merge(cells[['recording_id', 'pair']].drop_duplicates(), on='recording_id', validate='many_to_one')
        else:
            samples['pair'] = samples.subject
    elif not cells.empty:
        samples['cells'] = 1
    selected = cells if config['view'] == 'cells' else samples
    positioned = points(selected, groups)
    hits = []
    if config['view'] == 'cells':
        for group, prepared in zip(groups, positioned['groups']):
            rows = selected[selected.group.eq(group) & np.isfinite(selected.value)]
            for x, row in zip(prepared['x'], rows.to_dict('records')):
                hits.append(dict(x=float(x), y=float(row['value']), run=row['run'], stem=row['stem'], identity=int(row['identity'])))
    timeline = []
    if config['view'] == 'time_course':
        if config['unit'] == 'cell':
            units = frame[['group', 'hours', 'value']]
        else:
            key = 'recording_id' if config['unit'] == 'movie' else 'subject'
            if key == 'subject' and frame.subject.eq('').any():
                raise ValueError('Assign subjects before using subject time courses')
            units = frame.groupby(['group', key, 'hours'], sort=False).value.agg(config['aggregate']).reset_index()
        for group in groups:
            grouped = units[units.group.eq(group)].groupby('hours').value
            centre = grouped.agg(config['aggregate'])
            band = config['band']
            if band == 'iqr':
                low, high = grouped.quantile(.25), grouped.quantile(.75)
            else:
                spread = grouped.std() if band == 'sd' else grouped.sem() if band == 'sem' else centre*0
                low, high = centre-spread, centre+spread
            table = pd.DataFrame(dict(hours=centre.index, mean=centre.to_numpy(), lo=low.to_numpy(), hi=high.to_numpy(), n=grouped.count().to_numpy()))
            # Explicit blank rows break lines at missing native sampling times.
            clock = sorted(set(clocks))
            if config['start'] is not None:clock=[hour for hour in clock if hour>=config['start']]
            if config['end'] is not None:clock=[hour for hour in clock if hour<=config['end']]
            if config['frame'] is not None:clock=sorted(frame.hours.unique())
            table = table.set_index('hours').reindex(clock).rename_axis('hours').reset_index()
            timeline.append(dict(group=group, table=table))
    ylabel = metric_label or config['metric'].replace('_', ' ')
    if config['normalization'] != 'none':
        ylabel += ' (' + config['normalization'] + ')'
    return dict(config=config, groups=groups, labels=labels, colours=colours,
                positioned=positioned, hits=hits, cells=cells, samples=samples,
                timeline=timeline, ylabel=ylabel,
                counts=dict(cells=len(cells), finite_cells=int(np.isfinite(cells.value).sum()),
                            recordings=int(frame.recording_id.nunique()), samples=len(samples), observations=len(frame)))


def prepare(source, options):
    """Registered figure entry point; reads saved tables without fitting/tests."""
    from hashlib import sha256
    from .saved import Tables
    from .prepared import PreparedViews
    from .contrast import prepare as prepare_effects
    from ..measure.comparison_conditions import frame_conditions
    recordings = options.get('comparison_recordings') or [dict(run=str(source.run), stem=source.stem)]
    blocks = []
    for row in recordings:
        if not row.get('include',True):continue
        saved = Tables(row.get('run') or source.run, row.get('stem') or source.stem)
        block = saved.table('cell_frame').copy()
        if row.get('identities') is not None:
            block = block[block.identity.isin(row['identities'])]
        spec = saved.movie.get('spec', {})
        condition = saved.movie.get('condition') or spec.get('condition') or 'Unassigned'
        if isinstance(condition, dict):
            condition = condition.get('condition') or 'Unassigned'
        block['recording_id'] = row.get('id') or sha256((str(saved.run)+'\n'+saved.stem).encode()).hexdigest()[:20]
        block['condition'] = frame_conditions(row, block.identity, condition)
        block['subject'] = row.get('subject') or saved.movie.get('subject') or spec.get('subject') or ''
        block['pair'] = row.get('pair') or ''
        block['run'], block['stem'] = str(saved.run), saved.stem
        block['recording'] = saved.stem
        for key, value in row.get('factors', {}).items():
            block['factor:'+key] = value
        source.sources.extend(saved.sources)
        blocks.append(block)
    if not blocks:raise ValueError('Include at least one recording')
    frame = pd.concat(blocks, ignore_index=True)
    data = prepare_frames(frame, options.get('comparison_settings') or {}, conditions=options.get('comparison_conditions') or [])
    effects, _ = prepare_effects(source, dict(order='family', metrics=[]))
    views = {}
    for view in VIEWS:
        views[view] = {**data, 'config': {**data['config'], 'view': view}}
        if view == 'cells':
            views[view]['positioned'] = points(data['cells'], data['groups'])
        if view == 'time_course':
            views[view] = prepare_frames(frame, {**data['config'], 'view': view}, conditions=options.get('comparison_conditions') or [])
        if view == 'effects':
            views[view]['effects'] = effects['effects']
    return PreparedViews(views, wording=dict(title='Cell and group comparison')), data['cells']
