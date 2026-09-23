"""Saved supported pair effects on measured, independent coordinate frames."""
from pymicroglia._sources import source_file
from pathlib import Path
import math
import pandas as pd
from pymicroglia.pipelines._contracts import content_id
from pymicroglia.pipelines.coordination.inputs import read_inputs
from pymicroglia.pipelines.coordination.evidence import read_evidence
from pymicroglia.pipelines._screening import _json_value, file_hash
import pymicroglia.pipelines.coordination.display as display
SLUG = 'spatial-coordination-connections'
CLAIM = 'Saved pair-level support connects measured cell locations within each recording; temporal ordering and opposing associations retain their separate meanings.'
GROUP = ['source_run', 'movie', 'question', 'representation', 'adjustment', 'statistic', 'reference', 'target', 'model_id', 'evidence_kind', 'reference_state_id', 'target_state_id', 'distance_effect', 'coordination_question', 'coordination_measure']

def options(presentation):
    declared = presentation.as_dict().get('maps', {})
    if not isinstance(declared, dict) or set(declared) - {'edge_limit', 'position', 'snapshot_hours', 'text'}:
        raise ValueError('Map presentation accepts edge_limit, position, snapshot_hours and text')
    settings = {'edge_limit': 50, 'position': 'median', 'snapshot_hours': None, **{k: v for k, v in declared.items() if k != 'text'}}
    limit = settings['edge_limit']
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError('edge_limit must be a positive integer')
    if settings['position'] not in {'median', 'snapshot'}:
        raise ValueError('Map position must be median or snapshot')
    time = settings['snapshot_hours']
    if settings['position'] == 'snapshot' and (isinstance(time, bool) or not isinstance(time, (int, float)) or (not math.isfinite(time))):
        raise ValueError('Snapshot maps require one finite original snapshot_hours value')
    if settings['position'] == 'median' and time is not None:
        raise ValueError('snapshot_hours requires snapshot position mode')
    return (settings, declared.get('text', {}))

def version():
    from pymicroglia.pipelines._versions import rendering
    return rendering(__file__)

def positions(prepared, settings):
    result = prepared['layouts'].copy()
    result['position_observation_id'] = None
    if settings['position'] == 'median':
        return result
    points = prepared['geometry']
    points = points.loc[points.hours.eq(settings['snapshot_hours'])]
    if points.duplicated(['source_run', 'movie', 'identity']).any():
        raise ValueError('Snapshot has repeated original cell/time positions')
    lookup = {tuple((row[key] for key in ['source_run', 'movie', 'identity'])): row for row in _json_value(points.to_dict('records'))}
    for index, row in result.iterrows():
        point = lookup.get(tuple((row[key] for key in ['source_run', 'movie', 'identity'])))
        valid = point is not None and point['valid'] is True
        result.loc[index, ['x', 'y']] = [point['x'], point['y']] if valid else [None, None]
        result.loc[index, 'position_observation_id'] = point['observation_id'] if point is not None else None
        result.loc[index, 'position_meaning'] = 'Original measured position at ' + str(settings['snapshot_hours']) + ' h; missing at this exact time stays unavailable'
    return result

def data(prepared, evidence, selected, settings):
    """Display caps never enter the saved scientific selection or geometry."""
    chosen = {member['pair_id']: member for member in selected}
    pair_ids = set(evidence['pair_decisions'].loc[evidence['pair_decisions'].supported_effect_ids.map(bool), 'pair_id'])
    if set(chosen) != pair_ids:
        raise ValueError('Map selection differs from the saved supported pair population')
    wanted = {effect for member in chosen.values() for effect in member['supported_effect_ids']}
    effects = evidence['effects']
    supported = effects.loc[effects.effect_id.isin(wanted)]
    if set(supported.effect_id) != wanted or not supported.supported.all() or (not supported.evidence_level.eq('pair').all()):
        raise ValueError('Map edges require exact corrected pair-level effects; recording support cannot supply edges')
    membership = evidence['effect_pair_membership']
    layout = positions(prepared, settings)
    groups = {}
    records = {}
    statistics = []
    pages = []
    for row in _json_value(supported.to_dict('records')):
        definition = {key: row.get(key) for key in GROUP}
        group = content_id(definition)
        groups.setdefault(group, {'definition': definition, 'effects': []})['effects'].append(row)
    for group, bucket in sorted(groups.items()):
        definition = bucket['definition']
        movie = definition['movie']
        source = definition['source_run']
        local = layout.loc[layout.movie.eq(movie) & layout.source_run.eq(source)]
        lookup = {int(row['identity']): row for row in _json_value(local.to_dict('records'))}
        nodes = []
        for row in lookup.values():
            node = display.entry(**row, kind='node', group_id=group)
            records[node['entry_id']] = node
            nodes.append(node['entry_id'])
        all_effects = sorted(bucket['effects'], key=lambda row: row['effect_id'])
        shown = all_effects[:settings['edge_limit']]
        edges = []
        for row in shown:
            a, b = [lookup[int(row[role + '_identity'])] for role in ['reference', 'target']]
            ids = membership.loc[membership.effect_id.eq(row['effect_id']), 'pair_id'].tolist()
            if not ids or not set(ids) <= set(chosen):
                raise ValueError('Supported map effect lost its original endpoint reports')
            drawable = all((isinstance(v, (int, float)) and math.isfinite(v) for v in [a['x'], a['y'], b['x'], b['y']]))
            item = display.entry(**{**row, 'requested_pair_ids': ids}, kind='edge', group_id=group, x=a['x'], y=a['y'], x2=b['x'], y2=b['y'], value=row.get('effect'), drawable=drawable, reference_position_observation_id=a['position_observation_id'], target_position_observation_id=b['position_observation_id'])
            records[item['entry_id']] = item
            edges.append(item['entry_id'])
            statistics.append(item)
        footnote = 'Curves: corrected pair-level evidence; their bend has no physical meaning. Opposing association is not antiphase. Arrows show resolved temporal ordering, not causality. '
        footnote += 'Positions: coordinate-wise recording medians; lines do not establish continuous proximity.' if settings['position'] == 'median' else 'Positions: actual measurements at ' + str(settings['snapshot_hours']) + ' h; no nearest-time substitution.'
        if definition['question'] == 'proximity':
            footnote += ' Proximity effects describe saved changing-distance windows, not persistent neighbours.'
        page = {'view': 'connections', 'group_id': group, 'definition': definition, 'title': 'Supported cell-pair relationships | ' + movie, 'claim': CLAIM, 'footnote': footnote, 'entry_ids': nodes + edges, 'supported_effects': len(all_effects), 'shown_effects': len(shown), 'omitted_effect_ids': [row['effect_id'] for row in all_effects[len(shown):]], 'unit': prepared['provenance']['resolved_request']['request']['geometry']['unit'], 'definition_name': prepared['provenance']['resolved_request']['request']['geometry']['definition'], 'settings': settings, 'nodes': len(nodes), 'unplaced_nodes': int(local[['x', 'y']].isna().any(axis=1).sum())}
        pages.append(page)
    if not pages:
        item = display.entry(kind='diagnostic', status='no_supported_pairs', reason='No corrected pair-level evidence qualifies for a connection map', effect=None, p_value=None, q_value=None)
        records[item['entry_id']] = item
        statistics = [item]
        pages = [{'view': 'connections', 'title': 'No supported pair connections', 'claim': CLAIM, 'footnote': 'Recording-level evidence and geometric proximity cannot certify individual connections.', 'entry_ids': [item['entry_id']], 'supported_effects': 0, 'shown_effects': 0, 'omitted_effect_ids': [], 'settings': settings}]
    return (pd.DataFrame(records.values()), pd.DataFrame(statistics), pages)

def build(ctx):
    from pymicroglia.visualisation.panels import coordination_maps
    return display.build(ctx, coordination_maps)

def produce(context):
    from pymicroglia.visualisation.panels import coordination_maps
    settings, text = options(context.presentation)
    prepared = read_inputs(context.saved('pair-inputs'))
    evidence = read_evidence(context.saved('coordination-evidence'), context.saved('pair-inputs').outcome.scientific_id)
    values, statistics, pages = data(prepared, evidence, context.selection.members, settings)
    return display.produce(context, values=values, statistics=statistics, metadata={'settings': settings, 'evidence_id': context.saved('coordination-evidence').outcome.scientific_id}, pages=pages, slugs=[SLUG] * len(pages), panel=coordination_maps, sources=[__file__], text=text)
