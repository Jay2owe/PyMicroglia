"""Original cross-cell observations and geometry, before relationship estimates."""
from __future__ import annotations
from pymicroglia._results import read_document
from pymicroglia._sources import source_file

from importlib.metadata import version as library_version
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd

from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id
from pymicroglia.pipelines.coordination.options import CoordinationPairKey, EndpointKey, QUESTIONS
from pymicroglia.pipelines.relationships.inputs import _kind, match_observations, paired_support, MATCH_COLUMNS
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, read_verified_tables, write_table


KEYS = ['source_run', 'movie', 'identity']
PAIR_KEYS = ['source_run', 'movie', 'pair_id', 'reference_identity', 'target_identity',
    'reference', 'target', 'reference_measurement_id', 'target_measurement_id',
    'reference_endpoint_id', 'target_endpoint_id', 'oriented', 'sample', 'sample_confirmed', 'condition']
TRACE_COLUMNS = KEYS + ['measurement', 'measurement_id', 'endpoint_id', 'table', 'observation_id',
    'source_position', 'source_observation_id', 'source_scope', 'sequence_index', 'frame_index', 'hours', 'raw_value', 'raw_kind', 'time_kind', 'within_range',
    'raw_valid', 'clock_valid', 'processed_value', 'processed_valid']
GEOMETRY_COLUMNS = KEYS + ['geometry_id', 'observation_id', 'frame_index', 'hours', 'raw_x', 'raw_y', 'x', 'y',
    'x_observation_id', 'y_observation_id', 'x_sequence_index', 'y_sequence_index', 'x_scale', 'y_scale', 'input_unit', 'unit', 'definition',
    'within_range', 'valid', 'inside_field', 'near_field_edge', 'reason']


def _table(rows, columns=()):
    return pd.DataFrame(rows, columns=list(dict.fromkeys([*columns, *(key for row in rows for key in row)])))


def input_settings(resolved):
    request = resolved.request
    return {'inputs': resolved.inputs.as_dict(), 'reference': [m.as_dict() for m in resolved.reference_measurements],
        'target': [m.as_dict() for m in resolved.target_measurements], 'coordinates': [m.as_dict() for m in resolved.coordinates],
        'external_reference_columns': [m.as_dict() for m in resolved.reference_columns],
        'pairs': [pair.as_dict() for pair in request.measurement_pairs], 'geometry': request.geometry.as_dict(),
        'support': request.support.as_dict(), 'time_range_hours': request.time_range_hours,
        'questions': {q: request.questions[q]['enabled'] for q in QUESTIONS}, 'conditions': request.conditions.as_dict()}


def identity(context):
    read_verified_tables(context.table_paths, context.request.inputs.table_hashes)
    return content_id({'settings': input_settings(context.request),
        'libraries': {name: library_version(name) for name in ['numpy', 'pandas']},
        'code': {name: file_hash(source_file(name)) for name in
            ['coordination_inputs.py', 'coordination_options.py', 'relationship_inputs.py', 'contracts.py', 'screening.py']}})


def observations(cell, measurement, source, bounds):
    """Never reorder a physical clock to conceal a reset or interpolate a gap."""
    endpoint = EndpointKey(cell, measurement)
    scoped = source.stem.eq(cell.movie)
    if 'identity' in measurement.grain: scoped &= source.identity.eq(cell.identity)
    positions = np.flatnonzero(scoped)
    chosen = source.iloc[positions].copy(); chosen['_source_position'] = positions
    trace = bool({'hours', 'frame_index'} & set(measurement.grain))
    metadata = {**cell.as_dict(), 'measurement': measurement.column, 'measurement_id': measurement.record_id,
        'endpoint_id': endpoint.record_id, 'table': measurement.table, 'unit': measurement.unit,
        'label': measurement.label, 'kind': 'trace' if trace else 'scalar', 'source_observations': len(chosen),
        'source_scope': 'cell' if 'identity' in measurement.grain else 'recording_reference'}
    if not trace:
        value = chosen[measurement.column].to_numpy(dtype=float, na_value=np.nan)[0] if len(chosen) else np.nan
        scalar = {**metadata, 'value': value, 'value_kind': _kind(value)}
        return pd.DataFrame(columns=TRACE_COLUMNS), {**metadata, 'clock_valid': None,
            'valid_observations': int(np.isfinite(value)), 'status': 'available' if np.isfinite(value) else 'missing',
            'reason': 'Original whole-recording scalar; no temporal repetition or invented window'}, scalar
    if 'frame_index' in chosen: chosen = chosen.sort_values('frame_index', kind='stable')
    hours = chosen.hours.to_numpy(dtype=float, na_value=np.nan)
    raw = chosen[measurement.column].to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(hours)
    # A source with hour-only grain has no separate acquisition order; preserve
    # source row order and require the recorded finite clock to increase.
    clock_valid = bool(np.all(np.diff(hours[finite]) > 0))
    if 'frame_index' in chosen:
        frames = chosen.frame_index.to_numpy(dtype=float, na_value=np.nan)
        clock_valid = clock_valid and bool(np.all(np.isfinite(frames) & (frames >= 0) & (frames == np.floor(frames))))
    selected = finite.copy()
    if bounds is not None: selected &= (hours >= bounds[0]) & (hours < bounds[1])
    valid = selected & np.isfinite(raw) & clock_valid
    rows = []
    for ordinal, (_, row) in enumerate(chosen.iterrows()):
        row_key = {key: _json_value(row[key]) for key in measurement.grain}
        rows.append({**cell.as_dict(), 'measurement': measurement.column, 'measurement_id': measurement.record_id,
            'endpoint_id': endpoint.record_id, 'table': measurement.table,
            'observation_id': content_id({'endpoint': endpoint.record_id, 'row': row_key}),
            'source_observation_id': content_id({'source_run': cell.source_run, 'movie': cell.movie, 'measurement': measurement.record_id, 'row': row_key}),
            'source_scope': metadata['source_scope'],
            'source_position': int(row['_source_position']), 'sequence_index': ordinal, 'frame_index': row.get('frame_index'),
            'hours': hours[ordinal], 'raw_value': raw[ordinal], 'raw_kind': _kind(raw[ordinal]), 'time_kind': _kind(hours[ordinal]),
            'within_range': bool(selected[ordinal]), 'raw_valid': bool(np.isfinite(raw[ordinal])), 'clock_valid': clock_valid,
            # The matching adapter's field names are reused for ORIGINAL values.
            # Representation changes belong to the next scientific producer.
            'processed_value': raw[ordinal] if valid[ordinal] else np.nan, 'processed_valid': bool(valid[ordinal])})
    observed = hours[valid]
    status = 'invalid_clock' if not clock_valid else 'available' if len(observed) else 'missing'
    return pd.DataFrame(rows, columns=TRACE_COLUMNS), {**metadata, 'clock_valid': clock_valid,
        'valid_observations': len(observed), 'invalid_time_observations': int((~finite).sum()),
        'start_hours': float(observed[0]) if len(observed) else None, 'end_hours': float(observed[-1]) if len(observed) else None,
        'span_hours': float(observed[-1]-observed[0]) if len(observed) else 0., 'status': status,
        'reason': 'Original recorded observations' if clock_valid else 'Nonincreasing clock or invalid frame key; original rows retained, temporal matching withheld'}, None


def geometry(cell, x, y, settings):
    """Combine coordinate observations only at their exact recorded time/key."""
    records = {}
    for axis, frame in [('x', x), ('y', y)]:
        for row in frame.to_dict('records'):
            # Missing-time coordinates never match across tables just because
            # both are missing; their own observation identity keeps them apart.
            key = (row['frame_index'], row['hours']) if np.isfinite(row['hours']) else (axis, row['observation_id'])
            records.setdefault(key, {})[axis] = row
    scales = settings['recording_scales'].get(cell.movie, settings['scale'])
    geometry_id = content_id({'cell': cell, 'settings': settings.as_dict()})
    rows = []
    for record in records.values():
        first = record.get('x') or record['y']; a, b = record.get('x'), record.get('y')
        raw_x = a['raw_value'] if a else np.nan; raw_y = b['raw_value'] if b else np.nan
        vx, vy = raw_x*scales[0], raw_y*scales[1]
        valid = bool(a and b and a['processed_valid'] and b['processed_valid'])
        inside = edge = None
        if settings['field'] and np.isfinite(vx) and np.isfinite(vy):
            xmin, xmax, ymin, ymax = settings['field']['bounds']; margin = settings['field']['edge_margin']
            inside = bool(xmin <= vx <= xmax and ymin <= vy <= ymax)
            edge = bool(inside and min(vx-xmin, xmax-vx, vy-ymin, ymax-vy) <= margin)
            valid &= inside
        rows.append({**cell.as_dict(), 'geometry_id': geometry_id,
            'observation_id': content_id({'geometry': geometry_id, 'x': a['observation_id'] if a else None, 'y': b['observation_id'] if b else None}),
            'frame_index': first['frame_index'], 'hours': first['hours'], 'raw_x': raw_x, 'raw_y': raw_y, 'x': vx, 'y': vy,
            'x_observation_id': a['observation_id'] if a else None, 'y_observation_id': b['observation_id'] if b else None,
            'x_sequence_index': a['sequence_index'] if a else None, 'y_sequence_index': b['sequence_index'] if b else None,
            'x_scale': scales[0], 'y_scale': scales[1], 'input_unit': settings['input_unit'], 'unit': settings['unit'],
            'definition': settings['definition'], 'within_range': bool(first['within_range']), 'valid': valid,
            'inside_field': inside, 'near_field_edge': edge, 'reason': 'Recorded contemporaneous coordinate pair' if valid else
                'Outside declared measured field' if inside is False else 'Missing, unmatched or temporally invalid coordinate observation'})
    return pd.DataFrame(rows, columns=GEOMETRY_COLUMNS)


def _adjacent(previous, current, max_gap):
    dt = float(current['hours']-previous['hours'])
    if not 0 < dt <= max_gap: return False, 'Invalid or excessive elapsed time'
    a, b = previous.get('frame_index'), current.get('frame_index')
    if a is not None and b is not None and np.isfinite(a) and np.isfinite(b) and b-a != 1:
        return False, 'Missing original frame between observations'
    for key in ['sequence_index', 'x_sequence_index', 'y_sequence_index']:
        a, b = previous.get(key), current.get(key)
        if a is not None and b is not None and b-a != 1:
            return False, 'Missing or invalid original observation between matched points'
    return True, 'Adjacent observed original positions'


def _matched(left, right, settings):
    matched, diagnostics = match_observations(left, right, 0., settings)
    lookup = {row['observation_id']: row for frame in [left, right] for row in frame.to_dict('records')}
    rows, intervals = [], []; previous = None; segment = 0
    for row in matched.to_dict('records'):
        a, b = lookup[row['reference_observation']], lookup[row['target_observation']]
        if previous is not None:
            pa, pb = lookup[previous['reference_observation']], lookup[previous['target_observation']]
            a_ok, a_reason = _adjacent(pa, a, settings['max_gap_hours']); b_ok, b_reason = _adjacent(pb, b, settings['max_gap_hours'])
            start, end = max(pa['hours'], pb['hours']), min(a['hours'], b['hours'])
            valid = bool(a_ok and b_ok and end > start)
            if not valid: segment += 1
            intervals.append({'first_reference_observation': pa['observation_id'], 'last_reference_observation': a['observation_id'],
                'first_target_observation': pb['observation_id'], 'last_target_observation': b['observation_id'],
                'start_hours': start, 'end_hours': end, 'jointly_observed_hours': float(end-start) if valid else 0.,
                'valid': valid, 'reason': 'Jointly supported original interval' if valid else a_reason if not a_ok else b_reason if not b_ok else 'No common interval',
                'segment': segment})
        rows.append({**row, 'segment': segment, 'reference_frame_index': a['frame_index'], 'target_frame_index': b['frame_index']})
        previous = row
    return rows, intervals, paired_support(matched, diagnostics, settings)


def joint_intervals(left, right, settings):
    """Intersect native supported intervals, independently of timestamp pairing.

    Staggered acquisition grids can overlap continuously even when no exact
    observation times match. This records exposure, not interpolated values.
    """
    spans = []; domains = []
    for frame in [left, right]:
        if frame.empty or ('clock_valid' in frame and not frame.clock_valid.all()): return []
        selected = frame.loc[frame.within_range & np.isfinite(frame.hours)].sort_values('hours')
        if selected.empty: return []
        domains.append((float(selected.hours.min()), float(selected.hours.max())))
        rows = selected.to_dict('records'); supported = []
        for a, b in zip(rows, rows[1:]):
            if a['processed_valid'] and b['processed_valid'] and _adjacent(a, b, settings['max_gap_hours'])[0]:
                supported.append((a, b))
        spans.append(supported)
    start, end = max(d[0] for d in domains), min(d[1] for d in domains)
    if end <= start: return []
    i = j = 0; observed = []
    while i < len(spans[0]) and j < len(spans[1]):
        a, b = spans[0][i]; c, d = spans[1][j]
        low, high = max(a['hours'], c['hours']), min(b['hours'], d['hours'])
        if high > low:
            observed.append({'start_hours': low, 'end_hours': high, 'jointly_observed_hours': float(high-low), 'valid': True,
                'first_reference_observation': a['observation_id'], 'last_reference_observation': b['observation_id'],
                'first_target_observation': c['observation_id'], 'last_target_observation': d['observation_id'],
                'reason': 'Intersection of original supported observation intervals; no values interpolated'})
        if b['hours'] <= d['hours']: i += 1
        else: j += 1
    complete = []; cursor = start
    def gap(a, b):
        return {'start_hours': a, 'end_hours': b, 'jointly_observed_hours': 0., 'valid': False,
            'first_reference_observation': None, 'last_reference_observation': None,
            'first_target_observation': None, 'last_target_observation': None,
            'reason': 'Unobserved interval: missing or invalid original observation, missing frame, field support or excessive gap'}
    for row in observed:
        if row['start_hours'] > cursor: complete.append(gap(cursor, row['start_hours']))
        complete.append(row); cursor = row['end_hours']
    if cursor < end: complete.append(gap(cursor, end))
    return complete


def moving_neighbours(resolved, positions):
    """Geometry-only visibility and neighbour membership at original pair times."""
    request = resolved.request; rows = []
    for movie in sorted({cell.movie for cell in resolved.inputs.cells}):
        members = [cell for cell in resolved.inputs.cells if cell.movie == movie]
        for a, b in combinations(members, 2):
            left, right = [positions[(movie, cell.identity)] for cell in [a, b]]
            matched, _, _ = _matched(left.assign(processed_value=left.x, processed_valid=left.valid),
                right.assign(processed_value=right.x, processed_valid=right.valid), request.support)
            lookup = {row['observation_id']: row for frame in [left, right] for row in frame.to_dict('records')}
            pair_id = content_id({'cells': [a, b], 'geometry': request.geometry.as_dict()})
            for row in matched:
                pa, pb = lookup[row['reference_observation']], lookup[row['target_observation']]
                rows.append({'source_run': a.source_run, 'movie': movie, 'geometry_pair_id': pair_id,
                    'reference_identity': a.identity, 'target_identity': b.identity, **row,
                    'distance': float(np.hypot(pa['x']-pb['x'], pa['y']-pb['y'])), 'unit': request.geometry['unit']})
    visible = {}
    for row in rows:
        for endpoint in ['reference', 'target']:
            visible.setdefault(row[endpoint+'_observation'], []).append(row['distance'])
    rule = request.geometry['neighbourhood']; lookup = {}
    for row in rows:
        a, b = row['reference_observation'], row['target_observation']; distance = row['distance']
        if rule['method'] == 'all': near = True
        elif rule['method'] == 'radius': near = distance <= rule['distance']
        else:
            thresholds = [sorted(visible[point])[min(rule['neighbours'], len(visible[point]))-1] for point in [a, b]]
            near = distance <= max(thresholds)
        row.update(near=near, reference_visible_cells=len(visible[a]), target_visible_cells=len(visible[b]),
            membership_meaning='Observed geometry-only membership at the two saved matched times; unobserved neighbours remain unknown')
        lookup[tuple(sorted((a, b)))] = row
    return rows, lookup


def prepare(resolved, tables):
    request = resolved.request; frames = []; metadata = []; scalars = []; geometries = []; endpoints = {}; by_meta = {}; positions = {}
    measurements = {m.record_id: m for m in (*resolved.reference_measurements, *resolved.target_measurements, *resolved.coordinates, *resolved.reference_columns)}
    cells = []
    for cell in resolved.inputs.cells:
        cells.append(cell.as_dict())
        for measurement in measurements.values():
            frame, info, scalar = observations(cell, measurement, tables[measurement.table], request.time_range_hours)
            frames.append(frame); metadata.append(info)
            if scalar is not None: scalars.append(scalar)
            endpoints[(cell.movie, cell.identity, measurement.record_id)] = frame
            by_meta[(cell.movie, cell.identity, measurement.record_id)] = info
        x, y = [endpoints[(cell.movie, cell.identity, m.record_id)] for m in resolved.coordinates]
        points = geometry(cell, x, y, request.geometry); geometries.append(points); positions[(cell.movie, cell.identity)] = points
    layouts = []; by_layout = {}
    for cell in resolved.inputs.cells:
        valid = positions[(cell.movie, cell.identity)]; valid = valid.loc[valid.valid]
        layout = {**cell.as_dict(), 'valid_positions': len(valid), 'x': float(valid.x.median()) if len(valid) else None,
            'y': float(valid.y.median()) if len(valid) else None, 'unit': request.geometry['unit'], 'definition': request.geometry['definition'],
            'position_meaning': 'Coordinate-wise median over this cell recorded valid positions in the declared window; not its position at one shared time',
            'first_hours': float(valid.hours.min()) if len(valid) else None, 'last_hours': float(valid.hours.max()) if len(valid) else None,
            'field_known': bool(request.geometry['field']), 'edge_observations': int(valid.near_field_edge.eq(True).sum()) if len(valid) else 0}
        layouts.append(layout); by_layout[(cell.movie, cell.identity)] = layout
    # Geometry-only ranks, evaluated once before reading any pair measurement.
    distances = {}; neighbours = {}; rule = request.geometry['neighbourhood']
    for movie in sorted({cell.movie for cell in resolved.inputs.cells}):
        members = [cell for cell in resolved.inputs.cells if cell.movie == movie]
        for a, b in combinations(members, 2):
            pa, pb = by_layout[(movie, a.identity)], by_layout[(movie, b.identity)]
            distance = float(np.hypot(pa['x']-pb['x'], pa['y']-pb['y'])) if pa['x'] is not None and pb['x'] is not None else None
            distances[(movie, min(a.identity, b.identity), max(a.identity, b.identity))] = distance
        for cell in members:
            available = sorted(d for (m, a, b), d in distances.items() if m == movie and cell.identity in (a, b) and d is not None)
            neighbours[(movie, cell.identity)] = available[min(rule.get('neighbours', 1), len(available))-1] if available else None
    moving_rows, moving_lookup = moving_neighbours(resolved, positions)
    inventory = []; matches = []; intervals = []; matching_intervals = []; pair_positions = []; geometry_intervals = []; support = []; pair_definitions = []
    samples = {sample.movie: sample for sample in resolved.inputs.samples}
    reference = {m.column: m for m in resolved.reference_measurements}; target = {m.column: m for m in resolved.target_measurements}
    seen = set()
    for movie in sorted(samples):
        members = [cell for cell in resolved.inputs.cells if cell.movie == movie]
        for pair in request.measurement_pairs:
            ma, mb = reference[pair['reference']], target[pair['target']]
            oriented = ma != mb or request.questions['delay']['enabled']
            iterator = combinations(members, 2) if ma == mb else permutations(members, 2)
            for a, b in iterator:
                key = CoordinationPairKey(EndpointKey(a, ma), EndpointKey(b, mb), oriented)
                if key.record_id in seen: continue
                seen.add(key.record_id); pair_definitions.append({'pair_id': key.record_id, 'key': key.as_dict()})
                sample = samples[movie]
                base = {'source_run': a.source_run, 'movie': movie, 'pair_id': key.record_id,
                    'reference_identity': a.identity, 'target_identity': b.identity, 'reference': ma.column, 'target': mb.column,
                    'reference_measurement_id': ma.record_id, 'target_measurement_id': mb.record_id,
                    'reference_endpoint_id': key.reference.record_id, 'target_endpoint_id': key.target.record_id, 'oriented': oriented,
                    'sample': sample.sample, 'sample_confirmed': sample.confirmed, 'condition': request.conditions.get(movie)}
                left, right = [endpoints[(movie, cell.identity, measurement.record_id)] for cell, measurement in [(a, ma), (b, mb)]]
                rows, aligned_spans, temporal = _matched(left, right, request.support)
                spans = joint_intervals(left, right, request.support)
                matches.extend({**base, **row} for row in rows); intervals.extend({**base, **row} for row in spans)
                matching_intervals.extend({**base, **{key: value for key, value in row.items() if key != 'jointly_observed_hours'},
                    'matched_interval_hours': row['jointly_observed_hours']} for row in aligned_spans)
                ga, gb = [positions[(movie, cell.identity)] for cell in [a, b]]
                def as_trace(frame):
                    return frame.assign(processed_value=frame.x, processed_valid=frame.valid)
                # Geometry uses one canonical cell orientation regardless of
                # which measurement is assigned the scientific reference role.
                geo_rows, _, geo_support = _matched(as_trace(ga if a.identity < b.identity else gb),
                    as_trace(gb if a.identity < b.identity else ga), request.support)
                if a.identity > b.identity:
                    geo_rows = [{**row, **{endpoint+'_'+suffix: row[other+'_'+suffix]
                        for endpoint, other in [('reference', 'target'), ('target', 'reference')]
                        for suffix in ['observation', 'hours', 'value', 'frame_index']},
                        'matching_error_hours': -row['matching_error_hours']} for row in geo_rows]
                    geo_support = paired_support(pd.DataFrame(geo_rows, columns=MATCH_COLUMNS),
                        {'ambiguous_observations': geo_support['ambiguous_observations']}, request.support)
                geo_spans = joint_intervals(as_trace(ga), as_trace(gb), request.support)
                geo_lookup = {row['observation_id']: row for frame in [ga, gb] for row in frame.to_dict('records')}
                for row in geo_rows:
                    pa, pb = geo_lookup[row['reference_observation']], geo_lookup[row['target_observation']]
                    moving = moving_lookup[tuple(sorted((row['reference_observation'], row['target_observation'])))]
                    pair_positions.append({**base, **row, 'reference_x': pa['x'], 'reference_y': pa['y'], 'target_x': pb['x'], 'target_y': pb['y'],
                        'distance': float(np.hypot(pa['x']-pb['x'], pa['y']-pb['y'])), 'unit': request.geometry['unit'],
                        'near': moving['near'], 'geometry_pair_id': moving['geometry_pair_id'],
                        'reference_visible_cells': moving['reference_visible_cells'] if a.identity < b.identity else moving['target_visible_cells'],
                        'target_visible_cells': moving['target_visible_cells'] if a.identity < b.identity else moving['reference_visible_cells'],
                        'reference_edge': pa['near_field_edge'], 'target_edge': pb['near_field_edge'],
                        'distance_meaning': 'Distance between recorded coordinate points at the two saved matched times'})
                geometry_intervals.extend({**base, **row} for row in geo_spans)
                distance = distances[(movie, min(a.identity, b.identity), max(a.identity, b.identity))]
                near = None
                if distance is not None:
                    near = True if rule['method'] == 'all' else distance <= rule['distance'] if rule['method'] == 'radius' else (
                        distance <= neighbours[(movie, a.identity)] or distance <= neighbours[(movie, b.identity)])
                eligible = rule['method'] == 'all' or rule.get('comparison') == 'retain_distant' or near is True
                inventory.append({**base, 'static_distance': distance, 'distance_unit': request.geometry['unit'],
                    'static_distance_meaning': 'Distance between coordinate-wise median cell positions; not a detected contact or persistent proximity',
                    'near': near, 'geometry_population_eligible': eligible, 'geometry_points': len(geo_rows),
                    'paired_observations': len(rows), 'jointly_observed_hours': sum(row['jointly_observed_hours'] for row in spans),
                    'geometry_observed_hours': sum(row['jointly_observed_hours'] for row in geo_spans)})
                infos = [by_meta[(movie, cell.identity, measurement.record_id)] for cell, measurement in [(a, ma), (b, mb)]]
                for question in QUESTIONS:
                    enabled = request.questions[question]['enabled']; state, reason = temporal['status'], temporal['reason']
                    if question == 'characteristics':
                        enough = all(info['status'] == 'available' and (info['kind'] == 'scalar' or (
                            info['valid_observations'] >= request.support['min_observations'] and info['span_hours'] >= request.support['min_span_hours'])) for info in infos)
                        state = 'eligible' if enough and geo_support['status'] == 'eligible' else 'insufficient'
                        reason = ('Original scalar/trace support and overlapping measured geometry; summary definition is evaluated in the scientific producer'
                            if state == 'eligible' else 'Insufficient original characteristic observations or overlapping measured geometry')
                    elif question == 'proximity' and geo_support['status'] != 'eligible':
                        state, reason = 'insufficient_geometry', 'Too few contemporaneous recorded position pairs'
                    if any(info['status'] == 'invalid_clock' for info in infos):
                        state, reason = 'invalid_clock', 'One or both endpoint physical clocks are invalid'
                    if not eligible: state, reason = 'outside_geometry_population', 'Excluded by the declared geometry-only population rule'
                    if not enabled: state, reason = 'disabled', 'Question explicitly disabled'
                    support.append({**base, 'question': question, 'requested': enabled, **temporal, 'status': state, 'reason': reason,
                        'geometry_status': geo_support['status'], 'method_eligibility_established': False})
    frames = [frame for frame in frames if not frame.empty]; geometries = [frame for frame in geometries if not frame.empty]
    outputs = {'cells': _table(cells, KEYS), 'endpoints': _table(metadata, KEYS+['measurement', 'measurement_id', 'endpoint_id', 'table', 'unit', 'kind', 'status', 'valid_observations']),
        'traces': pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TRACE_COLUMNS),
        'scalars': _table(scalars, KEYS+['measurement_id', 'endpoint_id', 'value', 'value_kind']),
        'geometry': pd.concat(geometries, ignore_index=True) if geometries else pd.DataFrame(columns=GEOMETRY_COLUMNS),
        'layouts': _table(layouts, KEYS+['x', 'y', 'valid_positions', 'unit', 'definition', 'position_meaning']),
        'inventory': _table(inventory, PAIR_KEYS+['static_distance', 'distance_unit', 'static_distance_meaning', 'near', 'geometry_population_eligible',
            'geometry_points', 'paired_observations', 'jointly_observed_hours', 'geometry_observed_hours']),
        'matches': _table(matches, PAIR_KEYS+['reference_observation', 'target_observation', 'reference_hours', 'target_hours',
            'reference_value', 'target_value', 'matching_error_hours', 'segment', 'reference_frame_index', 'target_frame_index']),
        'intervals': _table(intervals, PAIR_KEYS+['start_hours', 'end_hours', 'jointly_observed_hours', 'valid', 'reason']),
        'matching_intervals': _table(matching_intervals, PAIR_KEYS+['start_hours', 'end_hours', 'matched_interval_hours', 'valid', 'reason']),
        'pair_positions': _table(pair_positions, PAIR_KEYS+['reference_hours', 'target_hours', 'reference_observation', 'target_observation',
            'reference_x', 'reference_y', 'target_x', 'target_y', 'distance', 'unit', 'segment']),
        'geometry_intervals': _table(geometry_intervals, PAIR_KEYS+['start_hours', 'end_hours', 'jointly_observed_hours', 'valid', 'reason']),
        'moving_neighbours': _table(moving_rows, ['source_run', 'movie', 'geometry_pair_id', 'reference_identity', 'target_identity',
            'reference_observation', 'target_observation', 'reference_hours', 'target_hours', 'distance', 'near', 'reference_visible_cells', 'target_visible_cells']),
        'support': _table(support, PAIR_KEYS+['question', 'requested', 'status', 'reason', 'paired_observations', 'overlap_span_hours',
            'geometry_status', 'method_eligibility_established'])}
    return outputs, pair_definitions


def produce(context):
    tables = read_verified_tables(context.table_paths, context.request.inputs.table_hashes)
    outputs, definitions = prepare(context.request, tables)
    context.output.mkdir(parents=True); refs = []
    for name, frame in outputs.items():
        path = context.output/(name+'.json'); path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id, columns=tuple(frame.columns)))
    for name, data in {'pair_definitions': definitions, 'provenance': {'schema_version': 1, 'scientific_id': context.scientific_id,
        'design_id': context.saved('coordination-design').outcome.scientific_id, 'resolved_request': context.request.as_dict(),
        'input_settings': input_settings(context.request), 'values': 'Original measured values only; no transformations, relationship tests or optional artifact reads',
        'time_window': 'Start inclusive, end exclusive; no observations beyond the recorded endpoints',
        'joint_time': 'Intersection of native adjacent observed intervals independently of aligned point matches; no values interpolated',
        'calibration': 'Explicit declared per-axis conversion of recorded coordinates, with per-recording overrides; field bounds use output units',
        'nearest_neighbours': 'Either endpoint within the other endpoint k-nearest distance threshold; distance ties all retained',
        'population_vs_moving_neighbours': 'A radius/nearest population restriction uses static median positions; separate moving membership uses actual matched positions and never changes that population silently',
        'scientific_tests_performed': False}}.items():
        path = context.output/(name+'.json'); _write_json(path, data)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id))
    return StepResult(context.step.name, context.scientific_id, 'completed',
        'Saved complete within-recording pairs, original measurements, calibrated geometry and gap-respecting observation support', tuple(refs))


def read_inputs(saved):
    import json
    provenance = read_document(saved.artifact('provenance'))
    if provenance.get('schema_version') != 1 or provenance.get('scientific_id') != saved.outcome.scientific_id:
        raise ValueError('Coordination input identity or schema mismatch')
    names = ['cells', 'endpoints', 'traces', 'scalars', 'geometry', 'layouts', 'inventory', 'matches', 'intervals',
        'matching_intervals', 'pair_positions', 'geometry_intervals', 'moving_neighbours', 'support']
    result = {name: read_table(saved.artifact(name)) for name in names}
    if result['inventory'].pair_id.duplicated().any(): raise ValueError('Duplicate full pair identities')
    if result['traces'].observation_id.duplicated().any(): raise ValueError('Duplicate original endpoint observation identities')
    result['pair_definitions'] = read_document(saved.artifact('pair_definitions'))
    definitions = {row['pair_id']: row['key'] for row in result['pair_definitions']}
    if len(definitions) != len(result['pair_definitions']) or set(definitions) != set(result['inventory'].pair_id):
        raise ValueError('Full pair definitions do not match the complete inventory')
    expected = {}
    for pair_id, key in definitions.items():
        if content_id(key) != pair_id: raise ValueError('Full endpoint definition has changed')
        a, b = key['reference'], key['target']
        if (a['cell']['source_run'], a['cell']['movie']) != (b['cell']['source_run'], b['cell']['movie']) or a['cell'] == b['cell']:
            raise ValueError('Saved pair crosses a recording or refers to one cell twice')
        expected[pair_id] = {'source_run': a['cell']['source_run'], 'movie': a['cell']['movie'],
            'reference_identity': a['cell']['identity'], 'target_identity': b['cell']['identity'],
            'reference': a['measurement']['column'], 'target': b['measurement']['column'],
            'reference_measurement_id': content_id(a['measurement']), 'target_measurement_id': content_id(b['measurement']),
            'reference_endpoint_id': content_id(a), 'target_endpoint_id': content_id(b), 'oriented': key['oriented']}
    for name in ['inventory', 'matches', 'intervals', 'matching_intervals', 'pair_positions', 'geometry_intervals', 'support']:
        for row in result[name].to_dict('records'):
            if row['pair_id'] not in expected or any(row[field] != value for field, value in expected[row['pair_id']].items()):
                raise ValueError(name + ': saved row does not match its exact endpoint pair')
    for name, source in [('matches', 'traces'), ('pair_positions', 'geometry')]:
        lookup = result[source].set_index('observation_id').to_dict('index')
        for row in result[name].to_dict('records'):
            for endpoint in ['reference', 'target']:
                observed = lookup.get(row[endpoint+'_observation'])
                if observed is None or observed['source_run'] != row['source_run'] or observed['movie'] != row['movie'] or observed['identity'] != row[endpoint+'_identity']:
                    raise ValueError(name + ': original observation does not belong to the recorded endpoint')
                if source == 'traces' and observed['endpoint_id'] != row[endpoint+'_endpoint_id']:
                    raise ValueError('Matched original observation belongs to another endpoint measurement')
                if observed['hours'] != row[endpoint+'_hours']:
                    raise ValueError(name + ': matched original time changed')
    result['provenance'] = provenance
    return result
