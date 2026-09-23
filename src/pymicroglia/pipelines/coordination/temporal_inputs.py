"""Saved temporal representations and explicitly constructed shared references."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pymicroglia.pipelines._contracts import content_id
from pymicroglia.pipelines.coordination.inputs import _adjacent, _table
from pymicroglia.pipelines.relationships.inputs import match_observations


ADJUSTMENT_MEANING = ('Association of explicitly reference-subtracted signals. Subtracting a noisy or population-derived '
    'reference can itself change signs or induce association; this is not a conditional-independence or causal test.')


def prepare_series(resolved, prepared):
    """Keep original levels alongside the requested representation, without tests."""
    request = resolved.request
    measurements = {m.record_id: m for m in (*resolved.reference_measurements, *resolved.target_measurements, *resolved.reference_columns)}
    views = ['raw'] + ([] if request.representation == 'raw' else [request.representation])
    groups = {endpoint: frame for endpoint, frame in prepared['traces'].groupby('endpoint_id', sort=False)}
    frames = {}; records = []; processing = []
    for endpoint in prepared['endpoints'].to_dict('records'):
        if endpoint['measurement_id'] not in measurements: continue
        original = groups.get(endpoint['endpoint_id'], prepared['traces'].iloc[:0]).sort_values('sequence_index')
        measurement = measurements[endpoint['measurement_id']]
        for view in views:
            settings = {'representation': view, 'detrending': resolved.detrending.as_dict() if view == 'detrended' else {},
                'increment': getattr(request, 'increment', 'difference') if view == 'changes' else None,
                'max_gap_hours': request.support['max_gap_hours']}
            series_id = content_id({'endpoint': endpoint['endpoint_id'], 'inputs': prepared['provenance']['scientific_id'], 'settings': settings})
            frame = original.copy(); frame['original_within_range'] = frame.within_range
            frame['representation'] = view; frame['adjustment'] = 'none'; frame['series_id'] = series_id
            frame['previous_observation_id'] = None; frame['increment_start_hours'] = np.nan; frame['increment_hours'] = np.nan
            frame['processing_reason'] = 'Original measured level'
            status, reason, workbench = endpoint['status'], endpoint['reason'], None
            if view == 'detrended':
                frame['processed_value'] = np.nan; frame['processed_valid'] = False
                if endpoint['status'] == 'invalid_clock':
                    status, reason = 'invalid_clock', 'Original clock invalid; no detrending performed'
                elif original.processed_valid.sum() < 2:
                    status, reason = 'insufficient', 'Fewer than two finite original values; no detrending performed'
                else:
                    import pymicroglia.workbench as circadian
                    selected = original.loc[original.within_range]
                    hours = selected.hours.to_numpy(dtype=float, na_value=np.nan)
                    values = selected.raw_value.to_numpy(dtype=float, na_value=np.nan); values[~np.isfinite(values)] = np.nan
                    try: workbench = circadian.detrend_trace(hours, values, resolved.detrending.as_dict())
                    except circadian.cw.WorkbenchInputError as error:
                        status, reason = 'processing_unavailable', str(error)
                    else:
                        clock = np.asarray(workbench['processed_trace']['hours'], dtype=float)
                        output = np.asarray(workbench['values'], dtype=float)
                        if (clock.shape != output.shape or len(clock) != len(hours) or
                                not np.allclose(clock, hours, rtol=0., atol=1e-9)):
                            raise ValueError('Workbench returned an invalid detrended observation clock')
                        lookup = dict(zip(clock, output))
                        for index, row in selected.iterrows():
                            value = lookup.get(float(row.hours), np.nan)
                            # An interpolated native value cannot turn an
                            # originally missing measurement into observed data.
                            if row.raw_valid and np.isfinite(value):
                                frame.loc[index, 'processed_value'] = value; frame.loc[index, 'processed_valid'] = True
                        reason = 'Workbench detrending mapped to exact original observed timestamps; original missing values remain missing'
                frame['processing_reason'] = reason
            elif view == 'changes':
                frame['processed_value'] = np.nan; frame['processed_valid'] = False
                rows = list(original.iterrows()); increment = settings['increment']
                for ordinal, (index, row) in enumerate(rows):
                    previous = rows[ordinal-1][1] if ordinal else None
                    if previous is None or not previous.within_range:
                        frame.loc[index, 'within_range'] = False
                        frame.loc[index, 'processing_reason'] = 'No preceding observation within the declared window to define an increment'
                        continue
                    frame.loc[index, 'previous_observation_id'] = previous.observation_id
                    frame.loc[index, 'increment_start_hours'] = previous.hours
                    frame.loc[index, 'increment_hours'] = row.hours-previous.hours
                    good, why = _adjacent(previous.to_dict(), row.to_dict(), request.support['max_gap_hours'])
                    if row.processed_valid and previous.processed_valid and good:
                        value = row.raw_value-previous.raw_value
                        if increment == 'rate': value /= row.hours-previous.hours
                        if np.isfinite(value):
                            frame.loc[index, 'processed_value'] = value; frame.loc[index, 'processed_valid'] = True
                        frame.loc[index, 'processing_reason'] = 'Original measured increment' if increment == 'difference' else 'Original measured increment divided by its observed elapsed hours'
                    else:
                        frame.loc[index, 'processing_reason'] = why if not good else 'Missing original value at one increment endpoint'
                reason = 'Signed original adjacent-observation increments; gaps and unknown endpoints retained'
            unit = measurement.unit
            if view == 'changes' and settings['increment'] == 'rate' and unit: unit += ' per hour'
            frame['value_unit'] = unit
            frames[(endpoint['endpoint_id'], view)] = frame
            record = {**endpoint, 'series_id': series_id, 'representation': view, 'adjustment': 'none', 'value_unit': unit,
                'valid_values': int(frame.processed_valid.sum()), 'analysis_positions': int(frame.within_range.sum()),
                'status': status, 'reason': reason, 'settings': settings,
                'source_scope': 'cell' if 'identity' in measurement.grain else 'recording_reference'}
            records.append(record); processing.append({**record, 'workbench_result': workbench})
    return frames, _table(records, ['series_id', 'endpoint_id', 'representation', 'adjustment', 'status', 'reason']), processing


def adjusted_pair(resolved, pair, frames, metadata):
    """Subtract a fixed-coefficient reference and save every original contributor."""
    request = resolved.request; settings = request.shared_reference; view = request.representation
    if settings['method'] == 'none': return {}, [], [], []
    meta = {row['endpoint_id']: row for row in metadata.to_dict('records') if row['representation'] == view}
    columns = {m.column: m for m in resolved.reference_columns}
    outputs = {}; summaries = []; memberships = []; definitions = []
    for role in ['reference', 'target']:
        owner_id = pair[role+'_endpoint_id']; original = frames[(owner_id, view)]; owner = meta[owner_id]
        if settings['method'] == 'leave_pair_out_mean':
            donors = [row for row in meta.values() if row['movie'] == pair['movie'] and row['measurement_id'] == owner['measurement_id']
                and row['identity'] not in {pair['reference_identity'], pair['target_identity']}]
            minimum = settings['min_cells']
        else:
            external = columns[settings['measurements'][owner['measurement']]]
            donors = [row for row in meta.values() if row['movie'] == pair['movie'] and row['identity'] == owner['identity']
                and row['measurement_id'] == external.record_id]
            minimum = 1
        series_id = content_id({'pair': pair['pair_id'], 'endpoint_series': owner['series_id'], 'shared_reference': settings.as_dict(),
            'donor_series': [row['series_id'] for row in donors]})
        frame = original.copy(); frame['series_id'] = series_id; frame['adjustment'] = 'shared_reference'
        frame['processed_value'] = np.nan; frame['processed_valid'] = False
        frame['reference_value'] = np.nan; frame['reference_members'] = 0; frame['reference_status'] = 'insufficient'
        frame['pair_id'] = pair['pair_id']; frame['endpoint_role'] = role
        # Align the reference on the owner's recorded CLOCK, including times
        # where the owner's own signal is missing. No signal value is invented.
        clock = original.assign(processed_value=original.hours,
            processed_valid=original.within_range & original.clock_valid & np.isfinite(original.hours.to_numpy(dtype=float, na_value=np.nan)))
        contributors = {}
        for donor in donors:
            source = frames[(donor['endpoint_id'], view)]
            matched, _ = match_observations(clock, source, 0., request.support)
            source_rows = source.set_index('observation_id').to_dict('index')
            for row in matched.to_dict('records'):
                origin = source_rows[row['target_observation']]
                contributor = {'pair_id': pair['pair_id'], 'adjusted_series_id': series_id, 'endpoint_role': role,
                    'owner_observation_id': row['reference_observation'], 'owner_hours': row['reference_hours'],
                    'donor_series_id': donor['series_id'], 'donor_endpoint_id': donor['endpoint_id'],
                    'donor_observation_id': row['target_observation'], 'donor_source_observation_id': origin.get('source_observation_id', row['target_observation']),
                    'donor_source_run': donor['source_run'], 'donor_movie': donor['movie'],
                    'donor_identity': donor['identity'] if donor['source_scope'] == 'cell' else None,
                    'donor_source_scope': donor['source_scope'], 'donor_measurement': donor['measurement'], 'donor_table': donor['table'],
                    'donor_hours': row['target_hours'], 'donor_value': row['target_value'], 'matching_error_hours': row['matching_error_hours']}
                contributors.setdefault(row['reference_observation'], []).append(contributor); memberships.append(contributor)
        for index, row in original.iterrows():
            members = contributors.get(row.observation_id, []); available = len(members) >= minimum
            reference_value = float(np.mean([member['donor_value'] for member in members])) if available else None
            reference_status = 'available' if available else 'insufficient'
            frame.loc[index, 'reference_members'] = len(members); frame.loc[index, 'reference_status'] = reference_status
            if reference_value is not None:
                frame.loc[index, 'reference_value'] = reference_value
                if row.processed_valid and np.isfinite(row.processed_value-reference_value):
                    frame.loc[index, 'processed_value'] = row.processed_value-reference_value; frame.loc[index, 'processed_valid'] = True
            summaries.append({'pair_id': pair['pair_id'], 'adjusted_series_id': series_id, 'endpoint_role': role,
                'owner_endpoint_id': owner_id, 'owner_observation_id': row.observation_id, 'hours': row.hours,
                'reference_value': reference_value, 'members': len(members), 'required_members': minimum,
                'member_series_ids': [member['donor_series_id'] for member in members], 'status': reference_status,
                'owner_value_available': bool(row.processed_valid), 'method': settings['method'], 'adjustment': 'subtract',
                'reason': 'Declared reference available at the original owner timestamp' if available else 'Insufficient observed reference contributors at this timestamp'})
        outputs[role] = frame
        definitions.append({**owner, 'pair_id': pair['pair_id'], 'endpoint_role': role, 'series_id': series_id,
            'adjustment': 'shared_reference', 'valid_values': int(frame.processed_valid.sum()),
            'status': owner['status'] if owner['status'] not in {'available', 'missing'} else 'available' if frame.processed_valid.any() else 'insufficient',
            'reason': 'Explicit reference-subtracted observations; missing reference support remains missing',
            'reference_settings': settings.as_dict(), 'reference_population': [row['series_id'] for row in donors],
            'interpretation': ADJUSTMENT_MEANING})
    return outputs, summaries, memberships, definitions
