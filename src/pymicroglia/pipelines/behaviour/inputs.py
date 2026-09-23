"""Freeze original state observations, protected groups and learning membership."""
from __future__ import annotations
from pymicroglia._sources import source_file

from importlib.metadata import version as library_version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pymicroglia.pipelines.behaviour.options import ObservationKey, feature_settings
from pymicroglia.pipelines._contracts import ArtifactRef, CellKey, Settings, StepResult, content_id
from pymicroglia.pipelines.relationships.inputs import KEYS, TRACE_COLUMNS, _prepare_trace
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_verified_tables, write_table


OBS_COLUMNS = KEYS + ['observation_id', 'frame_index', 'hours', 'time_status', 'within_range',
    'observed_features', 'feature_fraction', 'learning_feature_eligible', 'assignment_feature_eligible',
    'status', 'reason', 'vector_id', 'sample', 'sample_confirmed', 'condition', 'group', 'group_id',
    'role', 'learning_selected', 'previous_observation_id', 'elapsed_hours', 'adjacent', 'adjacency_reason']


def implementation_version():
    from pymicroglia.pipelines._screening import producer_identity
    return content_id({'code': {p.name: file_hash(p) for p in (Path(__file__),
        source_file('behaviour_options.py'), source_file('relationship_inputs.py'))},
        'libraries': {name: library_version(name) for name in ('numpy', 'pandas', 'scikit-learn')},
        'native_processing': producer_identity()})


def partitions(resolved):
    """Assign complete groups before looking at values or fitting preprocessing."""
    request = resolved.request
    samples = {sample.movie: sample for sample in resolved.inputs.samples}
    unit = request.validation['unit']
    cells = []
    for cell in resolved.inputs.cells:
        sample = samples[cell.movie]
        group = sample.sample if unit == 'biological_sample' and sample.confirmed else cell.movie if unit == 'recording' else None
        cells.append({**cell.as_dict(), 'sample': sample.sample, 'sample_confirmed': sample.confirmed,
            'condition': request.conditions.get(cell.movie), 'group': group,
            'group_id': content_id({'source_run': cell.source_run, 'unit': unit, 'group': group}) if group is not None else None,
            'role': 'assignment_only', 'reason': 'Unconfirmed biological sample' if group is None else 'Group not assigned to a learning or validation role'})
    groups = sorted({cell['group'] for cell in cells if cell['group'] is not None})
    split = request.validation['split']
    roles, problem = {}, None
    if split['method'] == 'explicit':
        roles = split['roles']
        unknown = set(roles) - set(groups)
        if unknown: raise ValueError('Validation roles name absent groups: ' + ', '.join(sorted(unknown)))
    elif len(groups) < 3:
        problem = 'Fewer than three complete groups are available for protected learning, development and confirmation'
    else:
        from sklearn.model_selection import GroupShuffleSplit
        indices = np.arange(len(groups))
        try:
            initial = GroupShuffleSplit(n_splits=1, test_size=split['confirmation_fraction'], random_state=split['seed'])
            remainder, confirm = next(initial.split(indices, groups=groups))
            secondary = GroupShuffleSplit(n_splits=1,
                test_size=split['development_fraction'] / (1 - split['confirmation_fraction']), random_state=split['seed'])
            train, develop = next(secondary.split(remainder, groups=np.asarray(groups)[remainder]))
        except ValueError as error:
            problem = 'Requested grouped fractions cannot leave all three protected roles: ' + str(error)
        else:
            for role, chosen in (('learning', remainder[train]), ('development', remainder[develop]), ('confirmation', confirm)):
                roles.update({groups[index]: role for index in chosen})
    for cell in cells:
        if cell['group'] in roles:
            cell.update(role=roles[cell['group']], reason='Declared complete-group validation role')
    records = pd.DataFrame(cells, columns=KEYS + ['sample', 'sample_confirmed', 'condition', 'group', 'group_id', 'role', 'reason'])
    counts = {role: len({cell['group'] for cell in cells if cell['role'] == role}) for role in ('learning', 'development', 'confirmation')}
    enough = all(counts[role] >= required for role, required in request.validation['min_groups'].items())
    design = {'unit': unit, 'group_counts': counts, 'required_group_counts': request.validation['min_groups'],
        'status': 'ready' if enough else 'inconclusive', 'reason': problem or ('Protected grouped roles available' if enough else 'Too few independent groups for the declared validation design'),
        'independence_justification': request.validation['independence_justification'],
        'scope': 'Across explicitly confirmed biological samples' if unit == 'biological_sample' else 'Across declared independent recordings; no automatic biological-sample replication',
        'values_used_to_assign_roles': False, 'split': split}
    return records, design


def learning_membership(observations, settings):
    """Unique capped rows, round-robin cells, then equal total sample budgets."""
    columns = ['observation_id', *KEYS, 'group_id', 'balance_group_id', 'learning_order', 'learning_weight']
    if observations.empty: return pd.DataFrame(columns=columns)
    eligible = observations.loc[observations.role.eq('learning') & observations.learning_feature_eligible].copy()
    rng = np.random.default_rng(settings['seed'])
    pools = {}
    for key, frame in eligible.groupby(KEYS, sort=True):
        if len(frame) < settings['min_observations']: continue
        order = rng.permutation(frame.observation_id.sort_values().to_numpy())[:settings['max_observations_per_cell']]
        if settings['balance'] == 'sample_cell':
            if not frame.sample_confirmed.iloc[0]: continue
            group = content_id({'source_run': key[0], 'sample': frame['sample'].iloc[0]})
        else: group = content_id(list(key))
        pools.setdefault(group, []).append(list(order))
    if not pools: return pd.DataFrame(columns=columns)
    budget = min(sum(len(cell) for cell in cells) for cells in pools.values())
    selected, balance_groups = [], []
    for group in sorted(pools):
        remaining = [list(cell) for cell in pools[group]]
        count = 0
        while count < budget:
            for cell in remaining:
                if cell and count < budget:
                    selected.append(cell.pop()); balance_groups.append(group); count += 1
    members = observations.set_index('observation_id').loc[selected].reset_index()
    members['balance_group_id'] = balance_groups
    members['learning_order'] = np.arange(len(members)); members['learning_weight'] = 1.
    return members[columns]


def observation_tables(resolved, traces, cell_partitions):
    request = resolved.request
    prepared_id = content_id(feature_settings(resolved))
    feature_names = [feature.column for feature in resolved.features]
    feature_columns = ['feature:' + name for name in feature_names]
    mappings = {(row['movie'], row['identity']): row for row in cell_partitions.to_dict('records')}
    observations, raw_rows, feature_rows, masks = [], [], [], []
    for key, frame in traces.groupby(KEYS + ['frame_index'], sort=True, dropna=False):
        source_run, movie, number, frame_index = key
        cell = CellKey(source_run, movie, number)
        # Missingness of features never chooses which timestamp is authoritative.
        clocks = frame[['hours', 'time_kind']].drop_duplicates().sort_values(['time_kind', 'hours'])
        valid_clock = len(clocks) == 1 and clocks.time_kind.iloc[0] == 'finite'
        time = float(clocks.hours.iloc[0]) if valid_clock else None
        clock_status = 'recorded' if valid_clock else 'conflicting_feature_clocks' if len(clocks) > 1 else 'invalid_time'
        # Validate the declared frame coordinate even when its clock is invalid.
        numeric_key = ObservationKey(cell, frame_index, time if valid_clock else 0.)
        obs_id = numeric_key.record_id if valid_clock else content_id({'cell': cell, 'frame_index': numeric_key.frame_index,
            'clocks': _json_value(clocks.to_dict('records'))})
        inside = valid_clock and (request.time_range_hours is None or request.time_range_hours[0] <= time < request.time_range_hours[1])
        raw, features, mask = {'observation_id': obs_id}, {'observation_id': obs_id}, {'observation_id': obs_id}
        for name, column in zip(feature_names, feature_columns):
            chosen = frame.loc[frame.measurement.eq(name)]
            if len(chosen) > 1: raise ValueError('Multiple source rows define one feature at a frame observation')
            raw[column] = chosen.raw_value.iloc[0] if len(chosen) else np.nan
            features[column] = chosen.processed_value.iloc[0] if len(chosen) else np.nan
            mask[column] = bool(len(chosen) and chosen.processed_valid.iloc[0])
        count = sum(mask[column] for column in feature_columns); fraction = count / len(feature_columns)
        learnable = bool(inside and fraction >= 1 - request.learning['missing']['max_fraction'])
        assignable = bool(inside and fraction >= request.assignment['min_observed_fraction'])
        status = 'eligible' if assignable else 'invalid_time' if not valid_clock else 'outside_range' if not inside else 'missing_features'
        reason = {'eligible': 'Original requested feature vector', 'invalid_time': clock_status,
            'outside_range': 'Outside the declared half-open analysis interval', 'missing_features': 'Too few observed features for the declared assignment policy'}[status]
        partition = mappings[movie, number]
        observations.append({**cell.as_dict(), 'observation_id': obs_id, 'frame_index': numeric_key.frame_index, 'hours': time,
            'time_status': clock_status, 'within_range': bool(inside), 'observed_features': count, 'feature_fraction': fraction,
            'learning_feature_eligible': learnable, 'assignment_feature_eligible': assignable, 'status': status, 'reason': reason,
            'vector_id': content_id({'observation': obs_id, 'prepared_inputs': prepared_id, 'values': _json_value(features)}),
            **{name: partition[name] for name in ('sample', 'sample_confirmed', 'condition', 'group', 'group_id', 'role')},
            'learning_selected': False, 'previous_observation_id': None, 'elapsed_hours': None, 'adjacent': False, 'adjacency_reason': 'Start of recorded cell'})
        raw_rows.append(raw); feature_rows.append(features); masks.append(mask)
    observations = pd.DataFrame(observations, columns=OBS_COLUMNS)
    for _, cell in observations.groupby(KEYS, sort=True):
        previous = None
        for index, row in cell.sort_values('frame_index').iterrows():
            if previous is not None:
                delta = row.hours - previous.hours if row.time_status == previous.time_status == 'recorded' else None
                reason = 'consecutive'
                if delta is None or delta <= 0: reason = 'Invalid or nonincreasing clock'
                elif row.frame_index != previous.frame_index + 1: reason = 'Missing frame index'
                elif delta > request.statistics['max_gap_hours']: reason = 'Recorded time gap'
                elif not (row.within_range and previous.within_range): reason = 'Outside analysis range'
                observations.loc[index, ['previous_observation_id', 'elapsed_hours', 'adjacent', 'adjacency_reason']] = [
                    previous.observation_id, delta, reason == 'consecutive', reason]
            previous = row
    return observations, pd.DataFrame(raw_rows, columns=['observation_id', *feature_columns]), \
        pd.DataFrame(feature_rows, columns=['observation_id', *feature_columns]), pd.DataFrame(masks, columns=['observation_id', *feature_columns])


def produce(context):
    resolved, request = context.request, context.request.request
    tables = read_verified_tables(context.table_paths, resolved.inputs.table_hashes)
    partition, design = partitions(resolved)
    # Reuse the established original-clock Workbench adapter. This performs no
    # imputation, shared scaling or model fit and each complete cell belongs to
    # one protected role before any temporal preprocessing is evaluated.
    preparation = SimpleNamespace(detrending=resolved.detrending, request=SimpleNamespace(
        representation=request.representation, time_range_hours=request.time_range_hours,
        support={'max_gap_hours': request.statistics['max_gap_hours']}))
    traces, inventory, processing = [], [], []
    for cell in resolved.inputs.cells:
        for feature in resolved.features:
            frame, details, _, action = _prepare_trace(preparation, cell, feature, tables[feature.table])
            traces.append(frame); inventory.append(details); processing.append(action)
    traces = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame(columns=TRACE_COLUMNS)
    observations, raw, features, masks = observation_tables(resolved, traces, partition)
    members = learning_membership(observations, request.learning)
    observations['learning_selected'] = observations.observation_id.isin(members.observation_id)
    cell_inventory = partition.copy()
    for field, source in (('observations', observations), ('eligible_observations', observations.loc[observations.assignment_feature_eligible]),
        ('learning_observations', members)):
        counts = source.groupby(KEYS).size().to_dict()
        cell_inventory[field] = [counts.get(tuple(row[name] for name in KEYS), 0) for row in cell_inventory.to_dict('records')]
    cell_inventory['status'] = np.where(cell_inventory.observations.gt(0), 'observed', 'missing_observations')
    eligible_counts = observations.loc[observations.learning_feature_eligible].groupby(KEYS).size().to_dict()
    reasons = []
    for row in cell_inventory.to_dict('records'):
        reason = 'Protected ' + row['role'] + ' role' if row['role'] != 'learning' else \
            'Confirmed biological sample required for sample_cell balancing' if request.learning['balance'] == 'sample_cell' and not row['sample_confirmed'] else \
            'Too few eligible observations for the declared learning-cell minimum' if eligible_counts.get(tuple(row[k] for k in KEYS), 0) < request.learning['min_observations'] else 'Eligible learning cell; exact subsample membership saved separately'
        reasons.append(reason)
    cell_inventory['learning_reason'] = reasons
    design.update(learning_observations=len(members), unique_learning_observations=members.observation_id.nunique(),
        learning_cells=len(members[KEYS].drop_duplicates()), learning_groups=members.group_id.nunique(), balance_groups=members.balance_group_id.nunique(),
        balancing='Unique rows capped per eligible learning cell; round-robin cells within confirmed samples; equal total sample budgets. Cell balance treats each cell as its own group. Unconfirmed samples cannot enter sample_cell-balanced learning.',
        learning_minimum_scope='Minimum eligible observations per cell before learning subsampling',
        observed_time_unchanged_by_learning=True)
    output_tables = {'observations': observations, 'raw_features': raw, 'features': features, 'feature_masks': masks,
        'source_traces': traces, 'feature_inventory': pd.DataFrame(inventory, columns=list(dict.fromkeys(
            KEYS + ['measurement', 'table', 'label', 'unit', 'trace_id', 'kind', 'status', 'reason', 'source_observations', 'valid_raw', 'valid_processed']
            + [key for row in inventory for key in row]))), 'cell_inventory': cell_inventory,
        'partitions': partition, 'learning_members': members}
    context.output.mkdir(parents=True)
    refs = []
    for name, frame in output_tables.items():
        path = context.output / (name + '.json'); path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id, columns=tuple(frame.columns)))
    for name, document in {'partition_design': design, 'processing': processing,
        'provenance': {'schema_version': 1, 'scientific_id': context.scientific_id,
            'design_id': context.saved('behaviour-design').outcome.scientific_id, 'resolved_request': resolved.as_dict(),
            'model_fitted': False, 'learned_transform_fitted': False, 'imputation_performed': False,
            'feature_columns': {feature.column: 'feature:' + feature.column for feature in resolved.features},
            'window': 'Original frame observations; requested time range is start inclusive and end exclusive',
            'alignment': 'Common source/movie/cell/frame key; conflicting feature clocks remain invalid with original source rows retained',
            'missing_time_exposure': 'No inferred observation or interval across absent frames or declared gaps'}}.items():
        path = context.output / (name + '.json'); _write_json(path, document)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id))
    return StepResult(context.step.name, context.scientific_id, 'completed',
        'Saved original feature observations, protected group roles and separate balanced learning membership', tuple(refs))
