"""Classify original pixel occupancy within declared windows and prepare flow geometry."""
from __future__ import annotations
from typing import Any,Mapping,Sequence
import numpy as np
import pandas as pd
import tifffile
from .radial import _selected_identities
from .prepared import PreparedViews
FATE_ROLES={'core':'circadian_green','fringe':'orange','transient':'teal','vacated':'raw','transferred':'plum'}

def pixel_fate_states(labels: Any, event_windows: Mapping[str, Sequence[int]], *, identities: Sequence[int] | None=None, core_fraction: float=0.8, transient_fraction: float=0.2) -> tuple[np.ndarray, pd.DataFrame]:
    """Classify selected pixels within any named sequence of frame windows.

    Each mapping entry is an event name and the frames belonging to the window
    that starts at that event. With more than one identity, a change in modal
    owner is reported as a transfer. With one identity, the same calculation
    describes that cell without inventing transfers to cells outside the
    selection.
    """
    values = np.asarray(labels)
    if values.ndim != 3:
        raise ValueError('pixel fate labels must have shape (frame, row, column)')
    if not 0.0 <= float(transient_fraction) < float(core_fraction) <= 1.0:
        raise ValueError('pixel fate thresholds must satisfy 0 <= transient < core <= 1')
    if len(event_windows) < 2:
        raise ValueError('pixel fate needs at least two event windows')
    windows: list[tuple[str, np.ndarray]] = []
    for name, indices in event_windows.items():
        frame_indices = np.asarray(indices, dtype=int)
        if frame_indices.ndim != 1 or frame_indices.size == 0:
            raise ValueError(f'pixel fate event {name!r} has no frames')
        if np.any(frame_indices < 0) or np.any(frame_indices >= values.shape[0]):
            raise ValueError(f'pixel fate event {name!r} contains a frame outside the stack')
        if len(np.unique(frame_indices)) != len(frame_indices):
            raise ValueError(f'pixel fate event {name!r} repeats a frame')
        windows.append((str(name), frame_indices))
    available = np.asarray([int(value) for value in np.unique(values) if value], dtype=int)
    selected = available if identities is None else np.asarray(identities, dtype=int)
    unknown = sorted(set(selected.tolist()) - set(available.tolist()))
    if unknown:
        raise ValueError(f'pixel fate identities are absent from the stack: {unknown}')
    selected_labels = np.where(np.isin(values, selected), values, 0)
    selected_mask = selected_labels > 0
    used_frames = np.unique(np.concatenate([indices for _, indices in windows]))
    union = selected_mask[used_frames].any(axis=0)
    coordinates = np.flatnonzero(union)
    class_names = list(FATE_ROLES)
    class_index = {name: index for index, name in enumerate(class_names)}
    states = np.full((len(coordinates), len(windows)), class_index['vacated'], dtype=int)
    previous_owner = np.zeros(len(coordinates), dtype=int)
    ever_occupied = np.zeros(len(coordinates), dtype=bool)
    summaries: list[dict[str, Any]] = []
    for event_index, (event_name, frame_indices) in enumerate(windows):
        frequency = selected_mask[frame_indices].mean(axis=0).ravel()[coordinates]
        current = frequency > 0
        states[current & (frequency >= core_fraction), event_index] = class_index['core']
        states[current & (frequency > transient_fraction) & (frequency < core_fraction), event_index] = class_index['fringe']
        states[current & (frequency <= transient_fraction), event_index] = class_index['transient']
        states[~current & ~ever_occupied, event_index] = class_index['vacated']
        if len(selected) > 1 and len(coordinates):
            owners = np.zeros(len(coordinates), dtype=int)
            flat = selected_labels[frame_indices].reshape(len(frame_indices), -1)[:, coordinates]
            for pixel in range(flat.shape[1]):
                nonzero = flat[:, pixel][flat[:, pixel] > 0]
                if len(nonzero):
                    owner_values, counts = np.unique(nonzero, return_counts=True)
                    owners[pixel] = int(owner_values[np.argmax(counts)])
            transferred = current & (previous_owner > 0) & (owners > 0) & (owners != previous_owner)
            states[transferred, event_index] = class_index['transferred']
            previous_owner = np.where(owners > 0, owners, previous_owner)
        ever_occupied |= current
        for class_name, code in class_index.items():
            pixels = int(np.count_nonzero(states[:, event_index] == code))
            summaries.append({'event_index': event_index, 'event': event_name, 'class': class_name, 'pixels': pixels, 'share': float(pixels / len(coordinates)) if len(coordinates) else np.nan, 'frames': int(len(frame_indices)), 'first_frame': int(frame_indices.min()), 'last_frame': int(frame_indices.max())})
    return (states, pd.DataFrame(summaries))

def fate_transition_table(states: Any, *, event_labels: Sequence[str] | None=None, class_names: Sequence[str] | None=None) -> pd.DataFrame:
    """Exact pixel counts connecting every class across adjacent events."""
    values = np.asarray(states)
    if values.ndim != 2:
        raise ValueError('pixel fate states must be an observation-by-event matrix')
    classes = list(class_names or FATE_ROLES)
    events = list(event_labels or [f'Event {index + 1}' for index in range(values.shape[1])])
    if len(events) != values.shape[1]:
        raise ValueError('event_labels must contain one label per event')
    rows = []
    for event_index in range(values.shape[1] - 1):
        for source, source_name in enumerate(classes):
            for target, target_name in enumerate(classes):
                rows.append({'from_event_index': event_index, 'to_event_index': event_index + 1, 'from_event': events[event_index], 'to_event': events[event_index + 1], 'from_class': source_name, 'to_class': target_name, 'pixels': int(np.count_nonzero((values[:, event_index] == source) & (values[:, event_index + 1] == target)))})
    return pd.DataFrame(rows)

def geometry(states,labels):
    observations,stages=states.shape;offsets=[];bars=[]
    for stage in range(stages):
        cursor=0.;starts={}
        for code,name in enumerate(FATE_ROLES):
            count=int(np.count_nonzero(states[:,stage]==code));starts[code]=cursor
            bars.append(dict(stage=stage,y=cursor,height=count,colour=FATE_ROLES[name]))
            cursor+=count
        offsets.append(starts)
    outgoing=[dict(o) for o in offsets];incoming=[dict(o) for o in offsets];paths=[]
    for stage in range(stages-1):
        for source,name in enumerate(FATE_ROLES):
            for target in range(len(FATE_ROLES)):
                count=int(np.count_nonzero((states[:,stage]==source)&(states[:,stage+1]==target)))
                if not count:continue
                y0a=outgoing[stage][source];y0b=y0a+count;y1a=incoming[stage+1][target];y1b=y1a+count
                outgoing[stage][source]=y0b;incoming[stage+1][target]=y1b
                x0=stage+.025;x1=stage+1-.025;c1=x0+(x1-x0)*.42;c2=x1-(x1-x0)*.42
                paths.append(dict(vertices=[(x0,y0a),(c1,y0a),(c2,y1a),(x1,y1a),(x1,y1b),(c2,y1b),(c1,y0b),(x0,y0b),(x0,y0a)],colour=FATE_ROLES[name]))
    return dict(bars=bars,paths=paths,labels=labels,observations=observations)


def prepare(source,options):
    label_path = source.input_path('labels')
    if label_path is None:
        raise ValueError('pixel-fate-flow needs the labels input recorded in the run manifest')
    labels = tifffile.imread(label_path)
    frame_hours = np.arange(labels.shape[0], dtype=float) * source.interval / 60.0
    requested_names = [str(value) for value in options.get('events')]
    requested_times = np.asarray(options.get('event_times'), dtype=float)
    if requested_names and (not requested_times.size):
        raise ValueError('--events needs matching --event-times')
    if requested_times.size:
        if requested_times.size < 2:
            raise ValueError('--event-times needs at least two event starts')
        if not np.isfinite(requested_times).all():
            raise ValueError('--event-times must contain finite hours')
        if np.any(np.diff(requested_times) <= 0):
            raise ValueError('--event-times must be in strictly increasing order')
        if requested_times[0] < 0 or requested_times[-1] > frame_hours[-1]:
            raise ValueError(f'--event-times must lie between 0 and {frame_hours[-1]:g} hours')
        event_names = requested_names or [f'Event {index + 1}' for index in range(len(requested_times))]
        if len(event_names) != len(requested_times):
            raise ValueError('--events and --event-times must contain the same number of values')
        if len(set(event_names)) != len(event_names):
            raise ValueError('--events names must be unique')
        first_frames = np.searchsorted(frame_hours, requested_times, side='left')
        if np.any(np.diff(first_frames) <= 0):
            raise ValueError('two --event-times resolve to the same recorded frame')
        bins = [np.arange(first_frames[index], first_frames[index + 1] if index + 1 < len(first_frames) else len(labels), dtype=int) for index in range(len(first_frames))]
        display_hours = requested_times
        event_mode = 'user-selected event windows'
    else:
        n_stages = max(2, int(options.get('stages')))
        bins = [np.asarray(values, dtype=int) for values in np.array_split(np.arange(labels.shape[0]), n_stages)]
        if any((not len(values) for values in bins)):
            raise ValueError(f'--stages cannot exceed the {len(labels)} recorded frames')
        event_names = [f'Stage {index + 1}' for index in range(n_stages)]
        display_hours = np.asarray([frame_hours[values].mean() for values in bins])
        event_mode = 'evenly spaced recording stages'
    event_windows = dict(zip(event_names, bins, strict=True))
    event_labels = [f'{name}\n{hour:g} h' for name, hour in zip(event_names, display_hours, strict=True)]
    event_table = pd.DataFrame([{'event_index': index, 'event': name, 'event_time_hours': float(display_hours[index]), 'requested_start_hours': float(requested_times[index]) if requested_times.size else np.nan, 'first_frame': int(indices.min()), 'last_frame': int(indices.max()), 'window_start_hours': float(frame_hours[indices.min()]), 'window_end_hours': float(frame_hours[indices.max()] + source.interval / 60.0), 'frames': int(len(indices))} for index, (name, indices) in enumerate(event_windows.items())])
    thresholds = options.get('thresholds')
    if len(thresholds) not in {2, 4}:
        raise ValueError('--thresholds needs core,transient or primary_core,primary_transient,alternate_core,alternate_transient')
    primary = thresholds[:2]
    alternate = thresholds[2:] if len(thresholds) == 4 else [0.7, 0.3]
    for pair in (primary, alternate):
        if not 0 <= pair[1] < pair[0] <= 1:
            raise ValueError('each --thresholds pair must satisfy 0 <= transient < core <= 1')
    support = pd.DataFrame({'identity': [int(value) for value in np.unique(labels) if value]})
    support['observed_frames'] = support['identity'].map(lambda identity: int(np.count_nonzero(np.any(labels == identity, axis=(1, 2)))))
    selected = _selected_identities(support, options.get('cells'))
    class_labels = list(FATE_ROLES)
    primary_pooled, primary_composition = pixel_fate_states(labels, event_windows, core_fraction=float(primary[0]), transient_fraction=float(primary[1]))
    primary_matrices = {0: primary_pooled}
    composition_tables = [primary_composition.assign(identity=0)]
    for identity in selected:
        matrix, composition = pixel_fate_states(labels, event_windows, identities=[identity], core_fraction=float(primary[0]), transient_fraction=float(primary[1]))
        primary_matrices[identity] = matrix
        composition_tables.append(composition.assign(identity=identity))
    alternate_matrix, alternate_composition = pixel_fate_states(labels, event_windows, core_fraction=float(alternate[0]), transient_fraction=float(alternate[1]))
    tables=[]
    for identity,matrix in primary_matrices.items():
        table=fate_transition_table(matrix,event_labels=event_names,class_names=class_labels)
        table.insert(0,'identity',identity)
        times=dict(zip(event_names,display_hours,strict=True))
        table['from_event_time_hours']=table.from_event.map(times);table['to_event_time_hours']=table.to_event.map(times)
        tables.append(table)
    table=pd.concat(tables,ignore_index=True)
    sensitivity=fate_transition_table(alternate_matrix,event_labels=event_names,class_names=class_labels)
    shares=primary_composition.pivot(index='event',columns='class',values='pixels').reindex(index=event_names,columns=class_labels).fillna(0)
    shares.insert(0,'hours',display_hours);shares=shares.reset_index()
    return PreparedViews({'flow':dict(table=table,**geometry(primary_pooled,event_labels)),
        'shares':dict(table=shares,classes=class_labels,colours=list(FATE_ROLES.values())),
        'sensitivity':dict(table=sensitivity,**geometry(alternate_matrix,event_labels))},
        auxiliary={'event_windows.csv':event_table,'pixel_fate_composition.csv':pd.concat(composition_tables,ignore_index=True),'sensitivity_composition.csv':alternate_composition,'sensitivity_flow.csv':sensitivity},
        wording=dict(title='Pixel fate across selected windows',subtitle=f'Core/transient threshold pairs {primary[0]:g}/{primary[1]:g} and {alternate[0]:g}/{alternate[1]:g}.',footnote='Every selected start opens a window; the final window ends with the recording. Transfer means a change in the modal cell occupying that pixel.')),None
