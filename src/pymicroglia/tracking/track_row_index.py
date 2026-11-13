"""Transient weak seats and persistent duplicate sites: per-track and per-owner rows from one grouping.

remove_transient_weak_seats._measure_tracks filtered the whole observation
table once per track (obs[obs.track_id == track_id]) and read each track's
positions through a two-column table. Grouping the table once by an integer
track_id gives each track's rows in table order with the same index labels,
columns and dtypes; tracks with no observation keep the original filter. A
track's x and y converted to float one column at a time hold the same values
as the two-column conversion, and the steps and norms read only values.
Each track's rows are only read after their frame sort, so rows whose integer
frames already strictly increase skip it (track_sort_skips.frame_ordered).

recover_persistent_duplicate_soma_sites.discover filtered a frame's strong
rows by owner for every seed and later frame (future[future.owner.eq(name)]).
Grouping a frame's rows once by their integer owner gives the same rows in
the same order; each lookup returns a copy, as the original did, and an owner
absent from the frame keeps the original filter. Only frames the search
reaches (the first fourteen) are grouped.
"""
import __future__
from contextlib import contextmanager
import hashlib
import importlib
import inspect
from pathlib import Path

import numpy as np
try:
    from .track_sort_skips import frame_ordered as _frame_ordered_helper
except ImportError:
    from track_sort_skips import frame_ordered as _frame_ordered_helper

SPECS = {
    'remove_transient_weak_seats': dict(
        source='bf3ae88d107ff4bbd589de3110efde18de8cd156927bdfb0d6f1461861a7e520',
        function='_measure_tracks',
        edits=(('    obs = pd.DataFrame(observations)\n',
                '    obs = pd.DataFrame(observations)\n    _measured_rows = _track_rows(obs)\n'),
               ('        measured = obs[obs.track_id == track_id]\n',
                '        measured = _measured_rows(obs, track_id)\n'),
               ('            xy = group[["x", "y"]].to_numpy(float)\n',
                '            xy = _track_xy(group)\n'),
               ('        group = group.sort_values("frame")\n',
                '        group = _frame_ordered(group)\n'))),
    'recover_persistent_duplicate_soma_sites': dict(
        source='7a8d8df34533804b8e9481bd21b6bb87bf8adf1e093f0818d0e8dae752f1ea08',
        function='discover',
        edits=(('    by_frame = {int(frame): group for frame, group in strong.groupby("frame")}\n',
                '    by_frame = {int(frame): group for frame, group in strong.groupby("frame")}\n'
                '    _owner_rows = _frame_owner_rows()\n'),
               ('                others = future[future.owner.eq(name)].copy()\n',
                '                others = _owner_rows(future, name)\n'))),
}


def helpers(stats):
    def _track_rows(obs):
        groups = None
        if ('track_id' in obs.columns and obs.columns.is_unique
                and obs['track_id'].dtype.kind in 'iu'):
            groups = {int(key): rows for key, rows in obs.groupby('track_id', sort=False)}

        def measured(obs, track_id):
            rows = groups.get(track_id) if groups is not None and type(track_id) is int else None
            if rows is None:
                stats['tracks_filtered'] += 1
                return obs[obs.track_id == track_id]
            stats['tracks_grouped'] += 1
            return rows
        return measured

    def _track_xy(group):
        if (group.columns.is_unique and 'x' in group.columns and 'y' in group.columns
                and group['x'].dtype.kind in 'iuf' and group['y'].dtype.kind in 'iuf'):
            return np.stack((group['x'].to_numpy(float), group['y'].to_numpy(float)), axis=1)
        return group[["x", "y"]].to_numpy(float)

    def _frame_owner_rows():
        held = {}

        def owner_rows(future, name):
            entry = held.get(id(future))
            if entry is None or entry[0] is not future:
                groups = None
                if (future.columns.is_unique and 'owner' in future.columns
                        and future['owner'].dtype.kind in 'iu'):
                    groups = {int(key): rows for key, rows in future.groupby('owner', sort=False)}
                entry = held[id(future)] = (future, groups)
            rows = None
            if entry[1] is not None and isinstance(name, (int, np.integer)) and not isinstance(name, bool):
                rows = entry[1].get(int(name))
            if rows is None:
                stats['owners_filtered'] += 1
                return future[future.owner.eq(name)].copy()
            stats['owners_grouped'] += 1
            return rows.copy()
        return owner_rows

    return dict(_track_rows=_track_rows, _track_xy=_track_xy, _frame_owner_rows=_frame_owner_rows,
                _frame_ordered=_frame_ordered_helper(stats))


@contextmanager
def install(stats=None):
    """Replace the verified modules' functions while the corrections run."""
    stats = stats if stats is not None else {}
    for key in ('versions', 'tracks_grouped', 'tracks_filtered', 'owners_grouped', 'owners_filtered',
                'in_order', 'sorted'):
        stats.setdefault(key, 0)
    added = helpers(stats)
    restore = []
    try:
        for name, spec in SPECS.items():
            try:
                module = importlib.import_module(name)
                if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != spec['source']:
                    continue
                original = getattr(module, spec['function'])
                if getattr(original, '__wrapped__', None) is not None:
                    continue
                source = inspect.getsource(original)
            except (ImportError, OSError, TypeError, AttributeError):
                continue
            if any(source.count(old) != 1 for old, _ in spec['edits']):
                continue
            for old, new in spec['edits']:
                source = source.replace(old, new)
            live = vars(module)
            live.update(added)
            namespace = {}
            exec(compile(source, f'<track-row-index:{name}>', 'exec',
                         flags=__future__.annotations.compiler_flag, dont_inherit=True), live, namespace)
            replacement = namespace[spec['function']]
            replacement.__wrapped__ = original
            restore.append((module, spec['function'], original))
            setattr(module, spec['function'], replacement)
            stats['versions'] += 1
        yield stats
    finally:
        for module, function, original in reversed(restore):
            setattr(module, function, original)
            for key in added:
                vars(module).pop(key, None)
