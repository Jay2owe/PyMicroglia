"""Per-track sorts that are skipped or done once in three corrections.

recover_bookended_owner_with_small_remnants._runs sorted every track's rows by
frame before cutting runs. When the frame column is a plain integer column
that already strictly increases, the sort keeps every row where it is, so the
rows are used as they are; reset_index(drop=True) then makes the same new
table. Other tracks are sorted as before.

precontact_soma_ownership (R205-A007) sorted every track's points by frame,
{int(k): q.sort_values('frame') for k, q in pts.groupby('track_id')}. When no
track repeats a frame (integer track and frame columns), each track's frames
sort to one order whatever the sort, so one stable sort of all points by
(track, frame) followed by the same groupby gives each track the same rows in
the same order, index, columns and dtypes. Repeated frames keep the per-track
sort.

process_inventory_tracked.scan indexed every track's points by frame for each
well, but reads only the two tracks of each audited pair. With an integer
track column, a track's table is built the first time it is read, from the
same groupby and the same set_index; a track that is not in the table raises
the same KeyError.

recover_persistent_resident_contacts._persistent_host_owners counted each
track's positive owners with value_counts. The top count, the second count and
(when the top is not tied) the top owner are the same from np.unique counts on
the same cast values. A tied top makes support equal second, which fails the
dominance test before the owner is read whenever the dominance exceeds 1; any
other dominance keeps value_counts.

restore_spatial_reciprocal_soma_paths_a002 weighted every frame's
component bincount by the integer pixel grids, which bincount converts to
float64 on every call; the grids are made float64 once. Integer coordinates
are exact in float64, so the sums are the same.

shared_birth_soma_ownership (R206-A002), silhouette_owner_continuity (R207-A003)
and recover_flanked_large_contact_interruptions sorted every track of a
groupby by frame. When the track and frame columns are integers and no track
repeats or reverses a frame in table order, every group already strictly
increases, so each sort keeps every row where it is and the groups are used
as they are (they are only read). Otherwise every group is sorted as before.

Seven more corrections sort every track the same two ways and only read the
result: dormant name basins, multistage handoffs and protected owner closure
rebind each loop group to its sort; short bookended somata cut _runs from the
sort and reset its index; retired names, bookended multi-owner somata and
co-observed bookended somata keep a dict of sorted tracks. The first four use
the per-group in-order check, the last three the whole-table one. The source
tables are written only before the groupby, and no sorted table is written.

Ten more follow the same rules: five loop rebinds (emergent late handoffs,
dormant raw soma bridges, repeated soma renaming, optical-core adjacent
pseudosomata, terminal exclusive somata) and bookended multi-owner episodes
use the per-group check; repeated-renaming and dim-displaced _runs iterate the
rows in the checked order; shared-parent somata (whose sorted tracks are then
indexed by frame), displaced somata and brief raw names use the whole-table
check.
"""
import __future__
from contextlib import contextmanager
import inspect
import sys

import numpy as np
try:
    from .module_scan import file_sha256 as _file_sha256, loaded as _loaded
except ImportError:
    from module_scan import file_sha256 as _file_sha256, loaded as _loaded


def _plain_int(column):
    return isinstance(column.dtype, np.dtype) and column.dtype.kind in 'iu'


def frame_ordered(stats):
    def _frame_ordered(group):
        if group.columns.is_unique and 'frame' in group.columns and _plain_int(group['frame']):
            frames = group['frame'].to_numpy()
            if len(frames) < 2 or bool(np.all(frames[1:] > frames[:-1])):
                stats['in_order'] += 1
                return group
        stats['sorted'] += 1
        return group.sort_values("frame")
    return _frame_ordered


def sorted_tracks(stats):
    def _sorted_tracks(pts):
        columns = pts.columns
        if (columns.is_unique and 'track_id' in columns and 'frame' in columns
                and _plain_int(pts['track_id']) and _plain_int(pts['frame'])):
            tracks, frames = pts['track_id'].to_numpy(), pts['frame'].to_numpy()
            order = np.lexsort((frames, tracks))
            tracks, frames = tracks[order], frames[order]
            if not np.any((tracks[1:] == tracks[:-1]) & (frames[1:] == frames[:-1])):
                stats['sorted_once'] += 1
                return {int(k): q for k, q in pts.take(order).groupby('track_id')}
        stats['per_track'] += 1
        return {int(k): q.sort_values('frame') for k, q in pts.groupby('track_id')}
    return _sorted_tracks


def tracks_in_order(stats):
    def _tracks_in_order(table, key):
        columns = table.columns
        if (columns.is_unique and key in columns and 'frame' in columns
                and _plain_int(table[key]) and _plain_int(table['frame'])):
            tracks, frames = table[key].to_numpy(), table['frame'].to_numpy()
            order = np.argsort(tracks, kind='stable')
            tracks, frames = tracks[order], frames[order]
            if not np.any((tracks[1:] == tracks[:-1]) & (frames[1:] <= frames[:-1])):
                stats['groups_in_order'] += 1
                return {int(k): g for k, g in table.groupby(key)}
        stats['groups_sorted'] += 1
        return {int(k): g.sort_values('frame') for k, g in table.groupby(key)}
    return _tracks_in_order


class _TrackFrames(dict):
    """{int(track): rows.set_index("frame")}, each built when first read."""

    def __init__(self, points, stats):
        super().__init__()
        self._groups = points.groupby("track_id")
        self._keys = set(np.unique(points["track_id"].to_numpy()).tolist())
        self._stats = stats

    def __missing__(self, track):
        if track not in self._keys:
            raise KeyError(track)
        self._stats['tracks_built'] += 1
        value = self[track] = self._groups.get_group(track).set_index("frame")
        return value


def track_frames(stats):
    def _track_frames(points):
        if (points.columns.is_unique and 'track_id' in points.columns and 'frame' in points.columns
                and _plain_int(points['track_id'])):
            stats['lazy'] += 1
            return _TrackFrames(points, stats)
        stats['eager'] += 1
        return {int(track): group.set_index("frame") for track, group in points.groupby("track_id")}
    return _track_frames


def host_counts(stats):
    def _host_counts(group, params):
        dominance = params.minimum_persistent_host_dominance
        if isinstance(dominance, (int, float)) and dominance > 1:
            values = group.accepted_owner.astype(int).to_numpy()
            if type(values) is np.ndarray and values.dtype.kind in 'iu':
                stats['host_unique'] += 1
                ids, tallies = np.unique(values[values > 0], return_counts=True)
                if not len(ids):
                    return None
                top = int(np.argmax(tallies))
                second = int(np.sort(tallies)[-2]) if len(tallies) > 1 else 0
                return int(ids[top]), int(tallies[top]), second
        stats['host_counts'] += 1
        counts = group[group.accepted_owner.astype(int).gt(0)] \
            .accepted_owner.astype(int).value_counts()
        if counts.empty:
            return None
        return int(counts.index[0]), int(counts.iloc[0]), (int(counts.iloc[1]) if len(counts) > 1 else 0)
    return _host_counts


SPECS = (
    dict(module='recover_bookended_owner_with_small_remnants', function='_runs',
         digests={'31f7e6cdc49e8e9f4c21bf3858997d2b500090877d884a9e7da60562ac30d52d'},
         edits=(('    rows = group.sort_values("frame").reset_index(drop=True)\n',
                 '    rows = _frame_ordered(group).reset_index(drop=True)\n'),),
         helper='_frame_ordered', factory=frame_ordered),
    dict(module='precontact_soma_ownership', function='preserve_precontact_owners',
         digests={'ecfe963cddb20fcca990d242796070c1097bb73c4504a01f9b30f7ce62d2b1e6'},
         edits=(("    by_track={int(k):q.sort_values('frame') for k,q in pts.groupby('track_id')}\n",
                 "    by_track=_sorted_tracks(pts)\n"),),
         helper='_sorted_tracks', factory=sorted_tracks),
    dict(module='process_inventory_tracked', function='scan',
         digests={'6864e0175fb3450ef76d97d5fd0e915399587b0b132f8713eae0f3b211f21b36'},
         edits=(('        track_points = {int(track): group.set_index("frame")\n'
                 '                        for track, group in points.groupby("track_id")}\n',
                 '        track_points = _track_frames(points)\n'),),
         helper='_track_frames', factory=track_frames),
    dict(module='recover_persistent_resident_contacts', function='_persistent_host_owners',
         digests={'b9b8abe3a7cef9ead503db73711daaa1a644202879f0e71774da7ebd50f15bdd'},
         edits=(('        counts = group[group.accepted_owner.astype(int).gt(0)] \\\n'
                 '            .accepted_owner.astype(int).value_counts()\n'
                 '        if counts.empty:\n'
                 '            continue\n'
                 '        owner, support = int(counts.index[0]), int(counts.iloc[0])\n'
                 '        second = int(counts.iloc[1]) if len(counts) > 1 else 0\n',
                 '        _found = _host_counts(group, params)\n'
                 '        if _found is None:\n'
                 '            continue\n'
                 '        owner, support, second = _found\n'),),
         helper='_host_counts', factory=host_counts),
    dict(module='restore_spatial_reciprocal_soma_paths_a002', function='correct',
         digests={'7d5c2be48bde80f567f244a428c8b6d538f7076b79ad05315b4b4e50cfe9f6c4'},
         edits=(('    yy, xx = np.indices(labels.shape[1:])\n',
                 '    yy, xx = np.indices(labels.shape[1:]).astype(float)\n'),),
         helper=None, factory=None),
    dict(module='shared_birth_soma_ownership', function='preserve_shared_birth_owners',
         digests={'24d9ae21149114ce9f86d15c896660840b1269ed92259501586129199bba9d82'},
         edits=(("    by_track={int(k):g.sort_values('frame') for k,g in pts.groupby('track_id')}\n",
                 "    by_track=_tracks_in_order(pts, 'track_id')\n"),),
         helper='_tracks_in_order', factory=tracks_in_order),
    dict(module='silhouette_owner_continuity', function='preserve_silhouette_owners',
         digests={'5f27548d7027c9e6017f20aa39697426160344b57ccc9bd7d29300d0d9b4862c'},
         edits=(("    by_track={int(k):g.sort_values('frame') for k,g in pts.groupby('track_id')}\n",
                 "    by_track=_tracks_in_order(pts, 'track_id')\n"),),
         helper='_tracks_in_order', factory=tracks_in_order),
    dict(module='recover_flanked_large_contact_interruptions', function='correct',
         digests={'f7683a6d56eb2643d266d6ab6b557e1629a8997d70ec2aa8aefff09f26556f1d'},
         edits=(('    by_track = {int(track):group.sort_values("frame")\n'
                 '                for track,group in scored.groupby("track_id")}\n',
                 '    by_track = _tracks_in_order(scored, "track_id")\n'),),
         helper='_tracks_in_order', factory=tracks_in_order),
    dict(module='recover_dormant_name_soma_basins', function='correct',
         digests={'96b3a0bf12c117d316b0900b002cefd2cb2d073d17389c3e737df1aca186ef2b'},
         edits=(('        group = group.sort_values("frame")\n',
                 '        group = _frame_ordered(group)\n'),),
         helper='_frame_ordered', factory=frame_ordered),
    dict(module='recover_multistage_reciprocal_handoff', function='correct',
         digests={'d6519dcd260e69ff014b0a675d73331a754265b3a2cccfe14e52d982f95c1a4a'},
         edits=(('        group = group.sort_values("frame")\n',
                 '        group = _frame_ordered(group)\n'),),
         helper='_frame_ordered', factory=frame_ordered),
    dict(module='complete_owner_handoff_episodes_r10_a002', function='correct',
         digests={'5305c91dc515bcfcf7612825fbdce55d4d6773edae27d71bfd86bfca8f8938f1'},
         edits=(('        group = group.sort_values("frame")\n',
                 '        group = _frame_ordered(group)\n'),),
         helper='_frame_ordered', factory=frame_ordered),
    dict(module='recover_short_bookended_soma_beside_transient_holders', function='_runs',
         digests={'344d7b9ad3cd09fe10eee1a58c2207244d68a9cc59ccf2acaeb2429a982b0024'},
         edits=(('    rows = group.sort_values("frame").reset_index(drop=True)\n',
                 '    rows = _frame_ordered(group).reset_index(drop=True)\n'),),
         helper='_frame_ordered', factory=frame_ordered),
    dict(module='recover_reciprocal_retired_name_a002', function='correct',
         digests={'d86704a5883b1ca6bb50017b7442b8bc0ade8286ff90142ff36e146aaef015dc'},
         edits=(('    by_track = {int(track): group.sort_values("frame")\n'
                 '                for track, group in scored.groupby("track_id")}\n',
                 '    by_track = _tracks_in_order(scored, "track_id")\n'),),
         helper='_tracks_in_order', factory=tracks_in_order),
    dict(module='bookended_multi_owner_soma_a002', function='correct',
         digests={'4988776f24a92151f26329f97cb4ffefad613f2308640ffa171b7f46dc2d3385'},
         edits=(("    groups={int(tr):g.sort_values('frame') for tr,g in scored.groupby('track_id')}\n",
                 "    groups=_tracks_in_order(scored, 'track_id')\n"),),
         helper='_tracks_in_order', factory=tracks_in_order),
    dict(module='coobserved_bookended_soma_memory_a001', function='correct',
         digests={'dafc2cfecd65ef3b2462716bc77e79f9b7973cf367ff97b0c0a2f4edc3174b73'},
         edits=(('    groups = {int(track): group.sort_values("frame")\n'
                 '              for track, group in visible.groupby("track_id")}\n',
                 '    groups = _tracks_in_order(visible, "track_id")\n'),),
         helper='_tracks_in_order', factory=tracks_in_order),
) + tuple(
    dict(module=module, function=function, digests={digest},
         edits=(('        group = group.sort_values("frame")\n', '        group = _frame_ordered(group)\n'),),
         helper='_frame_ordered', factory=frame_ordered)
    for module, function, digest in (
        ('recover_emergent_reciprocal_late_handoff', 'correct',
         '68a208d02d3d6444e537f32f0514719a8312b5b0bbba23079add47c13f2c85f4'),
        ('bridge_dormant_raw_soma_gaps', 'correct',
         'ab66a73777369674bfc9b717b3a56d4351fb84020e9e44d3079c21906e3a3fce'),
        ('recover_repeated_soma_renaming', 'correct',
         '565fb26bcfc32d94696251faf7b8f021d1b2d9c07d4e2d8d8df2e3ff5e8dfd92'),
        ('remove_optical_core_adjacent_pseudosomata', '_eligible_tracks',
         'ae7aa5e19da3e7d62e447b4a5f8a02040f0e4c00168b6b8c83313f65a7fd8802'),
        ('restore_terminal_exclusive_soma_from_shared_name', 'correct',
         '3701b823e12f2cb2aed2335e0f00a5fa918e7c389f7b0de7318bfe925794962e'))
) + (
    dict(module='bookended_multi_owner_soma_a002', function='episodes',
         digests={'4988776f24a92151f26329f97cb4ffefad613f2308640ffa171b7f46dc2d3385'},
         edits=(("        g=g.sort_values('frame')\n", "        g=_frame_ordered(g)\n"),),
         helper='_frame_ordered', factory=frame_ordered),
    dict(module='recover_repeated_soma_renaming', function='_runs',
         digests={'565fb26bcfc32d94696251faf7b8f021d1b2d9c07d4e2d8d8df2e3ff5e8dfd92'},
         edits=(('    for row in group.sort_values("frame").itertuples(index=False):\n',
                 '    for row in _frame_ordered(group).itertuples(index=False):\n'),),
         helper='_frame_ordered', factory=frame_ordered),
    dict(module='recover_dim_displaced_name_paths', function='_runs',
         digests={'eeadd6a3ea37ac9f903c5fcaa1515e72fd1f5317dfc97be86d7efde0f0f086e1'},
         edits=(('    for p in group.sort_values("frame").itertuples(index=False):\n',
                 '    for p in _frame_ordered(group).itertuples(index=False):\n'),),
         helper='_frame_ordered', factory=frame_ordered),
    dict(module='shared_parent_soma_ownership_a003', function='correct',
         digests={'40f469a5b5aae11f0967fd8fd19c03698ab99649a6fc662eedc49502a04ccdfc'},
         edits=(("    by_track={int(tr):g.sort_values('frame').set_index('frame',drop=False) for tr,g in visible.groupby('track_id')}\n",
                 "    by_track={tr:g.set_index('frame',drop=False) for tr,g in _tracks_in_order(visible,'track_id').items()}\n"),),
         helper='_tracks_in_order', factory=tracks_in_order),
    dict(module='overlapping_detector_soma_memory_a003', function='correct',
         digests={'5a4c49e32a9109e6925c594da2868c32d844b700947913901ab8d9d76abd1259'},
         edits=(("    groups={int(tr):g.sort_values('frame').set_index('frame',drop=False) for tr,g in scored.groupby('track_id')}\n",
                 "    groups={tr:g.set_index('frame',drop=False) for tr,g in _tracks_in_order(scored,'track_id').items()}\n"),),
         helper='_tracks_in_order', factory=tracks_in_order),
    dict(module='recover_displaced_soma_from_dim_name_reuse', function='correct',
         digests={'1b955773f3e4c43737bfe52596b317d48df35c25891d1e2c08c04b73f9f538ec'},
         edits=(('    by_track = {int(track): group.sort_values("frame")\n'
                 '                for track, group in strong.groupby("track_id")}\n',
                 '    by_track = _tracks_in_order(strong, "track_id")\n'),),
         helper='_tracks_in_order', factory=tracks_in_order),
    dict(module='restore_brief_name_in_raw_separable_contact', function='correct',
         digests={'ea9b9be0947d73b5b42ece34ff693e5bfe91dc0d4aae651d571b59e957971a15'},
         edits=(('    by_track = {int(track): group.sort_values("frame")\n'
                 '                for track, group in strong.groupby("track_id")}\n',
                 '    by_track = _tracks_in_order(strong, "track_id")\n'),),
         helper='_tracks_in_order', factory=tracks_in_order),
)
KEYS = ('versions', 'in_order', 'sorted', 'sorted_once', 'per_track', 'lazy', 'eager', 'tracks_built',
        'host_unique', 'host_counts', 'groups_in_order', 'groups_sorted')


def _digest(module):
    try:
        return _file_sha256(module.__file__)
    except (AttributeError, OSError, TypeError):
        return None


def install(stats=None):
    """Replace every loaded verified function (and its aliases) for the duration; restore afterwards."""
    return rewrite(SPECS, KEYS, '<track-sort-skips>', stats)


@contextmanager
def rewrite(specs, keys, filename, stats=None):
    """Install the specs' source edits (one spec per function), compiled under filename; restore afterwards."""
    stats = stats if stats is not None else {}
    for key in keys:
        stats.setdefault(key, 0)
    modules = [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']
    swap, added = {}, []
    for spec in specs:
        # A spec may name its module or hold the loaded module itself.
        targets = _loaded(spec['module']) if type(spec['module']) is str else [spec['module']]
        for module in {id(m): m for m in targets}.values():
            original = getattr(module, spec['function'], None)
            if (original is None or getattr(original, '__module__', None) != getattr(module, '__name__', None)
                    or getattr(original, '__wrapped__', None) is not None or id(original) in swap
                    or _digest(module) not in spec['digests']):
                continue
            try:
                source = inspect.getsource(original)
            except (OSError, TypeError):
                continue
            if any(source.count(old) != 1 for old, _ in spec['edits']):
                continue
            for old, new in spec['edits']:
                source = source.replace(old, new)
            if spec['helper'] is not None:
                vars(module)[spec['helper']] = spec['factory'](stats)
                added.append((module, spec['helper']))
            namespace = {}
            flags = original.__code__.co_flags & __future__.annotations.compiler_flag
            exec(compile(source, filename, 'exec', flags=flags, dont_inherit=True),
                 vars(module), namespace)
            replacement = namespace[spec['function']]
            replacement.__defaults__ = original.__defaults__
            replacement.__kwdefaults__ = original.__kwdefaults__
            replacement.__wrapped__ = original
            swap[id(original)] = (original, replacement)
            modules.append(module)
    stats['versions'] += len(swap)
    aliases = []
    if swap:
        for child in modules:
            names = getattr(child, '__dict__', None)
            if not names or swap.keys().isdisjoint(map(id, names.values())):
                continue
            for key, value in list(names.items()):
                entry = swap.get(id(value))
                # A module listed twice already holds the replacement the second time.
                if entry is not None and value is entry[0]:
                    aliases.append((child, key, value))
                    setattr(child, key, entry[1])
    try:
        yield stats
    finally:
        for child, key, value in reversed(aliases):
            setattr(child, key, value)
        for module, helper in added:
            vars(module).pop(helper, None)
