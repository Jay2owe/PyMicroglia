"""Index repeated companion-track lookups without changing Motion decisions.

This opt-in execution context leaves the shipped engine and its numerical
rules unchanged. Each immutable discovery table gets a private index, discarded
when discovery returns. Unusual dtypes retain the original implementation.
"""
from contextlib import contextmanager
import threading


class CompanionLookup:
    def __init__(self, attached, scaled_step):
        self.attached = attached
        self.scaled_step = scaled_step
        self.residents = {}
        self.owners = {}
        self.group_first = {}
        visible = attached[attached.physically_visible.astype(bool)]
        for track, group in visible.groupby('track_id'):
            first = int(group.frame.astype(int).min())
            self.group_first[int(track)] = first
            for owner, points in group.groupby('candidate_owner'):
                self.owners.setdefault(int(owner), []).append(
                    (int(track), len(points), first, tuple(points.itertuples(index=False))))

    def lookup(self, resident_track, owner_b, middle_frames, minimum_support,
               minimum_overlap, maximum_distance):
        resident_track = int(resident_track)
        if resident_track not in self.residents:
            rows = self.attached[self.attached.track_id.astype(int) == resident_track]
            seats = {}
            for row in rows.itertuples(index=False):
                seats.setdefault(row.frame, row)  # Original .loc keeps the first duplicate.
            self.residents[resident_track] = seats
        resident = self.residents[resident_track]
        frames = set(middle_frames)
        candidates = []
        for track, support, first, points in self.owners.get(int(owner_b), ()):
            if track == resident_track or support < minimum_support:
                continue
            overlap = 0
            for point in points:
                frame = int(point.frame)
                if frame in frames and frame in resident:
                    if self.scaled_step(resident[frame], point) <= maximum_distance:
                        overlap += 1
            if overlap >= minimum_overlap:
                candidates.append((overlap, support, track, first))
        if not candidates:
            pass
        # With zero support/overlap requirements the original also considers
        # visible tracks having no observations for the requested owner.
        if minimum_support <= 0 and minimum_overlap <= 0:
            present = {track for track, _, _, _ in self.owners.get(int(owner_b), ())}
            candidates.extend((0, 0, track, first) for track, first in self.group_first.items()
                              if track != resident_track and track not in present)
        if not candidates:
            return 0, 0, 0, -1
        overlap, support, track, first = max(candidates)
        return track, support, overlap, first


def _supported(table):
    return all(table[name].dtype.kind in 'iu' for name in
               ('track_id', 'frame', 'candidate_owner')) and table.physically_visible.dtype.kind == 'b'


@contextmanager
def install(module):
    """Temporarily index one frozen module; discovery lifetime prevents stale reuse."""
    original_discover = module.discover
    original_companion = module._companion
    local = threading.local()
    def discover(*args, **kwargs):
        previous = getattr(local, 'active', False)
        previous_index = getattr(local, 'index', None)
        local.active = True
        local.index = None
        try:
            return original_discover(*args, **kwargs)
        finally:
            local.active = previous
            local.index = previous_index
    def companion(attached, *args, **kwargs):
        if not getattr(local, 'active', False) or not _supported(attached):
            return original_companion(attached, *args, **kwargs)
        index = local.index
        if index is None or index.attached is not attached:
            index = local.index = CompanionLookup(attached, module._scaled_step)
        return index.lookup(*args, **kwargs)
    module.discover = discover
    module._companion = companion
    try:
        yield
    finally:
        module.discover = original_discover
        module._companion = original_companion


@contextmanager
def install_points(module):
    """Memoize the original point selector within an immutable discovery table."""
    original_discover = module.discover
    original_point = module._point
    local = threading.local()
    def discover(*args, **kwargs):
        previous = getattr(local, 'tables', None)
        local.tables = {}
        try:
            return original_discover(*args, **kwargs)
        finally:
            local.tables = previous
    def point(index, track, frame):
        tables = getattr(local, 'tables', None)
        if tables is None:
            return original_point(index, track, frame)
        identity = id(index)
        if identity not in tables:
            tables[identity] = (index, {})  # Retain the table so its identity cannot be reused.
        cache = tables[identity][1]
        key = (int(track), int(frame))
        if key not in cache:
            cache[key] = original_point(index, track, frame)
        return cache[key]
    module.discover = discover
    module._point = point
    try:
        yield
    finally:
        module.discover = original_discover
        module._point = original_point


@contextmanager
def indexed_owner_lookup():
    """Opt in around tracking.run; retain all native stages and their defaults."""
    from . import engine
    original_import = engine.importlib.import_module
    contexts = []
    def load(name, *args, **kwargs):
        module = original_import(name, *args, **kwargs)
        if name == 'pipeline':
            bounded = original_import('bounded_owner_excursion')
            context = install(bounded)
            context.__enter__()
            contexts.append(context)
            gap = original_import('gap_tolerant_reciprocal_seat_exchange')
            context = install_points(gap)
            context.__enter__()
            contexts.append(context)
        return module
    with engine._IMPORT_LOCK:
        engine.importlib.import_module = load
        try:
            yield
        finally:
            engine.importlib.import_module = original_import
            for context in reversed(contexts):
                context.__exit__(None, None, None)
