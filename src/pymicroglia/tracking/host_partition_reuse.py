"""Index presence by frame and retain only consumed host-partition frames."""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import types

import numpy as np
from scipy import ndimage as ndi

SOURCE = '34603089a862624829d01040ba36c00af126b76b787d9fa3abc0217af1b90658'


def frame_index(labels):
    result = {}
    for t, frame in enumerate(labels):
        for identity in np.unique(frame):
            if identity != 0:
                result.setdefault(int(identity), []).append(t)
    return {identity: np.asarray(result[identity], dtype=np.intp) for identity in sorted(result)}


class FrameTables:
    """Per-frame label areas, presence and boxed pixel lists of one label movie.

    Areas are np.unique counts, so equal to count_nonzero(frame == label); pixel
    lists come from the label's find_objects box in row-major order, so equal
    to np.column_stack(np.nonzero(frame == label)).
    """

    def __init__(self, labels):
        self.plain = labels.view(np.ndarray)
        self.areas = {}
        self.boxes = {}
        self.present_sets = {}

    def area(self, t, label):
        table = self.areas.get(t)
        if table is None:
            values, counts = np.unique(self.plain[t], return_counts=True)
            table = self.areas[t] = dict(zip(map(int, values), map(int, counts)))
        return table.get(int(label), 0)

    def present(self, t):
        found = self.present_sets.get(t)
        if found is None:
            self.area(t, 0)
            found = self.present_sets[t] = set(self.areas[t])
        return found

    def points(self, t, label):
        frame = self.plain[t]
        boxes = self.boxes.get(t)
        if boxes is None:
            boxes = self.boxes[t] = ndi.find_objects(frame)
        label = int(label)
        box = boxes[label - 1] if 0 < label <= len(boxes) else None
        if box is None:
            return np.column_stack(np.nonzero(frame == label))
        rows, columns = np.nonzero(frame[box] == label)
        return np.column_stack((rows + box[0].start, columns + box[1].start))


# Whole-frame tests in the candidate loop and their FrameTables equivalents.
BOXED = (
    ('    identity_frames = _frames(labels)\n',
     '    identity_frames = _frames(labels)\n    _tables = _frame_tables(labels)\n'),
    ('            np.count_nonzero(labels[t] == identity) for t in frames], float)',
     '            _tables.area(t, identity) for t in frames], float)'),
    ('            before_position = _centre(labels[t - 1] == identity)\n'
     '            after_position = _centre(labels[t + 1] == identity)',
     '            before_position = np.mean(_tables.points(t - 1, identity), axis=0)\n'
     '            after_position = np.mean(_tables.points(t + 1, identity), axis=0)'),
    ('            for host in sorted(set(map(int, np.unique(labels[t]))) - {0}):\n'
     '                if (not np.any(labels[t - 1] == host)\n'
     '                        or not np.any(labels[t + 1] == host)):',
     '            for host in sorted(_tables.present(t) - {0}):\n'
     '                if (host not in _tables.present(t - 1)\n'
     '                        or host not in _tables.present(t + 1)):'),
    ('                host_mask = labels[t] == host\n'
     '                host_distance = float(cKDTree(\n'
     '                    np.column_stack(np.nonzero(host_mask))).query(predicted)[0])',
     '                host_distance = float(cKDTree(\n'
     '                    _tables.points(t, host)).query(predicted)[0])'),
    ('                    np.count_nonzero(labels[t - 1] == host),\n'
     '                    np.count_nonzero(labels[t + 1] == host)]))\n'
     '                host_growth = float(host_mask.sum() - expected_host_area)',
     '                    _tables.area(t - 1, host),\n'
     '                    _tables.area(t + 1, host)]))\n'
     '                host_growth = float(_tables.area(t, host) - expected_host_area)'),
)


def _frames_only(split, *args, **kwargs):
    # The engine's _split_interval returns an accepted proposal as its written
    # frames when asked for frames_only (fold batch 17); an engine without it
    # returns the whole movie, whose frames are the same.
    if 'frames_only' in inspect.signature(split).parameters:
        return split(*args, frames_only=True, **kwargs)
    return split(*args, **kwargs)


def boxed_frame_tables(labels):
    """FrameTables for a 2-D-per-frame, non-negative integer movie; None otherwise."""
    if not isinstance(labels, np.ndarray) or labels.ndim != 3 or labels.dtype.kind not in 'iu':
        return None
    if labels.size and (labels.min() < 0 or labels.max() > 2**20):
        return None
    return FrameTables(labels)


@contextmanager
def install(module, history=None, *aliases):
    stats = dict(supported=False, presence_calls=0, partitions=0,
                 retained_frame_bytes=0, avoided_movie_bytes=0)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SOURCE:
        yield stats
        return
    original = module.optimise_host_merge_graph
    original_frames = module._frames
    source = inspect.getsource(module.host_merge_graph_candidates)
    needle = '                partitions[row_number] = partitioned'
    if source.count(needle) != 1:
        yield stats
        return
    source = source.replace(needle,
        '                partitions[row_number] = _retain_partition_frame(t, partitioned)')
    # The partition's only reader is the retained frame above, so an accepted
    # proposal is returned as its frames rather than a copied movie.
    call = '                partition = _split_interval(\n'
    if source.count(call) == 1 and source.count('partitioned') - source.count('can be partitioned') == 2:
        source = source.replace(call, '                partition = _frames_only(_split_interval,\n')
    plain_source = source
    if all(source.count(old) == 1 for old, _ in BOXED):
        for old, new in BOXED:
            source = source.replace(old, new)
        stats['boxed'] = 0
    stats['supported'] = True

    def frames(labels):
        if labels.ndim != 3 or labels.dtype.kind not in 'iu':
            return original_frames(labels)
        stats['presence_calls'] += 1
        return frame_index(labels)

    def retain(t, partitioned):
        frame = partitioned[t].copy()
        stats['partitions'] += 1
        stats['retained_frame_bytes'] += frame.nbytes
        stats['avoided_movie_bytes'] += partitioned.nbytes - frame.nbytes
        # The optimizer only reads this exact frame. Public candidate APIs keep
        # returning full movies; this dictionary is private to the optimizer.
        return {int(t): frame}

    def optimize(*args, **kwargs):
        namespace = dict(original.__globals__)
        namespace['_retain_partition_frame'] = retain
        namespace['_frames_only'] = _frames_only
        labels = args[0] if args else kwargs.get('labels')
        boxed = 'boxed' in stats and boxed_frame_tables(labels) is not None
        if boxed:
            stats['boxed'] += 1
            namespace['_frame_tables'] = boxed_frame_tables
        exec(compile(source if boxed else plain_source, '<compact-host-partitions>', 'exec'), namespace)
        function = types.FunctionType(original.__code__, namespace, original.__name__,
                                      original.__defaults__, original.__closure__)
        try:
            return function(*args, **kwargs)
        finally:
            namespace.clear()

    alias_targets=[target for target in (history,*aliases) if target is not None
        and getattr(target,'optimise_host_merge_graph',None) is original]
    module._frames = frames
    module.optimise_host_merge_graph = optimize
    for target in alias_targets:target.optimise_host_merge_graph=optimize
    try:
        yield stats
    finally:
        module._frames = original_frames
        module.optimise_host_merge_graph = original
        for target in alias_targets:target.optimise_host_merge_graph=original
