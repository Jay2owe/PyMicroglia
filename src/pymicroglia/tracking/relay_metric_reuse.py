"""Relay-pair metrics count and reuse each frame's union mask.

_relay_metrics built the persistent-or-relay union of every path frame, tested
it for any pixel and summed it, then for each overlap frame built the same union
again for the core count and two whole-frame identity masks for the distance.
Each path frame's union is now counted once (the same integer, zero exactly when
it has no pixel), the core count reads the path's own union, which holds the
same pixels and is never written, and the pair distance takes both cells'
pixels from one scan of that union: filtering the union's row-major pixels by
label keeps each cell's pixels in the order its own mask lists them, so the
nearest-neighbour query and its minimum are the same. Frames outside the path
keep the original expressions.
"""
import __future__
import hashlib
import inspect
from contextlib import contextmanager
from pathlib import Path

import numpy as np

SOURCE = '34603089a862624829d01040ba36c00af126b76b787d9fa3abc0217af1b90658'
EDITS = (
    ("    path: dict[int, np.ndarray] = {}\n"
     "    for t in range(max(0, relay_start - 1), min(len(labels), relay_end + 2)):\n"
     "        mask = (labels[t] == persistent) | (labels[t] == relay)\n"
     "        if np.any(mask):\n"
     "            path[t] = mask\n"
     "    path_areas = [int(mask.sum()) for mask in path.values()]\n",
     "    path: dict[int, np.ndarray] = {}\n"
     "    path_areas = []\n"
     "    for t in range(max(0, relay_start - 1), min(len(labels), relay_end + 2)):\n"
     "        frame = labels[t]\n"
     "        mask = (frame == persistent) | (frame == relay)\n"
     "        area = _relay_mask_area(mask)\n"
     "        if area:\n"
     "            path[t] = mask\n"
     "            path_areas.append(area)\n"),
    ("    distances = [_mask_distance(labels[t] == persistent, labels[t] == relay)\n"
     "                 for t in overlap]\n"
     "    core_counts = [_substantial_core_count(\n"
     "        (labels[t] == persistent) | (labels[t] == relay), minimum_core_px)\n"
     "                   for t in overlap]\n",
     "    distances = [_relay_pair_distance(path[int(t)], labels[t], persistent, relay)\n"
     "                 if int(t) in path else\n"
     "                 _mask_distance(labels[t] == persistent, labels[t] == relay)\n"
     "                 for t in overlap]\n"
     "    core_counts = [_substantial_core_count(\n"
     "        path[int(t)] if int(t) in path else\n"
     "        (labels[t] == persistent) | (labels[t] == relay), minimum_core_px)\n"
     "                   for t in overlap]\n"),
)


def _plain(array):
    return array.view(np.ndarray) if isinstance(array, np.ndarray) else np.asarray(array)


def helpers(module, stats):
    def mask_area(mask):
        return int(np.count_nonzero(_plain(mask)))

    def pair_distance(union, frame, first, second):
        stats['pair_distances'] += 1
        rows, columns = np.nonzero(_plain(union))
        values = _plain(frame)[rows, columns]
        is_first, is_second = values == first, values == second
        first_points = np.column_stack((rows[is_first], columns[is_first]))
        second_points = np.column_stack((rows[is_second], columns[is_second]))
        # The rest is _mask_distance's own calculation.
        if not len(first_points) or not len(second_points):
            return float("inf")
        if len(first_points) > len(second_points):
            first_points, second_points = second_points, first_points
        distances, _ = module.cKDTree(second_points).query(first_points, k=1)
        return float(np.min(distances))

    return dict(_relay_mask_area=mask_area, _relay_pair_distance=pair_distance)


@contextmanager
def install(module):
    stats = dict(supported=False, pair_distances=0)
    original = module._relay_metrics
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SOURCE:
        yield stats
        return
    source = inspect.getsource(original)
    if any(source.count(old) != 1 for old, _ in EDITS):
        yield stats
        return
    for old, new in EDITS:
        source = source.replace(old, new)
    names = vars(module)
    added = helpers(module, stats)
    names.update(added)
    namespace = {}
    exec(compile(source, '<relay-metric-reuse>', 'exec',
                 flags=__future__.annotations.compiler_flag, dont_inherit=True), names, namespace)
    replacement = namespace['_relay_metrics']
    replacement.__wrapped__ = original
    module._relay_metrics = replacement
    stats['supported'] = True
    try:
        yield stats
    finally:
        module._relay_metrics = original
        for name in added:
            names.pop(name, None)
