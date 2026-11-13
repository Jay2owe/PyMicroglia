"""Avoid impossible frame intersections and batch unchanged pair arithmetic."""
import ast
from bisect import bisect_right
from contextlib import contextmanager
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd


def frame_bits(arrays):
    if not arrays:
        return {}
    lower = min(int(v['frame'].min()) for v in arrays.values())
    upper = max(int(v['frame'].max()) for v in arrays.values())
    # Execution-memory bound only. Larger ranges retain the original search.
    if upper-lower > 65536:
        return {}
    result = {}
    for track, values in arrays.items():
        frames = values['frame']
        if len(np.unique(frames)) != len(frames):
            continue  # Preserve original assume_unique behavior for duplicates.
        bits = 0
        for frame in frames:
            bits |= 1 << (int(frame)-lower)
        result[track] = bits
    return result


def possible_overlap(bits, left, right, minimum):
    if left not in bits or right not in bits:
        return True
    return (bits[left] & bits[right]).bit_count() >= minimum


def finite_vector_median(values):
    array=np.asarray(values)
    if array.ndim!=1 or array.dtype!=np.dtype('float64') or not array.size or not np.isfinite(array).all():
        return np.median(values)
    middle=array.size//2
    indices=[middle] if array.size%2 else [middle-1,middle]
    partitioned=np.partition(array,indices)
    # Same NumPy mean and central order as median, including signed zeros.
    return np.mean(partitioned[indices[0]:indices[-1]+1])



def pair_summary_builder(arrays, summarize):
    # Private to one discovery. Track arrays are immutable during discovery.
    # Bounded tail cache changes execution work only; eviction recomputes exactly.
    from collections import OrderedDict
    full = {}
    tails = OrderedDict()
    def describe(frames, owners):
        summary = summarize(frames, owners)
        positive = np.flatnonzero(owners > 0)
        edges = None if not len(positive) else (
            frames[positive[0]], int(owners[positive[0]]),
            frames[positive[-1]], int(owners[positive[-1]]))
        return summary, edges
    def pair(left_track, right_track):
        left, right = arrays[left_track], arrays[right_track]
        if (int(left['frame'][0]), int(left['frame'][-1])) <= (int(right['frame'][0]), int(right['frame'][-1])):
            predecessor, successor = left_track, right_track
        else:
            predecessor, successor = right_track, left_track
        first, second = arrays[predecessor], arrays[successor]
        if predecessor not in full:
            full[predecessor] = describe(first['frame'], first['owner'])
        cut = int(first['frame'][-1]); key = (successor, cut)
        if key not in tails:
            select = second['frame'] > cut
            tails[key] = describe(second['frame'][select], second['owner'][select])
            if len(tails) > 4096:
                tails.popitem(last=False)
        else:
            tails.move_to_end(key)
        (a, ae), (b, be) = full[predecessor], tails[key]
        if ae is None:
            return b
        if be is None:
            return a
        with np.errstate(over='ignore'):
            joined = ae[3] == be[1] and be[0] == np.int64(ae[2]) + np.int64(1)
        counts = a[1].copy(); counts.update(b[1])
        runs = {owner: values.copy() for owner, values in a[2].items()}
        for owner, values in b[2].items():
            runs.setdefault(owner, []).extend(values[1:] if joined and owner == be[1] else values)
        return a[0] + b[0] - int(joined), counts, runs
    return pair


# Owner-relay discovery with every later partner of one reference track scored
# together (_batched_discover), on a dense track x frame grid. Separations are
# the same elementwise statements on the same values; the same-owner fraction
# is the same float count / length; the median takes the same central order
# statistics and the same mean of one or two values; the maximum does not
# depend on order. The virtual history is used only for its length and end
# frames, read from the sorted frame lists. Owner-run summaries use
# pair_summary_builder whenever an owner could be dedicated to the pair;
# otherwise only the run count is read, counted from per-track run starts. Rows
# are made in the same pair order. Duplicate or unsorted frames, non-finite
# geometry, wide frame ranges and non-positive overlap thresholds return None,
# so the compiled discovery runs. It is proven on this discovery source:
RELAY_SOURCE = 'e6f824267b3fce8b14762b8a6e7acc5a7ae34bbc67113eab5fb8337f4f5a6f4b'


def _grid(arrays, track_ids):
    """Dense presence, owner and geometry per track and frame, or None."""
    if not track_ids:
        return None
    for values in arrays.values():
        frames = values['frame']
        if len(frames) > 1 and not bool((frames[1:] > frames[:-1]).all()):
            return None
        if not all(bool(np.isfinite(values[name]).all()) for name in ('x', 'y', 'radius')):
            return None
    lower = min(int(arrays[t]['frame'][0]) for t in track_ids)
    upper = max(int(arrays[t]['frame'][-1]) for t in track_ids)
    if upper - lower > 65536:
        return None
    shape = (len(track_ids), upper - lower + 1)
    grid = dict(present=np.zeros(shape, bool), owner=np.zeros(shape, np.int64),
                x=np.zeros(shape), y=np.zeros(shape), radius=np.zeros(shape), lower=lower)
    for row, track in enumerate(track_ids):
        values = arrays[track]
        columns = values['frame'] - lower
        grid['present'][row, columns] = True
        for name in ('owner', 'x', 'y', 'radius'):
            grid[name][row, columns] = values[name]
    return grid


def _pair_measures(grid, row, columns, minimum_overlap):
    """Partner rows with enough overlap and their pair measures, in partner order."""
    later = grid['present'][row + 1:, columns]
    counts = later.sum(axis=1)
    chosen = np.flatnonzero(counts >= minimum_overlap)
    if not len(chosen):
        return []
    present = later[chosen]
    partners = chosen + row + 1
    counts = counts[chosen]
    block = np.ix_(partners, columns)
    left_owner = grid['owner'][row, columns]
    same = ((left_owner > 0) & (left_owner == grid['owner'][block]) & present).sum(axis=1)
    scale = np.maximum(grid['radius'][row, columns] + grid['radius'][block], 1.0)
    separations = np.hypot(grid['x'][row, columns] - grid['x'][block],
                           grid['y'][row, columns] - grid['y'][block]) / scale
    ordered = np.where(present, separations, np.inf)
    ordered.sort(axis=1)
    largest = np.where(present, separations, -np.inf).max(axis=1)
    middle = counts // 2
    lane = np.arange(len(chosen))
    upper = ordered[lane, middle]
    lower = ordered[lane, np.maximum(middle - 1, 0)]
    # np.mean of the one or two central values: their sum divided by the count.
    medians = np.where(counts % 2 == 1, upper, (lower + upper) / 2)
    first = present.argmax(axis=1)
    last = present.shape[1] - 1 - present[:, ::-1].argmax(axis=1)
    frames = columns + grid['lower']
    return list(zip(partners.tolist(), counts.tolist(), same.tolist(), medians.tolist(),
                    largest.tolist(), frames[first].tolist(), frames[last].tolist()))


class _RunCounts:
    """Owner-run counts of a track and of its tails, as _owner_run_summary counts them.

    A run starts at a positive owner whose owner differs from, or whose frame
    does not follow, the previous positive entry. A tail's first positive entry
    always starts a run. Joining follows pair_summary_builder.
    """

    def __init__(self, frames, owners):
        self.frames, self.owners = frames, owners
        n = len(frames)
        starts = [False] * n
        previous = None
        for index in range(n):
            if owners[index] > 0:
                starts[index] = (previous is None or owners[index] != owners[previous]
                                 or frames[index] != frames[previous] + 1)
                previous = index
        self.last = previous
        self.starts = starts
        self.after = [0] * (n + 1)  # starts at or after each position
        self.next = [n] * (n + 1)   # first positive position at or after each position
        for index in range(n - 1, -1, -1):
            self.after[index] = self.after[index + 1] + starts[index]
            self.next[index] = index if owners[index] > 0 else self.next[index + 1]

    def tail(self, start):
        """Run count and (first frame, first owner, last frame, last owner), or None."""
        first = self.next[start]
        if first == len(self.frames):
            return 0, None
        count = self.after[start] + (0 if self.starts[first] else 1)
        return count, (self.frames[first], self.owners[first],
                       self.frames[self.last], self.owners[self.last])


def _run_count_of(before, after, cut_index):
    a, ae = before.tail(0)
    b, be = after.tail(cut_index)
    if ae is None:
        return b
    if be is None:
        return a
    joined = ae[3] == be[1] and be[0] == ae[2] + 1
    return a + b - int(joined)


def _batched_discover(module, labels, points, params):
    """discover with every later partner of one reference track scored together, or None."""
    module.assert_target_free(params)
    scored = module.physical.attach_owners(points, labels)
    visible = scored[scored.physically_visible.astype(bool)].copy()
    groups = {int(track): group.sort_values("frame")
              for track, group in visible.groupby("track_id", sort=True)}
    arrays = {track: module._track_arrays(group) for track, group in groups.items()}
    owner_tracks = {
        int(owner): set(map(int, group.track_id))
        for owner, group in visible[visible.accepted_owner.astype(int) > 0]
        .groupby("accepted_owner", sort=True)
    }

    minimum_overlap = int(params.get("minimum_overlap_frames", 4))
    minimum_same_owner = float(params.get(
        "minimum_overlap_same_owner_fraction", 0.80))
    maximum_median_separation = float(params.get(
        "maximum_median_separation_sum_radii", 1.75))
    maximum_separation = float(params.get(
        "maximum_separation_sum_radii", 2.20))
    minimum_successor_extension = int(params.get(
        "minimum_successor_extension_frames", 8))
    minimum_support = int(params.get("minimum_dedicated_support_frames", 8))
    minimum_support_fraction = float(params.get(
        "minimum_dedicated_support_fraction", 0.15))
    minimum_runs = int(params.get("minimum_dedicated_runs", 2))
    minimum_transitions = int(params.get("minimum_owner_transitions", 4))
    minimum_competing = int(params.get("minimum_competing_owner_count", 1))
    minimum_competing_tracks = int(params.get(
        "minimum_competing_owner_tracks", 2))

    track_ids = sorted(groups)
    if not track_ids:
        return pd.DataFrame([]), scored
    grid = _grid(arrays, track_ids) if minimum_overlap >= 1 else None
    if grid is None:
        return None
    frames = {track: arrays[track]['frame'].tolist() for track in track_ids}
    if max(abs(grid['lower']), abs(grid['lower'] + grid['present'].shape[1])) > 1 << 62:
        return None  # pair_summary_builder joins frames in wrapping int64.
    runs = {track: _RunCounts(frames[track], arrays[track]['owner'].tolist()) for track in track_ids}
    # Owners that can be dedicated to a pair: every candidate owner is one of these.
    owned = {track: sorted({owner for owner in runs[track].owners if owner > 0})
             for track in track_ids}
    summary = pair_summary_builder(arrays, module._owner_run_summary)
    audits: list[dict] = []
    proposal_number = 0
    for left_index, left_track in enumerate(track_ids):
        columns = arrays[left_track]['frame'] - grid['lower']
        for (partner, overlap, same, median_separation, max_separation,
             first_common, last_common) in _pair_measures(grid, left_index, columns, minimum_overlap):
            right_track = track_ids[partner]
            same_fraction = same / overlap
            left, right = frames[left_track], frames[right_track]
            if (left[0], left[-1]) <= (right[0], right[-1]):
                predecessor, successor = left_track, right_track
            else:
                predecessor, successor = right_track, left_track
            before, after = frames[predecessor], frames[successor]
            tail = len(after) - bisect_right(after, before[-1])
            virtual_length = len(before) + tail
            virtual_last = after[-1] if tail else before[-1]
            extension = int(arrays[successor]["frame"][-1]
                            - arrays[predecessor]["frame"][-1])
            proposal_number += 1
            reasons: list[str] = []
            if same_fraction < minimum_same_owner:
                reasons.append("overlap_owner_not_conserved")
            if median_separation > maximum_median_separation:
                reasons.append("median_reference_separation_too_large")
            if max_separation > maximum_separation:
                reasons.append("maximum_reference_separation_too_large")
            if extension < minimum_successor_extension:
                reasons.append("successor_extension_too_short")

            pair = {left_track, right_track}
            if any(owner_tracks.get(owner, set()) <= pair
                   for owner in (*owned[left_track], *owned[right_track])):
                run_count, owner_counts, owner_run_frames = summary(left_track, right_track)
            else:
                # No owner can pass the first candidate test: only the count is read.
                run_count = _run_count_of(runs[predecessor], runs[successor], len(after) - tail)
                owner_counts = {}
            transitions = max(0, run_count - 1)
            candidates: list[dict] = []
            for owner, support in sorted(owner_counts.items()):
                owner_runs = owner_run_frames[owner]
                if not owner_tracks.get(owner, set()) <= pair:
                    continue
                if support < minimum_support:
                    continue
                if support / max(virtual_length, 1) < minimum_support_fraction:
                    continue
                if len(owner_runs) < minimum_runs:
                    continue
                competing = [
                    int(other) for other in owner_counts
                    if other != owner
                    and len(owner_tracks.get(int(other), set()))
                    >= minimum_competing_tracks]
                if len(competing) < minimum_competing:
                    continue
                candidates.append({
                    "dedicated_owner": owner,
                    "dedicated_support_frames": support,
                    "dedicated_support_fraction": support / virtual_length,
                    "dedicated_runs": len(owner_runs),
                    "first_dedicated_frame": owner_runs[0],
                    "competing_owners": competing,
                })
            if transitions < minimum_transitions:
                reasons.append("insufficient_owner_transitions")
            if len(candidates) != 1:
                reasons.append("dedicated_owner_not_unique")
            selected = candidates[0] if len(candidates) == 1 else {}
            eligible = not reasons
            audits.append({
                "proposal_id": f"P{proposal_number:04d}",
                "track_pair": str((left_track, right_track)),
                "predecessor_track": predecessor,
                "successor_track": successor,
                "overlap_first_frame": first_common,
                "overlap_last_frame": last_common,
                "overlap_frames": overlap,
                "overlap_same_owner_fraction": same_fraction,
                "median_separation_sum_radii": median_separation,
                "maximum_separation_sum_radii": max_separation,
                "successor_extension_frames": extension,
                "virtual_first_frame": before[0],
                "virtual_last_frame": virtual_last,
                "virtual_frames": virtual_length,
                "owner_runs": run_count,
                "owner_transitions": transitions,
                "dedicated_owner": selected.get("dedicated_owner", 0),
                "dedicated_support_frames": selected.get(
                    "dedicated_support_frames", 0),
                "dedicated_support_fraction": selected.get(
                    "dedicated_support_fraction", 0.0),
                "dedicated_runs": selected.get("dedicated_runs", 0),
                "first_dedicated_frame": selected.get(
                    "first_dedicated_frame", -1),
                "competing_owners": str(selected.get("competing_owners", [])),
                "eligible": eligible,
                "reason": "eligible" if eligible else "|".join(reasons),
            })
    return pd.DataFrame(audits), scored


@contextmanager
def faster_relay_pairs(module):
    original = module.discover
    tree = ast.parse(Path(module.__file__).read_text())
    function = next(v for v in tree.body if isinstance(v, ast.FunctionDef) and v.name == 'discover')
    arrays = [v for v in function.body if isinstance(v, ast.Assign) and ast.unparse(v.targets[0]) == 'arrays']
    assert len(arrays) == 1
    at = function.body.index(arrays[0])+1
    function.body[at:at] = ast.parse('_frame_bits = _make_frame_bits(arrays)\n_pair_summary = _make_pair_summary(arrays, _owner_run_summary)').body
    pairs = [v for v in ast.walk(function) if isinstance(v, ast.For) and ast.unparse(v.target) == 'right_track']
    assert len(pairs) == 1
    pairs[0].body[0:0] = ast.parse('if not _possible_overlap(_frame_bits, left_track, right_track, minimum_overlap):\n    continue').body
    inner = [v for v in pairs[0].body if isinstance(v, ast.For) and ast.unparse(v.target) == '(left_row, right_row)']
    assert len(inner) == 1
    index = pairs[0].body.index(inner[0])
    replacement = ast.parse('''same_owner = (left['owner'][left_indices] > 0) & (left['owner'][left_indices] == right['owner'][right_indices])
scale = np.maximum(left['radius'][left_indices] + right['radius'][right_indices], 1.0)
separations = np.hypot(left['x'][left_indices] - right['x'][right_indices], left['y'][left_indices] - right['y'][right_indices]) / scale
''').body
    pairs[0].body[index:index+1] = replacement
    medians=[node for node in ast.walk(function) if isinstance(node,ast.Call) and ast.unparse(node.func)=='np.median' and len(node.args)==1 and ast.unparse(node.args[0])=='separations']
    assert len(medians)==1,'Pair median call changed; equivalence review required'
    medians[0].func=ast.Name(id='_finite_vector_median',ctx=ast.Load())
    summaries=[node for node in ast.walk(function) if isinstance(node,ast.Call) and ast.unparse(node.func)=='_owner_run_summary' and [ast.unparse(arg) for arg in node.args]==['virtual_frames','virtual_owners']]
    assert len(summaries)==1
    summaries[0].func=ast.Name(id='_pair_summary',ctx=ast.Load())
    summaries[0].args=[ast.Name(id='left_track',ctx=ast.Load()),ast.Name(id='right_track',ctx=ast.Load())]
    namespace = dict(vars(module));namespace.update(_make_pair_summary=pair_summary_builder, _make_frame_bits=frame_bits, _possible_overlap=possible_overlap, _finite_vector_median=finite_vector_median)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(module.__file__)+'::pair_speed','exec'),namespace)
    compiled = namespace['discover']
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == RELAY_SOURCE:
        def discover(labels, points, params):
            result = _batched_discover(module, labels, points, params)
            return compiled(labels, points, params) if result is None else result
        discover.__wrapped__ = compiled
        module.discover = discover
    else:
        module.discover = compiled
    try:
        yield
    finally:
        module.discover = original
