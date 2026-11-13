"""Read Issue 012 ownership observations from find_objects without region objects.

observations built a skimage RegionProperties object for every connected
region only to read its coordinates and centroid, and compared the whole frame
with each region label when masks were requested. regionprops takes its boxes
from find_objects, its coordinates as the box offset plus argwhere of the
region image, and its centroid as the mean of those coordinates times unit
spacing plus a zero offset; the same statements are used here. Masks are the
region image placed in its box of an otherwise empty frame.

reconcile asks for one frame's observations without masks up to four times
(dormancy, overlap, continuation swaps, red extension), and the frame changes
only when a proposal is applied. The rows depend only on the frame's dtype,
shape and values, and no caller writes to a row, so the last frame's rows are
returned again (in a new list) when the next frame has the same dtype, shape
and bytes. The frame's bytes are held as a Python bytes object, so the
disk-backed working-array storage is not involved.

detect_new_dormancy measures full source metrics only for the names that pass
its overlap thresholds, and only when two or more pass (dormancy_sources).
overlap_proposals measures them only for the names its two row filters can
keep (overlap_sources).
"""
import __future__
import hashlib
import inspect
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
from scipy import ndimage as scipy_ndimage

SOURCE = 'c2343dee1275675c35597dcfed84babd2596652f3f33a95dc63cdbbca475a2dc'


def observations(module, stats):
    def rows_for(frame, with_masks=False):
        rows = []
        numbered = module.connected_labels(frame, background=0, connectivity=2)
        if numbered.ndim != 2 or numbered.dtype.kind not in 'iu':
            stats['fallback'] += 1
            return module._ownership_region_original(frame, with_masks)
        spacing = np.full(2, 1.0)
        offset = np.zeros((2,), dtype=int)
        if not with_masks:
            # Without masks each region needs only its coordinates. One row-major
            # nonzero, stably grouped by label, lists every region's pixels in
            # ascending label order and in row-major order within a region,
            # which is the box offset plus argwhere of the region image. Each
            # region gets its own (n, 2) intp array, and the centroid is the
            # same expression on an array of the same values and layout.
            # Row-major positions of the nonzero pixels split by the row length
            # give np.nonzero's rows and columns (intp), about four times faster.
            flat = numbered.ravel()
            positive = np.flatnonzero(flat != 0)
            ys, xs = np.divmod(positive, numbered.shape[1]) if numbered.size else np.nonzero(numbered)
            found = flat[positive]
            order = np.argsort(found, kind='stable')
            every = np.column_stack((ys[order], xs[order]))
            found = found[order]
            starts = np.flatnonzero(np.r_[True, found[1:] != found[:-1]]) if len(found) else found[:0]
            ends = np.r_[starts[1:], len(found)]
            for start, end in zip(starts.tolist(), ends.tolist()):
                points = every[start:end].copy()
                centre = tuple((points * spacing + offset).mean(axis=0))
                identity = int(frame[tuple(points[0])])
                rows.append(module.Observation(
                    identity=identity, mask=None, area=int(len(points)),
                    centroid=np.asarray(centre, float), pixels=points))
            stats['calls'] += 1
            stats['grouped'] = stats.get('grouped', 0) + 1
            return rows
        for index, box in enumerate(scipy_ndimage.find_objects(numbered)):
            if box is None:
                continue
            label = index + 1
            image = numbered[box] == label
            indices = np.argwhere(image)
            object_offset = np.array([box[0].start, box[1].start])
            points = object_offset + indices
            centre = tuple(((object_offset + indices) * spacing + offset).mean(axis=0))
            mask = None
            if with_masks:
                mask = np.zeros(numbered.shape, bool)
                mask[box] = image
            identity = int(frame[tuple(points[0])])
            rows.append(module.Observation(
                identity=identity, mask=mask, area=int(len(points)),
                centroid=np.asarray(centre, float), pixels=points))
        stats['calls'] += 1
        return rows

    held = [None]

    def remembered(frame, with_masks=False):
        if with_masks or not isinstance(frame, np.ndarray):
            return rows_for(frame, with_masks)
        plain = frame.view(np.ndarray)
        entry = held[0]
        if (entry is not None and entry[0] == plain.dtype.str and entry[1] == plain.shape
                and np.array_equal(plain, np.frombuffer(entry[2], plain.dtype).reshape(plain.shape))):
            stats['remembered'] = stats.get('remembered', 0) + 1
            return list(entry[3])
        rows = rows_for(frame, with_masks)
        held[0] = (plain.dtype.str, plain.shape, plain.tobytes(), tuple(rows))
        return rows
    return remembered


def source_metrics(module, stats):
    """source_metrics with the one-pixel dilation test read from a dilated box.

    With dilation 1 the original asks, for each destination pixel, whether the
    nearest source pixel is within 1 * sqrt(2). Pixel offsets have integer
    squared lengths 0, 1, 2, 4, ...; the tree's distance for 2 is the correctly
    rounded sqrt(2), equal to np.sqrt(2), so the test holds exactly for pixels
    in the source's 3 x 3 dilation. That dilation is formed once per source
    pixel array (source lookups are rebuilt per frame and never edited) and
    the same Boolean values are averaged in the same order.
    """
    original = module.source_metrics
    held = {}

    def near_source(source_pixels):
        found = held.get(id(source_pixels))
        if found is None or found[0] is not source_pixels:
            points = source_pixels.astype(np.int64)
            low = points.min(axis=0) - 1
            shape = tuple(int(v) for v in points.max(axis=0) - low + 2)
            box = np.zeros(shape, bool)
            box[tuple((points - low).T)] = True
            dilated = scipy_ndimage.binary_dilation(box, np.ones((3, 3), bool))
            if len(held) > 4096:
                held.clear()
            found = held[id(source_pixels)] = (source_pixels, low, dilated)
            stats['source_boxes'] += 1
        return found[1], found[2]

    def metrics(previous, item, identity, lag, dilation, source_lookup):
        source_pixels = source_lookup.get(identity)
        pixels = item.pixels
        if (dilation != 1 or not isinstance(source_pixels, np.ndarray) or len(source_pixels) == 0
                or source_pixels.ndim != 2 or source_pixels.shape[1] != 2
                or source_pixels.dtype.kind not in 'iu' or not isinstance(pixels, np.ndarray)
                or pixels.ndim != 2 or pixels.shape[1] != 2 or pixels.dtype.kind not in 'iu'
                or len(pixels) == 0):
            stats['metric_fallback'] += 1
            return original(previous, item, identity, lag, dilation, source_lookup)
        source_area = int(len(source_pixels))
        destination_values = previous[tuple(item.pixels.T)]
        overlap = int(np.count_nonzero(destination_values == identity))
        low, dilated = near_source(source_pixels)
        local = pixels.astype(np.int64) - low
        inside = np.all((local >= 0) & (local < np.array(dilated.shape)), axis=1)
        within = np.zeros(len(pixels), bool)
        within[inside] = dilated[tuple(local[inside].T)]
        destination_dilation = float(np.mean(within))
        source_fraction = overlap / source_area
        destination_fraction = overlap / max(item.area, 1)
        overlap_coefficient = overlap / max(min(source_area, item.area), 1)
        new_pixels = item.pixels[destination_values != identity]
        red = (float(np.mean(lag[tuple(new_pixels.T)] >= 1.5))
               if len(new_pixels) else 0.0)
        blue = float(np.mean(lag[tuple(source_pixels.T)] <= -1.5))
        distance = float(np.linalg.norm(source_pixels.mean(axis=0) - item.centroid))
        stats['metrics'] += 1
        return {
            "identity": identity,
            "source_area": source_area,
            "overlap_px": overlap,
            "source_fraction": source_fraction,
            "destination_fraction": destination_fraction,
            "overlap_coefficient": overlap_coefficient,
            "destination_dilation_fraction": destination_dilation,
            "distance_px": distance,
            "red_extension_fraction": red,
            "blue_release_fraction": blue,
        }
    return metrics


DORMANCY = ('        sources = []\n'
            '        for identity in sorted(set(map(int, np.unique(\n'
            '                previous[tuple(item.pixels.T)]))) - {0}):\n'
            '            values = source_metrics(previous, item, identity, lag, 1,\n'
            '                                    source_lookup)\n'
            '            if values is None:\n'
            '                continue\n'
            '            if (values["overlap_px"] >= int(cfg["minimum_overlap_px"])\n'
            '                    and values["source_fraction"] >= float(\n'
            '                        cfg["merge_absorption_fraction"])):\n'
            '                sources.append(values)\n',
            '        sources = _dormancy_sources(previous, item, lag, cfg, source_lookup)\n')


def _inside(points, shape):
    return (isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[1] == 2
            and points.dtype.kind in 'iu'
            and (not len(points) or (int(points.min()) >= 0
                                     and bool((points.max(axis=0) < np.array(shape)).all()))))


def dormancy_sources(module, stats):
    """detect_new_dormancy's qualifying sources, measuring only the names that qualify.

    A source qualifies on its overlap and source fraction alone, which are the
    count of the destination pixels holding its name and that count over its
    pixel count, as source_metrics computes them; the caller then reads the
    list only when two or more qualify. So each name is tested with those two
    expressions, the thresholds are read in the same order, and the full
    metrics (the same call) are measured only for a list the caller reads.
    source_metrics has no side effects, and with every source pixel inside the
    lag frame (reconcile's lookup is the previous frame's own pixels) the
    skipped calls could not have raised. Other inputs keep the original loop.
    """
    def _dormancy_sources(previous, item, lag, cfg, source_lookup):
        values_at = previous[tuple(item.pixels.T)]
        names = sorted(set(map(int, np.unique(values_at))) - {0})
        plain = (isinstance(previous, np.ndarray) and isinstance(lag, np.ndarray)
                 and previous.ndim == 2 and lag.shape == previous.shape)
        if not plain or not all(_inside(source_lookup.get(identity), lag.shape)
                                for identity in names if source_lookup.get(identity) is not None):
            stats['dormancy_fallback'] += 1
            sources = []
            for identity in names:
                values = module.source_metrics(previous, item, identity, lag, 1, source_lookup)
                if values is None:
                    continue
                if (values["overlap_px"] >= int(cfg["minimum_overlap_px"])
                        and values["source_fraction"] >= float(cfg["merge_absorption_fraction"])):
                    sources.append(values)
            return sources
        passed = []
        for identity in names:
            source_pixels = source_lookup.get(identity)
            source_area = int(len(source_pixels)) if source_pixels is not None else 0
            if source_area == 0:
                continue
            overlap = int(np.count_nonzero(values_at == identity))
            if (overlap >= int(cfg["minimum_overlap_px"])
                    and overlap / source_area >= float(cfg["merge_absorption_fraction"])):
                passed.append(identity)
        stats['dormancy_items'] += 1
        if len(passed) < 2:
            return passed
        stats['dormancy_measured'] += 1
        return [module.source_metrics(previous, item, identity, lag, 1, source_lookup) for identity in passed]
    return _dormancy_sources


OVERLAP = ('        rows = candidate_sources(previous, item, lag, 0, source_lookup)\n',
           '        rows = _overlap_sources(previous, item, lag, cfg, source_lookup)\n')


def overlap_sources(module, stats):
    """overlap_proposals' source rows, measuring only the rows its filters can keep.

    overlap_proposals reads its rows only through two filters: merge sources
    need overlap_px >= minimum_overlap_px and source_fraction >=
    merge_source_fraction; tenure bids need overlap_px >= minimum_overlap_px
    and source_fraction >= 0.6 (stationary) or >= 0.3 (high-overlap
    continuation). Both values are the count of the destination pixels
    holding the name and that count over its pixel count, as source_metrics
    computes them, so the rows dropped here are exactly the rows both filters
    drop, and the kept rows keep their order and are the same calls. With the
    thresholds readable and every source pixel inside the lag frame, the
    skipped calls could not have raised; other inputs keep candidate_sources.
    """
    def _overlap_sources(previous, item, lag, cfg, source_lookup):
        values_at = previous[tuple(item.pixels.T)]
        names = sorted(set(map(int, np.unique(values_at))) - {0})
        try:
            minimum, merge = int(cfg["minimum_overlap_px"]), float(cfg["merge_source_fraction"])
        except Exception:
            minimum = None
        plain = (minimum is not None and isinstance(previous, np.ndarray) and isinstance(lag, np.ndarray)
                 and previous.ndim == 2 and lag.shape == previous.shape)
        if not plain or not all(_inside(source_lookup.get(identity), lag.shape)
                                for identity in names if source_lookup.get(identity) is not None):
            stats['overlap_fallback'] += 1
            return module.candidate_sources(previous, item, lag, 0, source_lookup)
        rows = []
        for identity in names:
            source_pixels = source_lookup.get(identity)
            source_area = int(len(source_pixels)) if source_pixels is not None else 0
            if source_area == 0:
                continue
            overlap = int(np.count_nonzero(values_at == identity))
            fraction = overlap / source_area
            if overlap >= minimum and (fraction >= merge or fraction >= 0.3):
                rows.append(module.source_metrics(previous, item, identity, lag, 0, source_lookup))
            else:
                stats['overlap_skipped'] += 1
        stats['overlap_items'] += 1
        return rows
    return _overlap_sources


@contextmanager
def install(module):
    stats = dict(supported=False, calls=0, fallback=0, metrics=0, metric_fallback=0, source_boxes=0,
                 dormancy_items=0, dormancy_measured=0, dormancy_fallback=0,
                 overlap_items=0, overlap_skipped=0, overlap_fallback=0)
    if (hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SOURCE
            or '_ownership_region_original' in vars(module)):
        yield stats
        return
    original = module.observations
    module._ownership_region_original = original
    module.observations = observations(module, stats)
    original_metrics = module.source_metrics
    module.source_metrics = source_metrics(module, stats)
    original_dormancy = module.detect_new_dormancy
    source = inspect.getsource(original_dormancy)
    if source.count(DORMANCY[0]) == 1:
        vars(module)['_dormancy_sources'] = dormancy_sources(module, stats)
        namespace = {}
        exec(compile(source.replace(*DORMANCY), '<dormancy-source-prefilter>', 'exec',
                     flags=__future__.annotations.compiler_flag, dont_inherit=True), vars(module), namespace)
        module.detect_new_dormancy = namespace['detect_new_dormancy']
    original_overlap = module.overlap_proposals
    source = inspect.getsource(original_overlap)
    if source.count(OVERLAP[0]) == 1:
        vars(module)['_overlap_sources'] = overlap_sources(module, stats)
        namespace = {}
        exec(compile(source.replace(*OVERLAP), '<overlap-source-prefilter>', 'exec',
                     flags=__future__.annotations.compiler_flag, dont_inherit=True), vars(module), namespace)
        module.overlap_proposals = namespace['overlap_proposals']
    stats['supported'] = True
    try:
        yield stats
    finally:
        module.observations = original
        module.source_metrics = original_metrics
        module.detect_new_dormancy = original_dormancy
        module.overlap_proposals = original_overlap
        vars(module).pop('_ownership_region_original', None)
        vars(module).pop('_dormancy_sources', None)
        vars(module).pop('_overlap_sources', None)


@contextmanager
def install_loader(history):
    original = history._issue_module
    seen = set()
    stats = []
    with ExitStack() as stack:
        def load(*args, **kwargs):
            module = original(*args, **kwargs)
            if (id(module) not in seen and getattr(module, '__name__', '') == 'reconcile_domain_ownership'
                    and hasattr(module, 'observations')):
                counter = stack.enter_context(install(module))
                stats.append(counter)
                if counter['supported']:
                    seen.add(id(module))
            return module
        history._issue_module = load
        try:
            yield stats
        finally:
            history._issue_module = original
