"""Name every cell from its soma in the raw signal, not from the motion evidence.

A different approach from the host-election line. That line asks, after a merge
has already happened in the label map, which of two names should survive. This
one asks an earlier question: was it a merge at all?

The registered raw signal shows a bright soma for most cells. Where two of those
somas sit inside one connected motion blob with a deep intensity valley between
them, the blob is two cells that touch, not one cell. The valley is where the
boundary belongs, and each side keeps its own name.

Eight independently switchable rules, one attributable change each:

  A  soma-seat anchoring  - each tracked soma in the raw signal owns one name for
                            the whole life of that soma, and the body around it
                            carries that name. A name is not re-elected frame by
                            frame, so there is nothing for a passing cell to win.
  V  valley partition     - a component holding two seats is cut along the raw
                            intensity watershed between them, so each seat holds
                            its own side. Without V such a component is left as
                            the parent had it.
  Q  size-and-residency   - when two names have both worn one seat, the seat goes
     quarantine             to the name that wore it for more frames, ties broken
                            by the larger established body, and one name may hold
                            only one seat. A small newcomer cannot take an
                            established name off the cell that has been wearing it.
  F  full-region takeover - a name may take another name's body only by taking
                            effectively all of it. A partial claim on a body its
                            owner still holds is refused and given back, so cells
                            cannot slowly eat into each other.
  T  tenure claim         - a cell that has held its exact pixels for many frames
                            is not erased in one frame by a name with no claim
                            on them. The ground goes back.
  D  dim somas            - a cell too faint for the bright detector still sits on
                            a local maximum the signal falls away from. Those are
                            added as seats, so a dim cell is anchored too.
  B  prior-body barrier   - a cell too faint to show a soma is marked by the body
                            it held in the previous frame, so a seat's name
                            cannot flood across a neighbour that has no soma.
  G  shed the crumb      - a few pixels of a name left lying on a neighbour, too
                            small to be a body and detached from its own, go to
                            the neighbour they are touching.
  U  one body per name    - a name freed by a seat elsewhere does not stay behind
                            on a second, seatless body: that body reverts to
                            whichever name held it in the previous corrected frame.

With no rules enabled the stage reproduces the parent labels byte for byte.

The stage only ever renames pixels between identities that already exist in the
parent. It never creates an identity and never moves a pixel between the assigned
and the unclaimed map, so foreground accounting is exact by construction.
"""
from __future__ import annotations

import csv
import hashlib
import heapq
import inspect
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from merge_partition_speed import BoundedNdimage
from array_bounds import occupied_bounds, occupied_coordinates
ndi = BoundedNdimage()
from skimage.segmentation import watershed as _skimage_watershed

from project_paths import find_project_root


ROOT = find_project_root(Path(__file__))
sys.path.insert(0, str(ROOT / "code"))
from common import load_stack, save_stack  # noqa: E402


# skimage multiplies the markers by the mask, pads every array by one zero pixel
# and floods only masked pixels, in a heap ordered by image value and then push
# age. Markers are pushed in raster order and neighbours in a fixed direction
# order, so inside the mask's bounding box the push order, ages and labels are
# those of the whole frame; every pixel outside the box is unmasked and keeps
# the zero the markers-times-mask product gave it. That was checked for
# skimage 0.26's source; any other version floods the whole frame.
_WATERSHED_SOURCE = '189a88d34dff61b3b05f2a4b869c3b8c9c4c09f0e7542365ab344723e9ba6483'
_VALIDATE_SOURCE = '303650ca5c63beb2d34140b9f3df9707f4c72238f8995e347168d41578279532'


def _box_flooding_proven(flood) -> bool:
    from skimage.segmentation import _watershed
    return (flood is _watershed.watershed
            and hashlib.sha256(inspect.getsource(flood).encode()).hexdigest() == _WATERSHED_SOURCE
            and hashlib.sha256(inspect.getsource(_watershed._validate_inputs).encode()).hexdigest()
            == _VALIDATE_SOURCE)


def _boxed_watershed(flood):
    """skimage's watershed flooded inside the mask's bounding box and pasted into
    zeros of the result's dtype; other masks, markers, images or options flood
    the whole frame."""
    def watershed(image, markers=None, connectivity=1, offset=None, mask=None,
                  compactness=0, watershed_line=False):
        plain = (type(mask) is np.ndarray and mask.dtype == np.bool_ and mask.ndim == 2
                 and type(markers) is np.ndarray and markers.dtype.kind in 'iu'
                 and type(image) is np.ndarray and image.dtype.kind in 'iuf'
                 and image.shape == markers.shape == mask.shape
                 and connectivity == 1 and offset is None and compactness == 0
                 and watershed_line is False)
        rows = np.flatnonzero(mask.any(axis=1)) if plain else ()
        if not len(rows):
            return flood(image, markers, connectivity, offset, mask, compactness, watershed_line)
        columns = np.flatnonzero(mask.any(axis=0))
        box = (slice(int(rows[0]), int(rows[-1]) + 1), slice(int(columns[0]), int(columns[-1]) + 1))
        inner = flood(image[box], markers[box], 1, None, mask[box], 0, False)
        output = np.zeros(mask.shape, inner.dtype)
        output[box] = inner
        return output
    return watershed


watershed = (_boxed_watershed(_skimage_watershed) if _box_flooding_proven(_skimage_watershed)
             else _skimage_watershed)


STRUCTURE = np.ones((3, 3), bool)


# --------------------------------------------------------------------------
# soma detection, the project's own accepted detector
# --------------------------------------------------------------------------

def _plain(array):
    """A plain ndarray over the same memory (memory-mapped and pooled arrays included), or None."""
    if not isinstance(array, np.ndarray):
        return None
    return array if type(array) is np.ndarray else array.view(np.ndarray)


def _plain_view(value):
    value = np.asarray(value)
    return value.view(np.ndarray) if type(value) is not np.ndarray else value


def _bool2(mask, shape):
    mask = _plain(mask)
    return mask is not None and mask.dtype == np.bool_ and mask.shape == shape


def _bounds(mask):
    rows = np.flatnonzero(mask.any(axis=1))
    if not len(rows):
        return None
    columns = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(columns[0]), int(columns[-1]) + 1


def _opening_2x2(above):
    """binary_opening(above, ones((2, 2))) with zero borders, as shifted ANDs and ORs.

    scipy erodes a 2 x 2 structure onto each pixel and its upper-left
    neighbours and dilates onto each pixel and its lower-right neighbours;
    pixels outside the frame are False for both.
    """
    padded = np.pad(above, ((1, 0), (1, 0)))
    eroded = padded[:-1, :-1] & padded[:-1, 1:] & padded[1:, :-1] & padded[1:, 1:]
    padded = np.pad(eroded, ((0, 1), (0, 1)))
    return padded[:-1, :-1] | padded[:-1, 1:] | padded[1:, :-1] | padded[1:, 1:]


def component_clearance(somas: np.ndarray, component: np.ndarray, limit: float) -> np.ndarray:
    """Exact requested clearance; outside-component values are never consumed."""
    seeds = occupied_bounds(somas)
    target = occupied_bounds(component)
    if seeds is None or target is None:
        return (ndi.distance_transform_edt(~somas) >= limit) & component
    y0, y1 = min(seeds[0], target[0]), max(seeds[1], target[1])
    x0, x1 = min(seeds[2], target[2]), max(seeds[3], target[3])
    region = (slice(y0, y1), slice(x0, x1))
    # All zero-distance seeds and every queried pixel are retained.
    distances = ndi.distance_transform_edt(~somas[region])
    result = np.zeros_like(component)
    result[region] = (distances >= limit) & component[region]
    return result


def soma_cores(frame_raw: np.ndarray, assigned: np.ndarray, percentile: float,
               minimum_area: int) -> list[dict]:
    """Bright soma cores in one frame.

    This is the detector already accepted for the Issue 013 two-soma gate, at the
    settings its own 54-point sweep chose, so seats are found by a measurement
    that was fixed before this approach existed. Cores are kept only where they
    stand on assigned foreground, because a soma with no body cannot be named.

    The 2 x 2 opening is taken as shifted ANDs and ORs, each core's area from
    one count of the labelled frame and each core inside its find_objects box;
    other inputs take the whole-frame form.
    """
    threshold = np.percentile(frame_raw, percentile)
    above = _plain_view(frame_raw >= threshold)
    if above.ndim == 2 and above.dtype == np.bool_:
        bright = _opening_2x2(above)
    else:
        bright = ndi.binary_opening(frame_raw >= threshold, np.ones((2, 2), bool))
    numbered, count = ndi.label(bright, structure=STRUCTURE)
    numbered = np.asarray(numbered)
    if (np.asarray(assigned).dtype != np.bool_ or np.shape(assigned) != numbered.shape
            or numbered.dtype.kind not in 'iu' or numbered.ndim != 2
            or np.shape(frame_raw) != numbered.shape):
        return _soma_cores_whole(frame_raw, assigned, percentile, minimum_area)
    # (numbered == number) & assigned holds exactly these pixels, all inside
    # the component's find_objects box; the mask, its row-major coordinates
    # and its raw values are taken there.
    held = _plain_view(assigned)
    raw = _plain_view(frame_raw)
    sizes = np.bincount(numbered[held], minlength=count + 1)
    boxes = ndi.find_objects(numbered)
    found = []
    for number in range(1, count + 1):
        area = int(sizes[number])
        if area < minimum_area:
            continue
        box = boxes[number - 1]
        local = (numbered[box] == number) & held[box]
        mask = np.zeros(numbered.shape, bool)
        mask[box] = local
        rows, columns = np.nonzero(local)
        rows = rows + box[0].start
        columns = columns + box[1].start
        found.append({
            "mask": mask,
            "row": float(rows.mean()),
            "column": float(columns.mean()),
            "peak": float(raw[box][local].max()),
            "area": area,
            "dim": False,
        })
    return found

def _soma_cores_whole(frame_raw: np.ndarray, assigned: np.ndarray, percentile: float,
                      minimum_area: int) -> list[dict]:
    """The detector over whole frames, for inputs the boxed form does not take."""
    threshold = np.percentile(frame_raw, percentile)
    bright = ndi.binary_opening(frame_raw >= threshold, np.ones((2, 2), bool))
    numbered, count = ndi.label(bright, structure=STRUCTURE)
    found = []
    for number in range(1, count + 1):
        mask = (numbered == number) & assigned
        area = int(mask.sum())
        if area < minimum_area:
            continue
        rows, columns = occupied_coordinates(mask)
        found.append({
            "mask": mask,
            "row": float(rows.mean()),
            "column": float(columns.mean()),
            "peak": float(frame_raw[mask].max()),
            "area": area,
            "dim": False,
        })
    return found


def dim_soma_cores(frame_raw: np.ndarray, assigned: np.ndarray,
                   minimum_prominence: float, smooth: float, peak_floor: float,
                   minimum_basin_px: int, exclude: np.ndarray) -> list[dict]:
    """Somas the bright detector misses, found by how far the signal drops around them.

    Jamie: "if there is a big drop in intensity within a blob". A dim cell has no
    absolute brightness to be found by, but it still sits on a local maximum that
    the signal falls away from on every side before it climbs to any brighter
    cell. That fall is the whole test: a maximum survives when the highest saddle
    joining it to a brighter maximum lies below `minimum_prominence` of its own
    height. Everything the bright detector already found is excluded, so this is
    strictly an addition and the accepted detector is never overruled.
    """
    work = ndi.gaussian_filter(frame_raw.astype(np.float32), smooth)
    work = np.where(assigned, work, 0.0)
    coordinates = np.array(np.nonzero(assigned)).T
    coordinates = coordinates[np.argsort(work[assigned])[::-1]]
    height, width = work.shape
    index = -np.ones(work.shape, np.int32)
    parent: list[int] = []
    peak: list[float] = []
    alive: list[bool] = []

    def find(key: int) -> int:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for row, column in coordinates:
        value = float(work[row, column])
        roots = set()
        for step_row in (-1, 0, 1):
            for step_column in (-1, 0, 1):
                next_row, next_column = row + step_row, column + step_column
                if (0 <= next_row < height and 0 <= next_column < width
                        and index[next_row, next_column] >= 0):
                    roots.add(find(int(index[next_row, next_column])))
        if not roots:
            parent.append(len(parent))
            peak.append(value)
            alive.append(True)
            index[row, column] = len(parent) - 1
            continue
        ordered = sorted(roots, key=lambda key: -peak[key])
        index[row, column] = ordered[0]
        for other in ordered[1:]:
            if value > (1.0 - minimum_prominence) * peak[other]:
                alive[other] = False      # a shallow saddle: the same cell
            parent[other] = ordered[0]

    live = [key for key in range(len(peak))
            if alive[key] and peak[key] >= peak_floor]
    if not live:
        return []
    markers = np.zeros(work.shape, np.int32)
    for number, key in enumerate(live, 1):
        markers[index == key] = number
    basins = watershed(-work, markers, mask=assigned)
    found = []
    for number, key in enumerate(live, 1):
        mask = basins == number
        if int(mask.sum()) < minimum_basin_px or (mask & exclude).any():
            continue
        rows, columns = occupied_coordinates(mask)
        weight = work[mask]
        found.append({
            "mask": mask,
            "row": float((rows * weight).sum() / weight.sum()),
            "column": float((columns * weight).sum() / weight.sum()),
            "peak": float(peak[key]),
            "area": int(mask.sum()),
            "dim": True,
        })
    return found


def valley_ratio(frame_raw: np.ndarray, region: np.ndarray,
                 first: np.ndarray, second: np.ndarray) -> float:
    """How deep the dip between two somas is, as a fraction of the weaker soma.

    The measure is the widest-path bottleneck: the highest intensity a path from
    one soma to the other can be forced to stay above, anywhere inside the shared
    body. A low ratio is Jamie's "big drop in intensity within a blob" and means
    the two somas are only joined through dark pixels.

    The search pushes only pixels of the first soma and expands only into the
    shared body, so it runs inside their joint bounding box on Python lists.
    Local coordinates keep the heap's (value, row, column) order, every float32
    value is held exactly as a Python float, and the weaker soma is still
    measured over both full masks. Other inputs take the whole-frame form.
    """
    raw = _plain(frame_raw)
    if not (raw is not None and raw.ndim == 2 and raw.dtype.kind in 'iuf'
            and _bool2(region, raw.shape) and _bool2(first, raw.shape)
            and _bool2(second, raw.shape)):
        return _valley_ratio_whole(frame_raw, region, first, second)
    region, first, second = _plain(region), _plain(first), _plain(second)
    box = _bounds(first)
    if box is None:
        return 1.0
    other = _bounds(region)
    if other is not None:
        box = (min(box[0], other[0]), max(box[1], other[1]),
               min(box[2], other[2]), max(box[3], other[3]))
    y0, y1, x0, x1 = box
    values = raw[y0:y1, x0:x1].astype(np.float32).tolist()
    inside = region[y0:y1, x0:x1].tolist()
    target = second[y0:y1, x0:x1].tolist()
    height, width = y1 - y0, x1 - x0
    best = [[-np.inf] * width for _ in range(height)]
    heap = []
    rows, columns = np.nonzero(first[y0:y1, x0:x1])
    for row, column in zip(rows.tolist(), columns.tolist()):
        value = values[row][column]
        if value > best[row][column]:
            best[row][column] = value
            heapq.heappush(heap, (-value, row, column))
    while heap:
        negative, row, column = heapq.heappop(heap)
        value = -negative
        if value < best[row][column]:
            continue
        if target[row][column]:
            weaker = min(float(raw[first].astype(np.float32).max()),
                         float(raw[second].astype(np.float32).max()))
            return float(value / weaker) if weaker else 1.0
        for step_row in (-1, 0, 1):
            next_row = row + step_row
            if not 0 <= next_row < height:
                continue
            inside_row, best_row, values_row = inside[next_row], best[next_row], values[next_row]
            for step_column in (-1, 0, 1):
                next_column = column + step_column
                if not 0 <= next_column < width or not inside_row[next_column]:
                    continue
                carried = min(value, values_row[next_column])
                if carried > best_row[next_column]:
                    best_row[next_column] = carried
                    heapq.heappush(heap, (-carried, next_row, next_column))
    return 1.0

def _valley_ratio_whole(frame_raw: np.ndarray, region: np.ndarray,
                        first: np.ndarray, second: np.ndarray) -> float:
    """The widest-path search over the whole frame, for inputs the boxed form does not take."""
    values = frame_raw.astype(np.float32)
    best = np.full(values.shape, -np.inf, np.float32)
    import heapq
    heap: list[tuple[float, int, int]] = []
    for row, column in zip(*np.nonzero(first)):
        value = values[row, column]
        if value > best[row, column]:
            best[row, column] = value
            heapq.heappush(heap, (-float(value), int(row), int(column)))
    height, width = values.shape
    while heap:
        negative, row, column = heapq.heappop(heap)
        value = -negative
        if value < best[row, column]:
            continue
        if second[row, column]:
            weaker = min(float(values[first].max()), float(values[second].max()))
            return float(value / weaker) if weaker else 1.0
        for step_row in (-1, 0, 1):
            for step_column in (-1, 0, 1):
                next_row, next_column = row + step_row, column + step_column
                if not (0 <= next_row < height and 0 <= next_column < width):
                    continue
                if not region[next_row, next_column]:
                    continue
                carried = min(value, float(values[next_row, next_column]))
                if carried > best[next_row, next_column]:
                    best[next_row, next_column] = carried
                    heapq.heappush(heap, (-carried, next_row, next_column))
    return 1.0


# --------------------------------------------------------------------------
# seat tracks
# --------------------------------------------------------------------------

def track_seats(per_frame: list[list[dict]], maximum_step: float,
                maximum_gap: int) -> list[dict]:
    """Link somas across frames into seats by proximity.

    A soma moves slowly between frames, so nearest-inside-a-radius is enough and
    is deliberately weaker than the tracker this project already has: a seat is
    evidence about where a cell is, not a second opinion on identity.
    """
    tracks: list[dict] = []
    for index, somas in enumerate(per_frame):
        candidates = []
        for track in tracks:
            if index - track["frames"][-1] > maximum_gap + 1:
                continue
            for position, soma in enumerate(somas):
                distance = float(np.hypot(soma["row"] - track["row"],
                                          soma["column"] - track["column"]))
                if distance <= maximum_step:
                    candidates.append((distance, id(track), position, track))
        used_tracks: set[int] = set()
        used_somas: set[int] = set()
        for distance, key, position, track in sorted(candidates,
                                                     key=lambda item: item[0]):
            if key in used_tracks or position in used_somas:
                continue
            used_tracks.add(key)
            used_somas.add(position)
            soma = somas[position]
            track["frames"].append(index)
            track["masks"].append(soma["mask"])
            track["row"], track["column"] = soma["row"], soma["column"]
            track["peaks"].append(soma["peak"])
        for position, soma in enumerate(somas):
            if position in used_somas:
                continue
            tracks.append({
                "frames": [index], "masks": [soma["mask"]],
                "row": soma["row"], "column": soma["column"],
                "peaks": [soma["peak"]], "dim": [soma["dim"]],
            })
    for number, track in enumerate(tracks):
        track["seat"] = number
    return tracks


def established_area(labels: np.ndarray, identity: int, indices: list[int]) -> int:
    """Median body area of a name over the frames a seat was alive."""
    areas = [int(np.count_nonzero(labels[index] == identity)) for index in indices]
    areas = [area for area in areas if area > 0]
    return int(np.median(areas)) if areas else 0


def long_enough(track: dict, minimum_frames: int,
                minimum_dim_frames: int) -> bool:
    """A seat found only by prominence must prove itself for longer.

    A bright core is the accepted detector's own output. A dim maximum is a
    weaker claim, so it is asked to persist before it may name anything: the
    first dim lattice awarded 111 seats and doubled the duplicate rows.
    """
    wanted = (minimum_dim_frames if all(track.get("dim", [False]))
              else minimum_frames)
    return len(track["frames"]) >= wanted


MAXIMUM_COUNTED_LABEL = 1 << 13  # per-frame counts held for one call: at most ~30 MB


def seat_claims(labels: np.ndarray, tracks: list[dict], minimum_frames: int,
                minimum_dim_frames: int) -> list[dict]:
    """Every name that has worn a seat, with how long it wore it and how big it is.

    A soma's vote is the name most often under it, ties going to the name seen
    first in raster order (Counter.most_common), from one np.unique with first
    indices and counts; a name's established area counts its pixels from one
    bincount per frame. Other label stacks take the original form.
    """
    movie = _plain(labels)
    if not (movie is not None and movie.ndim == 3 and movie.dtype.kind in 'iu'):
        return _seat_claims_whole(labels, tracks, minimum_frames, minimum_dim_frames)
    counted = {}

    def area(index, identity):
        counts = counted.get(index)
        if counts is None:
            frame = movie[index]
            top = int(frame.max()) if frame.size else 0
            usable = frame.dtype.kind == 'u' or (frame.size and int(frame.min()) >= 0)
            counts = (np.bincount(frame.ravel(), minlength=1)
                      if usable and top < MAXIMUM_COUNTED_LABEL else False)
            counted[index] = counts
        if counts is False:
            return int(np.count_nonzero(movie[index] == identity))
        return int(counts[identity]) if 0 <= identity < len(counts) else 0

    claims = []
    for track in tracks:
        if not long_enough(track, minimum_frames, minimum_dim_frames):
            continue
        votes: Counter = Counter()
        for index, mask in zip(track["frames"], track["masks"]):
            if not _bool2(mask, labels.shape[1:]):
                names = Counter(int(value) for value in labels[index][mask] if value)
                if names:
                    votes[names.most_common(1)[0][0]] += 1
                continue
            # The mask's pixels in raster order, from its flat indices.
            picked = movie[index].ravel()[np.flatnonzero(_plain(mask))]
            picked = picked[picked != 0]
            if picked.size:
                found, first_at, counts = np.unique(picked, return_index=True, return_counts=True)
                best = max(range(found.size), key=lambda j: (counts[j], -first_at[j]))
                votes[int(found[best])] += 1
        for identity, frames in votes.items():
            areas = [area(index, identity) for index in track["frames"]]
            areas = [value for value in areas if value > 0]
            claims.append({
                "seat": track["seat"],
                "identity": identity,
                "frames_worn": frames,
                "seat_frames": len(track["frames"]),
                "first": track["frames"][0],
                "last": track["frames"][-1],
                "established_px": int(np.median(areas)) if areas else 0,
            })
    return claims

def _seat_claims_whole(labels: np.ndarray, tracks: list[dict], minimum_frames: int,
                       minimum_dim_frames: int) -> list[dict]:
    """Seat claims read one name at a time, for label stacks the counted form does not take."""
    claims = []
    for track in tracks:
        if not long_enough(track, minimum_frames, minimum_dim_frames):
            continue
        votes: Counter = Counter()
        for index, mask in zip(track["frames"], track["masks"]):
            names = Counter(int(value) for value in labels[index][mask] if value)
            if names:
                votes[names.most_common(1)[0][0]] += 1
        for identity, frames in votes.items():
            claims.append({
                "seat": track["seat"],
                "identity": identity,
                "frames_worn": frames,
                "seat_frames": len(track["frames"]),
                "first": track["frames"][0],
                "last": track["frames"][-1],
                "established_px": established_area(labels, identity,
                                                   track["frames"]),
            })
    return claims


def award_seats(claims: list[dict], rules: set[str], minimum_frames_worn: int,
                minimum_fraction_worn: float) -> dict[int, int]:
    """Give each seat one name and each name one seat.

    Rule Q is Jamie's quarantine stated where it can be checked: residency first,
    size second. A name that has worn a seat for sixty frames cannot lose it to a
    name that wore it for three, whatever the motion evidence did on the frame
    they touched. Without Q the seat simply takes whichever name it wore most,
    and a name may end up on two somas at once.

    One name may still hold several seats when those seats never overlap in time,
    because that is one cell whose soma was lost and found again rather than two
    cells.
    """
    ranked = sorted(
        (claim for claim in claims
         if claim["frames_worn"] >= minimum_frames_worn
         and claim["frames_worn"] / claim["seat_frames"] >= minimum_fraction_worn),
        key=lambda claim: (-claim["frames_worn"], -claim["established_px"],
                           claim["identity"]))
    owner: dict[int, int] = {}
    if "Q" not in rules:
        for claim in ranked:
            owner.setdefault(claim["seat"], claim["identity"])
        return owner
    held: dict[int, list[tuple[int, int]]] = {}
    for claim in ranked:
        if claim["seat"] in owner:
            continue
        spans = held.get(claim["identity"], [])
        if any(claim["first"] <= last and first <= claim["last"]
               for first, last in spans):
            continue          # the name is already on a soma at the same time
        owner[claim["seat"]] = claim["identity"]
        held.setdefault(claim["identity"], []).append((claim["first"],
                                                       claim["last"]))
    return owner


# --------------------------------------------------------------------------
# per-frame enforcement
# --------------------------------------------------------------------------

def partition_component(frame_raw: np.ndarray, component: np.ndarray,
                        seats: list[dict], maximum_ratio: float,
                        ) -> tuple[list[tuple[int | None, np.ndarray]],
                                   list[tuple[int, int]]]:
    """Cut one object into one region per soma, on the raw intensity between them.

    Every soma found in the object takes part, named or not. An unnamed soma is
    still a cell, and its territory is not the named cell's to take: leaving it
    out is what let one name swallow a neighbour the parent had kept separate.
    Only regions whose seat carries a name are renamed by the caller; the rest
    keep what the parent called them.

    Two somas joined over a shallow saddle are not evidence of anything - the raw
    signal cannot say whether that is one cell with a bright process or two cells
    lying across each other - so they share one region. That region is named only
    if exactly one name is claimed inside it.
    """
    parent = {seat["seat"]: seat["seat"] for seat in seats}

    def find(key: int) -> int:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    grouped: list[tuple[int, int]] = []
    for first in range(len(seats)):
        for second in range(first + 1, len(seats)):
            if seats[first].get("prior") or seats[second].get("prior"):
                continue          # a body is not a soma; it is never grouped away
            ratio = valley_ratio(frame_raw, component,
                                 seats[first]["mask"], seats[second]["mask"])
            if ratio > maximum_ratio:
                parent[find(seats[first]["seat"])] = find(seats[second]["seat"])
                grouped.append((seats[first]["seat"], seats[second]["seat"]))

    groups: dict[int, list[dict]] = {}
    for seat in seats:
        groups.setdefault(find(seat["seat"]), []).append(seat)

    def speaker(members: list[dict]) -> int | None:
        if any(seat.get("prior") for seat in members):
            return None            # this region is another cell's, held from last frame
        named = [seat for seat in members if seat["owned"]]
        if len(named) != 1:
            return None            # nobody, or two names that cannot be told apart
        return named[0]["seat"]

    if len(groups) == 1:
        only = next(iter(groups))
        return [(speaker(groups[only]), component)], grouped

    markers = np.zeros(component.shape, np.int32)
    for number, key in enumerate(sorted(groups), 1):
        for seat in groups[key]:
            markers[seat["mask"] & component] = number
    basins = watershed(-frame_raw.astype(np.int32), markers, mask=component)
    return ([(speaker(groups[key]), basins == number)
             for number, key in enumerate(sorted(groups), 1)], grouped)


LARGEST_BOXED_LABEL = 1 << 22


def _widened(box, shape):
    return tuple(slice(max(0, part.start - 1), min(size, part.stop + 1)) for part, size in zip(box, shape))


def _union(first, second):
    if first is None:
        return second
    return tuple(slice(min(a.start, b.start), max(a.stop, b.stop)) for a, b in zip(first, second))


def shed_crumbs(current: np.ndarray, minimum_body: int, minimum_piece: int,
                ) -> tuple[np.ndarray, list[dict]]:
    """A name does not keep a crumb that is lying on another cell.

    Moving a name onto the soma it belongs to can leave a few pixels of it
    stranded on a neighbour, which the strict duplicate audit counts as the name
    appearing in two places. A piece too small to be a body, detached from the
    name's real body and touching another name, is given to the neighbour it is
    touching. Nothing is deleted: the piece changes name, it does not change
    from assigned to unclaimed.

    Each identity is labelled in its box widened by the 3 x 3 reach; its
    components, their raster numbering, each crumb's ring and the ring's raster
    order are those of the whole frame. A receiving identity's box grows by the
    giver's box, so it still holds all of its pixels.
    """
    result = current.copy()
    plain = np.asarray(result)
    plain = plain.view(np.ndarray) if isinstance(plain, np.memmap) else plain
    identities = sorted(set(map(int, np.unique(result))) - {0})
    if (plain.ndim != 2 or plain.dtype.kind not in 'iu' or np.shape(STRUCTURE) != (3, 3)
            or (identities and (identities[0] < 0 or identities[-1] >= LARGEST_BOXED_LABEL))
            or (plain.size and int(plain.min()) < 0)):
        return _shed_crumbs_whole(current, minimum_body, minimum_piece)
    boxes = list(ndi.find_objects(plain))
    shed = []
    for identity in identities:
        window = _widened(boxes[identity - 1], plain.shape)
        local = plain[window]
        inside = local == identity
        # Every component is smaller than the whole identity, so an
        # identity below the body floor is skipped as the whole form skips it.
        if int(np.count_nonzero(inside)) < minimum_body:
            continue
        numbered, count = ndi.label(inside, structure=STRUCTURE)
        sizes = np.bincount(np.asarray(numbered).ravel(), minlength=count + 1)
        areas = {number: int(sizes[number]) for number in range(1, count + 1)}
        if not areas or max(areas.values()) < minimum_body:
            continue
        for number, area in areas.items():
            if area >= minimum_body or area < minimum_piece:
                continue
            piece = numbered == number
            edge = ndi.binary_dilation(piece, STRUCTURE) & ~piece
            neighbours = Counter(int(value) for value in local[edge]
                                 if value not in (0, identity))
            if not neighbours:
                continue
            successor = neighbours.most_common(1)[0][0]
            local[piece] = successor
            boxes[successor - 1] = _union(boxes[successor - 1], boxes[identity - 1])
            shed.append({"identity": identity, "given_to": successor,
                         "pixels": area})
    return result, shed


def _shed_crumbs_whole(current: np.ndarray, minimum_body: int, minimum_piece: int,
                       ) -> tuple[np.ndarray, list[dict]]:
    """Crumb shedding over whole frames, for label maps the boxed form does not take."""
    result = current.copy()
    shed = []
    for identity in sorted(set(map(int, np.unique(result))) - {0}):
        numbered, count = ndi.label(result == identity, structure=STRUCTURE)
        sizes = np.bincount(np.asarray(numbered).ravel(), minlength=count + 1)
        areas = {number: int(sizes[number]) for number in range(1, count + 1)}
        if not areas or max(areas.values()) < minimum_body:
            continue
        for number, area in areas.items():
            if area >= minimum_body or area < minimum_piece:
                continue
            piece = numbered == number
            edge = ndi.binary_dilation(piece, STRUCTURE) & ~piece
            neighbours = Counter(int(value) for value in result[edge]
                                 if value not in (0, identity))
            if not neighbours:
                continue
            successor = neighbours.most_common(1)[0][0]
            result[piece] = successor
            shed.append({"identity": identity, "given_to": successor,
                         "pixels": area})
    return result, shed


def tenure_claim(labels: np.ndarray, minimum_frames: int, minimum_piece: int,
                surrender_fraction: float, offset: int, reach: int = 2,
                ) -> tuple[np.ndarray, list[dict]]:
    """A cell that has held its ground for a long time is not erased in one frame.

    So the claim is counted per pixel, not per cell: how many consecutive frames
    the current holder has held this exact pixel. A name that has held its ground
    for `minimum_frames` and is then left with nothing by a taker that has no
    claim there at all has not lost a contest - it has been erased - and the
    ground goes straight back.

    Deliberately narrow. It fires only when the long-standing cell would keep
    nothing. A cell that keeps part of its body is having an argument, not being
    erased, and is left alone.
    """
    result = labels.copy()
    tenure = np.zeros(labels.shape[1:], np.int32)
    holder = result[0].copy()
    tenure[holder > 0] = 1
    returned = []
    for index in range(1, result.shape[0]):
        current = result[index]
        for loser in sorted(set(map(int, np.unique(holder))) - {0}):
            body = holder == loser
            area = int(body.sum())
            if area < minimum_piece:
                continue
            standing = int(np.count_nonzero(current[body] == loser))
            if standing > surrender_fraction * area:
                continue                      # it kept a body: an argument, not an erasure
            claim = body & (tenure >= minimum_frames)
            if int(claim.sum()) < minimum_piece:
                continue                      # no long-standing claim to defend
            # Give the cell back one body, not a handful of scraps: only the
            # largest connected part of the claim is restored. Restoring every
            # scattered fragment would create unnecessary duplicate components.
            numbered, count = ndi.label(claim, structure=STRUCTURE)
            if not count:
                continue
            largest = max(range(1, count + 1),
                          key=lambda number: int(np.count_nonzero(numbered == number)))
            claim = numbered == largest
            if int(claim.sum()) < minimum_piece:
                continue
            for taker in sorted(set(map(int, np.unique(current[claim]))) - {0, loser}):
                piece = claim & (current == taker)
                pixels = int(piece.sum())
                if pixels < minimum_piece:
                    continue
                current[piece] = loser
                returned.append({
                    "source_imagej_frame": index + offset + 1,
                    "identity": loser, "taken_by": taker,
                    "pixels_returned": pixels,
                    "claim_frames": int(tenure[piece].max()),
                    "body_px": area})
        result[index] = current
        # A claim is on a place, not on an exact pixel. A cell that shifts by a
        # pixel has not started again, and a cell briefly stood on by a neighbour
        # has not given anything up - so the claim survives as long as the
        # claimant is still standing within reach of the ground. Counting exact
        # exact-pixel counting would understate the duration of a stable claim.
        kept = np.zeros(current.shape, bool)
        for identity in sorted(set(map(int, np.unique(holder))) - {0}):
            near = ndi.binary_dilation(current == identity, structure=STRUCTURE,
                                       iterations=reach) if reach else current == identity
            kept |= (holder == identity) & near
        tenure = np.where(kept, tenure + 1, 1)
        holder = np.where(kept, holder, current)
    return result, returned


def refuse_partial_takeovers(previous: np.ndarray, current: np.ndarray,
                             surrender_fraction: float, minimum_body: int,
                             ) -> tuple[np.ndarray, list[dict]]:
    """Give back ground taken from a name that kept the rest of its body.

    Jamie: "its win or lose not gain ground, its not a slow war its hostile
    takeover or nothing". A name that ends the frame still standing on its own
    body did not lose a contest, so any piece of that body another name is now
    wearing is handed straight back. A name that lost effectively all of its body
    did lose, and that transfer stands as a genuine takeover.

    Ground is only reclaimed inside the object the defender is still standing in,
    so refusing a nibble can never leave a name holding a detached island.
    """
    result = current.copy()
    returned = []
    for identity in sorted(set(map(int, np.unique(previous))) - {0}):
        body = previous == identity
        area = int(body.sum())
        if area < minimum_body:
            continue
        kept = int(np.count_nonzero(result[body] == identity))
        if kept / area <= surrender_fraction:
            continue                      # the whole body went: a genuine takeover
        standing = result == identity
        if not standing.any():
            continue
        for intruder in sorted(set(map(int, np.unique(result[body]))) - {0, identity}):
            piece = body & (result == intruder)
            if not piece.any():
                continue
            # Only the part of the claim that the defender can actually hold on
            # to: the piece must join its own body. Ground handed back across a
            # gap would leave the name standing in two places, which is the very
            # duplicate the controls forbid.
            joined, count = ndi.label(piece | standing, structure=STRUCTURE)
            touching = set(map(int, np.unique(joined[standing]))) - {0}
            piece = piece & np.isin(joined, sorted(touching))
            pixels = int(piece.sum())
            if not pixels:
                continue
            result[piece] = identity
            standing = standing | piece
            returned.append({"identity": identity, "intruder": intruder,
                             "pixels_returned": pixels,
                             "body_px": area, "kept_px": kept})
    return result, returned


def free_second_bodies(current: np.ndarray, seat_regions: np.ndarray,
                       evicted: dict[int, list[int]], minimum_body: int,
                       minimum_stranded: int,
                       ) -> tuple[np.ndarray, list[dict]]:
    """Unwind the displacement: give the stranded body to the name that was evicted.

    When a seat takes its name back, the parent's other use of that name is left
    behind on some unrelated cell. That is the second half of Jamie's cascade -
    "cell 6 becomes 19 which causes 13 to then become what 19 actually was". The
    body it is standing on belongs to whichever name the seat has just evicted,
    provided that name has no body of its own left anywhere, so no name is given
    a second body in order to cure a second body.

    A body test and each piece's size read one bincount of the labelled name; a
    piece is the seat's own body when its number lies under the seat's region,
    and seat membership of a name reads the region's values once. Other label
    maps take the whole-frame form.
    """
    plain_current, regions = _plain(current), _plain(seat_regions)
    if not (plain_current is not None and plain_current.ndim == 2 and plain_current.dtype.kind in 'iu'
            and regions is not None and regions.shape == plain_current.shape):
        return _free_second_bodies_whole(current, seat_regions, evicted, minimum_body, minimum_stranded)
    result = current.copy()
    plain = _plain(result)
    freed = []

    def has_a_body(name: int) -> bool:
        mask = plain == name
        total = int(np.count_nonzero(mask))
        if total < minimum_body:
            return False                # no piece can be larger than all of them
        numbered, count = ndi.label(mask, structure=STRUCTURE)
        if count < 2:
            return count == 1           # one piece holds every pixel
        sizes = np.bincount(_plain(numbered)[mask], minlength=count + 1)
        return bool((sizes[1:] >= minimum_body).any())

    present = set(map(int, np.unique(seat_regions)))
    homeless = {name for names in evicted.values() for name in names
                if name not in present and not has_a_body(name)}
    for identity in sorted(present - {0}):
        options = [name for name in evicted.get(identity, []) if name in homeless]
        if not options:
            continue
        mask = plain == identity
        numbered, count = ndi.label(mask, structure=STRUCTURE)
        numbered = _plain(numbered)
        sizes = np.bincount(numbered[mask], minlength=count + 1)
        own = set(map(int, np.unique(numbered[regions == identity])))
        for number in range(1, count + 1):
            if int(sizes[number]) < minimum_stranded:
                continue
            if number in own:
                continue                    # this is the seat's own body
            if not options:
                break
            successor = options.pop(0)
            homeless.discard(successor)
            result[numbered == number] = successor
            freed.append({"identity": identity, "given_to": successor,
                          "pixels": int(sizes[number])})
    return result, freed

def _free_second_bodies_whole(current: np.ndarray, seat_regions: np.ndarray,
                              evicted: dict[int, list[int]], minimum_body: int,
                              minimum_stranded: int,
                              ) -> tuple[np.ndarray, list[dict]]:
    """Freeing second bodies one mask at a time, for label maps the counted form does not take."""
    result = current.copy()
    freed = []

    def has_a_body(name: int) -> bool:
        numbered, count = ndi.label(result == name, structure=STRUCTURE)
        return any(int(np.count_nonzero(numbered == number)) >= minimum_body
                   for number in range(1, count + 1))

    homeless = {name for names in evicted.values() for name in names
                if not (seat_regions == name).any() and not has_a_body(name)}
    for identity in sorted(set(map(int, np.unique(seat_regions))) - {0}):
        options = [name for name in evicted.get(identity, []) if name in homeless]
        if not options:
            continue
        numbered, count = ndi.label(result == identity, structure=STRUCTURE)
        for number in range(1, count + 1):
            piece = numbered == number
            if int(piece.sum()) < minimum_stranded:
                continue
            if (seat_regions[piece] == identity).any():
                continue                    # this is the seat's own body
            if not options:
                break
            successor = options.pop(0)
            homeless.discard(successor)
            result[piece] = successor
            freed.append({"identity": identity, "given_to": successor,
                          "pixels": int(piece.sum())})
    return result, freed


# --------------------------------------------------------------------------
# stage entry point
# --------------------------------------------------------------------------

def _seat_overlaps(numbered: np.ndarray, count: int, seats: list[dict]) -> list:
    """Every seat's overlap with each numbered component, from one count of the
    component numbers under the seat's pixels (index 0 is background)."""
    numbered = np.asarray(numbered)
    rows = []
    for seat in seats:
        mask = np.asarray(seat["mask"])
        if (mask.shape == numbered.shape and mask.dtype == np.bool_
                and numbered.dtype.kind in 'iu' and count >= 0):
            rows.append(np.bincount(numbered[mask], minlength=count + 1))
        else:
            rows.append([0] + [int(np.count_nonzero(seat["mask"] & (numbered == number)))
                               for number in range(1, count + 1)])
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run(upstream_dir, params: dict, out) -> dict:
    rules = {token.strip().upper() for token in params["rules"] if token.strip()}
    cfg = params["seats"]
    offset = int(params["source_frame_offset"])
    resolve = lambda key: (ROOT / params[key]).resolve()  # noqa: E731

    labels = load_stack(resolve("parent_labels"))
    raw = load_stack(resolve("registered_raw"))
    unclaimed = load_stack(resolve("parent_unclaimed"))
    frames = labels.shape[0]

    result = labels.copy()
    seat_rows: list[dict] = []
    partition_rows: list[dict] = []
    returned_rows: list[dict] = []
    freed_rows: list[dict] = []
    residency: list[dict] = []

    if "A" in rules:
        per_frame = []
        for index in range(frames):
            assigned = labels[index] > 0
            bright = soma_cores(raw[index], assigned, float(cfg["percentile"]),
                                int(cfg["minimum_core_area_px"]))
            somas = list(bright)
            if "D" in rules:
                covered = np.zeros(assigned.shape, bool)
                for core in bright:
                    covered |= core["mask"]
                somas.extend(dim_soma_cores(
                    raw[index], assigned,
                    float(cfg["dim_minimum_prominence"]),
                    float(cfg["dim_smooth_px"]),
                    float(cfg["dim_peak_floor"]),
                    int(cfg["dim_minimum_basin_px"]),
                    ndi.binary_dilation(covered, STRUCTURE)))
            per_frame.append(somas)
        tracks = track_seats(per_frame, float(cfg["maximum_step_px"]),
                             int(cfg["maximum_gap_frames"]))
        claims = seat_claims(labels, tracks, int(cfg["minimum_seat_frames"]),
                             int(cfg["dim_minimum_seat_frames"]))
        owner = award_seats(claims, rules, int(cfg["minimum_frames_worn"]),
                            float(cfg["minimum_fraction_worn"]))
        by_seat = {track["seat"]: track for track in tracks}

        for track in tracks:
            if not long_enough(track, int(cfg["minimum_seat_frames"]),
                               int(cfg["dim_minimum_seat_frames"])):
                continue
            seat_rows.append({
                "seat": track["seat"],
                "first_source_imagej_frame": track["frames"][0] + offset + 1,
                "last_source_imagej_frame": track["frames"][-1] + offset + 1,
                "frames": len(track["frames"]),
                "row": round(track["row"], 1),
                "column": round(track["column"], 1),
                "median_peak": int(np.median(track["peaks"])),
                "dim_only": bool(all(track.get("dim", [False]))),
                "owner_identity": owner.get(track["seat"], 0),
                "names_worn": ";".join(
                    f"{claim['identity']}:{claim['frames_worn']}"
                    for claim in sorted(
                        (claim for claim in claims if claim["seat"] == track["seat"]),
                        key=lambda claim: -claim["frames_worn"])),
            })

        # Every tracked seat marks out its own territory, whether or not a name
        # was awarded to it. A seat with no name still stops its neighbour's name
        # flooding across it: it keeps what the parent called it.
        seats_by_frame: dict[int, list[dict]] = {index: [] for index in range(frames)}
        for track in tracks:
            for index, mask in zip(track["frames"], track["masks"]):
                seats_by_frame[index].append({
                    "seat": track["seat"], "mask": mask,
                    "frames": len(track["frames"]),
                    "owned": track["seat"] in owner})

        for index in range(frames):
            frame = result[index]
            assigned = frame > 0
            numbered, count = ndi.label(assigned, structure=STRUCTURE)
            seat_regions = np.zeros(frame.shape, np.uint16)
            evicted: dict[int, list[int]] = {}
            # The component mask is built only for a component some seat
            # overlaps enough, which is the only place it is used.
            overlaps = _seat_overlaps(numbered, count, seats_by_frame[index])
            for number in range(1, count + 1):
                here = [seat for seat, overlap in zip(seats_by_frame[index], overlaps)
                        if int(overlap[number])
                        >= int(cfg["minimum_seat_overlap_px"])]
                if not here:
                    continue
                component = numbered == number
                if not any(seat["owned"] for seat in here):
                    continue
                if "B" in rules and index:
                    # A cell too faint to show a soma is still marked by where it
                    # stood last frame. Without this a seat's name can flood across
                    # its neighbour. Only ground well clear of every soma counts:
                    # a name wrapped around somebody else's soma is
                    # wearing it, not neighbouring it, and that is the case the
                    # seat exists to correct.
                    somas_here = np.zeros(component.shape, bool)
                    for seat in here:
                        somas_here |= seat["mask"]
                    clear = component_clearance(somas_here, component, float(
                        cfg["prior_marker_clearance_px"]))
                    owners_here = {owner[seat["seat"]] for seat in here
                                   if seat["owned"]}
                    previous_frame = result[index - 1]
                    wearing = set()
                    for seat in here:
                        if not seat["owned"]:
                            continue
                        for name in set(map(int, np.unique(
                                previous_frame[seat["mask"] & component]))) - {0}:
                            if int(np.count_nonzero(
                                    (previous_frame == name) & seat["mask"]
                                    & component)) >= int(cfg["minimum_seat_overlap_px"]):
                                wearing.add(name)
                    for name in sorted(set(map(int, np.unique(previous_frame[component])))
                                       - {0} - owners_here - wearing):
                        held = (previous_frame == name) & component & clear
                        if int(held.sum()) < int(cfg["minimum_body_px"]):
                            continue
                        here.append({"seat": -name, "mask": held, "frames": 0,
                                     "owned": False, "prior": True})
                if len(here) > 1 and "V" not in rules:
                    continue
                regions, grouped = partition_component(
                    raw[index], component, here,
                    float(cfg["maximum_valley_ratio"]))
                for first, second in grouped:
                    partition_rows.append({
                        "source_imagej_frame": index + offset + 1,
                        "seats": f"{first}+{second}",
                        "component_px": int(component.sum()),
                        "action": "one cell: the saddle between these somas is shallow"})
                for seat, region in regions:
                    if seat is None or not region.any():
                        continue
                    seat_regions[region] = owner[seat]
                    before = Counter(int(value) for value in frame[region] if value)
                    frame[region] = owner[seat]
                    lost = {name: pixels for name, pixels in before.items()
                            if name != owner[seat]}
                    if lost:
                        evicted.setdefault(owner[seat], []).extend(
                            name for name, _ in sorted(lost.items(),
                                                       key=lambda item: -item[1]))
                        partition_rows.append({
                            "source_imagej_frame": index + offset + 1,
                            "seats": str(seat),
                            "component_px": int(region.sum()),
                            "action": f"named {owner[seat]}; took "
                                      + ";".join(f"{name}:{pixels}"
                                                 for name, pixels in lost.items())})
            if "G" in rules:
                frame, shed = shed_crumbs(frame, int(cfg["minimum_body_px"]),
                                          int(cfg["minimum_stranded_px"]))
                for row in shed:
                    row["source_imagej_frame"] = index + offset + 1
                    row["reason"] = "crumb on a neighbour"
                freed_rows.extend(shed)
            if "U" in rules:
                frame, freed = free_second_bodies(
                    frame, seat_regions, evicted, int(cfg["minimum_body_px"]),
                    int(cfg["minimum_stranded_px"]))
                for row in freed:
                    row["source_imagej_frame"] = index + offset + 1
                freed_rows.extend(freed)
            result[index] = frame

    if "T" in rules:
        result, reclaimed = tenure_claim(
            result, int(cfg["claim_minimum_frames"]),
            int(cfg["claim_minimum_piece_px"]),
            float(cfg["claim_surrender_fraction"]), offset,
            int(cfg["claim_reach_px"]))
        returned_rows.extend(reclaimed)

    if "F" in rules:
        for index in range(1, frames):
            corrected, returned = refuse_partial_takeovers(
                result[index - 1], result[index],
                float(cfg["surrender_fraction"]),
                int(cfg["minimum_body_px"]))
            for row in returned:
                row["source_imagej_frame"] = index + offset + 1
            returned_rows.extend(returned)
            result[index] = corrected

    # -- residency: a name with no body of its own, but whose last body is now
    #    inside another name's, is recorded as living there rather than lost ----
    names_by_frame = [set(map(int, np.unique(result[index]))) - {0}
                      for index in range(frames)]
    present = {identity: [index for index in range(frames)
                          if identity in names_by_frame[index]]
               for identity in sorted(set(map(int, np.unique(labels))) - {0})}
    for identity, indices in present.items():
        if not indices or indices[-1] >= frames - 1:
            continue
        last = indices[-1]
        body = result[last] == identity
        for index in range(last + 1, frames):
            hosts = Counter(int(value) for value in result[index][body] if value)
            if not hosts:
                break
            host, pixels = hosts.most_common(1)[0]
            if pixels / max(int(body.sum()), 1) < float(cfg["residency_fraction"]):
                break
            residency.append({"source_imagej_frame": index + offset + 1,
                              "identity": identity, "host_identity": host,
                              "held_px": pixels})

    if result.max() > np.iinfo(np.uint16).max:
        raise RuntimeError("identity value overflow")
    if not np.array_equal(result > 0, labels > 0):
        raise RuntimeError("the assigned pixel set changed; accounting is not exact")
    if not set(map(int, np.unique(result))) <= set(map(int, np.unique(labels))):
        raise RuntimeError("an identity was created")

    stem = params["stem"]
    label_path = out.out / f"{stem}.tif"
    save_stack(label_path, result.astype(np.uint16), params["frame_interval_min"])
    unclaimed_path = out.out / f"{stem}_unclaimed_original_ids.tif"
    save_stack(unclaimed_path, unclaimed.astype(np.uint16), params["frame_interval_min"])

    outputs = {
        "labels": label_path,
        "unclaimed": unclaimed_path,
        "seat_table": write_csv(
            out.out / "seat_table.csv", seat_rows,
            ["seat", "first_source_imagej_frame", "last_source_imagej_frame",
             "frames", "row", "column", "median_peak", "dim_only",
             "owner_identity", "names_worn"]),
        "partition_actions": write_csv(
            out.out / "partition_actions.csv", partition_rows,
            ["source_imagej_frame", "seats", "component_px", "action"]),
        "returned_ground": write_csv(
            out.out / "returned_ground.csv", returned_rows,
            ["source_imagej_frame", "identity", "intruder", "taken_by",
             "pixels_returned", "claim_frames", "body_px", "kept_px"]),
        "freed_bodies": write_csv(
            out.out / "freed_bodies.csv", freed_rows,
            ["source_imagej_frame", "identity", "given_to", "pixels", "reason"]),
        "residency_registry": write_csv(
            out.out / "residency_registry.csv", residency,
            ["source_imagej_frame", "identity", "host_identity", "held_px"]),
    }
    summary = {
        "rules": "".join(sorted(rules)),
        "seats_tracked": len(seat_rows),
        "seats_awarded": sum(1 for row in seat_rows if row["owner_identity"]),
        "pixels_renamed": int(np.count_nonzero(result != labels)),
        "frames_changed": int(sum(1 for index in range(frames)
                                  if not np.array_equal(result[index], labels[index]))),
        "identities_before": int(len(set(map(int, np.unique(labels))) - {0})),
        "identities_after": int(len(set(map(int, np.unique(result))) - {0})),
        "partition_actions": len(partition_rows),
        "ground_returned_rows": len(returned_rows),
        "second_bodies_freed": len(freed_rows),
        "residency_rows": len(residency),
        "identical_to_parent": bool(np.array_equal(result, labels)),
    }
    (out.out / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    return {"outputs": outputs, "summary": summary}

