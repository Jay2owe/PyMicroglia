# Copied from Motion manual_editing/trajectory_review/edit_tools.py (sha256 2e87e776d68614b6...) for the PyMicroglia
# correction reviewer; only the import lines differ. PyMicroglia never imports Motion.
"""Build the reviewer's manual edits (review_data.apply_ops) from clicks and drawn lines.

frame(f) returns the full current labels of frame f (decisions and earlier edits applied);
raw(f) the matching raw image. Frames are label frames, 0-based. The split itself is the
Motion manual editor's engine (manual_splitting): a drawn line marks the two sides in the
first frame, and each later frame starts from the previous frame's sides.
"""
import numpy as np
from scipy import ndimage as ndi

from .manual_splitting import SplitRequest, propose_forced_split

MARGIN = 6           # px around the host given to the split engine
NEAR = 40            # px: an existing number this close to the cut-off piece can take it over
MIN_PIECE = 3        # px, smallest piece either side of a cut


def run_end(frame, n, f, last_frame):
    """Last frame of number n's unbroken run starting at f (f itself if n is absent there)."""
    end = f
    while end < last_frame and (frame(end+1) == n).any():
        end += 1
    return end


def present(frame, numbers, first, last):
    """Numbers (of the given ones) that appear anywhere in frames first..last."""
    found = set()
    for f in range(first, last+1):
        found |= set(np.intersect1d(frame(f), list(numbers)).tolist())
    return found


def line_mask(points, shape, width=1):
    """Pixels on a polyline of (y, x) points, thickened by width."""
    m = np.zeros(shape, bool)
    for (ya, xa), (yb, xb) in zip(points, points[1:] or points):
        n = int(max(abs(yb-ya), abs(xb-xa)))+1
        ys = np.clip(np.round(np.linspace(ya, yb, n)).astype(int), 0, shape[0]-1)
        xs = np.clip(np.round(np.linspace(xa, xb, n)).astype(int), 0, shape[1]-1)
        m[ys, xs] = True
    return ndi.binary_dilation(m, iterations=width) if width else m


def side_of(points, shape):
    """+1 / -1 for every pixel by which side of the line (first to last point) it lies on."""
    (ya, xa), (yb, xb) = points[0], points[-1]
    yy, xx = np.indices(shape)
    return np.sign((xb-xa)*(yy-ya) - (yb-ya)*(xx-xa))


def host_under(lab, points):
    """The number most of the line crosses; for a line drawn in the gap between two pieces of one
    number, the nearest number with pixels on both sides of the line. 0 when there is none."""
    line = line_mask(points, lab.shape, 0)
    vals = lab[line]
    vals = vals[vals > 0]
    if vals.size:
        return int(np.bincount(vals).argmax())
    dist = ndi.distance_transform_edt(~line)
    side = side_of(points, lab.shape)
    best = (np.inf, 0)
    for n in np.unique(lab[(dist <= NEAR/2) & (lab > 0)]):
        m = lab == n
        if (side[m] > 0).any() and (side[m] < 0).any():
            best = min(best, (float(dist[m].min()), int(n)))
    return best[1]


def _box(mask):
    ys, xs = np.nonzero(mask)
    return (max(int(ys.min())-MARGIN, 0), min(int(ys.max())+MARGIN+1, mask.shape[0]),
            max(int(xs.min())-MARGIN, 0), min(int(xs.max())+MARGIN+1, mask.shape[1]))


def _centre(mask):
    ys, xs = np.nonzero(mask)
    return int(round(ys.mean())), int(round(xs.mean()))


def split_sides(lab, points, flip=False):
    """(host, sides, box) for a line drawn in one frame: sides is the host's box with 1 for the side that keeps
    the number (the larger, or the smaller with flip) and 2 for the cut-off side."""
    host = host_under(lab, points)
    if not host:
        raise ValueError('Draw the line across a cell')
    y0, y1, x0, x1 = _box(lab == host)
    part = _drawn_sides(lab[y0:y1, x0:x1] == host, [(y-y0, x-x0) for y, x in points], host)
    if flip:
        part = np.where(part == 1, 2, np.where(part == 2, 1, 0))
    return host, part, (y0, y1, x0, x1)


def split(frame, raw, f0, last, points, known, flip=False, give=None):
    """Split the cell under a line drawn in frame f0, following it to frame `last` at most.

    Returns (edit, message). The larger side keeps the cell's number (the smaller with flip - Jamie
    2026-11-18: choose which side keeps it); the other side takes `give` when given, else the
    nearest number that ends just before f0 or starts just after the split and is not in use
    over the split frames, otherwise a new number above every number in `known`."""
    lab = frame(f0)
    host = host_under(lab, points)
    if not host:
        raise ValueError('Draw the line across a cell')
    keep, piece = None, None
    runs, stop = {}, ''
    for f in range(f0, last+1):
        lab = lab if f == f0 else frame(f)
        hmask = lab == host
        if not hmask.any():
            stop = f'{host} ends at frame {f-1}'
            break
        y0, y1, x0, x1 = _box(hmask)
        h = hmask[y0:y1, x0:x1]
        if f == f0:
            part = _drawn_sides(h, [(y-y0, x-x0) for y, x in points], host)
            if flip:
                part = np.where(part == 1, 2, np.where(part == 2, 1, 0))
        else:
            part = _follow_pieces(h, keep[y0:y1, x0:x1], piece[y0:y1, x0:x1])
            if part is None:
                part = _follow_touching(h, keep, piece, hmask, (y0, y1, x0, x1), np.nan_to_num(raw(f)[y0:y1, x0:x1]))
            if part is None:
                stop = f'at frame {f} the cut-off part is gone or cannot be told apart'
                break
        child = part == 2
        runs[str(f)] = [[y+y0, x+x0, n] for y, x, n in _runs(child)]
        keep = np.zeros(lab.shape, bool); piece = np.zeros(lab.shape, bool)
        keep[y0:y1, x0:x1] = part == 1
        piece[y0:y1, x0:x1] = child
    end = max(int(k) for k in runs)
    child = int(give) if give else _child_number(frame, f0, end, runs, known | {host}, host)
    edit = dict(op='split', host=host, child=child, first=f0, last=end, runs=runs)
    msg = f'Split {host}: the cut-off part is {child}' + (f'; stopped because {stop}' if stop else '')
    return edit, msg


def _drawn_sides(h, line, host):
    """Frame with the drawn line. A line across the cell: the two biggest parts it leaves are the
    sides and every other host pixel (the line itself, fragments) joins the nearer side. A line in
    the gap between pieces: each pixel's side of the line. Side 1 is the larger one."""
    cut = line_mask(line, h.shape)
    if (h & cut).any():
        parts, n = ndi.label(h & ~cut, structure=np.ones((3, 3), bool))
        sizes = np.bincount(parts.ravel())[1:]
        if n < 2 or np.sort(sizes)[-2] < MIN_PIECE:
            raise ValueError(f'The line must cut all the way across {host}')
        big = np.argsort(sizes)[::-1][:2]+1
        seeds = np.zeros(h.shape, np.int32)
        seeds[parts == big[0]] = 1
        seeds[parts == big[1]] = 2
        _, (iy, ix) = ndi.distance_transform_edt(seeds == 0, return_indices=True)
        part = np.where(h, seeds[iy, ix], 0)
    else:
        side = side_of(line, h.shape)
        part = np.where(h, np.where(side > 0, 1, 2), 0)
        if min((part == 1).sum(), (part == 2).sum()) < MIN_PIECE:
            raise ValueError(f'Draw the line between the parts of {host}')
        if (part == 2).sum() > (part == 1).sum():
            part = np.where(h, 3-part, 0)
    return part


def _follow_pieces(h, keep, piece):
    """Host made of separate pieces: each piece goes to the side it overlapped a frame earlier
    (or the nearer side). None when the pieces do not give both sides."""
    comps, n = ndi.label(h, structure=np.ones((3, 3), bool))
    if n < 2:
        return None
    part = np.zeros(h.shape, np.int32)
    centres = [np.argwhere(m).mean(0) if m.any() else None for m in (keep, piece)]
    for c in range(1, n+1):
        m = comps == c
        ov = [(m & keep).sum(), (m & piece).sum()]
        if ov[0] == ov[1]:
            yx = np.argwhere(m).mean(0)
            ov = [-np.inf if q is None else -np.hypot(*(yx-q)) for q in centres]
        part[m] = 1 + int(ov[1] > ov[0])
    return part if (part == 1).any() and (part == 2).any() else None


def _follow_touching(h, keep, piece, hmask, box, raw):
    """One host piece: each side starts from where it was a frame earlier, moved with the cell and
    shrunk; the Motion editor's engine then places the boundary by brightness (two bodies part at
    the dimmest path between them). None when a side has vanished or the split is not valid."""
    y0, y1, x0, x1 = box
    (py, px), (ny, nx) = _centre(keep | piece), _centre(hmask)
    assign = np.zeros(h.shape, np.int32)
    for i, prev in ((1, keep), (2, piece)):
        moved = _shift(prev, ny-py, nx-px)[y0:y1, x0:x1] & h
        core = ndi.binary_erosion(moved)
        assign[(core if core.any() else moved) & (assign == 0)] = i
    if not (assign == 1).any() or not (assign == 2).any():
        return None
    seeds = tuple(tuple(int(v) for v in np.argwhere(assign == i)[0]) for i in (1, 2))
    # The engine reads ownership as side index (1, 2) or child number, so it works on stand-in
    # numbers 3 (side 1) and 4 (side 2) that cannot be mistaken for sides.
    req = SplitRequest(host_identity=3, start_imagej_frame=1, end_imagej_frame=1, child_identities=(3, 4),
                       method='marker_controlled_watershed', seeds={1: seeds}, manual_assignments={1: assign},
                       minimum_child_area_px=MIN_PIECE)
    prop = propose_forced_split(np.where(h, 3, 0).astype(np.uint16)[None], raw.astype(np.float32)[None], req)
    if not prop.per_frame[0]['valid']:
        return None
    return np.where(prop.labels[0] == 3, 1, np.where(prop.labels[0] == 4, 2, 0))


def _shift(mask, dy, dx):
    out = np.zeros_like(mask)
    H, W = mask.shape
    out[max(dy, 0):H+min(dy, 0), max(dx, 0):W+min(dx, 0)] = mask[max(-dy, 0):H-max(dy, 0), max(-dx, 0):W-max(dx, 0)]
    return out


def _runs(mask):
    out = []
    for y in np.flatnonzero(mask.any(1)):
        d = np.diff(np.concatenate([[0], mask[y].astype(np.int8), [0]]))
        out += [(int(y), int(a), int(b-a)) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    return out


def _child_number(frame, f0, end, runs, known, host):
    first = np.zeros(frame(f0).shape, bool)
    for y, x, n in runs[str(f0)]:
        first[y, x:x+n] = True
    cy, cx = _centre(first)
    near = {}
    for f in (f0-1, end+1):
        if f < 0:
            continue
        try:
            lab = frame(f)
        except (IndexError, KeyError):
            continue
        ys, xs = np.nonzero(lab)   # every number's nearest pixel to the centre, in one pass (speed pass 2026-11-20)
        vals = lab[ys, xs].astype(np.int64)
        if not vals.size:
            continue
        best = np.full(int(vals.max()) + 1, np.inf)
        np.minimum.at(best, vals, np.hypot(ys-cy, xs-cx))
        for v in np.flatnonzero(np.isfinite(best)):
            v, d = int(v), float(best[v])
            if v and v != host and d <= NEAR:
                near[v] = min(near.get(v, d), d)
    busy = present(frame, near, f0, end) if near else set()
    free = sorted((d, v) for v, d in near.items() if v not in busy)
    return int(free[0][1]) if free else int(max(known))+1


def stray_pieces(labels, skip=None, min_share=0.3):
    """Piece edits for pieces that carry a number the same spot does not hold either side.

    A piece (connected part of number n in frame f) moves to number c when c covers at least
    min_share of the piece's pixels in frame f-1 and in frame f+1, beats n there on both sides,
    and c is not elsewhere in frame f (unless touching the piece). Never makes numbers or changes
    outlines. skip(f, n, c) can veto a move (for example, one that redoes a denied join)."""
    S = np.ones((3, 3), bool)
    edits = []
    for f in range(1, len(labels)-1):
        lab = labels[f]
        for n, sl in enumerate(ndi.find_objects(lab), 1):
            if sl is None:
                continue
            sl = tuple(slice(max(s.start-3, 0), s.stop+3) for s in sl)
            parts, k = ndi.label(lab[sl] == n, structure=S)
            for j in range(1, k+1):
                P = parts == j
                size = int(P.sum())
                best = []
                for g in (f-1, f+1):
                    v = labels[g][sl][P]
                    counts = np.bincount(v, minlength=n+1) if v.size else np.zeros(n+1, int)
                    counts[0] = 0
                    c = int(counts.argmax())
                    best.append((c, int(counts[c]), int(counts[n]) if n < len(counts) else 0))
                (c1, k1, n1), (c2, k2, n2) = best
                if c1 != c2 or c1 in (0, n) or min(k1, k2) < min_share*size or n1 >= k1 or n2 >= k2:
                    continue
                if skip and skip(f, n, c1):
                    continue
                elsewhere = lab == c1
                if elsewhere.any():
                    near = np.zeros(lab.shape, bool)
                    near[sl] = ndi.binary_dilation(P, iterations=2)
                    if not (elsewhere & near).any():
                        continue
                full = np.zeros(lab.shape, bool)
                full[sl] = P
                edits.append(dict(op='piece', host=n, child=c1, first=f, last=f,
                                  runs={str(f): [[int(y), int(x), int(m)] for y, x, m in _runs(full)]},
                                  by='stray-piece rule'))
    return edits
