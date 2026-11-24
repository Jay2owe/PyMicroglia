# Copied from Motion manual_editing/trajectory_review/note_actions.py (sha256 5b127e4a4110477a...) for the PyMicroglia
# correction reviewer; only the import lines differ. PyMicroglia never imports Motion.
"""Jamie's review phrases as reusable actions, for the reviewer's buttons and for saved notes.

  keep split               the clicked cell keeps its number over the reviewed snippet and is cut back
                           out whenever it merges with a neighbour (the neighbour keeps its own number)
  keep (N) whole recording the same over every frame
  N here / this is N / should be N / was N / keep N
                           the clicked cell takes N (with keep split / whole recording: over those frames,
                           otherwise over the run of frames it carries its clicked number)
  delete / not a cell      remove the clicked piece
"changes to N" and "swaps to N" describe a mistake; they and anything else are left for a person.

    python note_actions.py A3 [--out plan.json]   saved notes -> note_plan.py entries; lists the rest
"""
import argparse, json, re
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

EIGHT = np.ones((3, 3), bool)


def window(data, item):
    """The reviewer's snippet for a correction item: a few frames before the join to just past it."""
    T, j = data['T'], item['frame']
    last = max(data['tracks'][k][2] for k, _, _ in item['renames'])
    return max(0, j-6), min(T-1, max(j+6, min(last, j+30)+2))


def _neighbour(lab_prev, rest, exclude):
    """The number a merged-in neighbour carried in the frame before, or 0."""
    v, c = np.unique(lab_prev[rest], return_counts=True)
    keep = (v > 0) & ~np.isin(v, list(exclude))
    return int(v[keep][c[keep].argmax()]) if keep.any() else 0


def _nearest(lab, prev, reach):
    """The outline piece of similar size whose centre is nearest the lost cell's last centre, within reach px."""
    cy, cx = np.argwhere(prev).mean(0); size = prev.sum()
    y0, x0 = int(max(cy-reach, 0)), int(max(cx-reach, 0))
    box = lab[y0:int(cy+reach)+1, x0:int(cx+reach)+1]
    best, n, far = None, 0, reach**2
    for v in np.unique(box[box > 0]):
        comp, _ = ndi.label(lab == v, structure=EIGHT)
        for i in np.unique(comp[y0:int(cy+reach)+1, x0:int(cx+reach)+1]):
            m = comp == i
            if i == 0 or not 0.4*size <= m.sum() <= 2.5*size:
                continue
            d = ((np.argwhere(m).mean(0)-[cy, cx])**2).sum()
            if d < far:
                best, n, far = m, int(v), d
    return best, n


def _by_peaks(img, part, prev):
    """A merged outline cut between its two brightest spots, the side overlapping prev kept; whole if one spot."""
    from skimage.feature import peak_local_max
    sm = ndi.gaussian_filter(np.nan_to_num(img).astype(float), 1.5)
    pk = peak_local_max(np.where(part, sm, 0), min_distance=5, labels=part.astype(int), num_peaks=2)
    if len(pk) < 2:
        return part
    markers = np.zeros(part.shape, np.int32)
    for i, (py, px) in enumerate(pk, 1):
        markers[py, px] = i
    ws = watershed(-sm, markers, mask=part)
    return ws == (1 if (prev & (ws == 1)).sum() >= (prev & (ws == 2)).sum() else 2)


def _n(mask) -> int:
    """Pixels in a mask (count_nonzero: same number as .sum(), several times faster; speed pass 2026-11-20)."""
    return int(np.count_nonzero(mask))


def _label_cell(lab, n):
    """ndi.label(lab == n, EIGHT) worked out on the box holding n only (same pieces and numbering; speed pass
    2026-11-20: labelling the whole frame twice per frame was half of Keep's time)."""
    mask = lab == n
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    out = np.zeros(lab.shape, np.int32)
    if not rows.size:
        return out, 0
    box = slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)
    out[box], k = ndi.label(mask[box], structure=EIGHT)
    return out, k


def keep_cell(frame, raw, f, y, x, a, b, to=None, same=False, claimed=None, path=None, reach=None, gap=3):
    """Follow the cell under (y, x) in frame f over frames a-b and give it one number (default: its own).

    The cell is followed by overlap (half the smaller outline; lost after gap frames with no match). When it
    merges with a neighbour it is cut back out along the dim valley, the neighbour keeping the number it had
    the frame before. Any other holder of the number in a frame takes the cell's old number (a local swap).
    same: stop where the cell's number changes. claimed {frame: numbers}: stop at a cell already named
    there, and record this one (with reach: stop on the pixels an earlier keep followed instead). reach: when the overlap finds nothing, take the similar-sized outline piece
    whose centre is nearest the last one, within reach px (for a cell that jumps or drops out a frame).
    Returns one 'keep' edit, or None when nothing changes."""
    lab = frame(f)
    pts = np.argwhere(lab > 0)
    y, x = pts[((pts-[y, x])**2).sum(1).argmin()]
    first = int(lab[y, x])
    X = to or first
    parts, _ = _label_cell(lab, first)
    start = parts == parts[y, x]
    moves = {}
    claimed = {} if claimed is None else claimed
    for frames in (range(f, b+1), range(f-1, a-1, -1)):
        prev, prev_new, misses = start, None, 0
        for g in frames:
            lab = frame(g)
            if g == f:
                part, n = start, first
            else:
                c = np.bincount(lab[prev].ravel())   # as np.unique(..., return_counts=True), faster
                v = np.flatnonzero(c)
                c, v = c[v][v > 0], v[v > 0]
                part = None
                if len(v):
                    n = int(v[c.argmax()])
                    if same and n != first:
                        break
                    parts, _ = _label_cell(lab, n)
                    ov = np.bincount(parts[prev], minlength=parts.max()+1); ov[0] = 0
                    cand = parts == ov.argmax()
                    if ov.max() >= 0.5*min(_n(prev), _n(cand)) and _n(cand) >= 15:
                        part = cand
                if part is None and reach:
                    part, n = _nearest(lab, prev, reach)
                    if part is not None and same and n != first:
                        break
                if part is None:
                    misses += 1
                    if misses > gap:
                        break
                    continue
                if reach and n != X and _n(part & claimed.get(('px', g), False)) > 0.5*_n(part):
                    break   # with reach: stop on the pixels an earlier keep followed, not on its number
                if not reach and part is not None and n != X and n in claimed.get(g, ()):
                    break   # that cell was named by an earlier note
                whole = part
                if _n(part) > 1.4*_n(prev) and _n(part & ~prev) > 40:   # merged: cut it back out
                    markers = np.zeros(lab.shape, np.int32)
                    markers[ndi.binary_erosion(prev & part) | (prev & part & (_n(part) < 60))] = 1
                    markers[part & ~ndi.binary_dilation(prev, iterations=3)] = 2
                    if (markers == 1).any() and (markers == 2).any():
                        ws = watershed(-np.nan_to_num(raw(g)).astype(float), markers, mask=part)
                        part = ws == 1
                    if reach and _n(part) > 2*_n(prev):   # with reach: one bright spot is one grown cell,
                        part = _by_peaks(raw(g), whole, prev)   # two are cut between them
                    if (_n(part) > 2*_n(prev) and not reach) or not part.any():   # merged and cannot be cut out:
                        misses += 1                                   # leave this frame, look again in the next
                        if misses > gap:
                            break
                        continue
            misses = 0
            comp, _ = _label_cell(lab, n)
            rest = np.isin(comp, np.unique(comp[part])) & ~part   # what was cut off the same outline
            ops = []
            if n != X:
                ops.append([n, X, part])
                other = (lab == X) & ~part
                if other.any():
                    ops.append([X, n, other])
            elif rest.any() and prev_new is not None:   # a neighbour merged in under X gets its number back
                m = _neighbour(prev_new, rest, {X})
                if m:
                    ops.append([X, m, rest])
            if ops:
                moves[str(g)] = [[h, c, _runs(m)] for h, c, m in ops]
            new = lab.copy()
            for h, c, m in ops:
                new[m & (lab == h)] = c
            claimed.setdefault(g, set()).add(X)
            claimed[('px', g)] = claimed.get(('px', g), np.zeros(part.shape, bool)) | part
            if path is not None:
                path.append((g, *np.argwhere(part).mean(0), int(_n(part)), n))
            if not (reach and _n(part) < 0.5*_n(prev)):   # with reach, a fragment keeps the last full outline
                prev = part
            prev_new = new
    if not moves:
        return None
    done = sorted(map(int, moves))
    return dict(op='keep', cell=int(X), first=done[0], last=done[-1], moves=moves)


def _runs(mask):
    from . import correction_rules as rd
    return rd.row_runs(mask)


NUM = r'(\d{1,4})'


def plan_entry(note, data, items):
    """A note_plan.py entry for one saved note, or (None, reason) when it needs a person."""
    t = note['text'].lower().replace('whoel', 'whole').replace('hwole', 'whole').replace('recoridng', 'recording')
    if note.get('y') is None:
        return None, 'no cell clicked'
    f, y, x = note['frame'], note['y'], note['x']
    if re.search(r'\b(changes|swaps) to\b', t):
        return None, 'describes a mistake'
    if re.search(r'\bdelete\b|not a cell', t):
        return dict(note=note['n'], piece=[f, y, x], to=0), 'delete the clicked piece'
    to = None
    for pat in (rf'this is (?:also )?{NUM}\b', rf'should be {NUM}\b', rf'\b{NUM} (?:is )?here\b', rf'\bwas {NUM}\b',
                rf'\bkeep {NUM}\b', rf'^\s*{NUM}\s*$', rf'^{NUM}\b(?! and)'):
        m = re.search(pat, t)
        if m:
            to = int(m.group(1)); break
    whole = 'whole' in t
    keep = whole or 'keep split' in t or 'keep this split' in t or 'keep here' in t or 'keep it' in t
    if keep:
        a, b = (0, data['T']-1) if whole else window(data, items[note['item']])
        e = dict(note=note['n'], keep=[f, y, x], frames=[a, b])
        if to:
            e['to'] = to
        return e, ('keep' + (f' as {to}' if to else '') + (' over the whole recording' if whole else ' split over the snippet'))
    if to:
        return dict(note=note['n'], keep=[f, y, x], to=to, same=True), f'this run is {to}'
    return None, 'needs a person'


def main():  # Motion command line, kept for parity; needs Motion's review_data
    import review_data as rd
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('well'); p.add_argument('--out', type=Path)
    a = p.parse_args()
    data = rd.load(a.well)
    items = {it['key']: it for it in data['items']}
    plan, rest = [], []
    for i, n in enumerate(rd.read_notes().get(a.well, []), 1):
        e, why = plan_entry(dict(n, n=i), data, items)
        (plan if e else rest).append(e or (i, n['frame'], n['text'], why))
        if e:
            print(f"#{i} f{n['frame']} {n['text']!r}: {why}")
    print(f'{len(plan)} notes turned into plan entries; {len(rest)} left for a person:')
    for i, f, text, why in rest:
        print(f'  #{i} f{f} {text!r} ({why})')
    out = a.out or rd.REVIEW/f'{a.well}_note_actions.json'
    out.write_text('[\n' + ',\n'.join(json.dumps(e) for e in plan) + '\n]\n')
    print('plan written to', out)


if __name__ == '__main__':
    main()
