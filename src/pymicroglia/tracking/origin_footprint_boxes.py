"""Recovery origin and raw-footprint certificates: name boxes from one find_objects pass per frame.

certify_recovery_origins and certify_raw_soma_footprints found each measured
name's bounding rectangle in each of its frames with a whole-frame
np.nonzero(observed[t] == name). ndi.find_objects gives the same rectangle
(first to last row and column holding the name) for every name of a frame in
one pass; the rectangle is only used to slice the mask and the raw movie, so
every test is unchanged. Boxes are kept per frame for one call. Frames that
are not non-negative integer images keep the original statement.

With the same boxes, each name's per-frame mask is built inside its box of an
otherwise empty frame (equal to observed[t] == name, since the name occurs
nowhere else), and the consecutive-frame contact test np.any(masks[t] &
masks[previous]) is read on the two boxes' overlap (False when they do not
overlap) and kept per frame pair, which the backward pass asks again. Frame
names come from the frame's histogram (ascending, as np.unique). The raw
footprint loop builds each frame's footprint once and takes floor, ceil and
the finite test of each point's floats with math (same integers and verdicts).
"""
from contextlib import contextmanager
import math
import hashlib
import inspect
from pathlib import Path
import sys

import numpy as np
try:
    from .point_disks import footprints as _footprints
except ImportError:
    from point_disks import footprints as _footprints

MAXIMUM_LABEL = 1 << 20
NONZERO = '            yy,xx=np.nonzero(mask);small=mask[yy.min():yy.max()+1,xx.min():xx.max()+1]\n'
BOXED = '            _box=_name_box(observed,t,name,mask);small=mask[_box]\n'
VERSIONS = {
    '2d7103d27a7c69558b86a2c3cb50453751bd9629babafb0417d322f6405190a4': ('certify_recovery_origins', [(NONZERO, BOXED)]),
    '42c17adec5c34e2b0be9c39491919d2c8a52a95d90a7d397d7fc0a7aa6cd0a92': ('certify_raw_soma_footprints', [
        (NONZERO, BOXED),
        ('            patch=raw[t,yy.min():yy.max()+1,xx.min():xx.max()+1]\n',
         '            patch=raw[t,_box[0],_box[1]]\n')]),
}


OPTIONAL = (
    # The measured footprint loop as one call (see point_disks); applied before
    # the statement edits below, which then find nothing in it.
    ('    for q in selected.itertuples(index=False):\n'
     "        t=int(q.frame);x=float(q.x);y=float(q.y);radius=float(getattr(q,'radius_px',0.))\n"
     '        if not 0<=t<len(raw) or not np.isfinite([x,y,radius]).all() or radius<0:continue\n'
     '        y0=max(0,int(np.floor(y-radius)));y1=min(raw.shape[1],int(np.ceil(y+radius))+1)\n'
     '        x0=max(0,int(np.floor(x-radius)));x1=min(raw.shape[2],int(np.ceil(x+radius))+1)\n'
     '        if y0>=y1 or x0>=x1:continue\n'
     '        yy,xx=np.ogrid[y0:y1,x0:x1]\n'
     '        coverage=footprints.setdefault(t,np.zeros(raw.shape[1:],bool))\n'
     '        coverage[y0:y1,x0:x1]|=(yy-y)**2+(xx-x)**2<=radius**2\n',
     '    footprints=_point_disks(selected,raw,False)\n'),
    ('        masks={t:observed[t]==name for t in frames}\n',
     '        masks={t:_name_mask(observed,t,name) for t in frames}\n'),
    ('np.any(masks[t]&masks[previous])', '_pair_any(observed,name,t,previous,masks)'),
    ('        for n in np.unique(frame):\n            if n:actual',
     '        for n in _frame_names(frame):\n            if n:actual'),
    ('not np.isfinite([x,y,radius]).all()', 'not (_isfinite(x) and _isfinite(y) and _isfinite(radius))'),
    ('y0=max(0,int(np.floor(y-radius)));y1=min(raw.shape[1],int(np.ceil(y+radius))+1)',
     'y0=max(0,_floor(y-radius));y1=min(raw.shape[1],_ceil(y+radius)+1)'),
    ('x0=max(0,int(np.floor(x-radius)));x1=min(raw.shape[2],int(np.ceil(x+radius))+1)',
     'x0=max(0,_floor(x-radius));x1=min(raw.shape[2],_ceil(x+radius)+1)'),
    ('        coverage=footprints.setdefault(t,np.zeros(raw.shape[1:],bool))\n',
     '        coverage=footprints.get(t)\n'
     '        if coverage is None:coverage=footprints[t]=np.zeros(raw.shape[1:],bool)\n'),
)


def frame_names(frame):
    plain = np.asarray(frame)
    if plain.ndim == 2 and plain.dtype.kind == 'u' and plain.dtype.itemsize <= 4 and plain.size:
        return np.flatnonzero(np.bincount(plain.ravel()))
    return np.unique(frame)


def name_boxes(stats):
    held = {}
    pairs = {}

    def frame_boxes(observed, t):
        if held.get('movie') is not observed:
            held.clear()
            pairs.clear()
            held['movie'] = observed
        if t not in held:
            plain = np.asarray(observed[t])
            boxes = None
            if plain.ndim == 2 and plain.dtype.kind in 'iu' and plain.size:
                if int(plain.min()) >= 0 and int(plain.max()) < MAXIMUM_LABEL:
                    from scipy import ndimage
                    boxes = ndimage.find_objects(plain)
                    stats['frames'] += 1
            held[t] = boxes
        return held[t]

    def box_of(observed, t, name):
        boxes = frame_boxes(observed, t)
        value = int(name)
        if boxes is not None and 0 < value <= len(boxes):
            return boxes[value - 1]
        return False

    def _name_mask(observed, t, name):
        box = box_of(observed, t, name)
        if box is False:
            stats['mask_direct'] += 1
            return observed[t] == name
        mask = np.zeros(np.shape(observed[t]), bool)
        if box is not None:
            mask[box] = observed[t][box] == name
        stats['mask_boxed'] += 1
        return mask

    def _pair_any(observed, name, t, previous, masks):
        key = (int(name), min(t, previous), max(t, previous))
        if key in pairs and pairs[key][0] is masks:
            stats['pairs_reused'] += 1
            return pairs[key][1]
        first, second = box_of(observed, t, name), box_of(observed, previous, name)
        if first is False or second is False or first is None or second is None:
            value = np.any(masks[t] & masks[previous])
        else:
            rows = slice(max(first[0].start, second[0].start), min(first[0].stop, second[0].stop))
            cols = slice(max(first[1].start, second[1].start), min(first[1].stop, second[1].stop))
            if rows.start >= rows.stop or cols.start >= cols.stop:
                value = np.False_
            else:
                value = np.any(masks[t][rows, cols] & masks[previous][rows, cols])
        pairs[key] = (masks, value)
        stats['pairs_measured'] += 1
        return value

    def _name_box(observed, t, name, mask):
        boxes = frame_boxes(observed, t)
        value = int(name)
        if boxes is not None and 0 < value <= len(boxes) and boxes[value - 1] is not None:
            stats['boxed'] += 1
            return boxes[value - 1]
        stats['fallback'] += 1
        yy, xx = np.nonzero(mask)
        return np.s_[yy.min():yy.max()+1, xx.min():xx.max()+1]

    def release():
        held.clear()
        pairs.clear()
    return _name_box, release, _name_mask, _pair_any


def adapt(module, stats):
    path = getattr(module, '__file__', None)
    try:
        found = VERSIONS.get(hashlib.sha256(Path(path).read_bytes()).hexdigest())
    except (OSError, TypeError):
        return None
    if found is None:
        return None
    name, replacements = found
    original = getattr(module, name, None)
    if (original is None or getattr(original, '__module__', None) != module.__name__
            or getattr(original, '__wrapped__', None) is not None):
        return None
    try:
        source = inspect.getsource(original)
    except (OSError, TypeError):
        return None
    for old, new in replacements:
        if source.count(old) != 1:
            return None
        source = source.replace(old, new)
    for old, new in OPTIONAL:
        if source.count(old) == 1:
            source = source.replace(old, new)
    for key in ('mask_boxed', 'mask_direct', 'pairs_reused', 'pairs_measured'):
        stats.setdefault(key, 0)
    namespace = dict(vars(module))
    box, release, mask, pair = name_boxes(stats)
    namespace['_name_box'] = box
    namespace.update(_name_mask=mask, _pair_any=pair, _frame_names=frame_names,
                     _isfinite=math.isfinite, _floor=math.floor, _ceil=math.ceil, _point_disks=_footprints)
    exec(compile(source, '<origin-footprint-boxes>', 'exec'), namespace)
    rewritten = namespace[name]

    def replacement(*args, **kwargs):
        try:
            return rewritten(*args, **kwargs)
        finally:
            release()
    replacement.__name__ = name
    replacement.__wrapped__ = original
    return original, replacement


@contextmanager
def install(stats=None):
    """Replace loaded verified versions and their imported aliases; restore afterwards."""
    stats = stats if stats is not None else {}
    for key in ('versions', 'frames', 'boxed', 'fallback'):
        stats.setdefault(key, 0)
    swap = {}
    for module in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        if not (hasattr(module, 'certify_recovery_origins') or hasattr(module, 'certify_raw_soma_footprints')):
            continue
        found = adapt(module, stats)
        if found is not None:
            swap[id(found[0])] = found
    aliases = []
    if swap:
        stats['versions'] += len(swap)
        for child in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
            names = getattr(child, '__dict__', None)
            if not names:
                continue
            if not swap or swap.keys().isdisjoint(map(id, names.values())):  # alias loop skip
                continue
            for name, value in list(names.items()):
                found = swap.get(id(value))
                if found is not None and value is found[0]:
                    aliases.append((child, name, value))
                    setattr(child, name, found[1])
    try:
        yield stats
    finally:
        for child, name, value in reversed(aliases):
            setattr(child, name, value)
