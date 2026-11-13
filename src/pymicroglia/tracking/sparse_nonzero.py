"""Pixel coordinates of a small component in a whole-frame mask, from its bounding box.

Several corrections take the coordinates of one component's whole-frame
Boolean mask with np.where(mask) or np.nonzero(mask), which walks every pixel
of the frame twice (count, then fill). The rows that hold a set pixel and,
within them, the columns that do, are found with two any() reductions; every
set pixel lies in that box, so np.nonzero of the box gives the same pixels in
the same row-major order, and adding the box's first row and column gives the
same coordinates (np.intp, as np.nonzero returns). The arrays are new
contiguous arrays rather than column views of one buffer; the values are the
same. Masks that are not plain 2-D Boolean ndarrays, small masks and empty
masks keep the original call (np.where through the array pool, or np.nonzero).

The issue-012 domain ownership module's identity_points takes every named
pixel of a 2-D integer label frame; it finds them in the frame's != 0
Boolean array (read through a plain view), the same positions in the same
order. Other frames keep np.nonzero(frame).
"""
from contextlib import contextmanager

try:
    from .track_sort_skips import rewrite
except ImportError:
    from track_sort_skips import rewrite

import numpy as np

KEYS = ('versions', 'boxed', 'whole')
SMALL = 4096


def sparse_nonzero(stats):
    def _sparse_nonzero(mask, original):
        if type(mask) is np.ndarray and mask.ndim == 2 and mask.dtype == np.bool_ and mask.size >= SMALL:
            rows = np.flatnonzero(mask.any(axis=1))
            if len(rows):
                r0, r1 = int(rows[0]), int(rows[-1]) + 1
                band = mask[r0:r1]
                columns = np.flatnonzero(band.any(axis=0))
                c0, c1 = int(columns[0]), int(columns[-1]) + 1
                yy, xx = np.nonzero(band[:, c0:c1])
                stats['boxed'] += 1
                return yy + r0, xx + c0
        stats['whole'] += 1
        return original(mask)
    return _sparse_nonzero


def _spec(module, function, digest, *edits):
    return dict(module=module, function=function, digests={digest}, edits=edits,
                helper='_sparse_nonzero', factory=sparse_nonzero)


SPECS = (
    _spec('recurrent_core_memory_a002', 'correct',
          '043b4af314083c0be0acdd734110b5708b40d81def9bf2d4a946add0e48fb052',
          ('yy,xx=np.where(mask)\n',
           'yy,xx=_sparse_nonzero(mask,np.where)\n'),
          ('yy,xx=np.where(mask);key=',
           'yy,xx=_sparse_nonzero(mask,np.where);key=')),
    _spec('recovery_measured_history', 'constrain_measured_history',
          '3dc280bd8df368e6a6c5d5dbad8dcd84a3442317b25f8157bbdae7a9cd467eb5',
          ('yy,xx=np.nonzero(mask);ref=',
           'yy,xx=_sparse_nonzero(mask,np.nonzero);ref=')),
    _spec('recovery_continuous_history', 'constrain_continuous_history',
          'c1889a5465b1c09635f34eeb19c40061d66a025e6b6e8ed652c2f86ec43fa034',
          ('yy,xx=np.nonzero(mask);ref=',
           'yy,xx=_sparse_nonzero(mask,np.nonzero);ref=')),
    _spec('separable_merge_recovery', '_marker_pixel',
          'd6f2f7ac9be0d2d5da3be647b066473a7cbd72ec278979c7a644cfb6d4df3086',
          ('yy, xx = np.nonzero(mask)\n',
           'yy, xx = _sparse_nonzero(mask, np.nonzero)\n')),
    _spec('recovery_detached_inference', 'constrain_detached_inference',
          'ea1e43f2299d10dcfdcd72d234fd0b63a9994d39e9484014641373b5d0af1022',
          ('yy, xx = np.nonzero(source)\n',
           'yy, xx = _sparse_nonzero(source, np.nonzero)\n')),
    _spec('recovery_envelope_inference', 'constrain_envelope_inference',
          'b2cf95b98cc20ffaab901308a0fef75e4f77c62c75efe48fa8c2f94ae289001a',
          ('yy, xx = np.nonzero(source)\n',
           'yy, xx = _sparse_nonzero(source, np.nonzero)\n')),
    _spec('recover_bookended_owner_with_small_remnants', '_small_untracked_remnants',
          '31f7e6cdc49e8e9f4c21bf3858997d2b500090877d884a9e7da60562ac30d52d',
          ('yy, xx = np.where(remnant)\n',
           'yy, xx = _sparse_nonzero(remnant, np.where)\n')),
    _spec('process_inventory_temporal', 'measure',
          '176a73d7c37f8cb8d2cf1f725f66950f88e5bd631d0187cb93cd5acaa9827acf',
          ('sy, sx = np.where(small)\n',
           'sy, sx = _sparse_nonzero(small, np.where)\n')),
    _spec('process_inventory_intensity', 'measure',
          '5adedde22dfd8bd9b849958ce20d442a3d84c955b9a1562677992f568b1e005f',
          ('sy, sx = np.where(small)\n',
           'sy, sx = _sparse_nonzero(small, np.where)\n')),
    _spec('recover_dormant_name_soma_basins', '_basin',
          '96b3a0bf12c117d316b0900b002cefd2cb2d073d17389c3e737df1aca186ef2b',
          ('ys, xs = np.where(mask)\n',
           'ys, xs = _sparse_nonzero(mask, np.where)\n')),
    _spec('process_inventory_shape', 'geometry',
          '148dcf5f3cee800cb8bb99fca7adbefc013f1acbcdd2040ca14d22400c309831',
          ('ys, xs = np.where(mask)\n',
           'ys, xs = _sparse_nonzero(mask, np.where)\n')),
    _spec('extend_approved_process_paths', '_largest_centre',
          'aa48d6bd4a30058c853e5e9406d9b764f0d3d336731bc8a57f568ecb6ccb67de',
          ('yy, xx = np.where(parts == int(np.argmax(sizes[1:])+1))\n',
           'yy, xx = _sparse_nonzero(parts == int(np.argmax(sizes[1:])+1), np.where)\n')),
)


# The issue-012 domain ownership module, loaded through the accepted history.
ISSUE = 'c2343dee1275675c35597dcfed84babd2596652f3f33a95dc63cdbbca475a2dc'
ISSUE_SPECS = tuple(_spec(None, function, ISSUE,
                          ('np.column_stack(np.nonzero(mask))', 'np.column_stack(_sparse_nonzero(mask,np.nonzero))'))
                    for function in ('centroid', 'unclaimed_observations')) + (
    # identity_points takes every named pixel of a whole label frame: no small
    # box, but NumPy finds the same positions five times faster in a Boolean
    # array than in integer labels (a nonzero integer is a True Boolean).
    dict(module=None, function='identity_points', digests={ISSUE}, helper=None, factory=None,
         edits=(('    pixels = np.column_stack(np.nonzero(frame))\n',
                 "    pixels = np.column_stack(np.nonzero(frame.view(np.ndarray) != 0)\n"
                 "        if isinstance(frame, np.ndarray) and frame.ndim == 2 and frame.dtype.kind in 'iu'\n"
                 "        else np.nonzero(frame))\n"),)),)


@contextmanager
def install(stats=None):
    """Rewrite the verified functions' coordinate calls for the duration; restore afterwards."""
    with rewrite(SPECS, KEYS, '<sparse-nonzero>', stats) as found:
        yield found


@contextmanager
def install_loader(history, stats=None):
    """Rewrite the domain ownership module each time the accepted history loads it; restore afterwards."""
    from contextlib import ExitStack
    stats = stats if stats is not None else {}
    for key in KEYS:
        stats.setdefault(key, 0)
    original = history._issue_module
    with ExitStack() as stack:
        def load(*args, **kwargs):
            module = original(*args, **kwargs)
            if all(callable(getattr(module, spec['function'], None)) for spec in ISSUE_SPECS):
                specs = tuple(dict(spec, module=module) for spec in ISSUE_SPECS)
                stack.enter_context(rewrite(specs, KEYS, '<sparse-nonzero>', stats))
            return module
        history._issue_module = load
        try:
            yield stats
        finally:
            history._issue_module = original
