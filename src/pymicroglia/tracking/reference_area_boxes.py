"""Stable dual-soma reference areas from owner components labelled once, in the owner's box.

recover_stable_dual_soma_contacts._reference_area labelled the whole frame's
owner mask for every strong observed point of a track and kept the pixel count
of the component physical._component_at picks. Its one caller computes every
track's area in one dictionary comprehension at the start of correct(), and
nothing writes labels between those calls, so within one correct() call each
(frame, owner) mask is labelled once, inside its bounding box
(censored_area_boxes.OwnerParts: same components, same raster order, same
point-to-component rule).

Three other contact recoveries call the same _reference_area in the same way:
one comprehension at the start of their correct(), before anything is written,
and none writes into its labels argument. Their correct() gets the same memo.
A correct() whose per-track sort track_sort_skips replaced is wrapped the same
way; any other already-wrapped correct() is left alone.
"""
from contextlib import ExitStack, contextmanager
import hashlib
from pathlib import Path
import sys

import numpy as np
try:
    from .module_scan import file_sha256 as _file_sha256
except ImportError:
    from module_scan import file_sha256 as _file_sha256

SOURCE = '12c1ecf9362a51f3e42846733f099c9b117f38fe341e8ac237d64a13358fefe7'
BODY = '5305c91dc515bcfcf7612825fbdce55d4d6773edae27d71bfd86bfca8f8938f1'
PHYSICAL = 'd6f2f7ac9be0d2d5da3be647b066473a7cbd72ec278979c7a644cfb6d4df3086'
CALLERS = {
    'recover_flanked_large_contact_interruptions':
        'f7683a6d56eb2643d266d6ab6b557e1629a8997d70ec2aa8aefff09f26556f1d',
    'recover_terminal_name_switch_contacts_a002':
        '61a0fd44981a4bd14fbf68e79a792df912293d17a91277fb030d62c296dc7038',
    'recover_persistent_resident_contacts':
        'b9b8abe3a7cef9ead503db73711daaa1a644202879f0e71774da7ebd50f15bdd',
}


def _digest(module):
    try:
        return _file_sha256(module.__file__)
    except (AttributeError, OSError, TypeError):
        return None


def owner_parts(module, labels, stats):
    try:
        from .censored_area_boxes import OwnerParts
    except ImportError:
        from censored_area_boxes import OwnerParts
    return OwnerParts(labels, module.body.STRUCTURE, stats)


def reference_area(memo, original_area, stats):
    """_reference_area answered from memo for memo.labels; any other movie uses the original."""
    def _reference_area(frames, group, owner, anchor):
        if frames is not memo.labels:
            stats['fallback'] += 1
            return original_area(frames, group, owner, anchor)
        areas = []
        for row in group.itertuples(index=False):
            if (int(row.accepted_owner) != owner or not bool(row.strong)
                    or bool(row.in_encounter) or str(row.state) != "observed"):
                continue
            # A component always holds a pixel: area 0 is the original's None.
            area = memo.area_at(int(row.frame), owner, float(row.x), float(row.y), 6)
            if area > 0:
                areas.append(area)
        return float(np.median(areas)) if areas else 0.0
    return _reference_area


def _verified(module):
    return (_digest(module) == SOURCE and _digest(module.body) == BODY
            and _digest(module.physical) == PHYSICAL and _digest(module.body.physical) == PHYSICAL)


def _sort_skipped(function):
    """True for a correct() that track_sort_skips recompiled from the caller's own
    source with only its per-track sort changed; the memo wraps it as it would the
    original."""
    code = getattr(function, '__code__', None)
    return code is not None and code.co_filename == '<track-sort-skips>'


@contextmanager
def install(module, stats, caller=None):
    """Wrap correct() of a verified module (or of a verified caller of its
    _reference_area); returns without change otherwise."""
    caller = module if caller is None else caller
    if ((getattr(caller.correct, '__wrapped__', None) is not None and not _sort_skipped(caller.correct))
            or not _verified(module)):
        yield
        return
    original_correct, original_area = caller.correct, module._reference_area

    def correct(labels, *args, **kwargs):
        if not (isinstance(labels, np.ndarray) and labels.ndim == 3
                and np.issubdtype(labels.dtype, np.integer)):
            return original_correct(labels, *args, **kwargs)
        memo = owner_parts(module, labels, stats)
        module._reference_area = reference_area(memo, original_area, stats)
        stats['calls'] += 1
        try:
            return original_correct(labels, *args, **kwargs)
        finally:
            module._reference_area = original_area
            memo.parts.clear()
    correct.__wrapped__ = original_correct
    caller.correct = correct
    try:
        yield
    finally:
        caller.correct = original_correct
        module._reference_area = original_area


@contextmanager
def install_loaded(stats):
    """install() on the loaded recover_stable_dual_soma_contacts and its verified callers."""
    module = sys.modules.get('recover_stable_dual_soma_contacts')
    if module is None:
        yield
        return
    with ExitStack() as stack:
        stack.enter_context(install(module, stats))
        for name, digest in CALLERS.items():
            caller = sys.modules.get(name)
            # Each caller reaches the area through its own `dual` import.
            if (caller is not None and _digest(caller) == digest
                    and getattr(caller, 'dual', None) is module):
                stack.enter_context(install(module, stats, caller))
        yield
