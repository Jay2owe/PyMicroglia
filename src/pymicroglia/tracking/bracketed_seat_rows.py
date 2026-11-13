"""Bracketed ownerless seat completion: visible ownerless rows per gap frame, once cast.

complete() asks, for every frame of every eligible gap, for the attached
points at that frame that are visible, physically visible, ownerless and
inferred to belong to the identity, recasting five whole-table columns each
time. complete() sets inferred_owner before the gap loop and never changes the
table inside it, so for one call the same five casts are made once (at the
first question, in the original order, so a failing cast raises there), each
frame's row positions are listed, and the answer is the copy of the rows at
the frame's positions that pass the same tests, in table order.

The rewritten complete() is compiled under the module's own file name with
every line where it was: the memory pool's audited presence sites
(array_storage.PRESENCE_SITES) match this file by name and line. pandas other
than the verified version runs the original.
"""
import __future__
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = {'bracketed_ownerless_seat_completion': 'd90391c2c821a24f109893af074516c43f278ff1a66df0e83c95a4b4c3c23d7f'}
PANDAS = '2.2.2'
MEMO_LINE = '    frame_rows: list[dict] = []\n'
ORIGINAL = '''                    rows = attached[
                        attached.frame.astype(int).eq(frame)
                        & attached.state.astype(str).isin(VISIBLE_STATES)
                        & attached.physically_visible.astype(bool)
                        & attached.candidate_owner.astype(int).eq(0)
                        & attached.inferred_owner.astype(int).eq(identity)].copy()
'''
# Six lines for six, so every later line keeps its number.
REPLACEMENT = '''                    rows = _seat_rows.visible(
                        attached, frame, identity, VISIBLE_STATES)
                    # The same selection: visible, physically visible,
                    # ownerless rows of this frame inferred for identity,
                    # copied in table order (bracketed_seat_rows).

'''


def _sha256(module):
    try:
        data = Path(module.__file__).read_bytes()
    except (OSError, AttributeError, TypeError):
        return None
    return hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()


class Memo:
    """Column casts and frame positions for one complete() call."""

    def __init__(self, stats):
        self.stats = stats
        self.table = None

    def visible(self, attached, frame, identity, states):
        if self.table is None or self.table[0] is not attached:
            frames = attached.frame.astype(int)
            state = attached.state.astype(str).isin(states)
            physical = attached.physically_visible.astype(bool)
            unowned = attached.candidate_owner.astype(int).eq(0)
            inferred = attached.inferred_owner.astype(int)
            positions = {}
            for index, value in enumerate(frames.tolist()):
                positions.setdefault(value, []).append(index)
            passing = (state & physical & unowned).to_numpy()
            self.table = (attached, {key: np.asarray(value, np.intp) for key, value in positions.items()},
                          passing, inferred.to_numpy())
            self.stats['tables'] += 1
        _, positions, passing, inferred = self.table
        found = positions.get(frame)
        if found is None:
            found = np.empty(0, np.intp)
        chosen = found[passing[found] & (inferred[found] == identity)]
        self.stats['rows'] += 1
        return attached.iloc[chosen].copy()


@contextmanager
def install(completion=None):
    """Replace bracketed_ownerless_seat_completion.complete when its source is the verified one."""
    stats = dict(supported=[], calls=0, tables=0, rows=0)
    restore = []
    try:
        if (completion is not None and pd.__version__ == PANDAS
                and _sha256(completion) == SOURCES['bracketed_ownerless_seat_completion']):
            original = completion.complete
            lines, first = inspect.getsourcelines(original)
            source = ''.join(lines)
            if source.count(MEMO_LINE) == 1 and source.count(ORIGINAL) == 1:
                source = source.replace(ORIGINAL, REPLACEMENT).replace(
                    MEMO_LINE, MEMO_LINE.rstrip('\n') + '; _seat_rows = _seat_rows_factory()\n')
                assert source.count('\n') == len(lines)

                def factory():
                    stats['calls'] += 1
                    return Memo(stats)
                globals_ = vars(completion)
                globals_['_seat_rows_factory'] = factory
                restore.append((globals_, '_seat_rows_factory'))
                namespace = {}
                exec(compile('\n' * (first - 1) + source, completion.__file__, 'exec',
                             flags=__future__.annotations.compiler_flag, dont_inherit=True), globals_, namespace)
                replacement = namespace['complete']
                replacement.__wrapped__ = original
                completion.complete = replacement
                restore.append((completion, 'complete', original))
                stats['supported'].append('bracketed_ownerless_seat_completion')
        yield stats
    finally:
        for item in reversed(restore):
            if len(item) == 2:
                item[0].pop(item[1], None)
            else:
                setattr(*item)
