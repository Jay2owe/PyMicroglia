"""Concurrent duplicate invasion: donor rows over the invasion, by position.

concurrent_duplicate_invasion.discover asks, for every owner transition and
every canonical donor of the invading owner, for the donor track's rows with
frame between the invasion's first and last frame (Series.between, inclusive),
and goes on only when their number equals the invasion's length. discover
never changes the track tables, so each donor's numeric frame column is read
once per call; the same inclusive comparison picks the positions, and only a
selection of the right length is formed, as the same rows in the same order
(iloc of the positions the boolean mask selects). Frame columns that are not
plain NumPy integer or floating point, and pandas other than the verified
version, use the original filter.
"""
import __future__
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = {'concurrent_duplicate_invasion': 'a51af3a01a5ffcdbc04ca50293958ecfee604bd2b81339df6739033faadb7df0'}
PANDAS = '2.2.2'
MEMO_LINE = '    rows: list[dict] = []\n'
ORIGINAL = '''                donor_interval = donor[donor.frame.between(
                    int(invasion["first"]), int(invasion["last"]))]
                if len(donor_interval) != int(invasion["frames"]):
                    continue
'''
REPLACEMENT = '''                donor_interval = _invasion_rows.between(
                    donor, int(invasion["first"]), int(invasion["last"]),
                    invasion)
                if donor_interval is None:
                    continue
'''


def _sha256(module):
    try:
        data = Path(module.__file__).read_bytes()
    except (OSError, AttributeError, TypeError):
        return None
    return hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()


class Memo:
    """Donor frame columns for one discover call."""

    def __init__(self, stats):
        self.stats = stats
        self.groups = {}

    def between(self, group, first, last, invasion):
        entry = self.groups.get(id(group))
        if entry is None or entry[0] is not group:
            column = group.frame
            numeric = isinstance(column.dtype, np.dtype) and column.dtype.kind in 'iuf'
            frames = column.to_numpy() if numeric else None
            entry = self.groups[id(group)] = (group, frames)
        frames = entry[1]
        if frames is None:
            self.stats['original'] += 1
            rows = group[group.frame.between(first, last)]
            return rows if len(rows) == int(invasion["frames"]) else None
        self.stats['indexed'] += 1
        positions = np.flatnonzero((frames >= first) & (frames <= last))
        if len(positions) != int(invasion["frames"]):
            return None
        return group.iloc[positions]


@contextmanager
def install(invasion=None):
    """Replace concurrent_duplicate_invasion.discover when its source is the verified one."""
    stats = dict(supported=[], calls=0, indexed=0, original=0)
    restore = []
    try:
        if invasion is not None and pd.__version__ == PANDAS and _sha256(invasion) == SOURCES['concurrent_duplicate_invasion']:
            original = invasion.discover
            source = inspect.getsource(original)
            if source.count(MEMO_LINE) == 1 and source.count(ORIGINAL) == 1:
                source = source.replace(ORIGINAL, REPLACEMENT).replace(
                    MEMO_LINE, MEMO_LINE + '    _invasion_rows = _invasion_rows_factory()\n')

                def factory():
                    stats['calls'] += 1
                    return Memo(stats)
                globals_ = vars(invasion)
                globals_['_invasion_rows_factory'] = factory
                restore.append((globals_, '_invasion_rows_factory'))
                namespace = {}
                exec(compile(source, '<invasion-donor-rows:discover>', 'exec',
                             flags=__future__.annotations.compiler_flag, dont_inherit=True), globals_, namespace)
                replacement = namespace['discover']
                replacement.__wrapped__ = original
                invasion.discover = replacement
                restore.append((invasion, 'discover', original))
                stats['supported'].append('concurrent_duplicate_invasion')
        yield stats
    finally:
        for item in reversed(restore):
            if len(item) == 2:
                item[0].pop(item[1], None)
            else:
                setattr(*item)
