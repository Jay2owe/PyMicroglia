"""Stage-boundary guards of the composition files: the same answers, one block of frames at a time.

Each composition checks a corrected movie against its parent with whole-movie
expressions such as np.any((candidate != labels) & protected) or
np.array_equal(np.unique(candidate), np.unique(labels)). Every expression makes
whole-movie temporaries (disk-backed under the working-storage pool), and the
name sets list every name of both movies. The helpers below evaluate the same
elementwise expression on consecutive blocks of frames and combine the blocks
(any: any block; count: sum; equality: every block), which is the same answer.
Name sets come from one histogram of the parent movie: outside changed pixels
the candidate equals the parent, so the candidate holds a name exactly when the
parent holds it outside changed pixels or a changed pixel takes it.

The recovery, current-path and contact sequences settle each stage's output
with whole-movie masks (eligible pixels, changed pixels, ledger updates) and
then rescan each frame once per changed (source, target) pair; settle applies
the same statements to blocks of frames and takes each pair's pixel count from
the frame's pair table.

The recovery, current-path and reviewed-ownership compositions hand each
frozen method read-only views of the whole movies they used to copy (see
lent_call).

Arrays of unequal shape, non-arrays, unusual dtypes, negative or very large
names all evaluate the original expression. Composition files are pinned by
content hash; only their top-level functions are rewritten, by exact patterns.
"""
from contextlib import contextmanager
import ast
import hashlib
import math
from pathlib import Path
import re
import sys

import numpy as np

BLOCK = 1 << 20
NAME_SPAN = 1 << 20
VERSIONS = {
    '07e7aba04a0d7cd9': 'anchored_memory_composition',
    '6f19463e63868487': 'detector_gap_composition',
    'c3202f4f685ce702': 'dim_exchange_composition',
    'b008871cc57fb9b5': 'duplicate_contact_composition',
    '615f8fb8bbfea56e': 'embedded_handoff_composition',
    '23788699169ddbc3': 'fragment_contact_composition',
    'c55c4793b1470fb2': 'handoff_contact_composition',
    'a22378fa0b3a5dc1': 'optical_source_composition',
    '42136a0960a0e90d': 'owner_episode_composition',
    'dd7a0dde8057e449': 'ownerless_contact_composition',
    '9f6ce75c6abb62f4': 'recurrent_branch_composition',
    'f6ab732db1a0dc53': 'review_union_composition',
    'f2c30afd0834afd0': 'shared_birth_composition',
    '3d2fc0b747fe6f02': 'terminal_owner_composition',
    'fc5c754b864926b2': 'repeated_name_composition',
    '3bb8482edb8b441d': 'reviewed_ownership_composition',
    '52da3dacfdd8d94f': 'recovery_sequence_composition',
    'ab528069c2fd630d': 'current_path_sequence_composition',
    'df371a14a7588237': 'contact_sequence_composition',
    'b0181022e0991bc7': 'process_group_composition_v2',
}
stats = dict(versions=0, functions=0, calls=0, fallback=0)


def _plain(*arrays, kinds='biuf'):
    shape = None
    found = []
    for value in arrays:
        if not isinstance(value, np.ndarray) or value.ndim < 1 or value.dtype.kind not in kinds:
            return None
        if shape is None:
            shape = value.shape
        elif value.shape != shape:
            return None
        found.append(value if type(value) is np.ndarray else value.view(np.ndarray))
    return found


def _blocks(shape):
    step = max(1, BLOCK // max(1, math.prod(shape[1:])))
    for start in range(0, shape[0], step):
        yield slice(start, start + step)


def _scan(arrays, block_test, original, kinds='biuf'):
    """True when block_test is true for any block; original() otherwise."""
    stats['calls'] += 1
    plain = _plain(*arrays, kinds=kinds)
    if plain is None:
        stats['fallback'] += 1
        return original()
    for part in _blocks(plain[0].shape):
        if block_test(*(value[part] for value in plain)):
            return True
    return False


def _buffered(count, test):
    """A block test whose Boolean temporaries are `count` arrays reused across one call's blocks.

    The test passes them as out= to the same ufuncs the expression calls, so
    every value is the same; blocks come largest first, so one allocation serves
    them all.
    """
    held = []

    def block_test(*values):
        shape = values[0].shape
        size = math.prod(shape)
        if not held or held[0].size < size:
            held[:] = [np.empty(size, np.bool_) for _ in range(count)]
        return test(*values, *(work[:size].reshape(shape) for work in held))
    return block_test


def changed_where(a, b, mask):
    """np.any((a != b) & mask)"""
    kind = getattr(getattr(mask, 'dtype', None), 'kind', None)
    if kind not in ('b', 'i', 'u'):
        return _fallback(lambda: np.any((a != b) & mask))
    if kind == 'b':
        # bool & bool is bitwise_and on bools; an integer mask keeps the
        # expression, whose & is bitwise on the mask's integers.
        test = _buffered(1, lambda x, y, m, p: bool(np.bitwise_and(np.not_equal(x, y, out=p), m, out=p).any()))
    else:
        test = lambda x, y, m: bool(np.any((x != y) & m))
    return _scan((a, b, mask), test, lambda: np.any((a != b) & mask))


def changed_where_positive(a, b, value):
    """np.any((a != b) & (value > 0))"""
    return _scan((a, b, value), lambda x, y, v: bool(np.any((x != y) & (v > 0))),
                 lambda: np.any((a != b) & (value > 0)))


def changed_where_nonzero(a, b, value):
    """np.any(changed & (value != 0)) with changed = a != b"""
    return _scan((a, b, value), lambda x, y, v: bool(np.any((x != y) & (v != 0))),
                 lambda: np.any((a != b) & (value != 0)))


def changed_where_equal(a, b, c, d):
    """np.any((a != b) & (c == d))"""
    return _scan((a, b, c, d), lambda x, y, u, v: bool(np.any((x != y) & (u == v))),
                 lambda: np.any((a != b) & (c == d)))


def any_changed(a, b):
    """bool(np.any(a != b))"""
    return _scan((a, b), lambda x, y: bool(np.any(x != y)), lambda: bool(np.any(a != b)))


def gained_foreground(before, after):
    """np.any((before == 0) & (after > 0))"""
    return _scan((before, after), lambda x, y: bool(np.any((x == 0) & (y > 0))),
                 lambda: np.any((before == 0) & (after > 0)))


def lost_foreground(before, after):
    """np.any((before > 0) & (after == 0))"""
    return _scan((before, after), lambda x, y: bool(np.any((x > 0) & (y == 0))),
                 lambda: np.any((before > 0) & (after == 0)))


def same_foreground(a, b):
    """np.array_equal(a > 0, b > 0)"""
    return not _scan((a, b), _buffered(2, lambda x, y, p, q: bool(
                         np.not_equal(np.greater(x, 0, out=p), np.greater(y, 0, out=q), out=p).any())),
                     lambda: not np.array_equal(a > 0, b > 0))


def same_union_foreground(a, b, c, d):
    """np.array_equal((a > 0) | (b > 0), (c > 0) | (d > 0))"""
    return not _scan((a, b, c, d), lambda w, x, y, z: bool(np.any(((w > 0) | (x > 0)) != ((y > 0) | (z > 0)))),
                     lambda: not np.array_equal((a > 0) | (b > 0), (c > 0) | (d > 0)))


def equal(a, b):
    """np.array_equal(a, b)"""
    return not _scan((a, b), lambda x, y: not bool((x == y).all()), lambda: not np.array_equal(a, b))


def survivor_changed(candidate, labels):
    """survivor = candidate > 0; np.any(survivor & (labels == 0)) or np.any(candidate[survivor] != labels[survivor])"""
    def original():
        survivor = candidate > 0
        return np.any(survivor & (labels == 0)) or np.any(candidate[survivor] != labels[survivor])
    return _scan((candidate, labels),
                 lambda x, y: bool(np.any((x > 0) & ((y == 0) | (x != y)))), original)


def count_changed(a, b):
    """int((a != b).sum()), equal to np.count_nonzero(a != b)"""
    stats['calls'] += 1
    plain = _plain(a, b)
    if plain is None:
        stats['fallback'] += 1
        return int((a != b).sum())
    return sum(int(np.count_nonzero(x[part] != y[part])) for x, y in [plain] for part in _blocks(x.shape))


def check_comparable(a, b):
    """Statement a != b whose value is unused: evaluate it only where it could raise."""
    if _plain(a, b) is None:
        a != b


def check_positive(a):
    """Statement a > 0 whose value is unused: evaluate it only where it could raise."""
    if _plain(a) is None:
        a > 0


def _fallback(original):
    stats['calls'] += 1
    stats['fallback'] += 1
    return original()


def _bounded(value):
    if value.dtype.kind == 'u' and value.dtype.itemsize <= 2:
        return True
    return not value.size or (int(value.min()) >= 0 and int(value.max()) < NAME_SPAN)


def _add(total, counts):
    if total is None:
        return counts.astype(np.int64, copy=True)
    if len(counts) > len(total):
        total, counts = counts.astype(np.int64, copy=True), total
    total[:len(counts)] += counts
    return total


def _histogram(values):
    """np.bincount(values) for non-negative values: same counts, same length."""
    # count_nonzero counts the elements where values != 0; the mask is built
    # only for the sparse branch that selects with it.
    nonzero = int(np.count_nonzero(values))
    if 2 * nonzero >= values.size:
        return np.bincount(values)
    # Mostly background: count the nonzero names and give zero the remainder.
    counts = np.bincount(values[values != 0]) if nonzero else np.zeros(1, np.int64)
    if not len(counts):
        counts = np.zeros(1, np.int64)
    counts[0] += values.size - nonzero
    return counts


def _presence(candidate, parent):
    plain = _plain(candidate, parent, kinds='iu')
    if plain is None or not plain[0].size:
        return None
    x, y = plain
    if not (np.can_cast(x.dtype, np.intp, 'safe') and np.can_cast(y.dtype, np.intp, 'safe')):
        return None
    counts = removed = added = None
    for part in _blocks(x.shape):
        new, old = x[part].ravel(), y[part].ravel()
        if not _bounded(old):
            return None
        counts = _add(counts, _histogram(old))
        changed = new != old
        if changed.any():
            gained, lost = new[changed], old[changed]
            if not _bounded(gained):
                return None
            removed = _add(removed, np.bincount(lost))
            added = _add(added, np.bincount(gained))
    width = max(len(value) for value in (counts, removed, added) if value is not None)
    full = lambda value: np.pad(value, (0, width - len(value))) if value is not None else np.zeros(width, np.int64)
    counts, removed, added = full(counts), full(removed), full(added)
    return (counts - removed > 0) | (added > 0), counts > 0


def same_names(a, b):
    """np.array_equal(np.unique(a), np.unique(b))"""
    stats['calls'] += 1
    found = _presence(a, b)
    if found is None:
        stats['fallback'] += 1
        return np.array_equal(np.unique(a), np.unique(b))
    return bool(np.array_equal(found[0], found[1]))


def added_names(a, b):
    """set(map(int, np.unique(a))) - set(map(int, np.unique(b))); its length is len(np.setdiff1d(np.unique(a), np.unique(b)))"""
    stats['calls'] += 1
    found = _presence(a, b)
    if found is None:
        stats['fallback'] += 1
        return set(map(int, np.unique(a))) - set(map(int, np.unique(b)))
    return set(map(int, np.flatnonzero(found[0] & ~found[1])))


def _settle_original(labels, ledger, after, mode, saved, restored):
    candidate = labels.copy()
    eligible = (labels == restored) if mode.startswith('restored') else (labels == saved) & (restored == saved)
    if mode.endswith('+all'):
        eligible = np.ones(labels.shape, dtype=bool)
    candidate[eligible] = after[eligible]
    changed = candidate != labels
    new_ledger = ledger.copy()
    new_ledger[candidate > 0] = 0
    removed = changed & (candidate == 0)
    new_ledger[removed] = labels[removed]
    applications = []
    for t in range(len(labels)):
        for source, target in np.unique(np.stack((labels[t][changed[t]], candidate[t][changed[t]]), axis=1), axis=0):
            applications.append(dict(frame=t+1, source=int(source), target=int(target),
                changed_pixels=int(((labels[t] == source) & (candidate[t] == target) & changed[t]).sum())))
    return candidate, new_ledger, applications


def settle(labels, ledger, after, mode, saved, restored):
    """A sequence composition's accepted output, new ledger and applications (see _settle_original).

    The eligible mask, masked copies, changed mask and ledger updates are the
    same elementwise statements applied to blocks of frames. Each changed
    (source, target) pair's pixel count is the number of changed pixels holding
    that pair, which is the row's count in the frame's pair table.
    """
    stats['settle_calls'] = stats.get('settle_calls', 0) + 1
    plain = _plain(labels, ledger, after, saved, restored)
    if (plain is None or labels.ndim != 3 or labels.dtype.kind not in 'iu'
            or mode not in ('saved', 'restored', 'saved+all', 'restored+all')):
        stats['settle_fallback'] = stats.get('settle_fallback', 0) + 1
        return _settle_original(labels, ledger, after, mode, saved, restored)
    candidate = labels.copy()
    new_ledger = ledger.copy()
    copies = _plain(candidate, new_ledger)
    if copies is None or copies[0].dtype != labels.dtype:
        stats['settle_fallback'] = stats.get('settle_fallback', 0) + 1
        return _settle_original(labels, ledger, after, mode, saved, restored)
    L, _, A, S, R = plain
    C, N = copies
    applications = []
    for part in _blocks(L.shape):
        old, new = L[part], C[part]
        if mode.endswith('+all'):
            eligible = np.ones(old.shape, dtype=bool)
        elif mode == 'restored':
            eligible = old == R[part]
        else:
            eligible = (old == S[part]) & (R[part] == S[part])
        # Masked assignment casts unsafely; copyto with the same casting writes the same values.
        np.copyto(new, A[part], casting='unsafe', where=eligible)
        changed = new != old
        ledger_part = N[part]
        ledger_part[new > 0] = 0
        removed = changed & (new == 0)
        ledger_part[removed] = old[removed]
        for index in np.flatnonzero(changed.reshape(len(changed), -1).any(axis=1)):
            here = changed[index]
            pairs, counts = np.unique(np.stack((old[index][here], new[index][here]), axis=1), axis=0, return_counts=True)
            frame = part.start + int(index) + 1
            for (source, target), count in zip(pairs, counts):
                applications.append(dict(frame=frame, source=int(source), target=int(target), changed_pixels=int(count)))
    return candidate, new_ledger, applications


# Recovery, current-path and reviewed-ownership compositions copy every whole
# movie they hand a frozen method ("Copies retain that isolation"), so the
# method cannot change the caller's arrays. A read-only view gives the same
# isolation without the copy: a method that only reads gets the same values,
# type, dtype, shape and contiguous layout. A write into a lent view raises
# NumPy's read-only ValueError (none of the lent methods' modules catch
# ValueError or Exception, or read the writeable flag); the call is then
# repeated from a fresh import of the method with copies, exactly as written.
READ_ONLY = ('read-only', 'not writeable')


def _lendable(value):
    # Lent only where the view has the copy's type: a plain array, or a working
    # array whose copy is another file-backed working array (large, with more
    # dimensions than its pool keeps in memory). Others are copied.
    if not (isinstance(value, np.ndarray) and value.ndim >= 1 and value.dtype.kind in 'biuf'
            and value.flags.c_contiguous):
        return False
    if type(value) is np.ndarray:
        return True
    pool = getattr(value, '_pool', None)
    return (type(value).__name__ == 'WorkingArray' and pool is not None and pool.large(value.shape, value.dtype)
            and value.ndim > pool._resident_ndim.get())


def _lent(value):
    view = value.view(type(value))
    view.setflags(write=False)
    return view


def lend(value):
    """A read-only view of an array the composition copied only to keep it unchanged; others are copied."""
    return _lent(value) if _lendable(value) else value.copy()


def _unshared(result, lent):
    # A returned array that is (part of) a lent input is copied, as the
    # original's returned array was the method's own copy.
    def fresh(value):
        if isinstance(value, np.ndarray) and any(np.may_share_memory(value, other) for other in lent):
            stats['lent_returned'] = stats.get('lent_returned', 0) + 1
            return value.copy()
        return value
    return tuple(map(fresh, result)) if type(result) is tuple else fresh(result)


def lent_call(spec, frozen, method, arrays, lent, make):
    """method(*copies of arrays, *rest, **keywords with copies of lent), as the composition wrote it.

    make() returns (rest, keywords): the composition's other, freshly copied
    arguments. Plain contiguous numeric arrays are lent as read-only views; a
    call that writes into one is repeated in a fresh import with copies.
    """
    values = list(arrays) + list(lent.values())
    if all(_lendable(value) for value in values):
        views = [_lent(value) for value in values]
        rest, keywords = make()
        try:
            result = method(*views[:len(arrays)], *rest,
                            **dict(keywords, **dict(zip(lent, views[len(arrays):]))))
        except ValueError as error:
            if not any(text in str(error) for text in READ_ONLY):
                raise
            stats['lent_retried'] = stats.get('lent_retried', 0) + 1
            with frozen(spec) as fresh:
                rest, keywords = make()
                return fresh(*[value.copy() for value in arrays], *rest,
                             **dict(keywords, **{key: value.copy() for key, value in lent.items()}))
        stats['lent'] = stats.get('lent', 0) + 1
        return _unshared(result, values)
    stats['lent_copied'] = stats.get('lent_copied', 0) + 1
    rest, keywords = make()
    return method(*[value.copy() for value in arrays], *rest,
                  **dict(keywords, **{key: value.copy() for key, value in lent.items()}))


LEND_BEFORE = ("        before = saved.copy()\n", "        before = _guard_scan.lend(saved)\n")
LEND_HISTORY = ("        kwargs = {'proposal_history': history.copy(deep=True)} if spec['uses_history'] else {}\n",
                "        kwargs = lambda: {'proposal_history': history.copy(deep=True)} if spec['uses_history'] else {}\n")
LEND_SEQUENCE = (
    "            after, audit = method(observed.copy(), saved.copy(), raw.copy(),\n"
    "                                  points.copy(deep=True), deepcopy(params), **kwargs)\n",
    "            after, audit = _guard_scan.lent_call(spec, frozen_method, method, (observed, saved, raw), {LENT},\n"
    "                                  lambda: ((points.copy(deep=True), deepcopy(params)), kwargs()))\n")
LEND_LABELS = (
    "        if spec['use_current_labels']:\n"
    "            kwargs['current_labels'] = labels.copy()\n"
    "        if spec['use_original_labels']:\n"
    "            kwargs['original_labels'] = restored.copy()\n",
    "        lent = {}\n"
    "        if spec['use_current_labels']:\n"
    "            lent['current_labels'] = labels\n"
    "        if spec['use_original_labels']:\n"
    "            lent['original_labels'] = restored\n")
LEND_REVIEWED = (
    "                candidate,audit=method(observed.copy(),saved.copy(),raw.copy(),points.copy(),deepcopy(params),\n"
    "                    proposal_history=history.copy(),current_labels=labels.copy(),original_labels=restored.copy())\n",
    "                candidate,audit=_guard_scan.lent_call(spec,frozen_method,method,(observed,saved,raw),\n"
    "                    dict(current_labels=labels,original_labels=restored),\n"
    "                    lambda:((points.copy(),deepcopy(params)),dict(proposal_history=history.copy())))\n")


SETTLE_TAIL = (
    "        candidate[eligible] = after[eligible]\n"
    "        changed = candidate != labels\n"
    "        new_ledger = ledger.copy()\n"
    "        new_ledger[candidate > 0] = 0\n"
    "        removed = changed & (candidate == 0)\n"
    "        new_ledger[removed] = labels[removed]\n"
    "        applications = []\n"
    "        for t in range(len(labels)):\n"
    "            for source, target in np.unique(np.stack((labels[t][changed[t]], candidate[t][changed[t]]), axis=1), axis=0):\n"
    "                applications.append(dict(frame=t+1, source=int(source), target=int(target),\n"
    "                    changed_pixels=int(((labels[t] == source) & (candidate[t] == target) & changed[t]).sum())))\n")
SETTLE_SCOPE = ("        candidate = labels.copy()\n"
                "        eligible = ((labels == restored) if spec['projection_scope'] == 'unrevised_original'\n"
                "                    else (labels == saved) & (restored == saved))\n")
SETTLE_CALL = "        candidate, new_ledger, applications = _guard_scan.settle(labels, ledger, after, {}, saved, restored)\n"
SETTLE_MODE = "('restored' if spec['projection_scope'] == 'unrevised_original' else 'saved')"

NAME = r'([A-Za-z_]\w*)'
PATTERNS = [
    (r'len\(np\.setdiff1d\(np\.unique\(N\), ?np\.unique\(N\)\)\)', r'len(_guard_scan.added_names(\1,\2))'),
    (r'set\(map\(int, ?np\.unique\(N\)\)\) - set\(map\(int, ?np\.unique\(N\)\)\)', r'_guard_scan.added_names(\1,\2)'),
    (r'np\.array_equal\(np\.unique\(N\), ?np\.unique\(N\)\)', r'_guard_scan.same_names(\1,\2)'),
    (r'np\.array_equal\(\(N > 0\) \| \(N > 0\), ?\(N > 0\) \| \(N > 0\)\)', r'_guard_scan.same_union_foreground(\1,\2,\3,\4)'),
    (r'np\.array_equal\(N ?> ?0, ?N ?> ?0\)', r'_guard_scan.same_foreground(\1,\2)'),
    (r'np\.array_equal\(N, ?N\)', r'_guard_scan.equal(\1,\2)'),
    (r'np\.any\(\(N != N\) & \(N > 0\)\)', r'_guard_scan.changed_where_positive(\1,\2,\3)'),
    (r'np\.any\(\(N != N\) & \(N == N\)\)', r'_guard_scan.changed_where_equal(\1,\2,\3,\4)'),
    (r'np\.any\(\(N != N\) & N\)', r'_guard_scan.changed_where(\1,\2,\3)'),
    (r'np\.any\(\(N == 0\) & \(N > 0\)\)', r'_guard_scan.gained_foreground(\1,\2)'),
    (r'np\.any\(\(N > 0\) & \(N == 0\)\)', r'_guard_scan.lost_foreground(\1,\2)'),
    (r'bool\(np\.any\(N != N\)\)', r'_guard_scan.any_changed(\1,\2)'),
    (r'int\(\(N != N\)\.sum\(\)\)', r'_guard_scan.count_changed(\1,\2)'),
    (r'np\.count_nonzero\(N != N\)', r'_guard_scan.count_changed(\1,\2)'),
]
PATTERNS = [(re.compile(r'(?<![\w.])' + pattern.replace('N', NAME)), new) for pattern, new in PATTERNS]
# A whole-movie temporary named once and only read by the guard that follows it.
NAMED = {
    'anchored_memory_composition': [
        ('    changed=candidate != labels\n', '    _guard_scan.check_comparable(candidate,labels)\n'),
        ('np.any(changed & protected)', '_guard_scan.changed_where(candidate,labels,protected)'),
        ('int(changed.sum())', '_guard_scan.count_changed(candidate,labels)')],
    'terminal_owner_composition': [
        ('    changed = candidate != labels\n', '    _guard_scan.check_comparable(candidate,labels)\n'),
        ('np.any(changed & protected)', '_guard_scan.changed_where(candidate,labels,protected)'),
        ('np.any(changed & (labels != 0))', '_guard_scan.changed_where_nonzero(candidate,labels,labels)'),
        ('np.any(changed & (ledger != 0))', '_guard_scan.changed_where_nonzero(candidate,labels,ledger)')],
    'optical_source_composition': [
        ('        difference = candidate != labels\n', '        _guard_scan.check_comparable(candidate,labels)\n'),
        ('            survivor = candidate > 0\n', '            _guard_scan.check_positive(candidate)\n'),
        ('np.any(survivor & (labels == 0)) or np.any(candidate[survivor] != labels[survivor])',
         '_guard_scan.survivor_changed(candidate,labels)'),
        ('np.any(difference & protected)', '_guard_scan.changed_where(candidate,labels,protected)'),
        ('np.count_nonzero(difference)', '_guard_scan.count_changed(candidate,labels)')],
    'recovery_sequence_composition': [
        ("        candidate = labels.copy()\n        eligible = (labels == saved) & (restored == saved)\n" + SETTLE_TAIL,
         SETTLE_CALL.format("'saved'")),
        LEND_BEFORE, LEND_HISTORY, (LEND_SEQUENCE[0], LEND_SEQUENCE[1].replace('{LENT}', '{}'))],
    'current_path_sequence_composition': [(SETTLE_SCOPE + SETTLE_TAIL, SETTLE_CALL.format(SETTLE_MODE)),
        LEND_BEFORE, LEND_HISTORY, LEND_LABELS, (LEND_SEQUENCE[0], LEND_SEQUENCE[1].replace('{LENT}', 'lent'))],
    'reviewed_ownership_composition': [LEND_REVIEWED],
    'contact_sequence_composition': [
        (SETTLE_SCOPE + "        if spec['projection_scope'] == 'rule_validated_complete_paths':\n"
         "            eligible = np.ones(labels.shape, dtype=bool)\n" + SETTLE_TAIL,
         SETTLE_CALL.format(SETTLE_MODE + " + ('+all' if spec['projection_scope'] == 'rule_validated_complete_paths' else '')"))],
}
SEQUENCE_TEMPORARIES = ('eligible', 'changed', 'removed')
TEMPORARIES = {'anchored_memory_composition': 'changed', 'terminal_owner_composition': 'changed',
               'optical_source_composition': ('difference', 'survivor'),
               'recovery_sequence_composition': SEQUENCE_TEMPORARIES,
               'current_path_sequence_composition': SEQUENCE_TEMPORARIES,
               'contact_sequence_composition': SEQUENCE_TEMPORARIES}


def rewrite(source, name):
    """Rewritten top-level function sources {function: source}, or None when a named rewrite does not apply."""
    source = source.replace('\r\n', '\n')
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    found = {}
    named = NAMED.get(name, [])
    used = [0] * len(named)
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.decorator_list:
            continue
        text = ''.join(lines[node.lineno - 1:node.end_lineno])
        new = text
        for index, (old, replacement) in enumerate(named):
            count = new.count(old)
            used[index] += count
            new = new.replace(old, replacement)
        for pattern, replacement in PATTERNS:
            new = pattern.sub(replacement, new)
        temporaries = TEMPORARIES.get(name, ())
        for temporary in (temporaries,) if isinstance(temporaries, str) else temporaries:
            if re.search(r'\b' + temporary + r'\b', new):
                return None
        if new != text:
            found[node.name] = new
    if any(count != 1 for count in used):
        return None
    return found


def adapt(module):
    """{function name: (original, rewritten)} for a pinned composition module, or None."""
    path = getattr(module, '__file__', None)
    try:
        source = Path(path).read_bytes()
    except (OSError, TypeError):
        return None
    name = VERSIONS.get(hashlib.sha256(source).hexdigest()[:16])
    if name is None or getattr(module, '__name__', None) != name:
        return None
    if getattr(module, '_guard_scan', sys.modules[__name__]) is not sys.modules[__name__]:
        return None
    found = rewrite(source.decode('utf-8'), name)
    if not found:
        return None
    result = {}
    for function, text in found.items():
        original = getattr(module, function, None)
        if (original is None or getattr(original, '__module__', None) != name
                or getattr(original, '__wrapped__', None) is not None):
            return None
        namespace = {}
        exec(compile(text, '<guard-scan:' + name + '>', 'exec'), module.__dict__, namespace)
        rewritten = namespace[function]
        rewritten.__wrapped__ = original
        result[function] = (original, rewritten)
    return result


@contextmanager
def install():
    """Rewrite loaded pinned compositions and their imported aliases; restore afterwards."""
    counts = stats
    swaps = {}
    added = []
    for module in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        if not isinstance(getattr(module, '__file__', None), str) or '_composition' not in Path(module.__file__).name:
            continue
        found = adapt(module)
        if not found:
            continue
        if '_guard_scan' not in module.__dict__:
            module._guard_scan = sys.modules[__name__]
            added.append(module)
        counts['versions'] += 1
        counts['functions'] += len(found)
        for original, rewritten in found.values():
            swaps[id(original)] = (original, rewritten)
    aliases = []
    if swaps:
        for child in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
            names = getattr(child, '__dict__', None)
            if not names:
                continue
            for key, value in list(names.items()):
                found = swaps.get(id(value))
                if found is not None and value is found[0]:
                    aliases.append((child, key, value))
                    setattr(child, key, found[1])
    try:
        yield counts
    finally:
        for child, key, value in reversed(aliases):
            setattr(child, key, value)
        for module in added:
            module.__dict__.pop('_guard_scan', None)
