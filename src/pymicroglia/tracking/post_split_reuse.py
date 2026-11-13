"""Post-split backfill discovery with per-track values formed once.

For every target track, discovery visited every other track and recomputed its
first and last frame, its visible-owner support and its visible row count from
the same unchanged group. Each value is formed by the same statement the first
time it is needed and reused. First claims and summary losses come from one
unique pass per frame instead of whole-movie comparisons per identity: an
identity is first claimed on the first frame holding it, and is lost on a frame
that held it before and not after.
"""
import __future__
import hashlib
import inspect
from contextlib import contextmanager
from pathlib import Path

import numpy as np

SOURCE = '8df31e69a7e6ee3b27d60bd94da70a62906c9c85e99fcc0ae55a8badd3fe883b'
DISCOVER = (
    ("    first_claim = {\n"
     "        identity: int(np.flatnonzero(np.any(\n"
     "            labels == identity, axis=(1, 2)))[0]) for identity in active}\n",
     "    first_claim = _first_claims(labels, active)\n"),
    ("    proposals: list[dict] = []\n",
     "    proposals: list[dict] = []\n    _track_memo = {}\n"),
    ("            other_first = int(other.frame.min())\n"
     "            other_last = int(other.frame.max())\n",
     "            other_first, other_last = _track_bounds(other_track, other, _track_memo)\n"),
    ("_owner_support(other, stable_owner) >= minimum_successor",
     "_track_support(other_track, other, stable_owner, _track_memo) >= minimum_successor"),
    ("                    and _owner_support(other, foreign_owner)\n"
     "                    == len(other[other.physically_visible.astype(bool)])):\n",
     "                    and _track_support(other_track, other, foreign_owner, _track_memo)\n"
     "                    == _track_visible(other_track, other, _track_memo)):\n"),
)
SUMMARY = (
    ("    named_losses = 0\n"
     "    for identity in active:\n"
     "        before = np.any(labels == identity, axis=(1, 2))\n"
     "        after = np.any(candidate == identity, axis=(1, 2))\n"
     "        named_losses += int(np.count_nonzero(before & ~after))\n",
     "    named_losses = _named_losses(labels, candidate, active)\n"),
)


def _plain(frame):
    return frame.view(np.ndarray) if isinstance(frame, np.ndarray) else np.asarray(frame)


def helpers(module, stats):
    def first_claims(labels, active):
        first = {}
        for t in range(len(labels)):
            for value in np.unique(_plain(labels[t])):
                first.setdefault(int(value), t)
        stats['first_claims'] += 1
        return {identity: int(first[identity]) for identity in active}

    def track_bounds(track, other, memo):
        key = ('bounds', track)
        if key not in memo:
            memo[key] = (int(other.frame.min()), int(other.frame.max()))
        return memo[key]

    def track_support(track, other, owner, memo):
        key = ('support', track, owner)
        if key not in memo:
            memo[key] = module._owner_support(other, owner)
        return memo[key]

    def track_visible(track, other, memo):
        key = ('visible', track)
        if key not in memo:
            memo[key] = len(other[other.physically_visible.astype(bool)])
        return memo[key]

    def named_losses(labels, candidate, active):
        losses = 0
        for t in range(len(labels)):
            before = set(map(int, np.unique(_plain(labels[t]))))
            after = set(map(int, np.unique(_plain(candidate[t]))))
            losses += len((before & active) - after)
        stats['summaries'] += 1
        return losses

    return dict(_first_claims=first_claims, _track_bounds=track_bounds,
                _track_support=track_support, _track_visible=track_visible,
                _named_losses=named_losses)


def _recompile(function, replacements, globals_, name):
    source = inspect.getsource(function)
    for old, new in replacements:
        if source.count(old) != 1:
            return None
        source = source.replace(old, new)
    namespace = {}
    exec(compile(source, name, 'exec', flags=__future__.annotations.compiler_flag,
                 dont_inherit=True), globals_, namespace)
    return namespace[function.__name__]


@contextmanager
def install(module):
    stats = dict(supported=False, first_claims=0, summaries=0)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SOURCE:
        yield stats
        return
    globals_ = vars(module)
    added = helpers(module, stats)
    globals_.update(added)
    originals = dict(discover=module.discover, summarize=module.summarize)
    replacements = dict(
        discover=_recompile(module.discover, DISCOVER, globals_, '<post-split-discover>'),
        summarize=_recompile(module.summarize, SUMMARY, globals_, '<post-split-summary>'))
    if any(function is None for function in replacements.values()):
        for name in added:
            globals_.pop(name, None)
        yield stats
        return
    for name, function in replacements.items():
        setattr(module, name, function)
    stats['supported'] = True
    try:
        yield stats
    finally:
        for name, function in originals.items():
            setattr(module, name, function)
        for name in added:
            globals_.pop(name, None)
