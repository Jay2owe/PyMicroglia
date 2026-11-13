"""Terminal-companion recovery with indexed track points and frame-local excess.

Discovery filtered a whole track group for every frame it inspected; a frame's
point is the group's only row on that frame, so each group's frames are
indexed once and the same row is taken by position. Component excess is a sum
of independent per-frame terms, so two stacks that agree outside some frames
compare, and differ, exactly as their sums over those frames do. Application
compares a trial with the current result over the proposal's frames, and the
summary compares the result with the input over the frames that changed.
Named losses use the per-frame count shared with the post-split step.
"""
import __future__
import hashlib
import inspect
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from . import post_split_reuse

SOURCE = 'da467a0fcaa7eb6655f33fbe4fe087ab6674f83ef156ed885f815ed8f7156941'
DISCOVER = (
    ("        last = int(group.frame.max())\n",
     "        last = int(group.frame.max())\n        _points = _PointIndex(group)\n", 1),
    ("_point(group, ", "_points(", 3),
)
RECOVER = (
    ("        if not reason and _excess_components(trial) > _excess_components(candidate):\n",
     "        if not reason and _excess_frames(trial, _span(proposal)) > _excess_frames(candidate, _span(proposal)):\n", 1),
)
SUMMARY = (
    ("    named_losses = 0\n"
     "    for identity in before_ids:\n"
     "        before = np.any(labels == identity, axis=(1, 2))\n"
     "        after = np.any(candidate == identity, axis=(1, 2))\n"
     "        named_losses += int(np.count_nonzero(before & ~after))\n",
     "    named_losses = _named_losses(labels, candidate, before_ids)\n", 1),
    ("            0, _excess_components(candidate) - _excess_components(labels)),\n",
     "            0, _excess_frames(candidate, _changed_frames(changed))\n"
     "            - _excess_frames(labels, _changed_frames(changed))),\n", 1),
)


class PointIndex:
    def __init__(self, group):
        self.group = group
        frames = group.frame.astype(int).to_numpy()
        values, first, counts = np.unique(frames, return_index=True, return_counts=True)
        self.positions = {int(value): int(index)
                          for value, index, count in zip(values, first, counts) if count == 1}

    def __call__(self, frame):
        position = self.positions.get(int(frame))
        return None if position is None else self.group.iloc[position]


def helpers(module, stats):
    def excess_frames(stack, frames):
        stats['excess'] += 1
        return module._excess_components([stack[t] for t in frames])

    def span(proposal):
        return range(int(proposal.first_frame), int(proposal.last_frame) + 1)

    def changed_frames(changed):
        return [t for t in range(len(changed)) if np.any(changed[t])]

    named = post_split_reuse.helpers(module, stats)['_named_losses']
    return dict(_PointIndex=PointIndex, _excess_frames=excess_frames, _span=span,
                _changed_frames=changed_frames, _named_losses=named)


@contextmanager
def install(module):
    stats = dict(supported=False, excess=0, summaries=0, first_claims=0)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SOURCE:
        yield stats
        return
    sources = {}
    for name, edits in (('discover', DISCOVER), ('recover', RECOVER), ('summarize', SUMMARY)):
        source = inspect.getsource(getattr(module, name))
        for old, new, count in edits:
            if source.count(old) != count:
                yield stats
                return
            source = source.replace(old, new)
        sources[name] = source
    globals_ = vars(module)
    added = helpers(module, stats)
    globals_.update(added)
    originals = {name: getattr(module, name) for name in sources}
    for name, source in sources.items():
        namespace = {}
        exec(compile(source, f'<terminal-companion-{name}>', 'exec',
                     flags=__future__.annotations.compiler_flag, dont_inherit=True), globals_, namespace)
        setattr(module, name, namespace[name])
    stats['supported'] = True
    try:
        yield stats
    finally:
        for name, function in originals.items():
            setattr(module, name, function)
        for name in added:
            globals_.pop(name, None)
