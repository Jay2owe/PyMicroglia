"""Bracketed-handoff discovery with per-track last frames and owner runs formed once.

For every successor track, discovery visited every other track and took that
track's last frame (up to three times) and, when the gap fitted, rebuilt its
owner runs. Both depend only on the unchanged track group, so each is formed
by the same expression the first time it is needed and reused. Runs are only
read after they are built. Summary losses use the per-frame count shared with
the post-split step.
"""
import __future__
import hashlib
import inspect
from contextlib import contextmanager
from pathlib import Path

from . import post_split_reuse

SOURCE = '675d83042ff96e188c57e641679e3f2dcec5dd4edac2da3dfe7d182773237acd'
DISCOVER = (
    ("    rows: list[dict] = []\n", "    rows: list[dict] = []\n    _track_memo = {}\n", 1),
    ("        runs = _positive_runs(successor)\n",
     "        runs = _track_runs(successor_id, successor, _track_memo)\n", 1),
    ("            predecessor_runs = _positive_runs(predecessor)\n",
     "            predecessor_runs = _track_runs(predecessor_id, predecessor, _track_memo)\n", 1),
    ("int(predecessor.frame.max())", "_track_last(predecessor_id, predecessor, _track_memo)", 3),
)
SUMMARY = post_split_reuse.SUMMARY


def helpers(module, stats):
    def track_last(track, group, memo):
        key = ('last', track)
        if key not in memo:
            memo[key] = int(group.frame.max())
        return memo[key]

    def track_runs(track, group, memo):
        key = ('runs', track)
        if key not in memo:
            memo[key] = module._positive_runs(group)
            stats['runs'] += 1
        return memo[key]

    named = post_split_reuse.helpers(module, stats)['_named_losses']
    return dict(_track_last=track_last, _track_runs=track_runs, _named_losses=named)


@contextmanager
def install(module):
    stats = dict(supported=False, runs=0, summaries=0)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SOURCE:
        yield stats
        return
    sources = {}
    for name, edits in (('discover', DISCOVER), ('summarize', tuple((old, new, 1) for old, new in SUMMARY))):
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
        exec(compile(source, f'<bracketed-handoff-{name}>', 'exec',
                     flags=__future__.annotations.compiler_flag, dont_inherit=True), globals_, namespace)
        setattr(module, name, namespace[name])
    from .execution import current
    policy = current()
    if policy is not None:
        from .parallel_handoff import discover
        serial = module.discover
        stats['parallel_calls'] = 0
        def parallel(scored, labels, params):
            return discover(module, sources['discover'], serial,
                            scored, labels, params, policy, stats)
        module.discover = parallel
    stats['supported'] = True
    try:
        yield stats
    finally:
        for name, function in originals.items():
            setattr(module, name, function)
        for name in added:
            globals_.pop(name, None)
