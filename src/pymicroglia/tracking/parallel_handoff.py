"""Process-parallel, read-only handoff discovery; label edits remain sequential.

Each task handles a contiguous set of successor tracks. Workers read one shared
memory-mapped label movie and each own their table. Results are joined in the
original successor order before proposal identifiers are assigned.
"""
import __future__
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import multiprocessing
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

_WORK = None


def _initialise(module_path, source, scored_path, labels_path, params, errors, promotion):
    global _WORK
    sys.path.insert(0, str(Path(module_path).parent))
    spec = importlib.util.spec_from_file_location('_parallel_handoff_engine', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from .bracketed_handoff_reuse import helpers
    module.__dict__.update(helpers(module, dict(runs=0, summaries=0)))
    loop = 'for successor_id, successor in sorted(groups.items()):'
    if source.count(loop) != 1:
        raise ValueError('Handoff discovery changed; parallel equivalence review required')
    source = source.replace(loop,
        'for successor_id, successor in sorted(groups.items()):\n'
        '        if successor_id not in _parallel_successors: continue')
    exec(compile(source, '<parallel-handoff-discover>', 'exec',
                 flags=__future__.annotations.compiler_flag, dont_inherit=True), module.__dict__)
    np.seterr(**errors)
    if promotion is not None:
        np._set_promotion_state(promotion)
    # Limit nested numerical libraries; the process pool owns this CPU budget.
    from threadpoolctl import threadpool_limits
    limiter = threadpool_limits(limits=1)
    _WORK = (module, pd.read_pickle(scored_path),
             np.load(labels_path, mmap_mode='r'), params, limiter)


def _discover(successors):
    module, scored, labels, params, _ = _WORK
    module._parallel_successors = frozenset(successors)
    return module.discover(scored, labels, params)


def discover(module, source, serial, scored, labels, params, policy, stats):
    tracks = sorted(int(value) for value in scored.loc[
        scored.physically_visible.astype(bool), 'track_id'].unique()) if not scored.empty else []
    requested = min(policy.count('identity_handoff'), len(tracks))
    # Each worker owns pandas groups and copies; mapped labels remain read-only.
    table_bytes = int(scored.memory_usage(index=True, deep=True).sum())
    per_worker = max(64 * 1024**2, table_bytes * 6)
    memory_workers = int(policy.memory_gb * 1024**3 // per_worker)
    count = min(requested, max(1, memory_workers))
    stats.update(requested_workers=requested, workers=count, table_bytes=table_bytes)
    if count < 2:
        return serial(scored, labels, params)
    chunks = [tracks[i * len(tracks) // count:(i + 1) * len(tracks) // count]
              for i in range(count)]
    with tempfile.TemporaryDirectory(prefix='motion-handoff-') as name:
        scratch = Path(name)
        scored_path, labels_path = scratch/'scored.pkl', scratch/'labels.npy'
        scored.to_pickle(scored_path)
        mapped = np.lib.format.open_memmap(labels_path, mode='w+',
                                           dtype=labels.dtype, shape=labels.shape)
        for frame in range(len(labels)):
            mapped[frame] = labels[frame]
        mapped.flush()
        del mapped
        getter = getattr(np, '_get_promotion_state', None)
        with ProcessPoolExecutor(max_workers=count,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=_initialise,
                initargs=(str(module.__file__), source, str(scored_path), str(labels_path),
                          params, np.geterr(), getter() if getter else None)) as executor:
            tables = list(executor.map(_discover, chunks))
    # Concatenation of nonempty chunks matches the serial constructor's dtypes.
    nonempty = [table for table in tables if not table.empty]
    if not nonempty:
        return tables[0]
    result = pd.concat(nonempty, ignore_index=True)
    result['proposal_id'] = [f'P{i+1:04d}' for i in range(len(result))]
    stats['parallel_calls'] += 1
    return result
