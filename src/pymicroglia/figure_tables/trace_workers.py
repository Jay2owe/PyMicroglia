"""Bounded processes run the existing independent cell/measurement fits."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
from queue import Empty

import numpy as np
import pandas as pd

_events = None
_session = None


def _priority(frame, metrics):
    """Broad spectra often require more model components; start them earlier.

    This inexpensive scheduling score never changes the data passed to the
    estimator. Missing samples/frames and fitted periods keep their full rules.
    """
    score = float('-inf')
    for metric in metrics:
        values = pd.to_numeric(frame[metric], errors='coerce').to_numpy(float)
        values = values[np.isfinite(values)]
        if len(values) < 8:
            continue
        power = np.abs(np.fft.rfft(values - np.mean(values)))[1:] ** 2
        total = np.sum(power)
        if not np.isfinite(total) or total <= 0:
            continue
        probability = power[power > 0] / total
        score = max(score, float(-np.sum(probability * np.log(probability))))
    return score


def _initialize(events, environment):
    from pymicroglia import workbench
    global _events, _session
    _events = events
    # Each process holds one nested provenance scope. The parent checks the
    # shared software files again after all workers finish before attaching.
    _session = workbench.analysis_session()
    capture = _session.__enter__()
    if capture.environment != environment:
        raise ValueError('Worker software environment differs from the parent analysis.')


def _cell(arguments):
    from .all_cell_traces import trace_data
    frame, metrics, identity, resolved, view, normalization, config, settings = arguments
    return trace_data(frame, metrics, [identity], resolved, view, normalization, config,
                      **settings, progress=_events.put, _adjust=False)


def parallel_traces(frame, metrics, identities, resolved, view, normalization, config,
                    settings, *, workers, progress):
    from pymicroglia import workbench
    workers = min(int(workers), len(identities) * len(metrics))
    context = multiprocessing.get_context('spawn')
    events = context.Queue()
    columns = list(dict.fromkeys(['identity', 'frame_index', 'hours', *metrics]))
    grouped = {int(identity): group[columns].copy()
               for identity, group in frame.groupby('identity', sort=False)
               if identity in identities}
    pending, results, active = {}, {}, {}
    counts = {identity: set() for identity in identities}
    # Different measurements of the same cell are independent fits. Do not
    # leave a worker idle while one cell runs several slow fits in sequence.
    tasks = [(identity, metric) for identity in identities for metric in metrics]
    inventory = iter(sorted(tasks, key=lambda task:
                            (-_priority(grouped[task[0]], [task[1]]), task[0], task[1])))

    def report(stage, identity=None, metric=None):
        if progress is not None:
            progress({'stage': stage, 'identity': identity, 'metric': metric,
                      'completed': sum(map(len, counts.values())), 'total': len(identities) * len(metrics),
                      'completed_cells': sum(len(count) == len(metrics) for count in counts.values()),
                      'total_cells': len(identities), 'workers': workers,
                      'active_cells': list(active.values())})

    def submit(pool):
        try:
            identity, metric = next(inventory)
        except StopIteration:
            return False
        args = (grouped[identity][['identity', 'frame_index', 'hours', metric]],
                [metric], identity, resolved, view, normalization, config, settings)
        key = (identity, metric)
        pending[pool.submit(_cell, args)] = key
        active[key] = {'identity': identity, 'stage': 'fitting', 'metric': metric}
        return True

    report('starting_workers')
    try:
        with workbench.analysis_session() as session, ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                 initializer=_initialize,
                                 initargs=(events, session.environment)) as pool:
            for _ in range(workers):
                submit(pool)
            while pending:
                completed, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                while True:
                    try:
                        event = events.get_nowait()
                    except Empty:
                        break
                    identity, metric = event['identity'], event['metric']
                    key = (identity, metric)
                    if key in results:
                        continue
                    if event['completed']:
                        counts[identity].add(metric)
                    active[key] = {name: event[name] for name in ('identity', 'stage', 'metric')}
                    report(event['stage'], identity, metric)
                for future in completed:
                    identity, metric = key = pending.pop(future)
                    results[key] = future.result()
                    counts[identity].add(metric)
                    active.pop(key, None)
                    submit(pool)
                    report('testing', identity, metric)
    finally:
        events.close()
        events.join_thread()
    # Restore the exact serial identity/metric order before correction/drawing.
    order = [(identity, metric) for identity in identities for metric in metrics]
    return (pd.concat([results[key][0] for key in order], ignore_index=True),
            # Infer columns over the complete family, as serial trace_data
            # does. Concatenating per-task frames can retain object dtype
            # from an all-missing task beside ordinary numeric estimates.
            pd.DataFrame([row for key in order
                          for row in results[key][1].to_dict('records')]))
