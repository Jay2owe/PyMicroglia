"""Optional wall-clock trace of correction operations, including failures.

Inclusive spans contain nested calls. Exclusive spans subtract those child
operations on the same thread; concurrent thread durations are never additive.
"""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import json
import threading
import time


@contextmanager
def time_history_stages(module, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # An explicit trace path owns one invocation; never overwrite prior evidence.
    handle = destination.open('x', encoding='utf8')
    lock = threading.Lock()
    state = threading.local()
    saved = []

    def record(value):
        with lock:
            handle.write(json.dumps(value, sort_keys=True)+'\n')
            handle.flush()

    def wrap(name, function):
        @wraps(function)
        def timed(*args, **kwargs):
            stack = getattr(state, 'stack', None)
            if stack is None:
                stack = state.stack = []
            frame = dict(children=0.)
            parent = stack[-1] if stack else None
            stack.append(frame)
            start = time.perf_counter()
            thread = threading.get_ident()
            record(dict(event='start', operation=name, thread=thread, monotonic_seconds=start))
            status = 'complete'
            try:
                return function(*args, **kwargs)
            except BaseException:
                status = 'failed'
                raise
            finally:
                elapsed = time.perf_counter()-start
                stack.pop()
                if parent is not None:
                    parent['children'] += elapsed
                record(dict(event='finish', operation=name, thread=thread,
                            status=status, inclusive_seconds=elapsed,
                            exclusive_seconds=elapsed-frame['children']))
        return timed

    try:
        for name, value in list(vars(module).items()):
            if callable(value) and (name.startswith(('_issue', '_well_')) or
                                    name in ('run_accepted_history', '_seat_identity', '_general_blob_residency')):
                saved.append((name, value))
                setattr(module, name, wrap(name, value))
        yield
    finally:
        for name, value in reversed(saved):
            setattr(module, name, value)
        handle.close()
