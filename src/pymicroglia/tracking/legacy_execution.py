"""Apply the new execution policy to an already pinned, older engine runtime.

This scoped bridge preserves the existing implementation files and checkpoint
signatures. It changes only the read-only discovery scheduling, with a separate
execution receipt. New installations use engine.run(execution=...) directly.
"""
from contextlib import contextmanager
import inspect

from .execution import activate


@contextmanager
def install(value):
    from . import bracketed_handoff_reuse as reuse, batched_detection_cores as detection
    from .parallel_handoff import discover
    original_install = reuse.install
    original_detection = detection.detection_workers
    with activate(value) as policy:
        @contextmanager
        def handoff(module):
            # Capture the original function before the pinned adapter compiles it.
            source = inspect.getsource(module.discover)
            for old, new, count in reuse.DISCOVER:
                if source.count(old) != count:
                    raise ValueError('Pinned handoff source changed; review the execution bridge')
                source = source.replace(old, new)
            with original_install(module) as stats:
                if not stats.get('supported'):
                    raise ValueError('Pinned handoff adapter does not support this source')
                serial = module.discover
                stats['parallel_calls'] = 0
                def parallel(scored, labels, params):
                    return discover(module, source, serial, scored, labels, params, policy, stats)
                module.discover = parallel
                try:
                    yield stats
                finally:
                    module.discover = serial

        def detection_workers(pool, workers=None):
            count = policy.count('detection')
            if workers is not None:
                count = min(count, max(1, int(workers)))
            return original_detection(pool, workers=count)

        reuse.install, detection.detection_workers = handoff, detection_workers
        try:
            yield policy
        finally:
            reuse.install, detection.detection_workers = original_install, original_detection
