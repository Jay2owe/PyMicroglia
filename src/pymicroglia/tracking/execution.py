"""Execution resources, separate from tracking's scientific parameters.

The engine already serializes its process-global imports. A scoped policy uses
the same restriction; independent recordings belong in separate processes.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import os
import threading

STAGES = ("detection", "identity_handoff")
_LOCK = threading.RLock()
_CURRENT = None


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def available_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


@dataclass(frozen=True)
class ExecutionPolicy:
    workers: int = 1
    stage_workers: dict = field(default_factory=dict)
    cpu_budget: int | None = None
    memory_gb: float = 4.0

    def __post_init__(self):
        _positive(self.workers, "workers")
        if self.cpu_budget is not None:
            _positive(self.cpu_budget, "cpu_budget")
        if isinstance(self.memory_gb, bool) or not 0 < self.memory_gb < float('inf'):
            raise ValueError("memory_gb must be finite and positive")
        overrides = dict(self.stage_workers)
        for stage, count in overrides.items():
            if stage not in STAGES:
                raise ValueError(f"Unknown parallel stage {stage!r}; choose {STAGES}")
            _positive(count, f"stage_workers[{stage!r}]")
        object.__setattr__(self, "stage_workers", overrides)

    def count(self, stage):
        if stage not in STAGES:
            raise ValueError(f"Unknown parallel stage: {stage}")
        return min(self.stage_workers.get(stage, self.workers),
                   self.cpu_budget or available_cpus(), available_cpus())

    def as_dict(self):
        return dict(workers=self.workers, stage_workers=dict(self.stage_workers),
                    cpu_budget=self.cpu_budget, memory_gb=self.memory_gb,
                    resolved={stage: self.count(stage) for stage in STAGES})


def current():
    return _CURRENT


@contextmanager
def activate(value=None):
    global _CURRENT
    if value is None:
        yield None
        return
    policy = value if isinstance(value, ExecutionPolicy) else ExecutionPolicy(**value)
    with _LOCK:
        previous, _CURRENT = _CURRENT, policy
        try:
            yield policy
        finally:
            _CURRENT = previous
