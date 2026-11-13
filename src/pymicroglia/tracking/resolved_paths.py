"""Loaded module paths resolved once per process.

Every adapter install walks sys.modules and resolves module files to find the
aliases inside the frozen code; the correction methods install many times per
run. Loaded module files do not move during a run, so a successful resolve of
an absolute path is kept and reused; a failed one is retried and raises as
before, and a relative path (which depends on the working folder) is resolved
every time.
"""
from pathlib import Path

_resolved = {}


def resolve(path):
    key = str(path)
    value = _resolved.get(key)
    if value is None:
        value = Path(path)
        if not value.is_absolute():
            return value.resolve()
        value = _resolved[key] = value.resolve()
    return value
