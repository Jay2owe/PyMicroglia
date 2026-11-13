"""File digests for the installers that scan every loaded module.

Several correction installers look at each loaded module that has a `correct`
(or another stage) function and hash its file to see whether it is a verified
version. They run before every correction stage, so the same hundred engine
files were read and hashed again and again. file_sha256 hashes a file once per
(path, size, modification time): the same hex digest, and the same OSError or
TypeError for a missing file or a module without a path.

The installers' scans also skip the objects Cython's pure-Python shadow puts in
sys.modules (cython.parallel, cython.cimports, ...). Their __getattr__ tries to
import the requested name as a top-level module, so every getattr(module,
'correct', None) searched the whole import path before failing; none of the
names looked up is an importable module, so the lookup always gave the default.

Inside held() (one stage's adapter installs, milliseconds apart) each path is
checked once: the installers that look at the same modules share its digest
instead of each re-reading its size and modification time. Errors are not
remembered, so a missing file still raises at every call.

loaded(name) also finds module objects that are no longer in sys.modules: an
isolated import (frozen_method) restores sys.modules afterwards, but a
composition imported meanwhile keeps the module object it imported as a global.
"""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import sys
from types import ModuleType

_DIGESTS = {}
_HELD = None
_INDEX = None


@contextmanager
def held():
    global _HELD, _INDEX
    outer = _HELD
    if outer is None:
        _HELD = {}
        _INDEX = None
    try:
        yield
    finally:
        if outer is None:
            _HELD = None
            _INDEX = None


def file_sha256(path):
    held_digests = _HELD
    if held_digests is not None:
        digest = held_digests.get(path) if isinstance(path, str) else None
        if digest is not None:
            return digest
    status = os.stat(path)
    key = (os.fspath(path), status.st_size, status.st_mtime_ns)
    digest = _DIGESTS.get(key)
    if digest is None:
        digest = _DIGESTS[key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if held_digests is not None and isinstance(path, str):
        held_digests[path] = digest
    return digest


def _held_modules():
    """Every module object a loaded module holds, by __name__, in scan order."""
    index = {}
    for module in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        names = getattr(module, '__dict__', None)
        if not names:
            continue
        for value in list(names.values()):
            if type(value) is ModuleType:
                held_name = value.__dict__.get('__name__')
                if type(held_name) is str:
                    index.setdefault(held_name, {}).setdefault(id(value), value)
    return index


def loaded(name):
    """Every distinct module object called name: sys.modules' entry and any a loaded module holds."""
    global _INDEX
    found = {}
    entry = sys.modules.get(name)
    if entry is not None:
        found[id(entry)] = entry
    if _HELD is not None and type(name) is str:
        # One scan serves every install of a held() block; the installs
        # import no module under a target's name.
        if _INDEX is None:
            _INDEX = _held_modules()
        for key, value in _INDEX.get(name, {}).items():
            found.setdefault(key, value)
        return list(found.values())
    for module in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        names = getattr(module, '__dict__', None)
        if not names:
            continue
        for value in list(names.values()):
            if type(value) is ModuleType and value.__dict__.get('__name__') == name:
                found.setdefault(id(value), value)
    return list(found.values())
