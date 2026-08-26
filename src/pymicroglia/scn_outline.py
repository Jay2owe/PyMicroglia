"""The accepted automatic SCN outline — the action, not the algorithm.

The suprachiasmatic nucleus outline moved to **PySCNSlice** on 2026-08-23. About
6,200 lines of it lived here, in a package named after a cell type the method has
nothing to do with, and it was already a leaf: nothing in PyMicroglia imported
it, and everything it needed from PyMicroglia was nine generic file helpers. No
science crossed the boundary, so the cut was free.

What stayed is this module, and it stayed for one reason: ``automatic_scn_outline``
is a **registered action**, and run records name it. Deleting the action would
have broken every stored record that says a run of it happened. So the action
still resolves, still reports its live defaults, and still runs — one import
further away.

``pyscnslice`` is a **hard optional dependency**, the same shape as
``circadian_workbench`` in :mod:`pymicroglia.rhythm` and the opposite of
``analysis_kit`` in :mod:`pymicroglia._optional`. Losing the audit layer costs a
run record and must be silent; losing the outline means the action cannot run at
all, so its absence is a named error rather than a quiet ``None``.

Install it with::

    pip install "PyMicroglia[scn]"

The delegates below are wrappers rather than plain re-exports, and the two lines
that look redundant are not. ``functools.wraps`` carries the real signature
across, which is what lets ``describe`` report the defaults the function will
actually apply instead of falling back to the catalogue. Reassigning
``__module__`` puts the wrapper back in this module's namespace, which is what
lets ``registry._covered_functions`` recognise it — otherwise ``discover`` would
start reporting the outline's own helpers as public functions nobody exposed.
"""

from __future__ import annotations

import functools
from typing import Any

__all__ = [
    "SCNSliceMissing",
    "METHOD_VERSION",
    "ACCEPTED_SETTINGS",
    "CROP_SCALES",
    "draw_scn_labels",
    "automatic_scn_outline",
]

_MISSING = (
    "The automatic SCN outline needs PySCNSlice: "
    'pip install "PyMicroglia[scn]". It is a hard optional dependency, not a '
    "soft one — the method moved out of this package rather than being "
    "reimplemented, and silently skipping an outline would be worse than "
    "failing."
)


class SCNSliceMissing(ImportError):
    """The outline was asked for on a machine without PySCNSlice."""


def _upstream():
    """``pyscnslice.outline``, or ``None`` when it is not installed.

    ``None`` rather than a raise, because this module has to import either way:
    ``registry`` treats an unimportable science module as *pending*, and pending
    means "this stage has not landed yet", which would be the wrong story for a
    dependency somebody simply has not installed.
    """
    try:
        from pyscnslice import outline as module
    except ImportError:
        return None
    return module


def _delegate(name: str):
    """One upstream function, callable from here under its own signature."""
    module = _upstream()
    if module is None:
        def unavailable(*args: Any, **kwargs: Any):
            raise SCNSliceMissing(_MISSING)

        unavailable.__name__ = name
        unavailable.__qualname__ = name
        unavailable.__doc__ = _MISSING
        return unavailable

    target = getattr(module, name)

    @functools.wraps(target)
    def delegate(*args: Any, **kwargs: Any):
        return target(*args, **kwargs)

    delegate.__module__ = __name__
    return delegate


_MODULE = _upstream()

#: Which frozen method would run. Empty when PySCNSlice is absent — the catalogue
#: carries this action's version too, and ``recording._method_version`` reads
#: that first, so a record still says what ran.
METHOD_VERSION: str = getattr(_MODULE, "METHOD_VERSION", "")

#: The accepted pixel values, for anyone comparing a run against them.
ACCEPTED_SETTINGS: dict[str, Any] = dict(
    getattr(_MODULE, "ACCEPTED_SETTINGS", {}) or {})

#: What ``tight``, ``standard`` and ``wide`` mean as a multiple of the outline.
CROP_SCALES: dict[str, float] = dict(getattr(_MODULE, "CROP_SCALES", {}) or {})

# The action keeps its own names; upstream renamed its functions in PySCNSlice
# 0.5.0, when the outline was split into modules named for what each does.
# `draw_scn_labels` is `outline.labels`, `automatic_scn_outline` is
# `outline.automatic`. The action is what a run record names, so it does not
# move with them.
draw_scn_labels = _delegate("labels")
automatic_scn_outline = _delegate("automatic")
