"""Identity tracking: the seam the tracker drops into.

The mask says which pixels are cell in one exposure and links nothing across
frames. Linking them -- through movement, merges, splits and temporary
invisibility -- is the Motion project's tracker, which is not ported: its
entry point imports over forty well-specific calibrations by name. What *is*
settled is its output, and :mod:`.contract` writes that down so the measure
step reads against it today and the tracker is swapped in later by changing
:data:`TRACKER_TARGET` alone.

This package is therefore a **seam**, and the registry treats it as one: its
functions resolve, but :func:`status` says whether the target behind them
does, and the ``track`` action is reported pending until it does. That is
Auto-Organotypic's shape for a missing instrument client -- a line at the top
of a run rather than an ImportError in the middle -- and the same shape
``motion_handoff`` used before this package existed.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping

from .contract import (DecisionTables, TrackingResult, expected_files,
                       sha256_of)
from .provenance import (PROVENANCE_ADDED, PROVENANCE_INFERRED,
                         PROVENANCE_UNRESOLVED)

__all__ = ["TRACKER_TARGET", "status", "run", "TrackingResult",
           "DecisionTables", "expected_files", "sha256_of",
           "PROVENANCE_INFERRED", "PROVENANCE_UNRESOLVED", "PROVENANCE_ADDED"]

#: The dotted ``module:attribute`` that will one day resolve to the tracker.
#: One name, read at call time, so a test or a later port changes it in one
#: place. Moved here from ``pipelines.motion_handoff.TARGET``, which now
#: re-exports it.
TRACKER_TARGET = "motion.pipeline:run"


def _split(target: str) -> tuple[str, str]:
    module, _, attribute = str(target).partition(":")
    return module, attribute


def _importable(module: str) -> bool:
    """Whether ``module`` can be imported, without importing it.

    A module already imported counts, which ``find_spec`` alone would not
    say for one placed in ``sys.modules`` by hand (a test's stand-in tracker
    has no spec and ``find_spec`` raises on it).
    """
    if module in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def status() -> tuple[str, str]:
    """``("ready", "")`` once the tracker resolves, else ``("pending", why)``.

    The reason names the dotted target and the module that is missing, in
    that order, so a reader of a run record can tell what to install. The
    registry reads this to report the ``track`` action pending.
    """
    module, attribute = _split(TRACKER_TARGET)
    if not module or not attribute:
        return "pending", f"{TRACKER_TARGET!r} is not a 'module:attribute' target."
    if not _importable(module):
        return "pending", (f"{TRACKER_TARGET} does not resolve: {module} is "
                           f"not installed.")
    return "ready", ""


def _resolve(target: str):
    """The callable behind ``module:attribute``, or ``None`` while pending."""
    module, attribute = _split(target)
    if not module or not attribute or not _importable(module):
        return None
    try:
        loaded = importlib.import_module(module)
    except ImportError:
        return None
    return getattr(loaded, attribute, None)


def run(inputs, folder, *, claim: str = "",
        tracker_options: Mapping[str, Any] | None = None) -> TrackingResult:
    """Track every recording ``inputs`` names, writing under ``folder``.

    ``inputs`` is the ``motion_inputs.json`` the handoff wrote: the registered
    stacks and their masks, pinned by SHA-256, and the output names the
    tracker is expected to produce. ``tracker_options`` reaches the tracker
    unchanged; its keys are the tracker's own and are checked against its
    live signature once it resolves, not here. ``claim`` is recorded by the
    action layer and accepted here so the catalogue and the signature agree.

    Raises :class:`pymicroglia.ActionPending` naming :data:`TRACKER_TARGET`
    while nothing resolves behind it. The tracker returns a mapping in the
    words of :class:`TrackingResult.from_mapping`, and that is what comes
    back -- files, never arrays.
    """
    # Imported here rather than at the top: the registry imports this package
    # while ``pymicroglia.run`` is still importing the registry.
    from ..run import ActionPending

    target = _resolve(TRACKER_TARGET)
    if target is None:
        _, why = status()
        raise ActionPending(
            f"track is pending: {why or TRACKER_TARGET + ' does not resolve.'} "
            "Install the tracker and re-run; the handoff file is what it reads.")
    returned = target(inputs, folder, **dict(tracker_options or {}))
    if isinstance(returned, TrackingResult):
        return returned
    return TrackingResult.from_mapping(returned, root=Path(folder))
