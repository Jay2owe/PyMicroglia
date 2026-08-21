"""The line between looking at data and measuring it.

``AGENTS.md`` fixes the processing order and states the rule in prose:

    VSI conversion -> registration -> cosmic-ray removal -> unsmoothed
    measurement, with display-only smoothing on a separate branch.

    Display smoothing must never feed masks, traces, amplitudes or statistics.

That sentence is the most consequential one in the project, and prose does not
enforce itself. This module is the sentence written as a refusal: a measurement
entry point calls ``require_measurement`` on its inputs, and anything from the
display branch stops the run.

**Two independent signals are checked**, because either alone can be lost. The
artefact's ``display_only`` flag survives a file being renamed; the
``_DISPLAY_ONLY`` mark in the name survives a sidecar being deleted or an array
being passed around without one. A file that has lost both was never going to be
catchable, which is why the display branch writes both.

**There is no ``force``.** A keyword that switched the guard off would be used,
once, in a hurry, and the resulting number would be indistinguishable from a
real one. If a legitimate case needs display-filtered input, that is a
conversation to have, not a flag to pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["DisplayOnlyInput", "require_measurement", "is_display_only",
           "DISPLAY_ONLY_MARK", "display_only_name"]

#: The mark a display-branch output carries in its own filename.
DISPLAY_ONLY_MARK = "_DISPLAY_ONLY"

_RULE = (
    "See AGENTS.md: VSI conversion -> registration -> cosmic-ray removal -> "
    "unsmoothed measurement, with display-only smoothing on a separate branch. "
    "Display smoothing must never feed masks, traces, amplitudes or statistics."
)


class DisplayOnlyInput(ValueError):
    """Display-filtered data was offered to a measurement."""


def display_only_name(name: str) -> str:
    """The same name, marked. Idempotent, so wrapping twice is harmless."""
    text = str(name)
    return text if DISPLAY_ONLY_MARK in text.upper() else f"{text}{DISPLAY_ONLY_MARK}"


def is_display_only(item: Any) -> bool:
    """Whether this input came off the display branch.

    Accepts whatever a caller has: a ``Series``, a stored artefact, a sidecar
    dictionary, or a path. Anything that cannot be interrogated is treated as a
    measurement input — the guard exists to catch a known mark, not to refuse
    things it does not recognise.
    """
    if item is None:
        return False
    if getattr(item, "display_only", False):
        return True
    if isinstance(item, dict) and item.get("display_only"):
        return True
    if isinstance(item, (str, Path)):
        return DISPLAY_ONLY_MARK in str(item).upper()
    for attribute in ("path", "name"):
        value = getattr(item, attribute, None)
        if value is not None and DISPLAY_ONLY_MARK in str(value).upper():
            return True
    record = getattr(item, "record", None)
    if isinstance(record, dict) and record.get("display_only"):
        return True
    return False


def require_measurement(*inputs: Any) -> None:
    """Refuse anything on the display branch. Returns nothing; raises or passes.

    Called by every measurement entry point on its inputs, before any work.
    """
    for item in inputs:
        if is_display_only(item):
            raise DisplayOnlyInput(
                f"{_describe(item)} is display-only and cannot feed a "
                f"measurement. {_RULE}")


def _describe(item: Any) -> str:
    for attribute in ("path", "name"):
        value = getattr(item, attribute, None)
        if value is not None:
            return str(value)
    if isinstance(item, dict):
        return str(item.get("path", item))
    return str(item)
