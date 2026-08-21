"""Soft imports.

The audit layer must never break the science. Every route from PyMicroglia to
``analysis_kit`` goes through :func:`kit`, which returns the module or ``None``
and never raises. Nothing in this package may ``import analysis_kit`` at module
scope; the self-containment test enforces that.

Contrast with ``pymicroglia.rhythm``, which imports ``circadian_workbench``
*hard*. Losing a run record is survivable and should be silent; silently
skipping a periodogram is not, so that one is allowed to fail loudly.

``analysis_kit.style`` is a third case. It arrives through this module because
the self-containment test allows no other route to the kit, but the figure code
treats a missing style as *fatal* rather than falling back — a figure drawn
without the shared palette is a figure in the wrong colours, which is the drift
the kit exists to stop. See ``visualisation.panels.FigureStyleMissing``.
"""

from __future__ import annotations

from typing import Any

_KIT: Any = None
_TRIED = False


def kit() -> Any:
    """The ``analysis_kit`` module, or ``None`` if it is not importable.

    Cached after the first attempt, so a missing kit costs one failed import per
    process rather than one per call.
    """
    global _KIT, _TRIED
    if not _TRIED:
        _TRIED = True
        try:
            import analysis_kit  # noqa: PLC0415 - deliberately local
            # The subpackages are not re-exported by the kit's __init__, and
            # every call site here reaches them as attributes (ak.capture,
            # ak.audit, ak.style). Importing them binds those names.
            import analysis_kit.audit  # noqa: F401, PLC0415
            import analysis_kit.capture  # noqa: F401, PLC0415
            import analysis_kit.style  # noqa: F401, PLC0415
        except Exception:
            _KIT = None
        else:
            _KIT = analysis_kit
    return _KIT


def kit_version() -> str:
    """The installed kit's version, or ``""`` when it is absent."""
    module = kit()
    return str(getattr(module, "__version__", "")) if module is not None else ""


def reset_cache() -> None:
    """Forget whether the kit was importable. For tests only."""
    global _KIT, _TRIED
    _KIT, _TRIED = None, False
