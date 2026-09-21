"""The measurement modules, one file each. Loading this package registers them.

A module registers itself with :func:`pymicroglia.measure.declare.measurement`
or :func:`~pymicroglia.measure.declare.derived` the moment it is imported, so
adding one is writing the file and naming it in :data:`MODULES`; nothing else
in the package changes. The twenty-three files below register twenty-four
modules (``surveillance`` registers ``surveillance`` and ``motion_evidence``),
ported from Motion's ``analysis/modules/`` on 2026-09-21 as stage 04 of the
Motion port.

Importing *this* package imports none of them. Six of the modules import
scikit-image at module level and three reach Circadian Workbench, and
``describe`` reads this package before anything is measured; it needs the
names, which are data here, not the arithmetic. :func:`load` imports every
file and is what a run calls first.
"""

from __future__ import annotations

import importlib

__all__ = ["MODULES", "MODULE_NAMES", "load"]

#: Every module file this package ships, in the order Motion imported them.
#: Data rather than imports, so that listing the names costs nothing.
MODULES: tuple[str, ...] = (
    "channels", "contacts", "coupling", "history", "intensity", "lifecycle",
    "morphology", "motility", "neighbours", "object_geometry", "objects",
    "presence", "provenance", "recurrence", "regimes", "rhythms",
    "sequence_distance", "sholl", "surveillance", "territory",
    "territory_shape", "trend", "walk",
)

#: The names the files above register, in Motion's ``enabled_modules`` order:
#: the fifteen pixel-reading modules first, then the nine derived ones. What
#: ``describe measure`` offers as the choices for ``enabled_modules``.
MODULE_NAMES: tuple[str, ...] = (
    "morphology", "intensity", "channels", "objects", "object_geometry",
    "motility", "surveillance", "motion_evidence", "presence", "lifecycle",
    "territory", "sholl", "contacts", "neighbours", "provenance",
    "regimes", "recurrence", "sequence_distance", "coupling", "rhythms",
    "trend", "territory_shape", "walk", "history",
)


def load() -> tuple[str, ...]:
    """Import every module file, registering each one. Idempotent.

    Python imports a module once per process, so calling this twice costs a
    dictionary lookup per name and registers nothing a second time.
    """
    for name in MODULES:
        importlib.import_module(f"{__name__}.{name}")
    return MODULES
