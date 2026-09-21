"""The measurement modules, one file each. Importing this package registers them.

Empty at stage 03 of the Motion port: the chassis is here, the twenty-four
modules follow in stage 04. A module registers itself with
:func:`pymicroglia.measure.declare.measurement` or :func:`~pymicroglia.measure.
declare.derived` at import time, so adding one is writing the file and
listing it below; nothing else in the package changes.
"""

from __future__ import annotations

#: Every module this package ships, in the order they are imported. Filled
#: in by stage 04; a test may register a stand-in without touching this list.
MODULES: tuple[str, ...] = ()

__all__ = ["MODULES"]
