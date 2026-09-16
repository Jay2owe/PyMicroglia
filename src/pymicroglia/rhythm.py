"""Periods and cosinor fits, borrowed rather than reimplemented.

Moved to **Auto-Organotypic** on 2026-08-24. This is the address it left behind, and
the line below makes this name *be* that module.

It went down with the instrumental control it refuses to work without, and the
dependency on ``circadian_workbench`` went with it. Routing the rhythm
statistics through this package made sense while it sat underneath; with it
above, asking a whole-tissue trace what its period is would have meant
installing the single-cell package to find out. It is ``auto_organotypic[rhythm]``
now, so the weight still lands only on somebody who wants a verdict. The moved
adapter calls Circadian Workbench's stable package-root facade instead of
knowing its engine modules.

``test_rhythm`` still measures traces itself when it is not handed any, and
that is per-cell work which stayed here. It reaches back for it by dotted name
rather than by import, so the arrow keeps pointing one way.
"""

import sys

from auto_organotypic import rhythm as _moved

sys.modules[__name__] = _moved
