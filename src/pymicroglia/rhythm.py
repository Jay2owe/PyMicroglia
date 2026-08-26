"""Periods and cosinor fits, borrowed rather than reimplemented.

Moved to **PySCNSlice** on 2026-08-24. This is the address it left behind, and
the line below makes this name *be* that module.

It went down with the instrumental control it refuses to work without, and the
dependency on ``circadian_workbench`` went with it. Routing the rhythm
statistics through this package made sense while it sat underneath; with it
above, asking a whole-tissue trace what its period is would have meant
installing the single-cell package to find out. It is ``pyscnslice[rhythm]``
now, so the weight still lands only on somebody who wants a verdict, and it is
still four functions behind one adapter: a third Lomb-Scargle in this lab
remains the exact drift analysis-kit was built to end.

``test_rhythm`` still measures traces itself when it is not handed any, and
that is per-cell work which stayed here. It reaches back for it by dotted name
rather than by import, so the arrow keeps pointing one way.
"""

import sys

from pyscnslice import rhythm as _moved

sys.modules[__name__] = _moved
