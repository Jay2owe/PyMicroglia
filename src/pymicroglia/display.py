"""Filters that change pixels for looking at, and must never be measured from.

Moved to **PySCNSlice** on 2026-08-24. This is the address it left behind, and
the line below makes this name *be* that module, so patching, ``is``
comparisons and identity checks all keep working.

The line the move drew is *longitudinal, and not about single cells*.
Pixels changed for looking at are a property of the recording,
not of anything living in it. The guard that keeps this branch
out of a number moved with it.
"""

import sys

from pyscnslice import display as _moved

sys.modules[__name__] = _moved
