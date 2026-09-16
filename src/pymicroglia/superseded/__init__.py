"""Methods this package no longer runs, kept because records of them exist.

Moved to **Auto-Organotypic** on 2026-08-24. This is the address it left behind, and
the line below makes this name *be* that module, so patching, ``is``
comparisons and identity checks all keep working.

The line the move drew is *longitudinal, and not about single cells*.
It runs nothing. It travelled with the cosmic-ray rule it
supersedes so that a stored record naming
``filtering.remove_cosmic_rays`` can still be replayed.
"""

import sys

from auto_organotypic import superseded as _moved

sys.modules[__name__] = _moved
