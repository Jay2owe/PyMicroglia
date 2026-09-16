"""Unmixing, and the one filter left that changes scientific pixels here.

Moved to **Auto-Organotypic** on 2026-08-24. This is the address it left behind, and
the line below makes this name *be* that module, so patching, ``is``
comparisons and identity checks all keep working.

The line the move drew is *longitudinal, and not about single cells*.
Spectral bleed between the channels of one recording is a
fact about how it was made. The superseded cosmic-ray method
this module re-exports for old run records came with it.
"""

import sys

from auto_organotypic import filtering as _moved

sys.modules[__name__] = _moved
