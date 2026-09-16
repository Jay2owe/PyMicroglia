"""Reading and writing image files, on this machine's terms.

Moved to **Auto-Organotypic** on 2026-08-24. This is the address it left behind, and
it does not merely forward: the line below makes this name *be* that module.

A forwarding shim -- ``import *`` plus a ``__getattr__`` -- reads every
attribute correctly and still has one gap that matters. ``monkeypatch.setattr``
on the shim rebinds a name the implementation never looks at, so a test that
replaces ``replace_with_retry`` to prove an interrupted write leaves no partial
file behind would quietly stop interrupting anything and pass for the wrong
reason. Replacing the entry in ``sys.modules`` leaves one module object with two
names, so patching, ``is`` comparisons and identity checks all keep working.

The line the move drew is *longitudinal, and not about single cells*. A long
path, a held handle and a half-written TIFF are facts about the machine rather
than about a cell, and both packages meet all three.
"""

import sys

from auto_organotypic import io as _moved

sys.modules[__name__] = _moved
