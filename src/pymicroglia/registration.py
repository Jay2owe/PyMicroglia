"""Translation-only registration, ported from the two engines that do it today.

Moved to **PySCNSlice** on 2026-08-24. This is the address it left behind, and
it does not merely forward: the line below makes this name *be* that module.

A forwarding shim -- ``import *`` plus a ``__getattr__`` -- reads every
attribute correctly and still has one gap that matters. ``monkeypatch.setattr``
on the shim rebinds a name the implementation never looks at, so a test that
replaces a function to prove some failure path works would quietly stop
exercising anything and pass for the wrong reason. One of them nearly did.
Replacing the entry in ``sys.modules`` leaves one module object under two
names, so patching, ``is`` comparisons and identity checks all keep working.

The line the move drew is *longitudinal, and not about single cells*.
Aligning every frame of a stack to one reference is a statement about
a time axis and about nothing else.
"""

import sys

from pyscnslice import registration as _moved

sys.modules[__name__] = _moved
