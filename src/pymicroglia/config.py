"""Where things live, and how much room they get.

Moved to **Auto-Organotypic** on 2026-08-24. This is the address it left behind, and
it does not merely forward: the line below makes this name *be* that module.

A forwarding shim -- ``import *`` plus a ``__getattr__`` -- reads every
attribute correctly and still has one gap that matters. ``monkeypatch.setattr``
on the shim rebinds a name the implementation never looks at, so a test that
replaces a function to prove some failure path works would quietly stop
exercising anything and pass for the wrong reason. One of them nearly did.
Replacing the entry in ``sys.modules`` leaves one module object under two
names, so patching, ``is`` comparisons and identity checks all keep working.

The line the move drew is *longitudinal, and not about single cells*.
The store this points at holds the derived arrays of recordings over
time. Which *project* it belongs to is claimed in this package's
``__init__``, which is where a package should say so -- and it is a
claim rather than an assignment, because a single process holds both
packages and an SCN run must keep its own store when it reaches in
here for the single-cell trace stage.
"""

import sys

from auto_organotypic import config as _moved

sys.modules[__name__] = _moved
