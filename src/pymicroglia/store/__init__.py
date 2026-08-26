"""The artefact store: keep the recipe, not the cooked dish.

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
It is what decides whether two artefacts are the same thing, which is
the one kind of code that must never exist twice: two copies would let
one package fail to recognise the other's work and rebuild it, and two
eviction budgets over one directory would have one deleting what the
other was about to read. Nothing about the key changed, so every
artefact already on disk still resolves.
"""

import sys

from pyscnslice import store as _moved

sys.modules[__name__] = _moved
