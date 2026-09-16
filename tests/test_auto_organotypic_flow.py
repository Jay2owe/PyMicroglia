"""Whether an Auto-Organotypic change reaches this package on its own.

Fourteen of PyMicroglia's modules are Auto-Organotypic's under an old name, and the
arrow runs downhill: that package never imports this one. The question this
file answers is the one that decides how much maintenance that costs —
**when Auto-Organotypic gains a function, or moves one, does PyMicroglia get it without
being edited?**

The mechanism is one line per moved module::

    sys.modules[__name__] = _moved

Not a forwarding shim. ``import *`` plus ``__getattr__`` reads every attribute
correctly and still has the gap that matters: the shim is a *second* module
object, so a name Auto-Organotypic adds tomorrow is a name the shim never
re-exports, and ``monkeypatch.setattr`` on it rebinds something the
implementation never looks at. Replacing the entry in ``sys.modules`` leaves one
module object under two names, and one object cannot drift from itself.

So there are three things to check and they are different in kind. That the
moved names really *are* one object, which is what makes a new function arrive
for free. That the pipeline Auto-Organotypic drives is fully resolvable here,
because its stages are reached by dotted name and a rename on that side turns
into a missing stage on this one — visible now, or nine days into an experiment.
And that the version actually installed is one the pin admits, because the pin
is the single place where "automatically" stops being true.
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
from pathlib import Path

import pytest

import pymicroglia

#: Every module in this package whose body hands its name to Auto-Organotypic.
#: Discovered rather than listed, so an eleventh one is covered the day it is
#: written and not the day somebody remembers this file.
HANDOVER = "sys.modules[__name__] = _moved"


def _delegating_modules() -> dict[str, str]:
    """``pymicroglia.<name>`` -> ``auto_organotypic.<name>``, read off the source."""
    root = Path(pymicroglia.__file__).parent
    found: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if HANDOVER not in text:
            continue
        moved = [line for line in text.splitlines()
                 if line.startswith("from auto_organotypic import ")
                 and line.endswith(" as _moved")]
        assert len(moved) == 1, f"{path} hands over without one clear source"
        name = moved[0].split()[3]
        relative = path.relative_to(root).with_suffix("")
        parts = [part for part in relative.parts if part != "__init__"]
        found[".".join(["pymicroglia", *parts])] = f"auto_organotypic.{name}"
    return found


def test_there_is_something_to_check():
    assert len(_delegating_modules()) >= 10


@pytest.mark.parametrize("here, there", sorted(_delegating_modules().items()))
def test_a_moved_module_is_the_module_it_moved_to(here, there):
    """One object under two names, not two objects kept in step by hand."""
    assert importlib.import_module(here) is importlib.import_module(there)


def test_a_function_auto_organotypic_grows_arrives_without_an_edit_here():
    """The property the whole arrangement exists for, done as an experiment.

    A name that did not exist when this package was written is readable through
    this package's name for it the moment Auto-Organotypic has it — no
    re-export, no version bump, no edit in this repository.
    """
    from auto_organotypic import series as upstream

    assert not hasattr(upstream, "a_function_invented_after_the_move")
    upstream.a_function_invented_after_the_move = lambda: "from upstream"
    try:
        from pymicroglia import series as here

        assert here.a_function_invented_after_the_move() == "from upstream"
    finally:
        del upstream.a_function_invented_after_the_move

    from pymicroglia import series as after

    assert not hasattr(after, "a_function_invented_after_the_move"), (
        "a removal must flow through as completely as an addition")


def test_a_replaced_function_is_replaced_for_both_names(monkeypatch):
    """Why the shim was not good enough, stated as the failure it caused.

    Patching a forwarding shim rebinds a name the implementation never reads, so
    a test that replaced a function to prove some failure path quietly stopped
    exercising anything and passed for the wrong reason. One of them nearly did.
    """
    from auto_organotypic import metadata as upstream
    from pymicroglia import metadata as here

    monkeypatch.setattr(here, "GAP_HOURS", 999.0)
    assert upstream.GAP_HOURS == 999.0


# ── the pipeline Auto-Organotypic drives ────────────────────────────────────
def test_every_stage_of_the_automated_pipeline_resolves():
    """Reached by dotted name, so a rename upstream is a missing stage here.

    A half-installed pipeline is meant to be a visible state at the top of a run
    rather than an ImportError in the middle of one. This is that check, run
    before the nine-day experiment rather than after it.
    """
    from auto_organotypic import pipeline

    rows = pipeline.describe()
    assert len(rows) >= 15
    pending = [row for row in rows
               if row.get("status") not in ("ready", "chosen at run time")]
    assert pending == [], f"stages that cannot run: {pending}"


def test_the_pipeline_still_never_reaches_back_up_into_this_package():
    """The arrow runs downhill, and a stage *shipped* pointing here reverses it.

    The single-cell analysis is reached by running PyMicroglia, not by installing
    it as an extra of the package underneath it. A dotted target naming this
    package, in that package's own table, would make the two mutually dependent
    and neither could then be installed on its own.

    Registration is the other direction and is fine: ``auto_microglia`` puts
    ``cell_masks`` into that chain itself, from here, at import. The target
    names this package because this package wrote it there — nothing in that
    wheel mentions it, and nothing in it goes looking. The test below proves
    that by asking a process that never imported PyMicroglia.
    """
    from auto_organotypic import pipeline

    from pymicroglia.pipelines import auto_microglia  # noqa: F401 - registers

    ours = pipeline.registered_stages()
    assert ours == {"cell_masks": "PyMicroglia"}, (
        "the stages this package adds, and who is recorded as owning them")
    for stage in pipeline.STAGES:
        if stage.name in ours:
            continue
        assert "pymicroglia" not in str(stage.target).lower(), stage.name


def test_auto_organotypic_alone_has_never_heard_of_this_package():
    """The claim the one above cannot make from inside a process that imported us.

    A fresh interpreter, importing only that package: its stage table must name
    nothing of ours and nothing must be registered. This is what "installable on
    its own" means, and it is the one thing a registration API could quietly
    break — an entry-point scan, a try/import, a name in the shipped table would
    each pass every other test in this file.
    """
    import subprocess
    import sys

    said = subprocess.run(
        [sys.executable, "-c",
         "from auto_organotypic import pipeline as p;"
         "print(bool(p.registered_stages()),"
         "      any('pymicroglia' in str(s.target).lower() for s in p.STAGES),"
         "      'pymicroglia' in __import__('sys').modules)"],
        capture_output=True, text=True, check=True).stdout.split()
    assert said == ["False", "False", "False"], (
        f"a bare Auto-Organotypic process knows about this package: {said}")


# ── where "automatically" stops ─────────────────────────────────────────────
def test_the_installed_auto_organotypic_is_one_the_pin_admits():
    """The one place an upstream change does *not* flow through on its own.

    ``Auto-Organotypic>=0.6,<0.7`` is deliberate — 0.6 dropped a verdict, renamed a
    field and moved the artefact store, and each of those reached this package
    as a test failure. The cap means a released 0.7 will not be picked up
    without a decision here, and this test is where that decision gets made:
    when it fails, read 0.7's changes and move the pin, rather than widening it
    to make the red go away.
    """
    from packaging.requirements import Requirement

    pins = [Requirement(line) for line in
            metadata.requires("PyMicroglia") or []
            if line.lower().startswith("auto-organotypic")]
    assert pins, "PyMicroglia must depend on Auto-Organotypic explicitly"

    installed = metadata.version("Auto-Organotypic")
    for pin in pins:
        if pin.marker is not None and not pin.marker.evaluate():
            continue                      # an extra this interpreter has not got
        assert pin.specifier.contains(installed), (
            f"{pin} does not admit the installed {installed}. Read what "
            f"changed upstream, then move the pin deliberately.")
