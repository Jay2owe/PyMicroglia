"""The kit is optional: without it the package still reads and still runs."""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest


BLOCKED = "analysis_kit"

# Captured at import time, before any test blocks the module. Grabbing it inside
# the without_kit fixture would hit that fixture's own import block and skip the
# one comparison that makes the local stand-in worth having.
try:
    from analysis_kit import ParamDoc as KIT_PARAMDOC
except ImportError:  # pragma: no cover - the kit is installed here
    KIT_PARAMDOC = None


@pytest.fixture
def without_kit(monkeypatch):
    """Make ``analysis_kit`` unimportable and reload the package under that.

    Simulating absence rather than uninstalling keeps the test runnable on a
    machine where the kit is installed, which is every machine here.
    """
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == BLOCKED or name.startswith(BLOCKED + "."):
            raise ImportError(f"{name} is blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    for name in [n for n in list(sys.modules) if n == BLOCKED or n.startswith(BLOCKED + ".")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name in [n for n in list(sys.modules) if n == "pymicroglia" or n.startswith("pymicroglia.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield importlib.import_module("pymicroglia")


def test_kit_returns_the_module_when_present():
    from pymicroglia import _optional

    _optional.reset_cache()
    module = _optional.kit()
    if module is None:
        pytest.skip("analysis_kit is not installed in this interpreter")
    assert _optional.kit_version()


def test_kit_never_raises_when_absent(without_kit):
    assert without_kit._optional.kit() is None
    assert without_kit._optional.kit_version() == ""


def test_import_and_harvest_still_work_without_the_kit(without_kit, fixtures):
    block = without_kit.harvest(fixtures / "fixture_engine.py")
    assert len(block) == 7
    assert block.by_constant("DEFAULT_THRESHOLD_SIGMA").default == 12.0


def test_local_paramdoc_matches_the_kit_field_for_field(without_kit):
    """The stand-in must be indistinguishable to a consumer.

    Same field names, same order, same defaults, same as_dict keys — otherwise
    a caller behaves differently depending on whether the audit layer happens to
    be installed, which is the one thing the soft import exists to prevent.
    """
    import dataclasses

    if KIT_PARAMDOC is None:  # pragma: no cover
        pytest.skip("analysis_kit is not installed in this interpreter")
    local, real = without_kit.ParamDoc, KIT_PARAMDOC

    def signature(cls):
        return [(f.name, f.type if isinstance(f.type, str) else str(f.type))
                for f in dataclasses.fields(cls)]

    assert [n for n, _ in signature(local)] == [n for n, _ in signature(real)]

    made_local = local(name="x", type="int", description="d")
    made_real = real(name="x", type="int", description="d")
    assert made_local.as_dict() == made_real.as_dict()


def test_cache_means_one_failed_import_not_many(monkeypatch):
    from pymicroglia import _optional

    _optional.reset_cache()
    calls = {"n": 0}
    real_import = builtins.__import__

    def counting(name, *args, **kwargs):
        if name == BLOCKED:
            calls["n"] += 1
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", counting)
    monkeypatch.delitem(sys.modules, BLOCKED, raising=False)

    assert _optional.kit() is None
    assert _optional.kit() is None
    assert _optional.kit() is None
    assert calls["n"] == 1
    _optional.reset_cache()
