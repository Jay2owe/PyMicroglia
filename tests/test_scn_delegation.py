"""The SCN outline left this package, and the action it backed did not.

On 2026-08-23 about 6,200 lines of suprachiasmatic nucleus outlining moved to
Auto-Organotypic. ``automatic_scn_outline`` is a registered action and run records
name it, so the action stayed and now delegates.

The parity evidence went with the code: Auto-Organotypic's own
``test_automatic_scn_outline_parity.py`` compares output bytes against ten
accepted fields. What is left to check here is the seam — that the action still
resolves, still reports the defaults the function will actually apply, and fails
by name rather than silently when the outline is not installed.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from pymicroglia import catalogue, knowledge, recording, scn_outline

SRC = Path(__file__).resolve().parents[1] / "src" / "pymicroglia"

auto_organotypic = pytest.importorskip(
    "auto_organotypic",
    reason='the SCN outline is an extra: pip install "PyMicroglia[scn]"')


# ------------------------------------------------------- the algorithm is gone
def test_the_outline_engine_is_no_longer_in_this_package():
    """A copy left behind is a fork waiting to happen.

    Two packages holding the same frozen method is exactly the drift the move
    was made to stop: the parity test would guard one of them and the other
    would quietly diverge.
    """
    assert not (SRC / "_accepted_scn").exists()
    assert not (SRC / "scn_orientation.py").exists()


def test_this_module_delegates_rather_than_computing():
    """No array work here. The whole file should be import plumbing."""
    tree = ast.parse((SRC / "scn_outline.py").read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert "numpy" not in imported and "scipy" not in imported
    assert "auto_organotypic" in imported


# --------------------------------------------------------------- the seam holds
def test_the_action_resolves_through_the_delegate():
    from pymicroglia.registry import REGISTRY

    assert "automatic_scn_outline" in REGISTRY.names()
    assert REGISTRY.resolve("automatic_scn_outline") is scn_outline.automatic_scn_outline
    assert scn_outline.automatic_scn_outline.__wrapped__ is \
        auto_organotypic.outline.automatic


def test_the_delegate_keeps_the_real_signature():
    """This is what stops ``describe`` reporting stale catalogue defaults.

    ``recording._signature_defaults`` reads the resolved callable. A wrapper
    taking ``*args, **kwargs`` would return nothing, the catalogue would answer
    instead, and the first time upstream changed a default an agent would be
    told the old one.
    """
    upstream_signature = inspect.signature(auto_organotypic.outline.automatic)
    assert inspect.signature(scn_outline.automatic_scn_outline) == \
        upstream_signature

    live = recording._signature_defaults("automatic_scn_outline")
    for name in (
        "orient_scn",
        "crop_mode",
        "scn_time",
        "selected_source_only",
        "hash_source",
        "anomaly_broad_max_turn_deg",
    ):
        assert live[name] == upstream_signature.parameters[name].default


def test_the_delegate_stays_in_this_module_for_the_coverage_check():
    """``registry._covered_functions`` matches on ``__module__``.

    A plain re-export would carry Auto-Organotypic's module name across, the action
    would cover nothing, and ``discover`` would start reporting the outline's
    public helpers as functions nobody exposed.
    """
    assert scn_outline.automatic_scn_outline.__module__ == "pymicroglia.scn_outline"

    from pymicroglia.registry import _covered_functions

    covered = _covered_functions()
    assert "automatic_scn_outline" in covered.get("automatic_scn_outline", ())


# ------------------------------------------------------ a missing outline says so
def test_a_missing_auto_organotypic_fails_by_name(monkeypatch):
    """Hard optional, like the workbench and unlike the audit layer.

    Losing a run record must be silent. Losing the method means the action
    cannot run at all, so it says which package to install rather than
    returning something empty.
    """
    monkeypatch.setattr(scn_outline, "_upstream", lambda: None)
    stub = scn_outline._delegate("automatic_scn_outline")

    with pytest.raises(scn_outline.AutoOrganotypicMissing, match="PyMicroglia\\[scn\\]"):
        stub("anything.tif")
    assert isinstance(scn_outline.AutoOrganotypicMissing(""), ImportError)


def test_the_module_still_imports_without_the_outline(monkeypatch):
    """Pending means "this stage has not landed", not "you did not pip install".

    ``registry._load_modules`` turns an unimportable science module into a
    pending action, and the fix hint it prints tells you to regenerate the
    catalogue. That would be the wrong instruction here, so the module imports
    either way and only the call fails.
    """
    monkeypatch.setattr(scn_outline, "_upstream", lambda: None)
    assert scn_outline._upstream() is None
    assert scn_outline._delegate("draw_scn_labels") is not None


# ------------------------------------------------------- what the catalogue says
def test_the_action_is_public_and_describes_the_valid_field():
    """Moved verbatim from the old parity file. It always tested this half."""
    entry = catalogue.action("automatic_scn_outline")
    assert entry["method"] == "scn_outline.automatic_scn_outline"
    assert entry["method_version"] == auto_organotypic.outline.METHOD_VERSION

    described = knowledge.describe("automatic_scn_outline")
    params = {row["name"]: row for row in described["params"]}
    assert described["pending"] is False
    assert params["valid_mask"]["required"] is False
    assert params["scn_channel"]["default"] is None
    assert "one-based" in params["scn_channel"]["description"].lower()
    assert params["scn_z"]["default"] is None
    assert params["scn_time"]["default"] == "mean"
    assert "maximum" in params["scn_time"]["description"].lower()
    assert params["selected_source_only"]["default"] is False
    assert "online-only" in params["selected_source_only"]["description"].lower()
    assert params["hash_source"]["default"] is True
    assert "still hashed" in params["hash_source"]["description"].lower()
    assert params["anomaly_broad_shortest_path"]["default"] is True
    assert params["anomaly_broad_max_turn_deg"]["default"] == 6.5
    assert params["stable_local_line_redetect"]["default"] is True
    assert params["open_enclosed_carve_channels"]["default"] is True
    assert params["orient_scn"]["default"] is True


def test_the_record_can_still_say_which_method_ran():
    """The field that decides whether two runs are comparable at all.

    It comes from the catalogue first, so it survives even on a machine without
    Auto-Organotypic — where the run would have failed anyway.
    """
    assert recording._method_version("automatic_scn_outline") == \
        auto_organotypic.outline.METHOD_VERSION
    assert scn_outline.METHOD_VERSION == auto_organotypic.outline.METHOD_VERSION
