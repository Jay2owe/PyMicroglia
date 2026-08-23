"""Recording a run, and the promise that failing to record never costs a result."""

from __future__ import annotations

import json
import os
import types

import pytest

from pymicroglia import recording
from pymicroglia.recording import NullRun, capture


def test_a_record_is_written_with_the_kit(tmp_path):
    from pymicroglia._optional import kit

    if kit() is None:  # pragma: no cover
        pytest.skip("analysis_kit is not installed in this interpreter")

    with capture("remove_cosmic_rays", {"seed_z": 8},
                 claim="unit test", output_roots=[tmp_path]) as run:
        run.result = {"ok": True}

    assert run.record["action"] == "remove_cosmic_rays"
    assert run.record["project"] == "pymicroglia"
    assert run.recorded is True


def test_the_record_carries_resolved_defaults_not_just_what_was_passed(tmp_path):
    """A run made with one argument and fifteen defaults must replay the same
    way after somebody changes a default. So the record stores the resolved
    values, not the caller's."""
    from pymicroglia._optional import kit

    if kit() is None:  # pragma: no cover
        pytest.skip("analysis_kit is not installed in this interpreter")

    with capture("remove_cosmic_rays", {"seed_z": 8},
                 output_roots=[tmp_path]) as run:
        run.result = None

    params = run.record["params"]
    assert params["seed_z"] == 8          # what the caller passed
    assert params["growth_px"] == 2                # a default, resolved in
    assert params["signal_channel"] == 1


def test_the_record_carries_the_entry_path_and_method_version(tmp_path, monkeypatch):
    from pymicroglia._optional import kit

    if kit() is None:  # pragma: no cover
        pytest.skip("analysis_kit is not installed in this interpreter")

    monkeypatch.setenv("PYMICROGLIA_ENTRY", "imagej")
    with capture("remove_cosmic_rays", {}, output_roots=[tmp_path]) as run:
        run.result = None

    assert run.record["entry_path"] == "imagej"
    assert run.record["method_version"] == "2026-08-21-selectable-replacement"


def test_the_equivalent_script_names_the_action_and_its_arguments(tmp_path):
    """Stage 12 turned the placeholder into a real call on the public API.

    What it must still do is what stage 02 asked of the placeholder: name the
    action's own function and every argument the run was made with. Whether it
    reproduces the run is asked in ``test_equivalent_script.py``, by running it.
    """
    from pymicroglia._optional import kit

    if kit() is None:  # pragma: no cover
        pytest.skip("analysis_kit is not installed in this interpreter")

    with capture("remove_cosmic_rays", {"seed_z": 8},
                 output_roots=[tmp_path]) as run:
        run.result = None

    script = run.record["script"]
    assert "from pymicroglia import cosmic" in script
    assert "cosmic.remove_cosmic_rays(" in script
    assert "seed_z=8" in script


def test_the_analysis_exception_is_never_swallowed(tmp_path):
    with pytest.raises(ZeroDivisionError):
        with capture("remove_cosmic_rays", {}, output_roots=[tmp_path]) as run:
            run.result = 1 / 0


def test_an_audit_failure_does_not_become_an_analysis_failure(tmp_path, monkeypatch):
    """The one bare except in the package, and the reason it is there."""
    from pymicroglia._optional import kit

    ak = kit()
    if ak is None:  # pragma: no cover
        pytest.skip("analysis_kit is not installed in this interpreter")

    def explode(*args, **kwargs):
        raise RuntimeError("the index is on fire")

    monkeypatch.setattr(ak.audit, "record_run", explode)

    with capture("remove_cosmic_rays", {}, output_roots=[tmp_path]) as run:
        run.result = "the science still happened"

    assert run.result == "the science still happened"
    assert run.recorded is False


def test_without_the_kit_capture_is_inert(monkeypatch, tmp_path):
    monkeypatch.setattr(recording, "kit", lambda: None)

    with capture("remove_cosmic_rays", {"seed_z": 8},
                 output_roots=[tmp_path]) as run:
        run.result = "still ran"

    assert isinstance(run, NullRun)
    assert run.result == "still ran"
    assert run.record == {}
    assert run.params["seed_z"] == 8      # defaults still resolved
    assert run.params["growth_px"] == 2


def test_a_run_appends_one_line_to_the_global_index(tmp_path, monkeypatch):
    from pymicroglia._optional import kit

    ak = kit()
    if ak is None:  # pragma: no cover
        pytest.skip("analysis_kit is not installed in this interpreter")

    index_root = tmp_path / "index"
    monkeypatch.setenv("ANALYSIS_KIT_INDEX", str(index_root))

    captured = {}
    real = ak.audit.record_run

    def spy(record, **kwargs):
        captured["kwargs"] = kwargs
        return real(record, **{**kwargs, "index_root": index_root})

    monkeypatch.setattr(ak.audit, "record_run", spy)

    with capture("remove_cosmic_rays", {}, claim="index check",
                 output_roots=[tmp_path / "out"]) as run:
        run.result = None

    assert captured["kwargs"]["claim"] == "index check"
    shards = list(index_root.glob("index-*.jsonl"))
    assert len(shards) == 1
    rows = [json.loads(line) for line in shards[0].read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["project"] == "pymicroglia"
    assert rows[0]["action"] == "remove_cosmic_rays"
    assert rows[0]["claim"] == "index check"


def test_entry_path_defaults_to_python(monkeypatch):
    monkeypatch.delenv("PYMICROGLIA_ENTRY", raising=False)
    assert recording.entry_path() == "python"
    assert recording.entry_path("cli") == "cli"
    monkeypatch.setenv("PYMICROGLIA_ENTRY", "imagej")
    assert recording.entry_path() == "imagej"
    assert recording.entry_path("cli") == "cli", "what was passed wins"


def test_the_command_line_does_not_leave_its_own_name_behind(tmp_path,
                                                             monkeypatch):
    """One CLI call in a process must not make every later run claim to be one.

    ``os.environ.setdefault`` would: it outlives the call that meant it, so an
    agent that ran one command and then used the API would have every one of
    those runs recorded as having come from the command line.
    """
    monkeypatch.delenv("PYMICROGLIA_ENTRY", raising=False)
    from pymicroglia.cli import main

    main(["run", "nope_not_an_action", "source=x.tif"])

    assert "PYMICROGLIA_ENTRY" not in os.environ
    assert recording.entry_path() == "python"


def test_null_run_has_the_shape_callers_rely_on():
    run = NullRun("x", {"a": 1})
    run.result = 3
    assert run.outputs == [] and run.record == {} and run.result == 3
