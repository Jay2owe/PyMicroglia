r"""The script inside a record, and whether it really does the run again.

A record that describes a run is a note. A record that reproduces it is a
result, and the only way to know which one this package writes is to take a
stored script, run it in a fresh interpreter, and compare what comes out.

Two failure modes are guarded here and neither is hypothetical:

**Late binding.** A script that named only the arguments the caller typed would
silently pick up whatever the defaults happen to be on the day it is replayed.
The script therefore carries every resolved value explicitly.

**Windows paths.** ``"X:\Unicode\..."`` in an ordinary double-quoted string is a
broken unicode escape, and ``\temp`` is a tab. Every path in this project
contains spaces and most contain one of those letters.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from pymicroglia import catalogue, recording
from pymicroglia.registry import REGISTRY
from tests_support import two_channel_stack

#: A path with every trap in it: spaces, a folder Python reads as an escape, a
#: drive letter, and one apostrophe.
AWKWARD = r"X:\Unicode Folder\Jamie's\temp\MCG 04 - 1 - 595.tif"


def _kit_or_skip():
    from pymicroglia._optional import kit

    installed = kit()
    if installed is None:  # pragma: no cover
        pytest.skip("analysis_kit is not installed in this interpreter")
    return installed


def _call_node(script: str) -> ast.Call:
    """The one call the script makes, parsed."""
    tree = ast.parse(script)
    calls = [node.value for node in tree.body
             if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)]
    assert len(calls) == 1, f"a reproduction script makes exactly one call, not {len(calls)}"
    return calls[0]


def _arguments(script: str) -> dict[str, object]:
    return {keyword.arg: ast.literal_eval(keyword.value)
            for keyword in _call_node(script).keywords}


# ---------------------------------------------------------------- path literals
@pytest.mark.parametrize("text", [
    AWKWARD,
    r"D:\new\tabs\ugly.tif",
    r"C:\a b\c",
    "/mnt/plain/posix.tif",
    "no path at all",
    r"trailing\\",
    'has a "quote" in it',
])
def test_a_path_literal_survives_the_round_trip(text):
    """Whatever it emits, reading it back gives the same string.

    Raw literals are the readable answer and cover almost every case; the two
    they cannot express — a trailing backslash and an embedded double quote —
    fall back to ``repr``, so the property holds for all of them.
    """
    assert ast.literal_eval(recording._text_literal(text)) == text


def test_a_windows_path_is_emitted_raw_rather_than_escaped():
    """Readability is the point: a person is meant to be able to read this."""
    emitted = recording._text_literal(AWKWARD)
    assert emitted.startswith('r"') and AWKWARD in emitted


@pytest.mark.parametrize("value", [
    {1, 2}, set(), frozenset({"a"}), (1,), (1.0, 2.0), [1, [2]],
    {"a": {"b": (1, 2)}}, None, True, 3.5, "plain",
])
def test_a_container_replays_as_the_type_it_was(value):
    """A tuple of display percentiles read back as a list, or a set read back as
    a list, is a different call from the one the record claims to reproduce."""
    replayed = eval(recording._literal(value))  # noqa: S307 - the point of the test
    assert replayed == value and type(replayed) is type(value)


def test_a_nested_path_escapes_as_carefully_as_a_top_level_one():
    """A region of interest arrives as ``{"path": ...}``, and a nested path
    breaks a script exactly as thoroughly as a top-level one."""
    script = recording.equivalent_script(
        "roi_overlay", {"source": AWKWARD, "roi": {"path": AWKWARD}})
    assert _arguments(script)["roi"] == {"path": AWKWARD}


# ------------------------------------------------------------ what it calls
def test_the_script_calls_the_public_api_the_action_binds_to():
    """Composed from public API calls, never a private helper.

    If building one ever needed a private helper the public API would be wrong,
    and that would be the thing to fix: a second implementation living in the
    audit layer is how a record starts quietly disagreeing with the run it
    claims to reproduce.
    """
    for name in REGISTRY.names():
        method = catalogue.action(name)["method"]
        script = recording.equivalent_script(name, {"source": AWKWARD})
        module_path, _, function = method.rpartition(".")
        leaf = module_path.rpartition(".")[2]
        assert f"{leaf}.{function}(" in script, name
        assert f"from pymicroglia" in script, name
        ast.parse(script)  # it is real code, not a description of code


def test_a_pipeline_that_is_not_a_registered_action_still_gets_a_script():
    """Two of the four pipelines compose existing actions and never reached the
    catalogue. They record runs all the same, and a record with no script is
    the one kind this stage exists to stop."""
    script = recording.equivalent_script("bioluminescence", {"source": AWKWARD})
    assert "from pymicroglia.pipelines import bioluminescence" in script
    assert "bioluminescence.run(" in script


def test_an_argument_with_no_source_form_is_refused_rather_than_emitted():
    """A script that does not parse is worse than a record saying it has none.

    The kit catches this and writes the reason into the record's script field,
    so the run is still recorded and the gap says what it is.
    """
    class Opaque:
        pass

    with pytest.raises(TypeError, match="no source form"):
        recording.equivalent_script("remove_cosmic_rays",
                                    {"source": AWKWARD, "engine": Opaque()})


def test_a_run_with_such_an_argument_is_still_recorded(tmp_path):
    _kit_or_skip()

    class Opaque:
        pass

    from pymicroglia.recording import capture

    with capture("remove_cosmic_rays", {"engine": Opaque()},
                 output_roots=[tmp_path]) as run:
        run.result = "the science still happened"

    assert run.result == "the science still happened"
    assert run.recorded is True
    assert "could not be built" in run.record["script"]


def test_an_action_nobody_has_ever_heard_of_falls_back_to_the_runner():
    script = recording.equivalent_script("not_an_action", {"a": 1})
    assert "run_action('not_an_action', a=1)" in script


def test_the_source_is_named_first_and_the_rest_are_settled():
    """Source leads because it is what a reader looks for; the rest are
    alphabetical so two runs of one action diff cleanly against each other."""
    script = recording.equivalent_script(
        "remove_cosmic_rays", {"zeta": 1, "alpha": 2, "source": AWKWARD})
    names = [keyword.arg for keyword in _call_node(script).keywords]
    assert names[0] == "source"
    assert names[1:] == sorted(names[1:])


# ------------------------------------------------------- resolved, not passed
def test_the_script_names_every_default_that_was_in_force():
    """The property gate 3 rests on: nothing in the script is late-bound.

    Every argument the function has a default for is written out explicitly, so
    changing that default afterwards cannot reach a script already stored.
    """
    resolved = recording._resolved("remove_cosmic_rays", {"source": AWKWARD})
    written = set(_arguments(recording.equivalent_script(
        "remove_cosmic_rays", resolved)))

    from pymicroglia import cosmic

    defaulted = {
        name for name, parameter
        in inspect.signature(cosmic.remove_cosmic_rays).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    assert defaulted <= written, f"late-bound after replay: {sorted(defaulted - written)}"


def test_the_defaults_come_from_the_function_not_from_the_catalogue():
    """The two disagree in a handful of places, and the record must say what ran.

    ``run_controls`` searches 16-32 h; the engine it was copied from searched
    15-40, and that is the number the catalogue still documents. A script built
    from the catalogue would replay a different analysis.
    """
    resolved = recording._resolved("run_controls", {"source": AWKWARD})
    assert resolved["ls_pmin"] == 16.0
    assert resolved["ls_pmax"] == 32.0
    assert catalogue.action("run_controls")["defaults"]["ls_pmin"] == 15.0


def test_a_stored_script_does_not_move_when_a_default_moves(monkeypatch):
    """Gate 3, stated as the thing it protects.

    The stored text still says 12.0 after the default becomes 99.0 — and a
    script built afterwards says 99.0, which is how you can tell the first one
    was pinned rather than merely unchanged.
    """
    stored = recording.equivalent_script(
        "remove_cosmic_rays",
        recording._resolved("remove_cosmic_rays", {"source": AWKWARD}))
    assert _arguments(stored)["seed_z"] == 12.0

    monkeypatch.setattr(
        recording, "_signature_defaults",
        lambda action: {"seed_z": 99.0, "signal_channel": 1})
    rebuilt = recording.equivalent_script(
        "remove_cosmic_rays",
        recording._resolved("remove_cosmic_rays", {"source": AWKWARD}))

    assert _arguments(stored)["seed_z"] == 12.0
    assert _arguments(rebuilt)["seed_z"] == 99.0


# ---------------------------------------------------------------- the round trip
def _file_digests(folder: Path) -> dict[str, str]:
    """Content hashes of everything the run wrote, sidecars excluded.

    A sidecar carries the moment it was written, so two identical runs differ
    there and nowhere else. The key it claims is compared separately, which is
    the part that matters.
    """
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.iterdir())
            if path.is_file() and not path.name.endswith(".artefact.json")}


def test_the_script_reproduces_the_run_in_a_fresh_interpreter(tmp_path, monkeypatch):
    """Gate 2, end to end: a stored script, a new process, an empty cache.

    The cache is deliberately thrown away between the two runs. A script that
    only ever reproduced a cache hit would prove nothing about the analysis.
    """
    _kit_or_skip()
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache_first"))

    from pymicroglia import run_action

    source = two_channel_stack(tmp_path, frames=6, height=48, width=48)
    results = tmp_path / "results"
    run_action("remove_cosmic_rays", source=str(source),
               output_dir=str(results), signal_channel=1,
               output_roots=[results])

    records = sorted((results / ".analysis-kit" / "records").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))

    before = _file_digests(results)
    keys = {item["stage"]: item["digest"] for item in record["artefacts"]}
    assert keys, "the run wrote artefacts and the record should list them"

    shutil.move(str(results), str(tmp_path / "first_run"))
    script = tmp_path / "reproduce.py"
    script.write_text(record["script"], encoding="utf-8")

    finished = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True,
        env={**os.environ,
             "PYMICROGLIA_STORE": str(tmp_path / "cache_second")})
    assert finished.returncode == 0, finished.stderr[-3000:]

    assert _file_digests(results) == before, (
        "the replayed run wrote different bytes than the run it reproduces")

    # Read the claims through Auto-Organotypic rather than globbing for them.
    # It writes one ``.auto-organotypic/artefacts.json`` per folder since
    # 2026-09-14, where it wrote a ``<name>.artefact.json`` twin per file
    # before, and ``sidecars_in`` reads either -- so this asserts the keys and
    # not the filing system they happen to be kept in.
    from auto_organotypic.store import tier_a

    replayed = {claim["stage"]: claim["digest"]
                for claim in tier_a.sidecars_in(results)}
    assert replayed, "the replayed run claimed no artefacts at all"
    for stage, digest in keys.items():
        assert replayed.get(stage) == digest, (
            f"{stage} was stored under a different key on replay")
