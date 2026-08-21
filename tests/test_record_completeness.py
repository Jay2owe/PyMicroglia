"""Every field a run record has to carry, on the way out and on the way down.

The record is what is left when the folder has been renamed, the person has
moved on and the parameters are a year out of date. So the test is not "does it
write something" but "does it write everything, including when the analysis
raised" — a run that broke is the one you most want to be able to look up.

The claim is the other half. It is the only field a person writes and the only
one that makes a list of a thousand runs skimmable, so the rule is checked from
both ends: an action whose point is the file it produces fills its own in, and
an action that concludes something refuses to run without one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pymicroglia import recording, registry
from pymicroglia.recording import ArtefactWatch, capture
from pymicroglia.registry import CLAIM_TEMPLATES, NEEDS_A_CLAIM, ClaimRequired
from tests_support import two_channel_stack

#: What the exit gate for this stage names, field by field. Kept as data so a
#: field that quietly stops being written fails here rather than in a year.
REQUIRED_FIELDS = (
    "action",
    "params",
    "method_version",
    "kit_version",
    "package_version",
    "duration_s",
    "outputs",
    "artefacts",
    "decisions",
    "entry_path",
    "claim",
    "script",
)


@pytest.fixture
def kit_installed():
    from pymicroglia._optional import kit

    installed = kit()
    if installed is None:  # pragma: no cover
        pytest.skip("analysis_kit is not installed in this interpreter")
    return installed


@pytest.fixture
def local_store(tmp_path, monkeypatch):
    """A throwaway artefact store, so one test never sees another's cache."""
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    return tmp_path


def _records_in(results: Path) -> list[dict]:
    folder = results / ".analysis-kit" / "records"
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(folder.glob("*.json"))]


# ------------------------------------------------------------- the whole record
def test_every_field_the_gate_names_is_present(kit_installed, tmp_path):
    with capture("remove_cosmic_rays", {"seed_z": 8},
                 claim="checking the record", output_roots=[tmp_path]) as run:
        run.result = {"ok": True}

    missing = [field for field in REQUIRED_FIELDS if field not in run.record]
    assert not missing, f"the record is missing {missing}"
    assert run.record["claim"] == "checking the record"
    assert run.record["entry_path"] == "python"
    assert run.record["package_version"]
    assert run.record["script"].startswith("#!/usr/bin/env python")


def test_every_registered_action_produces_a_complete_record(kit_installed,
                                                            tmp_path):
    """Gate 1 across all of them, not just the one that was convenient.

    Nothing is executed: ``capture`` builds the record around whatever runs
    inside it, so an empty body is enough to ask whether the record it assembles
    has every field. A missing ``method_version`` on one action out of
    twenty-six is exactly the kind of gap that only shows up years later.
    """
    from pymicroglia.registry import REGISTRY

    for name in REGISTRY.names():
        with capture(name, {"source": str(tmp_path / "stack.tif")},
                     claim="completeness sweep",
                     output_roots=[tmp_path / name]) as run:
            run.result = None
        missing = [field for field in REQUIRED_FIELDS if field not in run.record]
        assert not missing, f"{name} is missing {missing}"
        assert run.record["script"].strip(), f"{name} recorded no script"
        assert "could not be built" not in run.record["script"], name


#: Actions whose module declares which version of the method it implements. The
#: rest are display and figure code, which has none to declare, and an empty
#: string is the honest answer there rather than an invented one.
VERSIONED = {
    "register", "register_three_channel", "export_registered_stack",
    "remove_cosmic_rays", "unmix", "background", "segment", "export_roi",
    "extract_traces", "run_controls", "trace_panel", "red_only_video",
    "composite_video", "stack_to_mp4", "dluc_single_cell", "cry1_dluc_photon",
}


def test_the_record_says_which_version_of_the_method_ran():
    """The field that decides whether two runs are comparable at all.

    Only five actions inherited a version from the protocol script they were
    copied from; the rest declare it on the module that does the work, and a
    record built from the catalogue alone threw those away — including both
    pipelines, which are the runs most worth comparing across a version bump.
    """
    for name in sorted(VERSIONED):
        assert recording._method_version(name), f"{name} records no version"

    assert (recording._method_version("dluc_single_cell")
            == "2026-08-20-dluc-single-cell-one-outlier-rule")


def test_a_failed_run_is_still_recorded_with_its_error(kit_installed, tmp_path):
    """Gate 4. The analysis exception is never swallowed, and the record is
    still written on the way past — from a ``finally``, so nothing about the
    failure can skip it."""
    results = tmp_path / "results"
    results.mkdir()

    with pytest.raises(ZeroDivisionError):
        with capture("remove_cosmic_rays", {}, claim="a run that breaks",
                     output_roots=[results]) as run:
            (results / "half_written.tif").write_bytes(b"partial")
            run.result = 1 / 0

    assert run.recorded is True
    assert "ZeroDivisionError" in run.record["error"]
    written = [Path(path).name for path in run.record["outputs"]]
    assert "half_written.tif" in written, (
        "a partial run still reports the files it managed to write")

    stored = _records_in(results)
    assert len(stored) == 1 and "ZeroDivisionError" in stored[0]["error"]


def test_one_record_in_the_project_store_and_one_line_in_the_index(
        kit_installed, tmp_path, monkeypatch):
    """Gate 6. Two writes, and they carry the same run id."""
    index_root = tmp_path / "index"
    real = kit_installed.audit.record_run
    monkeypatch.setattr(
        kit_installed.audit, "record_run",
        lambda record, **kwargs: real(record, **{**kwargs,
                                                 "index_root": index_root}))

    results = tmp_path / "results"
    with capture("remove_cosmic_rays", {}, claim="index check",
                 output_roots=[results]) as run:
        run.result = None

    stored = _records_in(results)
    assert len(stored) == 1

    shards = list(index_root.glob("index-*.jsonl"))
    assert len(shards) == 1
    rows = [json.loads(line) for line
            in shards[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["run_id"] == stored[0]["run_id"] == run.record["run_id"]
    assert rows[0]["claim"] == "index check"
    assert Path(rows[0]["record"]).is_file(), "the index row points at the record"


# ------------------------------------------------------------------ artefacts
def test_the_record_lists_the_store_artefacts_with_their_digests(
        kit_installed, local_store):
    """The outputs say which files appeared; the artefacts say which of them the
    store will hand back on the next lookup, and under which key."""
    from pymicroglia import run_action

    source = two_channel_stack(local_store, frames=6, height=48, width=48)
    results = local_store / "results"
    run_action("remove_cosmic_rays", source=str(source), signal_channel=1,
               output_dir=str(results), output_roots=[results])

    record = _records_in(results)[0]
    artefacts = record["artefacts"]
    assert artefacts, "the run wrote three tier-A artefacts and listed none"

    stages = {item["stage"] for item in artefacts}
    assert {"cosmic_rays", "cosmic_rays_events", "cosmic_rays_summary"} <= stages
    for item in artefacts:
        assert item["digest"] and item["tier"] in {"A", "B"}
        assert Path(item["path"]).exists()
        assert item["method_version"] if item["tier"] == "A" else True


def test_a_second_run_over_the_same_folder_still_reports_what_it_wrote(
        kit_installed, local_store):
    """A cold re-run into the same folder replaces its artefacts in place.

    The index stamp is written to the second, so two writes inside one second
    are indistinguishable from it — and a set difference on paths would say the
    second run produced nothing at all. Inside the run's own output folders the
    artefact's own timestamp settles it.
    """
    from pymicroglia import store

    source = two_channel_stack(local_store, frames=5, height=32, width=32)
    results = local_store / "results"
    identity = store.fingerprint(source)

    def write_one(value):
        store.put("unit_test", identity, {"n": 1}, kind="scalars",
                  value={"v": value}, name="probe", output_dir=results)

    write_one(1)
    watch = ArtefactWatch([results]).arm()
    write_one(2)
    watch.disarm()

    assert [item["stage"] for item in watch.artefacts] == ["unit_test"]


def test_an_artefact_left_over_from_an_earlier_run_is_not_claimed(local_store):
    """The other half of the same rule: a run that only read the cache produced
    nothing, and must not be credited with what was already there."""
    from pymicroglia import store

    source = two_channel_stack(local_store, frames=5, height=32, width=32)
    results = local_store / "results"
    store.put("unit_test", store.fingerprint(source), {"n": 1}, kind="scalars",
              value={"v": 1}, name="probe", output_dir=results)

    watch = ArtefactWatch([results]).arm()
    watch.disarm()

    assert watch.artefacts == [] and watch.decisions == []


def test_a_decision_taken_during_a_run_is_listed_apart_from_the_artefacts(
        kit_installed, local_store):
    """A person answered a decision, so a re-run will not have to ask again.
    That is a different kind of fact from a derived table."""
    from pymicroglia import store

    source = two_channel_stack(local_store, frames=5, height=32, width=32)
    identity = store.fingerprint(source)
    decisions = local_store / "decisions"

    watch = ArtefactWatch().arm()
    store.decision("channel_assignment", identity, {"dluc": 2}, root=decisions)
    watch.disarm()

    assert [item["decision"] for item in watch.decisions] == ["channel_assignment"]
    assert watch.artefacts == [], "a decision is not an artefact"


def test_the_watch_never_raises_even_when_the_store_is_unreadable(
        local_store, monkeypatch):
    """A record must never cost an output. An artefact list that cannot be
    assembled is a thinner record; an exception here would be a lost run."""
    from pymicroglia import store

    def explode(*args, **kwargs):
        raise OSError("the manifest is on fire")

    monkeypatch.setattr(store.manifest, "load", explode)
    monkeypatch.setattr(store.tier_b, "entries", explode)

    watch = ArtefactWatch([local_store]).arm()
    watch.disarm()
    assert watch.artefacts == [] and watch.decisions == []

    with capture("remove_cosmic_rays", {}, output_roots=[local_store]) as run:
        run.result = "the science still happened"
    assert run.result == "the science still happened"
    assert run.record["artefacts"] == []


# ---------------------------------------------------------------------- claims
def test_every_registered_action_is_on_exactly_one_side_of_the_claim_rule():
    """An action added later must land in one list or the other on purpose.

    Without this it would silently fall through to demanding a claim, which
    looks like a decision and is not one.
    """
    from pymicroglia.registry import REGISTRY

    names = set(REGISTRY.names())
    overlap = sorted(set(CLAIM_TEMPLATES) & NEEDS_A_CLAIM)
    assert not overlap, f"{overlap} both have a template and demand one"

    uncovered = sorted(names - set(CLAIM_TEMPLATES) - NEEDS_A_CLAIM)
    assert not uncovered, (
        f"{uncovered} would demand a claim by accident. Give each a template, "
        "or list it in NEEDS_A_CLAIM because it concludes something.")


def test_a_mechanical_action_names_the_recording_in_its_own_claim():
    """The claim is where the file name lives, and the claim is what the index
    searches. ``runs --contains MCG_04`` is the whole reason."""
    claim, owed = registry.claim_for(
        "remove_cosmic_rays", "",
        {"source": r"D:\AI_Exports\MCG_04 - 1 - 595.ome.tif"})
    assert owed is False
    assert "MCG_04 - 1 - 595" in claim
    assert claim.startswith("cleaned the cosmic-ray spikes")


def test_what_the_caller_wrote_always_wins():
    claim, owed = registry.claim_for("remove_cosmic_rays", "  spike check  ",
                                     {"source": "x.tif"})
    assert (claim, owed) == ("spike check", False)


def test_an_action_that_concludes_something_refuses_an_empty_claim():
    """Refused at the front door, where a refusal costs nothing — rather than
    six hours later, or worse, recorded with boilerplate nobody can search."""
    with pytest.raises(ClaimRequired) as raised:
        registry.require_claim("dluc_single_cell", "")
    assert "one sentence" in str(raised.value)

    registry.require_claim("dluc_single_cell", "the KO line loses its rhythm")
    registry.require_claim("remove_cosmic_rays", "")


def test_a_claimless_interpretive_run_is_recorded_as_still_owing_one():
    """Calling the pipeline function directly is a conversation with yourself,
    so it is not refused — but the record says the claim is outstanding rather
    than inventing one."""
    claim, owed = registry.claim_for("run_controls", "", {"source": "x.tif"})
    assert (claim, owed) == ("", True)


def test_run_action_refuses_before_anything_is_written(tmp_path):
    from pymicroglia import run_action

    with pytest.raises(ClaimRequired):
        run_action("segment", source=str(tmp_path / "nothing.tif"))
    assert not list(tmp_path.iterdir()), "nothing should have been written"


def test_describe_says_which_actions_owe_a_claim():
    """An agent learns it from `describe`, not from a refusal after it has
    already assembled forty arguments."""
    from pymicroglia import describe

    assert describe("dluc_single_cell")["claim_required"] is True
    assert describe("remove_cosmic_rays")["claim_required"] is False
    assert describe("remove_cosmic_rays")["claim_template"]


# -------------------------------------------------------------------- notebook
def test_the_notebook_is_off_unless_it_is_asked_for(kit_installed, tmp_path):
    """A notebook is one JSON document rewritten in full on every append, and
    most runs are a step towards one worth narrating rather than the one."""
    with capture("remove_cosmic_rays", {}, output_roots=[tmp_path]) as run:
        run.result = None

    assert run.notebook is None
    assert not recording.notebook_for(tmp_path).exists()


def test_a_run_can_be_narrated_into_the_notebook(kit_installed, tmp_path):
    with capture("remove_cosmic_rays", {}, claim="spike check",
                 output_roots=[tmp_path], notebook=True,
                 request="is the top-left corner a cosmic ray?") as run:
        run.result = None

    path = recording.notebook_for(tmp_path)
    assert run.notebook == str(path) and path.is_file()

    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = "\n".join(str(cell["source"]) for cell in notebook["cells"])
    assert run.record["run_id"] in sources
    assert "is the top-left corner a cosmic ray?" in sources
    assert "cosmic.remove_cosmic_rays(" in sources, (
        "the notebook carries the script as a runnable cell")
    assert "spike check" in sources


def test_a_notebook_failure_never_costs_the_run(kit_installed, tmp_path,
                                                monkeypatch):
    monkeypatch.setattr(kit_installed.capture, "append_run",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            OSError("the notebook is locked")))

    with capture("remove_cosmic_rays", {}, output_roots=[tmp_path],
                 notebook=True) as run:
        run.result = "the science still happened"

    assert run.result == "the science still happened"
    assert run.recorded is True and run.notebook is None


# ------------------------------------------------------------- the front door
@pytest.fixture
def one_recorded_run(kit_installed, tmp_path):
    """A run in the quarantined index, for the commands that read it back."""
    results = tmp_path / "results"
    with capture("remove_cosmic_rays", {"seed_z": 8},
                 claim="a spike check on MCG_04", output_roots=[results]) as run:
        run.result = None
    return run.record["run_id"], results


def _cli(capsys, *argv):
    from pymicroglia.cli import main

    code = main(list(argv))
    return code, capsys.readouterr().out


def test_runs_lists_the_run_and_finds_it_by_what_it_claimed(capsys,
                                                            one_recorded_run):
    """Gate 5, and the question the index exists for. The recording's name is
    in the claim, so a search for the recording finds the run."""
    run_id, _ = one_recorded_run

    code, out = _cli(capsys, "runs", "--limit", "50")
    assert code == 0
    assert run_id in {row["run_id"] for row in json.loads(out)["runs"]}

    code, out = _cli(capsys, "runs", "--contains", "MCG_04", "--limit", "50")
    assert code == 0 and json.loads(out)["count"] >= 1

    code, out = _cli(capsys, "runs", "--action", "no_such_action")
    assert code == 0 and json.loads(out)["runs"] == []


def test_runs_can_print_one_line_per_run(capsys, one_recorded_run):
    run_id, _ = one_recorded_run
    code, out = _cli(capsys, "runs", "--limit", "50", "--table")
    assert code == 0 and run_id in out and not out.lstrip().startswith("{")


def test_result_returns_the_whole_record(capsys, one_recorded_run):
    run_id, _ = one_recorded_run
    code, out = _cli(capsys, "result", run_id)
    assert code == 0

    record = json.loads(out)["result"]["record"]
    assert record["run_id"] == run_id
    assert record["claim"] == "a spike check on MCG_04"

    code, out = _cli(capsys, "result", "20260101-000000-nosuch")
    assert code == 1 and json.loads(out)["error"] == "not_found"


def test_result_script_prints_something_that_can_be_redirected_and_run(
        capsys, one_recorded_run):
    """Bare, not JSON. The point of the flag is a runnable file, and a
    JSON-quoted one would not be."""
    import ast

    run_id, _ = one_recorded_run
    code, out = _cli(capsys, "result", run_id, "--script")
    assert code == 0
    assert out.startswith("#!/usr/bin/env python")
    ast.parse(out)


def test_notebook_appends_a_recorded_run_without_being_told_where(
        capsys, one_recorded_run):
    """The results folder is derived from where the record sits, so a folder
    copied to another machine still finds its own notebook."""
    run_id, results = one_recorded_run
    code, out = _cli(capsys, "notebook", run_id, "--request", "why the dip?")
    assert code == 0

    path = Path(json.loads(out)["notebook"])
    assert path == recording.notebook_for(results) and path.is_file()
    assert "why the dip?" in path.read_text(encoding="utf-8")


def test_note_appends_an_interpretation_that_cites_the_run(capsys,
                                                           one_recorded_run):
    run_id, results = one_recorded_run
    code, out = _cli(capsys, "note", "the corner spike is a ray, not a cell",
                     "--run", run_id, "--store", str(results))
    assert code == 0

    text = Path(json.loads(out)["notebook"]).read_text(encoding="utf-8")
    assert run_id in text and "not a cell" in text


def test_the_index_commands_say_what_is_missing_rather_than_failing(
        capsys, monkeypatch):
    """Without the audit layer these three have nothing to read, and say so in
    the terms that let somebody fix it."""
    from pymicroglia import cli

    monkeypatch.setattr(cli, "kit", lambda: None)
    for argv in (["runs"], ["result", "x"], ["note", "y"], ["notebook", "x"]):
        code, out = _cli(capsys, *argv)
        assert code == 1 and json.loads(out)["error"] == "kit_missing"


# ---------------------------------------------------------- without the kit
def test_without_the_kit_the_action_still_runs_and_doctor_says_so(
        monkeypatch, local_store):
    """Gate 7. The scientific output is produced; only the record is missing,
    and ``doctor`` complains about it rather than staying quiet."""
    from pymicroglia import knowledge, run

    monkeypatch.setattr(recording, "kit", lambda: None)
    monkeypatch.setattr(knowledge, "kit", lambda: None)

    source = two_channel_stack(local_store, frames=5, height=32, width=32)
    results = local_store / "results"
    cleaned = run.run_action("remove_cosmic_rays", source=str(source),
                             signal_channel=1, output_dir=str(results),
                             output_roots=[results])

    assert cleaned is not None
    assert list(results.glob("*.tif")), "the science still happened"
    assert not (results / ".analysis-kit").exists(), "and was not recorded"

    report = knowledge.doctor()
    assert report["ok"] is False
    assert any("analysis_kit is not importable" in complaint
               for complaint in report["complaints"])
