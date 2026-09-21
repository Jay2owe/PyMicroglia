"""The pipeline library, the review record, and the loop that closes a question.

Three things are being checked here and they are different in kind.

The **order** is a scientific constraint from ``AGENTS.md`` — registration
before cosmic-ray removal, measurement before display — so it is asserted
against the calls a run actually made rather than against a tuple a module
declares about itself. A future edit that reorders the calls fails here even if
it updates the docstring to match.

The **reuse** is the payoff for stages 03 to 10: change one segmentation
threshold and the registration and the cosmic-ray removal are read back rather
than recomputed. It is asserted by artefact access, not by wall-clock, because
a machine that happened to be busy is not evidence of anything.

The **closure** is stage 11's own new part: an answered question becomes a
stored decision, and the same question is never asked again — not after a
parameter change and not after a ``METHOD_VERSION`` bump, because a person
answered it once and a version number is not a reason to ask them twice.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pymicroglia import pipelines, review
from tests_support import oscillating_stack


# ── fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """A private cache and a private decisions folder for one test."""
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.setenv("PYMICROGLIA_DECISIONS", str(tmp_path / "decisions"))
    return tmp_path


SMALL = {"t0": None, "t1": None, "baselines": (6.0,), "detrends": ("cubic",),
         "ndecoy": 40, "skip_videos": True, "skip_control": True}


# ── the review record ───────────────────────────────────────────────────────
def test_a_confident_call_that_changes_nothing_is_a_note_not_a_question():
    """The bar, asserted. ``open_questions`` returning forty items is useless."""
    notes = review.Review()
    notes.note("cosmic_rays", chosen=0.06, confidence="high",
               why="0.06 % of pixel-frames replaced, which is the reference "
                   "dataset's own figure")
    assert notes.counts()["note"] == 1
    assert notes.open_questions() == []


def test_low_confidence_opens_a_question_and_so_does_changing_a_result():
    notes = review.Review()
    notes.note("window", chosen="72-210 h", confidence="low",
               why="the record is broken by two gaps")
    notes.note("channels", chosen={"dluc": 0}, confidence="high",
               changes_result=True, why="every mask depends on this")
    gates = [item.gate for item in notes.open_questions()]
    assert gates == ["window", "channels"] or gates == ["channels", "window"]
    assert all(item.severity == "check" for item in notes.open_questions())


def test_a_blocker_is_always_open_and_blocks():
    notes = review.Review()
    notes.flag("blocker", "control", "The instrumental control was skipped")
    assert notes.blocked() is True
    assert [item.gate for item in notes.open_questions()] == ["control"]


def test_an_answer_to_a_question_nobody_asked_is_refused():
    """A decision keyed on a question the pipeline never raises is a silent
    override: it looks authoritative and nothing shows why a run differed."""
    notes = review.Review()
    notes.note("window", chosen=72.0, confidence="low", why="short record")
    with pytest.raises(review.UnknownQuestion) as caught:
        notes.answer("channels", {"dluc": 1})
    assert "window" in str(caught.value)


def test_the_report_leads_with_the_section_the_workflow_names(tmp_path):
    notes = review.Review()
    notes.flag("check", "roi", "The outline is automatic and unchecked",
               question="Does it follow the structure?")
    text = notes.render(tmp_path / "REPORT.md", title="Test run")
    assert "## What needs your eye" in text
    assert "**Ask:** Does it follow the structure?" in text
    assert (tmp_path / "REPORT.md").is_file()


# ── the run-folder policy ───────────────────────────────────────────────────
def test_version_keeps_the_previous_run(tmp_path):
    first = pipelines.run_folder(tmp_path, "demo", "run", "version")
    first.path.mkdir(parents=True)
    second = pipelines.run_folder(tmp_path, "demo", "run", "version")
    assert second.label == "run_v2"
    assert first.path.is_dir(), "versioning must not touch the earlier run"


def test_error_refuses_an_existing_run(tmp_path):
    folder = pipelines.run_folder(tmp_path, "demo", "run", "error")
    folder.path.mkdir(parents=True)
    with pytest.raises(FileExistsError) as caught:
        pipelines.run_folder(tmp_path, "demo", "run", "error")
    assert "if_exists='version'" in str(caught.value)


def test_skip_hands_back_the_existing_folder_for_reuse(tmp_path):
    folder = pipelines.run_folder(tmp_path, "demo", "run", "skip")
    folder.path.mkdir(parents=True)
    pipelines.write_manifest(folder, {"objects": 3})
    again = pipelines.run_folder(tmp_path, "demo", "run", "skip")
    assert again.reuse is True
    assert pipelines.read_manifest(again) == {"objects": 3}


def test_overwrite_clears_it(tmp_path):
    folder = pipelines.run_folder(tmp_path, "demo", "run", "overwrite")
    folder.path.mkdir(parents=True)
    (folder.path / "old.txt").write_text("previous", encoding="utf-8")
    again = pipelines.run_folder(tmp_path, "demo", "run", "overwrite")
    assert again.label == "run"
    assert not (again.path / "old.txt").exists()


def test_an_unknown_policy_is_refused(tmp_path):
    with pytest.raises(ValueError) as caught:
        pipelines.run_folder(tmp_path, "demo", "run", "clobber")
    assert "overwrite" in str(caught.value)


def test_identical_settings_hash_to_the_same_run_name():
    left = pipelines.slug("mcg", {"k_mask": 8.0, "t0": 72.0})
    right = pipelines.slug("mcg", {"t0": 72.0, "k_mask": 8.0})
    assert left == right
    assert left != pipelines.slug("mcg", {"k_mask": 6.0, "t0": 72.0})


# ── the stage order ─────────────────────────────────────────────────────────
def test_cleaning_before_registering_is_refused():
    """``AGENTS.md`` fixes the order and this is that sentence as a test.

    On an unregistered stack the neighbouring frames show different tissue, so
    real motion is removed as if it were a spike.
    """
    with pytest.raises(pipelines.StageOrderError) as caught:
        pipelines.check_stage_order(["cosmic_rays", "register", "measure"])
    assert "before registration" in str(caught.value)


def test_measuring_after_the_display_filter_is_refused():
    with pytest.raises(pipelines.StageOrderError):
        pipelines.check_stage_order(["register", "display", "measure"])


def test_the_stage_log_refuses_the_wrong_order_as_it_happens():
    log = pipelines.StageLog()
    with log("cosmic_rays"):
        pass
    with pytest.raises(pipelines.StageOrderError):
        with log("register"):
            pass


def test_every_pipeline_declares_the_canonical_order():
    for row in pipelines.describe():
        assert row["available"], row
        stages = row["stages"]
        assert stages, row["name"]
        pipelines.check_stage_order(stages)
        if "cosmic_rays" in stages:
            assert stages.index("register") < stages.index("cosmic_rays")


# ── the file boundary that replaces PyFLASH's prefixes ──────────────────────
def test_no_pipeline_module_exceeds_eight_hundred_lines():
    """Gate 8, and the reason the folder has one module per pipeline.

    PyFLASH keeps six pipelines in one 7,728-line file and separates their
    helpers with a ``_corr_``/``_adj_``/``_ovw_`` prefix on every private
    function. Those prefixes exist because the file boundary does not. Here it
    does, so this is what keeps it doing its job.
    """
    folder = Path(pipelines.__file__).parent
    for path in sorted(folder.glob("*.py")):
        lines = len(path.read_text(encoding="utf-8").splitlines())
        assert lines <= 800, f"{path.name} is {lines} lines"


#: The files that do what ``dluc_pipeline.py`` did: the single-cell pipeline
#: and the two stretches of it that were split out.
#:
#: Named rather than globbed, and two things are deliberately outside it.
#: ``__init__.py`` is run bookkeeping -- manifests, stage order, run folders --
#: shared by every pipeline here, and the engine had no equivalent of it to
#: replace. The rest of the folder replaces nothing: ``auto_microglia`` wraps
#: another package's chain, ``cell_masks`` is a learned model. Counting either
#: against that engine's budget would fail this test for adding code the engine
#: never contained, which is not the risk it was written for.
THE_REBUILD = ("registered.py", "objects.py", "dluc_single_cell.py")


def test_the_rebuild_is_much_smaller_than_the_engine_it_replaces():
    """The risk the plan names: rebuilding should remove code, not move it.

    ``dluc_pipeline.py`` is 3,643 lines. If what does its job is anywhere near
    that, the package's parts are not being reused and something is wrong
    upstream rather than here.
    """
    folder = Path(pipelines.__file__).parent
    for name in THE_REBUILD:
        assert (folder / name).is_file(), f"{name} is gone; retire this list"
    total = sum(len((folder / name).read_text(encoding="utf-8").splitlines())
                for name in THE_REBUILD)
    assert total < 3643 // 2, (
        f"what replaced dluc_pipeline.py is {total} lines against its 3,643; "
        f"it should be a fraction of that, not most of it")


# ── a real run of the single-cell pipeline ──────────────────────────────────
@pytest.fixture
def one_run(isolated):
    from pymicroglia.pipelines import dluc_single_cell

    source = oscillating_stack(isolated)
    manifest = dluc_single_cell.run(source, output_dir=isolated / "out",
                                    if_exists="error", **SMALL)
    return source, manifest


def test_a_run_records_the_stages_it_performed_in_order(one_run):
    _, manifest = one_run
    order = [entry["stage"] for entry in manifest["stages"]]
    assert order.index("register") < order.index("cosmic_rays")
    assert order.index("cosmic_rays") < order.index("measure")
    assert order.index("measure") < order.index("display")
    pipelines.check_stage_order(order)


def test_the_report_and_the_manifest_both_carry_the_review(one_run):
    source, manifest = one_run
    folder = Path(manifest["folder"])
    assert (folder / "REPORT.md").is_file()
    assert "## What needs your eye" in (folder / "REPORT.md").read_text(
        encoding="utf-8")
    assert manifest["review"], "the manifest must carry the review"
    stored = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert stored["review"] == manifest["review"]


def test_the_open_questions_are_the_four_names_the_engine_uses(one_run):
    """Gate 4: channel assignment, the window, the outline, and admissibility.

    Asserted as a subset of the *possible* names rather than an exact list,
    because a recording with no marginal object correctly raises no
    admissibility question — the bar is what may open, not what must.
    """
    _, manifest = one_run
    gates = {item["gate"] for item in manifest["open_questions"]}
    assert gates, "a default run on an unchecked recording has questions"
    assert gates <= {"channels", "window", "roi", "admissibility", "cells",
                     "movement", "candidate_geometry", "control",
                     "cosmic_rays", "registration", "tissue_mask"}
    assert "channels" in gates
    assert "roi" in gates


def test_the_run_folder_names_itself_from_its_settings(one_run):
    source, manifest = one_run
    assert manifest["run_label"].startswith(Path(source).stem.split(".")[0])


def test_a_second_identical_run_refuses_rather_than_overwriting(one_run):
    from pymicroglia.pipelines import dluc_single_cell

    source, manifest = one_run
    with pytest.raises(FileExistsError):
        dluc_single_cell.run(source, output_dir=Path(manifest["folder"]).parents[1],
                             if_exists="error", **SMALL)


def test_skip_returns_the_previous_manifest_without_recomputing(one_run):
    from pymicroglia.pipelines import dluc_single_cell

    source, manifest = one_run
    again = dluc_single_cell.run(
        source, output_dir=Path(manifest["folder"]).parents[1],
        if_exists="skip", **SMALL)
    assert again["reused"] is True
    assert again["run_label"] == manifest["run_label"]
    assert again["summary"]["off_tissue_sigma"] == \
        manifest["summary"]["off_tissue_sigma"]


# ── reuse, asserted by artefact access ──────────────────────────────────────
def test_a_changed_threshold_reuses_registration_and_cosmic_ray_removal(
        one_run):
    """Gate 3, and the payoff for every stage from 03 onward.

    ``k_mask`` moves a segmentation threshold. Nothing upstream of segmentation
    depends on it, so the registered arrays and the cleaned bioluminescence must
    come back from the store rather than be rebuilt — which the run reports
    itself, per artefact, rather than being inferred from how long it took.
    """
    from pymicroglia.pipelines import dluc_single_cell

    source, manifest = one_run
    assert not any(manifest["summary"]["cached"].values()), \
        "the first run built everything"

    again = dluc_single_cell.run(
        source, output_dir=Path(manifest["folder"]).parents[1],
        if_exists="version", k_mask=7.0, **SMALL)
    cached = again["summary"]["cached"]
    assert cached["dluc_clean"] is True, \
        "cosmic-ray removal does not depend on a segmentation threshold"
    assert all(value for name, value in cached.items()
               if name.startswith("registered_")), \
        "registration does not depend on a segmentation threshold either"


def test_a_changed_cosmic_threshold_reuses_registration_but_not_the_cleaning(
        one_run):
    """The other direction, which is what makes the first test mean something."""
    from pymicroglia.pipelines import dluc_single_cell

    source, manifest = one_run
    again = dluc_single_cell.run(
        source, output_dir=Path(manifest["folder"]).parents[1],
        if_exists="version", cosmic_seed_z=10.0, **SMALL)
    cached = again["summary"]["cached"]
    assert cached["dluc_clean"] is False
    assert all(value for name, value in cached.items()
               if name.startswith("registered_"))


# ── the closure ─────────────────────────────────────────────────────────────
def test_answering_a_question_stops_it_being_asked_again(one_run, isolated):
    """Gate 5, both halves.

    The answer becomes a decision artefact keyed on the source, so the next run
    reads it instead of deciding — and keeps reading it after a
    ``METHOD_VERSION`` bump, because a version number is not a reason to ask a
    person the same question twice.
    """
    from pymicroglia import store
    from pymicroglia.pipelines import dluc_single_cell

    source, manifest = one_run
    assert "channels" in {item["gate"] for item in manifest["open_questions"]}

    notes = review.Review(source)
    notes.flag("check", "channels", "the channel assignment")
    notes.answer("channels", {"dluc": 0, "bf": 1, "struct": 1, "other": []})
    assert store.decision("channel_assignment", source) is None
    assert store.decision("channels", source) is not None

    again = dluc_single_cell.run(
        source, output_dir=Path(manifest["folder"]).parents[1],
        if_exists="version", **SMALL)
    assert "channels" not in {item["gate"]
                              for item in again["open_questions"]}

    # and again after the pipeline's method version moves on
    from pymicroglia.pipelines import registered as registered_module

    original = registered_module.__dict__.get("METHOD_VERSION")
    try:
        dluc_single_cell.METHOD_VERSION = "9999-01-01-a-later-version"
        after = dluc_single_cell.run(
            source, output_dir=Path(manifest["folder"]).parents[1],
            if_exists="version", **SMALL)
        assert "channels" not in {item["gate"]
                                  for item in after["open_questions"]}
    finally:
        dluc_single_cell.METHOD_VERSION = "2026-08-20-dluc-single-cell-one-outlier-rule"
        if original is not None:
            registered_module.METHOD_VERSION = original


def test_a_stored_answer_is_reported_as_answered_not_hidden(one_run):
    """An answered question leaves the *open* list and stays in the record.

    Dropping it entirely would make the review look like nothing was ever
    uncertain, which is the opposite of what it is for.
    """
    source, _ = one_run
    notes = review.Review(source)
    notes.flag("check", "roi", "the outline")
    notes.answer("roi", "the automatic one is correct")
    assert notes.open_questions() == []
    assert notes.as_records()[0]["answered"] is True
    assert notes.as_records()[0]["answer"] == "the automatic one is correct"


# ── the front half on its own ───────────────────────────────────────────────
def test_prepare_returns_the_array_the_parity_tests_need(isolated):
    """The registered, windowed, cosmic-cleaned bioluminescence, on its own.

    Stage 07's and stage 08's parity gates compare against exactly this array,
    which is why it is reachable without running the rest of the analysis.
    """
    from pymicroglia.pipelines import registered as prepare_module

    source = oscillating_stack(isolated)
    prepared = prepare_module.prepare(source, t0=None)
    frames, height, width = prepared.shape
    assert frames == 72
    assert height == width == 130 - 2 * prepared.crop_pad
    assert len(prepared.registered) == 2
    assert prepared.times_h[0] == pytest.approx(0.0)
    assert np.asarray(prepared.dluc).dtype == np.float32


def test_the_registered_arrays_are_the_only_source_downstream(isolated):
    """Rule 11: register once, then use the canonical arrays everywhere.

    Asserted through the view every measurement receives — it holds the
    registered arrays themselves, so nothing downstream is in a position to
    re-register a channel for a figure.
    """
    from pymicroglia.pipelines import registered as prepare_module

    source = oscillating_stack(isolated)
    prepared = prepare_module.prepare(source, t0=None)
    view = prepared.view()
    assert view.shape[1] == 2
    np.testing.assert_array_equal(view.frame(0, 0), prepared.registered[0][0])
    assert view.display_only is False


def test_a_stack_view_refuses_to_invent_a_source():
    """It has no file and no identity, and says so rather than guessing one."""
    view = pipelines.StackView([np.zeros((4, 8, 8), np.float32)])
    with pytest.raises(AttributeError) as caught:
        view.source
    assert "no source identity" in str(caught.value)


# ── discovery ───────────────────────────────────────────────────────────────
def test_the_pipelines_available_are_exactly_the_ones_named_and_no_others():
    """The list is closed: a sixth idea gets written down, not added.

    Spelled out rather than counted, so that adding one is two deliberate
    edits -- the module and this line -- instead of a number going up. The
    fifth, ``auto_microglia``, was added on 2026-09-15 because it is the only
    one that starts at the instrument rather than at a file somebody already
    has.
    """
    assert set(pipelines.available()) == set(pipelines.PIPELINE_NAMES)
    assert pipelines.PIPELINE_NAMES == (
        "auto_microglia", "dluc_single_cell", "cry1_dluc_photon",
        "bioluminescence", "phase_green_red")


def test_the_two_registered_protocols_are_no_longer_pending():
    """Gate 9: all thirteen registered protocols are backed.

    The Motion port's stage 02 added ``track``, pending by design behind a
    declared seam; nothing else may be pending, and no protocol is.
    """
    from pymicroglia import registry

    assert set(registry.pending()) <= {"track"}
    for name in registry.pending():
        answer = registry.seam_status(registry.REGISTRY.binds_to(name))
        assert answer is not None and answer[0] == "pending", name
    assert registry.REGISTRY.binds_to("dluc_single_cell") == \
        "pipelines.dluc_single_cell.run"
    assert registry.REGISTRY.binds_to("cry1_dluc_photon") == \
        "pipelines.cry1_dluc_photon.run"


# ── against the engines these pipelines replace ─────────────────────────────
CRY1_REFERENCE = Path("reference-data") / "cry1-photon"
CRY1_SOURCE = CRY1_REFERENCE / "MCG_05 - 1 - 26.tif"
CRY1_ENGINE = CRY1_REFERENCE / "engine"


def _csv_rows(path):
    import csv

    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_the_photon_pipeline_reproduces_its_engine(isolated):
    """Gate 2 for ``cry1_dluc_photon``: the registered stack, pixel for pixel.

    The smallest of the three reference recordings, chosen so this runs in about
    ten seconds and can stay in the ordinary suite rather than behind a flag.

    Every comparison here is against output the engine produced earlier. Nothing
    in this test runs the engine, which is the same rule every parity test in
    this package follows: a port is checked against a stored result, never
    against a second live implementation that might have drifted with it.
    """
    if not (CRY1_SOURCE.is_file() and CRY1_ENGINE.is_dir()):
        pytest.skip("the Cry1 photon reference recording is not on this machine")

    import tifffile

    from pymicroglia.pipelines import cry1_dluc_photon

    manifest = cry1_dluc_photon.run(CRY1_SOURCE, output_dir=isolated / "cry1",
                                    if_exists="overwrite", videos=False)
    mine = manifest["summary"]
    theirs = json.loads((CRY1_ENGINE / "processing_manifest.json").read_text(
        encoding="utf-8"))

    for key in ("frames", "crop_xyxy", "output_shape",
                "recorded_median_interval_seconds",
                "registration_residual_median_px",
                "registration_residual_p95_px",
                "registration_residual_max_px"):
        assert mine[key] == theirs[key], key

    folder = Path(manifest["folder"])
    ours = _csv_rows(folder / "registration_shifts_and_qc.csv")
    engine = _csv_rows(CRY1_ENGINE / "registration_shifts_and_qc.csv")
    assert len(ours) == len(engine) == theirs["frames"]
    for column in ("pair_shift_y_px", "pair_shift_x_px",
                   "cumulative_shift_y_px", "cumulative_shift_x_px",
                   "residual_magnitude_px"):
        np.testing.assert_array_equal(
            [float(row[column]) for row in ours],
            [float(row[column]) for row in engine], err_msg=column)

    ours = _csv_rows(folder / "photon_signal_qc.csv")
    engine = _csv_rows(CRY1_ENGINE / "photon_signal_qc.csv")
    for mine_column, their_column in (
            ("camera_baseline", "camera_baseline_p30"),
            ("isolated_event_threshold_counts",
             "isolated_event_threshold_counts"),
            ("isolated_event_pixels", "isolated_event_pixels"),
            ("positive_signal_sum", "positive_signal_sum"),
            ("positive_signal_percentile", "positive_signal_p99")):
        np.testing.assert_array_equal(
            [float(row[mine_column]) for row in ours],
            [float(row[their_column]) for row in engine],
            err_msg=mine_column)

    stem = CRY1_SOURCE.stem
    engine_stack = tifffile.imread(
        CRY1_ENGINE / f"{stem}_registered_raw_4channel.tif")
    ours_stack = tifffile.imread(manifest["outputs"]["registered"])
    assert ours_stack.shape == engine_stack.shape
    np.testing.assert_array_equal(ours_stack, engine_stack)


def test_the_registered_tiff_holds_raw_counts_and_says_so(isolated):
    """The scientific product carries no display processing, and records that.

    The photon products are good for looking at single photons arriving on a
    sensor and bad for measuring them, so they never go back into the TIFF. The
    manifest states it in a field rather than in prose nobody parses.
    """
    if not (CRY1_SOURCE.is_file() and CRY1_ENGINE.is_dir()):
        pytest.skip("the Cry1 photon reference recording is not on this machine")

    from pymicroglia.pipelines import cry1_dluc_photon

    manifest = cry1_dluc_photon.run(CRY1_SOURCE, output_dir=isolated / "cry1",
                                    if_exists="overwrite", videos=False)
    assert manifest["summary"]["scientific_tiff_photon_filtering"] == "none"
    assert manifest["summary"]["display_only"] is True
    order = [entry["stage"] for entry in manifest["stages"]]
    assert order.index("measure") < len(order)
    assert "display" not in order[:order.index("measure")]


# ── the general bioluminescence route ───────────────────────────────────────
def test_the_general_pipeline_runs_the_live_cosmic_method(isolated):
    """The other pipeline, executed rather than described.

    ``bioluminescence`` is not a registered action, so nothing else in this
    suite runs it end to end, and a change to the step it calls would go
    unnoticed until a real recording hit it. Segmentation and the rhythm test
    are off: what is checked is the order and the cosmic-ray call, and both
    happen before either.
    """
    from pymicroglia import cosmic
    from pymicroglia.pipelines import bioluminescence

    source = oscillating_stack(isolated, frames=12)
    manifest = bioluminescence.run(
        source, output_dir=isolated / "out", segment=False, test_rhythm=False,
        claim="the standard order, on a synthetic stack")

    order = [entry["stage"] for entry in manifest["stages"]]
    pipelines.check_stage_order(order)
    assert order.index("register") < order.index("cosmic_rays")
    assert order.index("cosmic_rays") < order.index("display")

    cleaned = manifest["summary"]["cosmic_rays"]
    assert cleaned["method_version"] == cosmic.METHOD_VERSION
    assert "percent_of_selected_channel" in cleaned


def test_the_general_pipeline_flags_a_cut_that_rewrites_too_much(isolated):
    """The check that was reading a key the summary does not have.

    This pipeline used to compare against ``percent_of_stack``, which no cosmic
    summary has ever carried, so the flag could not fire however much the
    filter rewrote. Forced here by dropping the cut to 2.5 noise units, where it
    starts eating the noise itself.
    """
    from pymicroglia.pipelines import bioluminescence

    source = oscillating_stack(isolated, frames=12)
    manifest = bioluminescence.run(
        source, output_dir=isolated / "out", segment=False, test_rhythm=False,
        seed_z=2.5, claim="what a cut this low does to the stack")

    flagged = [item for item in manifest["review"]
               if item["gate"] == "cosmic_rays"]
    assert flagged, "a filter rewriting most of the stack should be flagged"
    assert any("raise seed_z" in str(item.get("remedy", ""))
               for item in flagged)
