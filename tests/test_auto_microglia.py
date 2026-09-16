"""The claim ``auto_microglia`` makes, tested as a claim rather than described.

The claim is: *Auto-Organotypic's chain, with only the defaults changed*. Every
test here is one of the ways that could quietly stop being true.

**The chain is called, not copied.** One test replaces
``auto_organotypic.pipeline.run_pipeline`` with a recorder and checks that what
this pipeline hands it is the folder and the caller's keywords, untouched. If a
parameter ever gets intercepted, defaulted or renamed on the way through, the
sentence in the module docstring becomes false and this fails.

**The differences are the whole list.** ``differences()`` is the auditable
table, and a default changed without a row added to it is a difference nobody
can find. The test cross-checks the table against what a real run actually does
with ``skip``, so the two cannot drift.

**The order is mask, then measure.** The inverse of the rule
``check_stage_order`` enforces on ``dluc_single_cell``, and for the same reason:
here the mask is what finds the cells, so a measurement made before it measured
something else.

**Motion is pending, visibly.** The handoff is written, the stage says
``pending``, and the review carries a note that says what to install. A stage
that silently did nothing would be indistinguishable from a stage that worked.

The network is never run and neither is the chain: both are replaced. These are
claims about wiring, and a test that needed an instrument and 40 MB of weights
to check that ``outline`` defaults to ``False`` would be skipped everywhere it
matters.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pymicroglia import pipelines
from pymicroglia.pipelines import auto_microglia, motion_handoff


# ── a chain that is not the chain ───────────────────────────────────────────
@pytest.fixture
def fake_chain(monkeypatch, tmp_path):
    """``run_pipeline`` replaced by a recorder, returning one registered file.

    Returns the dict the recorder writes its call into, so a test can assert
    what was passed through rather than only what came back.
    """
    from auto_organotypic import pipeline as chain

    stack = tmp_path / "recordings" / "MCG 04 - 1 - 595.tif"
    stack.parent.mkdir(parents=True, exist_ok=True)
    stack.write_bytes(b"a registered stack, as far as this test is concerned")

    seen: dict[str, object] = {}

    def run_pipeline(folder, **options):
        seen["folder"] = folder
        seen["options"] = dict(options)
        recordings = [{"path": str(stack),
                       "interval_s": {"value": 1997.4},
                       "pixel_size_um": {"value": 2.0}}]
        stages = [{"stage": name, "seconds": 0.1}
                  for name in auto_microglia.AO_STAGES]

        # The one thing this stand-in must not fake: a stage registered into
        # the chain is run *by* the chain. Skipping that here would leave every
        # test below passing while the real pipeline masked nothing, because
        # the masking loop this pipeline used to own is gone.
        skipped = set(options.get("skip") or ())
        asked = set(options.get("stages") or ())
        for name, owner in chain.registered_stages().items():
            opt_in = chain._REGISTERED[name]["stage"].opt_in
            if name in skipped or (asked and name not in asked):
                continue
            if opt_in and not asked and f"{name}_options" not in options:
                continue      # opt-in there, and nobody named its options
            state = {"recordings": recordings, "folder": Path(folder),
                     "output_root": options.get("output_root") or folder}
            stages.append({"stage": name, "owner": owner, "status": "ok",
                           "seconds": 0.1,
                           **chain._RUNNERS[name](state, dict(options))})

        return {"version": "0.6.0", "stages": stages, "recordings": recordings}

    monkeypatch.setattr(chain, "run_pipeline", run_pipeline)
    seen["stack"] = stack
    return seen


@pytest.fixture
def fake_mask(monkeypatch):
    """``cell_masks`` replaced by something that writes two files and labels."""
    from pymicroglia.pipelines import cell_masks

    asked: list[dict] = []

    def mask_run(prepared, folder, notes=None, **settings):
        asked.append(dict(settings))
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        written = {}
        for key in ("mask_probability", "mask_cells", "mask"):
            path = folder / f"stub_{key}.tif"
            path.write_bytes(b"stub")
            written[key] = str(path)
        return {"outputs": written, "settings": {"cut": 0.8, "cells_found": 2},
                "warning": None, "reused": False}

    def mask_still(prepared, folder, notes=None, **settings):
        import tifffile

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "stub_cells_still.tif"
        labels = np.zeros((16, 16), np.uint16)
        labels[2:6, 2:6] = 1
        labels[9:13, 9:13] = 2
        # A real label image, because the run record carries the *path* to the
        # still mask and the measurement reads it back from there. An array
        # handed between two functions in one process was the old shape and is
        # exactly what a stage inside the chain cannot do.
        tifffile.imwrite(path, labels)
        return {"labels": labels,
                "outputs": {"mask_cells_still": str(path)},
                "settings": {"cut": 0.8, "cells_found": 2}, "warning": None}

    monkeypatch.setattr(cell_masks, "mask_run", mask_run)
    monkeypatch.setattr(cell_masks, "mask_still", mask_still)
    return asked


@pytest.fixture
def fake_measurement(monkeypatch):
    """``region_trace.run`` and the decoy test replaced; returns what was asked.

    The point of recording the call is ``labels=``: it is the whole reason this
    pipeline can skip the outline and still get Auto-Organotypic's traces, and
    a change that stopped passing it would leave a run measuring nothing while
    still writing a folder full of plausible output.
    """
    from auto_organotypic import region_trace

    asked: list[dict] = []

    def run(source, **settings):
        asked.append({"source": source, **settings})
        return {"regions": 2, "table": [{"label": 1}, {"label": 2}]}

    monkeypatch.setattr(region_trace, "run", run)
    monkeypatch.setattr(auto_microglia, "_admissible",
                        lambda *a, **k: {"admissible": 2, "tested": 2,
                                         "tissue_channel": 2,
                                         "decoys": "on tissue, absolute counts"})
    return asked


@pytest.fixture
def one_run(tmp_path, fake_chain, fake_mask, fake_measurement):
    """A whole default run over the fake chain, with nothing real computed."""
    manifest = auto_microglia.run(str(tmp_path / "source"),
                                  output_dir=tmp_path / "out",
                                  if_exists="error")
    return manifest, fake_chain, fake_measurement


# ── the chain is called, not copied ─────────────────────────────────────────
def test_every_keyword_the_caller_gives_reaches_the_chain_untouched(
        tmp_path, fake_chain, fake_mask, fake_measurement):
    """The sentence the module docstring makes, as an assertion.

    If a keyword is ever intercepted here -- read, defaulted, renamed -- then
    Auto-Organotypic's parameter block has a second home, and within a release
    the two disagree about something somebody depends on.
    """
    auto_microglia.run(str(tmp_path / "source"), output_dir=tmp_path / "out",
                       if_exists="error",
                       experiment="MCG_04", instrument="lumicycle",
                       broad_crop=True, video=False, since="2026-01-01",
                       force=True, trace_options={"detrend": "cubic"})
    passed = fake_chain["options"]
    assert passed["experiment"] == "MCG_04"
    assert passed["instrument"] == "lumicycle"
    assert passed["broad_crop"] is True
    assert passed["video"] is False
    assert passed["since"] == "2026-01-01"
    assert passed["force"] is True
    assert passed["trace_options"] == {"detrend": "cubic"}


def test_the_pipelines_own_keywords_are_not_passed_down_as_the_chains(
        tmp_path, fake_chain, fake_mask, fake_measurement):
    """``cell_masks``, ``motion`` and the rest are this package's, not its.

    Handing them to ``run_pipeline`` would be a ``TypeError`` today and a
    silently accepted unknown option the day that function grows ``**options``.
    """
    auto_microglia.run(str(tmp_path / "source"), output_dir=tmp_path / "out",
                       if_exists="error", cell_masks=False, cells=False,
                       motion=False, decoy_count=9, mask_threads=3)
    passed = fake_chain["options"]
    for ours in ("cell_masks", "cells", "motion", "decoy_count",
                 "mask_threads", "mask_cut", "region_traces", "outline",
                 "if_exists", "run_label", "claim"):
        assert ours not in passed, f"{ours} was handed to the chain"


# ── the differences are the whole list ──────────────────────────────────────
def test_the_declared_differences_are_the_ones_a_run_actually_makes(
        tmp_path, fake_chain, fake_mask, fake_measurement):
    """The table and the behaviour, checked against each other.

    ``differences()`` is what somebody reads to know how this differs from the
    chain. A row that is not honoured, or a skip with no row, makes that table
    a description of a previous version.
    """
    auto_microglia.run(str(tmp_path / "source"), output_dir=tmp_path / "out",
                       if_exists="error")
    skipped = set(fake_chain["options"]["skip"])
    assert skipped == set(auto_microglia.NOT_OURS)

    table = {row["setting"]: row for row in auto_microglia.differences()}
    turned_off = {name for name, row in table.items() if row["here"] is False}
    turned_on = {name for name, row in table.items() if row["here"] is True}
    assert turned_off == skipped
    assert turned_on == set(auto_microglia.OURS)
    assert set(table) == turned_off | turned_on, "a row is neither on nor off"


def test_every_difference_says_why_in_a_sentence_somebody_can_argue_with():
    """A table of changed defaults with no reasons is a list of preferences."""
    for row in auto_microglia.differences():
        assert len(row["why"].split()) >= 8, row


def test_asking_for_the_outline_back_gives_the_whole_scn_half(
        tmp_path, fake_chain, fake_mask, fake_measurement):
    """``outline=True`` on its own is a whole restoration, not a partial one.

    The riders follow it, so somebody who says "this recording does have an SCN"
    gets Auto-Organotypic's SCN half at Auto-Organotypic's settings, rather than
    an outline with nothing reading it.
    """
    auto_microglia.run(str(tmp_path / "source"), output_dir=tmp_path / "out",
                       if_exists="error", outline=True)
    assert fake_chain["options"]["skip"] == ()
    assert fake_chain["options"]["spatial"] is True


def test_a_rider_can_be_narrowed_without_giving_up_the_outline(
        tmp_path, fake_chain, fake_mask, fake_measurement):
    """Given explicitly, a rider means what it says: outline, traces, no grid."""
    auto_microglia.run(str(tmp_path / "source"), output_dir=tmp_path / "out",
                       if_exists="error", outline=True, spatial=False)
    assert set(fake_chain["options"]["skip"]) == {"spatial"}
    assert fake_chain["options"]["spatial"] is False


def test_the_unmodified_chain_is_reachable_without_leaving_this_package(
        tmp_path, fake_chain, fake_mask, fake_measurement):
    """Auto-Organotypic's own run, at its own defaults, through this pipeline.

    Turning this pipeline's three stages off and the SCN half back on leaves
    nothing of ours in the call: an empty ``skip`` and whatever the caller
    passed. So "PyMicroglia is where the microglia defaults live" never becomes
    "PyMicroglia is where you cannot get the ordinary chain", which would push
    somebody into maintaining a second entry point for SCN work.
    """
    from auto_organotypic import pipeline as chain

    auto_microglia.run(str(tmp_path / "source"), output_dir=tmp_path / "out",
                       if_exists="error", outline=True,
                       cell_masks=False, cells=False, motion=False)
    assert fake_chain["options"]["skip"] == ()
    assert "cell_masks_options" not in fake_chain["options"], (
        "the mask is opt-in in the chain, so not asking for it is the whole of "
        "turning it off -- adding a skip instead would leave this pipeline's "
        "fingerprints on a call that is meant to be the chain's own")

    reachable = {stage.name for stage in chain.STAGES}
    assert set(auto_microglia.AO_STAGES) <= reachable
    assert reachable & set(auto_microglia.OURS) == {auto_microglia.MASK_STAGE}, (
        "the mask is a stage of the chain, registered by this package; the "
        "other two are this package's own and must not collide with a name "
        "over there")
    assert chain.registered_stages()[auto_microglia.MASK_STAGE] == "PyMicroglia"


def test_every_opt_in_stage_of_the_chain_is_still_opt_in_from_here(
        tmp_path, fake_chain, fake_mask, fake_measurement):
    """The chain's own extras are asked for the chain's own way.

    ``image``, ``grid``, ``video`` and the rest are opt-in there and reach
    ``run_pipeline`` as themselves, not as some renamed local switch. The one
    exception is ``review``, whose name this package had already spent.
    """
    from auto_organotypic import pipeline as chain

    extras = ({stage.name for stage in chain.STAGES
               if getattr(stage, "opt_in", False)}
              - {"acquire"} - set(chain.registered_stages()))
    asked = {auto_microglia.CLAIMED.get(name) or name: True for name in extras}
    auto_microglia.run(str(tmp_path / "source"), output_dir=tmp_path / "out",
                       if_exists="error", outline=True,
                       cell_masks=False, cells=False, motion=False, **asked)
    for name in extras:
        assert fake_chain["options"][name] is True, name


def test_no_keyword_of_the_chains_is_swallowed_here_without_saying_so():
    """The collision check, so the next one fails a test instead of a run.

    ``review`` means a ``Review`` object in every pipeline of this package and a
    boolean stage in that one. The clash cost a crash -- ``'bool' object has no
    attribute 'as_records'`` -- from a keyword that looked like it was passing
    through. Any future name Auto-Organotypic adds that this signature already
    uses would do the same, silently, so it is caught here instead.
    """
    import inspect

    from auto_organotypic import pipeline as chain

    theirs = set(inspect.signature(chain.run_pipeline).parameters)
    ours = set(inspect.signature(auto_microglia.run).parameters)
    assert (theirs & ours) - set(auto_microglia.CLAIMED) == set(), (
        "a keyword of the chain's is captured here under the same name. Either "
        "rename this one and add it to CLAIMED, or pass it through explicitly.")
    for taken, instead in auto_microglia.CLAIMED.items():
        assert taken in theirs, f"CLAIMED lists {taken}, which the chain dropped"
        assert not instead or instead in ours, instead
    assert "review" in auto_microglia.CLAIMED, (
        "the collision this check was written for is gone from the list")


def test_the_stage_list_this_pipeline_advertises_is_the_chains_own():
    """``AO_STAGES`` is a copy, and the one thing here that cannot self-update.

    Everything else about the chain arrives by passing keywords through, so it
    follows upstream with no edit. This list is spelled out because ``describe``
    reads it, and a copy drifts — so a stage added or renamed upstream fails
    here rather than going unmentioned to whoever reads the row.
    """
    from auto_organotypic import pipeline as chain

    registered = set(chain.registered_stages())
    # Somebody else's registered stage is not this package's to advertise, and
    # its presence must not make this test fail: a third package installed
    # beside us is the case the whole registration API exists for.
    others = {name for name, owner in chain.registered_stages().items()
              if owner != "PyMicroglia"}
    theirs = tuple(name for name in chain.stage_names()
                   if name not in auto_microglia.NOT_OURS
                   and name not in registered)
    assert auto_microglia.AO_STAGES == theirs
    assert "cell_masks" in registered, (
        "the mask is registered on import of this module, so a run of this "
        "suite that did not import it would test a different pipeline")
    # And the advertised order is the chain's order, which is now where the
    # mask's place is decided -- not a tuple here that says it runs last.
    ours_after = tuple(name for name in auto_microglia.OURS
                       if name not in registered)
    assert auto_microglia.STAGES == tuple(
        name for name in chain.stage_names()
        if name not in auto_microglia.NOT_OURS
        and name not in others) + ours_after


def test_a_renamed_stage_upstream_stops_the_run_instead_of_outlining_everything(
        tmp_path, fake_chain, fake_mask, fake_measurement, monkeypatch):
    """The silent failure this pipeline is most exposed to, made loud.

    ``skip`` is filtered against the chain's stage list, so a name that no
    longer exists is ignored rather than refused. If ``outline`` were renamed,
    a run would outline every recording, orient and crop around a structure that
    is not there, and report success.
    """
    from auto_organotypic import pipeline as chain

    monkeypatch.setattr(chain, "stage_names",
                        lambda: ("index", "register", "scn_outline"))
    with pytest.raises(ValueError) as caught:
        auto_microglia.run(str(tmp_path / "source"),
                           output_dir=tmp_path / "out", if_exists="error")
    said = str(caught.value)
    assert "outline" in said and "no longer has" in said
    assert "scn_outline" in said, "it should say what the stages are now"


def test_the_chains_review_stage_is_reachable_under_its_new_name(
        tmp_path, fake_chain, fake_mask, fake_measurement):
    """And asking for it does not cost the run its own Review."""
    manifest = auto_microglia.run(
        str(tmp_path / "source"), output_dir=tmp_path / "out",
        if_exists="error", outline=True, chain_review=True,
        cell_masks=False, cells=False, motion=False)
    assert fake_chain["options"]["review"] is True
    assert isinstance(manifest["review"], list)


@pytest.mark.parametrize("asked", [{"region_traces": True}, {"spatial": True},
                                   {"region_traces": True, "spatial": True}])
def test_a_rider_without_the_outline_is_refused_at_the_keyword(tmp_path, asked):
    """Refused here rather than several stages later on a missing file.

    ``trace`` is one trace per region *of the outline* and ``spatial`` lays its
    squares across *its* crop. Without the outline neither has anything to read,
    and the failure would otherwise surface inside Auto-Organotypic, far from
    the keyword that caused it.
    """
    with pytest.raises(ValueError) as caught:
        auto_microglia.run(str(tmp_path / "source"),
                           output_dir=tmp_path / "out", if_exists="error",
                           **asked)
    assert "outline=True" in str(caught.value)


def test_a_skip_the_caller_asked_for_is_added_to_and_never_replaced(
        tmp_path, fake_chain, fake_mask, fake_measurement):
    """Ours are extra reasons to skip, not the only ones allowed."""
    auto_microglia.run(str(tmp_path / "source"), output_dir=tmp_path / "out",
                       if_exists="error", skip=("video",))
    assert set(fake_chain["options"]["skip"]) == {"video"} | set(
        auto_microglia.NOT_OURS)


# ── the mask finds the cells, so it comes first ─────────────────────────────
def test_the_mask_runs_before_the_measurement_and_a_run_records_that(one_run):
    manifest, _, _ = one_run
    order = [entry["stage"] for entry in manifest["stages"]]
    assert order.index("cell_masks") < order.index("cells")
    assert order.index("register") < order.index("cell_masks")
    auto_microglia._the_mask_feeds_the_measurement(order)


def test_measuring_before_masking_is_refused_rather_than_measured():
    """The rule, tested on the rule and not only on a run that obeys it."""
    with pytest.raises(pipelines.StageOrderError) as caught:
        auto_microglia._the_mask_feeds_the_measurement(
            ["register", "cells", "cell_masks"])
    assert "after the cell measurement" in str(caught.value)


def test_measuring_without_a_mask_refuses_and_says_which_switch_to_flip(
        tmp_path, fake_chain, fake_mask, fake_measurement):
    """There are no labels to measure, and the fix is one keyword."""
    with pytest.raises(ValueError) as caught:
        auto_microglia.run(str(tmp_path / "source"),
                           output_dir=tmp_path / "out", if_exists="error",
                           cell_masks=False, cells=True)
    assert "cell_masks=True" in str(caught.value)


def test_the_cells_are_measured_by_auto_organotypics_own_engine(one_run):
    """``labels=`` is what makes "imports the chain" true for the traces too.

    Without it this pipeline would need its own trace, detrend, control and
    rhythm code -- a second copy of the part of the science most likely to be
    revised.
    """
    _, chain, asked = one_run
    assert len(asked) == 1
    call = asked[0]
    assert call["source"] == str(chain["stack"])
    assert np.asarray(call["labels"]).max() == 2


# ── Motion is pending, and says so ──────────────────────────────────────────
def test_the_motion_stage_is_pending_and_names_what_would_make_it_ready(
        one_run):
    manifest, _, _ = one_run
    stage = next(entry for entry in manifest["stages"]
                 if entry["stage"] == "motion")
    assert stage["status"] == "pending"
    assert "not installed" in stage["reason"]


def test_the_pending_stage_reaches_the_review_and_not_only_the_manifest(
        one_run):
    """A stage that did nothing must be visible where a person actually looks."""
    manifest, _, _ = one_run
    motion = [row for row in manifest["review"] if row["gate"] == "motion"]
    assert motion, "the pending Motion stage left no note in the review"
    assert "Install the Motion project" in motion[0]["remedy"]


def test_motion_gets_a_file_it_can_verify_rather_than_a_folder_to_guess_at(
        one_run):
    manifest, chain, _ = one_run
    written = Path(manifest["outputs"]["motion"]["motion_inputs"])
    payload = json.loads(written.read_text(encoding="utf-8"))
    stem = Path(chain["stack"]).stem
    pinned = payload["pinned_files"][stem]
    assert len(pinned["registered_raw"]["sha256"]) == 64
    assert len(pinned["cell_mask"]["sha256"]) == 64
    assert payload["input_space"] == "registered"
    assert payload["frame_interval_min"] == pytest.approx(1997.4 / 60.0)


def test_the_evidence_motion_has_to_compute_itself_is_named_not_faked(
        one_run):
    """Empty entries would make this file look complete when it is not."""
    manifest, _, _ = one_run
    payload = json.loads(
        Path(manifest["outputs"]["motion"]["motion_inputs"]).read_text(
            encoding="utf-8"))
    assert set(payload["still_missing"]) == {
        "lag_float", "neutral_tracks", "trail_labels", "trail_ages",
        "motion_composite"}
    for entry in payload["pinned_files"].values():
        assert not set(entry) & set(payload["still_missing"])


def test_skipping_the_hashes_is_a_choice_that_shows_in_the_file(tmp_path):
    """``hashes=False`` writes nulls, which Motion will not accept.

    The option exists because hashing is a full read of every stack, and the
    cost is worth choosing to defer rather than discovering inside a run.
    """
    stack = tmp_path / "one.tif"
    stack.write_bytes(b"pixels")
    out = motion_handoff.write([{"path": str(stack)}], {}, tmp_path,
                               dataset="d", hashes=False)
    payload = json.loads(Path(out["outputs"]["motion_inputs"]).read_text(
        encoding="utf-8"))
    assert payload["pinned_files"]["one"]["registered_raw"]["sha256"] is None


def test_the_handoff_reports_pending_while_motion_is_not_installed():
    """Auto-Organotypic's shape for a missing package: a state, not a crash."""
    state, reason = motion_handoff.status()
    assert state == "pending"
    assert motion_handoff.TARGET.partition(":")[0] in reason


# ── the run record ──────────────────────────────────────────────────────────
def test_the_manifest_keeps_the_chains_own_record_rather_than_summarising_it(
        one_run):
    """What Auto-Organotypic decided is its record to make, and is kept whole.

    Re-describing it here would lose whatever that package adds next, and the
    loss would only show up when somebody went looking for it.
    """
    manifest, _, _ = one_run
    assert manifest["auto_organotypic_record"]["version"] == "0.6.0"
    assert manifest["summary"]["auto_organotypic"] == "0.6.0"
    assert manifest["method_version"] == auto_microglia.METHOD_VERSION


def test_a_run_without_a_folder_says_so_instead_of_guessing_one():
    with pytest.raises(ValueError) as caught:
        auto_microglia.run(None)
    assert "folder of recordings" in str(caught.value)


def test_the_pipeline_is_reachable_by_name_like_every_other(one_run):
    assert "auto_microglia" in pipelines.available()
    assert pipelines.get("auto_microglia") is auto_microglia


# ── not paying for the same mask twice ──────────────────────────────────────
def test_the_mask_step_reuses_by_default_and_force_turns_that_off(
        tmp_path, fake_chain, fake_mask, fake_measurement):
    """``force`` is the chain's word for "do it again", so it is the word here.

    A second one meaning the same thing is a second thing to remember, and the
    one somebody types is then the one that does nothing.
    """
    auto_microglia.run(str(tmp_path / "source"), output_dir=tmp_path / "a",
                       if_exists="error")
    assert fake_mask[0]["reuse"] is True

    auto_microglia.run(str(tmp_path / "source"), output_dir=tmp_path / "b",
                       if_exists="error", force=True)
    assert fake_mask[-1]["reuse"] is False
    assert fake_chain["options"]["force"] is True, "force still reaches the chain"


def test_an_unspecified_window_is_left_out_rather_than_passed_as_none(
        tmp_path, fake_chain, fake_mask, fake_measurement):
    """Its default is the duration the weights were trained on.

    That number lives next to the weights. Forwarding ``None`` would ask the
    mask to treat "not specified" as a number, which is a ``TypeError`` several
    stages into a run.
    """
    auto_microglia.run(str(tmp_path / "source"), output_dir=tmp_path / "out",
                       if_exists="error")
    assert "window_hours" not in fake_mask[0]

    auto_microglia.run(str(tmp_path / "source"), output_dir=tmp_path / "two",
                       if_exists="error", mask_window_h=6.0)
    assert fake_mask[-1]["window_hours"] == 6.0


def test_the_slowest_step_reports_progress_the_way_the_chain_does(
        tmp_path, fake_chain, fake_mask, fake_measurement):
    """Minutes per recording, against seconds for everything around it.

    A watcher told about every other stage and not this one reports a run that
    has stopped, so the entries are the shape Auto-Organotypic's own are.
    """
    seen: list[dict] = []
    auto_microglia.run(str(tmp_path / "source"), output_dir=tmp_path / "out",
                       if_exists="error", on_progress=seen.append)
    masked = [row for row in seen if row.get("stage") == "cell_masks"]
    assert masked, "the mask step told nobody it was working"
    assert set(masked[0]) >= {"stage", "status", "seconds", "recording"}
    assert masked[0]["of"] == 1
    assert fake_chain["options"]["on_progress"] is not None, (
        "the chain still gets its own progress callback")


def test_a_run_records_how_many_masks_it_did_not_have_to_make(
        tmp_path, fake_chain, fake_mask, fake_measurement):
    """In the manifest, because "it was quick" is not an explanation."""
    manifest = auto_microglia.run(str(tmp_path / "source"),
                                  output_dir=tmp_path / "out",
                                  if_exists="error")
    stage = next(entry for entry in manifest["stages"]
                 if entry["stage"] == "cell_masks")
    assert stage["recordings"] == 1
    assert stage["reused"] == 0


# ── the mask is a stage of the chain, not a loop after it ───────────────────
def test_the_run_record_names_this_package_as_the_masks_owner(one_run):
    """Who did what, in one record, for a run two packages performed.

    The reason the mask is registered rather than run in a loop afterwards:
    a run record that stops at ``split`` and a separate manifest that starts at
    the mask cannot be read as one run by anybody who was not there.
    """
    manifest, _chain, _measured = one_run
    stages = {row["stage"]: row for row in manifest["stages"]}
    assert auto_microglia.MASK_STAGE in stages
    assert stages[auto_microglia.MASK_STAGE]["owner"] == "PyMicroglia"
    assert stages[auto_microglia.MASK_STAGE]["recordings"] == 1
    assert "masks" not in stages[auto_microglia.MASK_STAGE], (
        "the mask's own paths are in outputs; a second copy in the stage log "
        "is the same list written twice in one manifest")


def test_the_mask_is_selectable_by_name_the_way_the_chains_own_stages_are(
        tmp_path, fake_chain, fake_mask, fake_measurement):
    """``stages=`` and ``since=`` reach it, which is what registration bought.

    Before it was a stage of the chain, ``stages=("cell_masks",)`` was a
    ``KeyError`` from a pipeline that had never heard the name.
    """
    from auto_organotypic import pipeline as chain

    assert "cell_masks" in chain.stage_names()
    auto_microglia.run(str(tmp_path / "source"), output_dir=tmp_path / "out",
                       if_exists="error", cells=False, motion=False,
                       stages=("index", "split", "cell_masks"))
    assert fake_chain["options"]["stages"] == ("index", "split", "cell_masks")


def test_a_misspelled_mask_setting_is_refused_against_the_real_function(
        tmp_path):
    """Checked against ``cell_masks.mask_run``'s live signature, not a list.

    The chain does this for its own stages and now does it for this one, which
    means the accepted names follow the function: a setting added to
    ``mask_run`` is accepted here the day it exists, and one misspelled costs a
    message rather than the registration that would have run before the mask
    was reached.
    """
    from auto_organotypic import pipeline as chain

    with pytest.raises(TypeError) as caught:
        chain._stage_options("cell_masks", {"cell_masks_options": {"cutt": 0.5}})
    said = str(caught.value)
    assert "cutt" in said
    assert "cut" in said and "grow_to" in said, "it should list what it takes"

    assert chain._stage_options(
        "cell_masks", {"cell_masks_options": {"threads": 2}}) == {"threads": 2}


def test_the_staleness_grid_can_say_whether_a_re_run_would_mask_again(
        tmp_path):
    """``what_would_run`` answers for this stage instead of saying ``unknown``.

    The question the grid exists for is what a re-run would cost, and the mask
    is the only step in this pipeline where the answer is minutes rather than
    seconds. This package can answer it because the recipe beside each mask
    says what produced it.
    """
    from auto_organotypic import pipeline as chain

    from pymicroglia.pipelines import cell_masks

    class Recorded:
        path = tmp_path / "MCG 04 - 1 - 595.tif"

    masks = tmp_path / "cell_masks"
    masks.mkdir()
    options = {"mask_output_dir": str(masks)}

    verdicts = chain._stage_verdicts("cell_masks", [Recorded()], options,
                                     tmp_path)
    status, why = verdicts[str(Recorded.path)]
    assert status == "stale" and "no mask" in why

    for kind in ("run", "still"):
        (masks / cell_masks.RECIPE_NAME.format(stem=Recorded.path.stem,
                                               kind=kind)).write_text("{}")
    verdicts = chain._stage_verdicts("cell_masks", [Recorded()], options,
                                     tmp_path)
    assert verdicts[str(Recorded.path)][0] == "fresh"


def test_importing_this_module_does_not_change_what_a_plain_chain_run_does(
        tmp_path):
    """The price of registering on import, and the check that it is not paid.

    The stage is opt-in over there, so a run of Auto-Organotypic on a machine
    with PyMicroglia installed performs exactly what it performed before. It is
    asked for by naming its options -- that package's own rule for its exports
    and its review, applied to a stage it did not write.
    """
    from auto_organotypic import pipeline as chain

    assert chain._REGISTERED["cell_masks"]["stage"].opt_in is True, (
        "registered on import and not opt-in would mean every existing "
        "Auto-Organotypic run on this machine quietly gained a stage")
    # That the flag is honoured rather than decorative is that package's rule
    # and is tested there, in ``test_registered_stages.py``. What is checked
    # here is that this package keeps its side of it: the stage declares
    # itself opt-in, and the pipeline asks for it by naming its options.


# ── and a stage somebody else registered ────────────────────────────────────
def _somebody_elses_stage(recordings, *, threshold: int = 1):
    """Stands in for a third package's entry point, so its target resolves."""
    return {"counted": len(list(recordings)), "threshold": threshold}


@pytest.fixture
def table():
    """Register into the chain's table, and put it back afterwards."""
    from auto_organotypic import pipeline as chain

    stages, runners = chain.STAGES, dict(chain._RUNNERS)
    registered = dict(chain._REGISTERED)
    yield chain
    chain.STAGES = stages
    chain._RUNNERS.clear()
    chain._RUNNERS.update(runners)
    chain._REGISTERED.clear()
    chain._REGISTERED.update(registered)


def test_a_third_packages_stage_runs_inside_this_pipeline_and_is_recorded(
        table, tmp_path, fake_chain, fake_mask, fake_measurement):
    """Not just ours: anybody's, through this pipeline, with nothing added here.

    The mask being registered would be a trick if this pipeline only tolerated
    its own. A third package installed beside this one adds a stage the same
    way, its settings reach it through the same pass-through that carries every
    other keyword, and it lands in this package's manifest with its own owner
    on it -- because the manifest is built from the chain's record rather than
    from a list of stage names kept here.
    """
    seen: list[dict] = []

    def count(state, options):
        seen.append(dict(options.get("cell_count_options") or {}))
        return {"counted": len(state["recordings"])}

    table.register_stage(
        table.Stage("cell_count", "SomebodyElse",
                    "tests.test_auto_microglia:_somebody_elses_stage",
                    "count objects in each recording",
                    needs="somebodyelse[count]", opt_in=True),
        count, after="split")

    manifest = auto_microglia.run(str(tmp_path / "source"),
                                  output_dir=tmp_path / "out",
                                  if_exists="error",
                                  cell_count_options={"threshold": 3})

    assert fake_chain["options"]["cell_count_options"] == {"threshold": 3}, (
        "a third package's settings must pass through untouched, like every "
        "other keyword this pipeline does not claim")
    assert seen == [{"threshold": 3}], "they did not reach the stage"

    stages = {row["stage"]: row for row in manifest["stages"]}
    assert stages["cell_count"]["owner"] == "SomebodyElse"
    assert stages["cell_count"]["counted"] == 1
    assert stages["cell_masks"]["owner"] == "PyMicroglia", (
        "and ours is still ours")


def test_a_third_packages_stage_does_not_disturb_the_order_checks(
        table, tmp_path, fake_chain, fake_mask, fake_measurement):
    """The two rules this pipeline enforces name stages; they do not count them.

    A name neither check has heard of is passed over rather than refused, which
    is what lets a stage arrive between the mask and the measurement without
    this package being edited to permit it.
    """
    table.register_stage(
        table.Stage("cell_count", "SomebodyElse",
                    "tests.test_auto_microglia:_somebody_elses_stage",
                    "count objects", needs="somebodyelse[count]", opt_in=True),
        lambda state, options: {"counted": len(state["recordings"])},
        after="cell_masks")

    manifest = auto_microglia.run(str(tmp_path / "source"),
                                  output_dir=tmp_path / "out",
                                  if_exists="error",
                                  cell_count_options={})
    order = [row["stage"] for row in manifest["stages"]]
    assert order.index("cell_masks") < order.index("cell_count") < order.index("cells")
