"""The tracking seam: the contract a tracker satisfies, and the pending state.

Stage 02 of the Motion port (``Auto-Organotypic/docs/consolidation/motion-port/``).
Nothing here tracks anything. What is tested is the shape a tracker hands
back, that the shape survives a round trip through a flat record, that a
finished Motion run folder and a Motion configuration both read onto it with
the right hashes, that the ``track`` action is *pending* rather than broken
while the tracker's dotted name does not resolve, and that the handoff file
now says what it expects back.
"""

from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import pytest

from pymicroglia import knowledge, registry, tracking
from pymicroglia.pipelines import motion_handoff
from pymicroglia.tracking import contract, provenance
from pymicroglia.tracking.contract import DecisionTables, TrackingResult

FIXTURE = Path(__file__).parent / "fixtures" / "motion_parity"
CONFIG = FIXTURE / "config.json"


@pytest.fixture(scope="module")
def movie() -> dict:
    """The one movie the stage-01 fixture configuration names."""
    document = json.loads(CONFIG.read_text(encoding="utf-8"))
    (entry,) = document["movies"]
    return entry


# ── the bit flags ───────────────────────────────────────────────────────────
def test_the_three_provenance_bits_are_the_ones_motion_writes():
    """Moved verbatim from ``Motion/analysis/io.py``; ``Motion/code/pipeline.py``
    keeps its own copy of the same three values, so these numbers are the
    contract between the two halves and may not move."""
    assert provenance.PROVENANCE_INFERRED == 0b001
    assert provenance.PROVENANCE_UNRESOLVED == 0b010
    assert provenance.PROVENANCE_ADDED == 0b100
    assert provenance.FLAGS == {"inferred": 1, "unresolved": 2, "added": 4}
    assert tracking.PROVENANCE_ADDED == provenance.PROVENANCE_ADDED


def test_unpack_splits_a_packed_sidecar_into_three_planes():
    np = pytest.importorskip("numpy")
    packed = np.array([[0b000, 0b001], [0b011, 0b111]], dtype=np.uint8)
    planes = provenance.unpack(packed)
    assert planes["inferred"].tolist() == [[False, True], [True, True]]
    assert planes["unresolved"].tolist() == [[False, False], [True, True]]
    assert planes["added"].tolist() == [[False, False], [False, True]]


# ── the contract ────────────────────────────────────────────────────────────
def test_a_result_is_files_never_arrays(tmp_path):
    labels = tmp_path / "a.tif"
    raw = tmp_path / "raw.tif"
    labels.write_bytes(b"labels")
    raw.write_bytes(b"raw")
    result = TrackingResult(stem="a", labels=str(labels), raw=raw)
    assert isinstance(result.labels, Path)
    assert result.files() == {"labels": labels, "raw": raw}
    record = result.as_dict()
    assert record["labels"] == str(labels)
    assert record["unclaimed"] is None and record["decisions"] is None
    assert all(isinstance(value, (str, int, dict, type(None)))
               for value in record.values())


def test_labels_and_raw_are_required():
    with pytest.raises(ValueError, match="raw"):
        TrackingResult(stem="a", labels=Path("a.tif"), raw=None)


def test_the_record_round_trips_through_from_mapping(tmp_path):
    files = {}
    for role in contract.ROLES:
        files[role] = tmp_path / f"{role}.tif"
        files[role].write_bytes(role.encode())
    decisions = tmp_path / "mid" / "accepted_history"
    decisions.mkdir(parents=True)
    result = TrackingResult(stem="cell", decisions=DecisionTables.motion(decisions),
                            source_frame_offset=2, **files).hashed()
    again = TrackingResult.from_mapping(result.as_dict())
    assert again == result
    assert again.sha256 == result.sha256
    assert set(again.sha256) == set(contract.ROLES)
    assert again.decisions.root == decisions
    assert again.decisions.tables == contract.MOTION_DECISION_TABLES


def test_from_mapping_refuses_a_changed_pinned_input(tmp_path):
    labels = tmp_path / "a.tif"
    raw = tmp_path / "raw.tif"
    labels.write_bytes(b"labels")
    raw.write_bytes(b"raw")
    with pytest.raises(ValueError, match="pinned"):
        TrackingResult.from_mapping({"stem": "a", "labels": labels, "raw": raw,
                                     "sha256": {"labels": "0" * 64}})


def test_the_decision_tables_are_data_on_the_contract():
    """The 14 paths from ``Motion/analysis/modules/history.py`` live here as a
    declaration a different tracker may replace, not as constants in the
    measure step."""
    tables = DecisionTables.motion("/nowhere")
    assert len(contract.MOTION_DECISION_TABLES) == 14
    assert len(tables.tables) == 14
    names = tables.names()
    assert {"gap_evidence", "residency_registry", "merge_events",
            "hierarchy_actions"} <= set(names)
    assert set(names) >= {"stationary_takeover_events", "stationary_component_runs"}
    assert all(rel.endswith(".csv") for _, rel, _ in tables.tables)
    assert all(grain == () for _, _, grain in tables.tables), \
        "a copied table's grain is the tracker's to name, not ours"
    assert tables.path("gap_evidence") == Path("/nowhere") / \
        "21_continuity_evidence/out/gap_evidence.csv"
    assert tables.path("stationary_takeover_events", "/labels") == \
        Path("/labels") / "stationary_takeover_events.csv"
    with pytest.raises(KeyError):
        tables.path("not_a_table")


def test_present_reports_only_what_is_on_disk(tmp_path):
    root = tmp_path / "mid" / "accepted_history"
    one = root / "21_continuity_evidence" / "out" / "gap_evidence.csv"
    one.parent.mkdir(parents=True)
    one.write_text("identity\n", encoding="utf-8")
    found = DecisionTables.motion(root).present()
    assert found == {"gap_evidence": one}


# ── the stage-01 fixture reads onto the contract ───────────────────────────
def test_from_mapping_reads_the_fixture_config_with_matching_hashes(movie):
    """Exit gate 4, by configuration: every file the fixture names, hashed to
    the pins the configuration carries."""
    result = TrackingResult.from_mapping(movie, root=FIXTURE)
    assert result.stem == "parity_A1"
    named = {role: movie[role] for role in contract.ROLES if movie.get(role)}
    assert set(result.files()) == set(named)
    for role, relative in named.items():
        assert result.files()[role] == FIXTURE / relative
        assert result.sha256[role] == movie["sha256"][role], role
    assert result.verify(movie["sha256"]) == {role: True for role in named}
    assert result.decisions is None, "the fixture has no tracker decision folder"


def test_from_folder_reads_a_motion_run_folder_with_matching_hashes(tmp_path, movie):
    """Exit gate 4, by run folder: the fixture's files laid out the way the
    Motion tracker writes them, then read back through the contract."""
    stem = movie["stem"]
    run = tmp_path / "m21_stationary_reconciliation" / "r01"
    for role, relative in contract.MOTION_OUTPUTS.items():
        target = run / relative.format(stem=stem)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FIXTURE / movie[role], target)
    decisions = run / contract.MOTION_DECISIONS_ROOT
    table = decisions / "21_continuity_evidence" / "out" / "gap_evidence.csv"
    table.parent.mkdir(parents=True)
    table.write_text("identity\n", encoding="utf-8")

    result = TrackingResult.from_folder(run, stem, raw=FIXTURE / movie["raw"])

    assert result.labels == run / "out" / f"{stem}.tif"
    assert result.unclaimed == run / "out" / f"{stem}_unclaimed_original_ids.tif"
    assert result.provenance == run / "out" / f"{stem}_provenance.tif"
    assert result.evidence == run / "qc" / f"{stem}_motion_evidence.tif"
    assert result.raw == FIXTURE / movie["raw"]
    assert set(result.files()) == set(contract.ROLES)
    for role in contract.ROLES:
        assert result.sha256[role] == movie["sha256"][role], role
    assert result.decisions.root == decisions
    assert result.decisions.present() == {"gap_evidence": table}


def test_from_folder_says_what_it_needs_when_something_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="out/ holds <stem>.tif"):
        TrackingResult.from_folder(tmp_path, "x", raw=tmp_path / "r.tif")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "x.tif").write_bytes(b"labels")
    with pytest.raises(ValueError, match="raw="):
        TrackingResult.from_folder(tmp_path, "x")
    partial = TrackingResult.from_folder(tmp_path, "x", raw=tmp_path / "r.tif",
                                         hashes=False)
    assert partial.unclaimed is None and partial.evidence is None
    assert partial.decisions is None


# ── pending, visibly ────────────────────────────────────────────────────────
def test_the_seam_reports_pending_and_names_the_dotted_target():
    state, reason = tracking.status()
    assert state == "pending"
    assert tracking.TRACKER_TARGET in reason
    assert motion_handoff.TARGET == tracking.TRACKER_TARGET == "motion.pipeline:run"
    assert motion_handoff.status() == tracking.status()


def test_run_raises_action_pending_naming_the_target(tmp_path):
    from pymicroglia import ActionPending

    with pytest.raises(ActionPending) as caught:
        tracking.run(tmp_path / "motion_inputs.json", tmp_path)
    assert tracking.TRACKER_TARGET in str(caught.value)


def test_the_track_action_is_pending_in_the_registry_and_in_describe():
    """Exit gate 2: ``describe track`` says pending and names the target."""
    assert "track" in registry.pending()
    assert registry.REGISTRY.binds_to("track") == "tracking.run"
    assert registry.REGISTRY.resolve("track") is None
    assert registry.seam_status("tracking.run") == tracking.status()

    payload = knowledge.describe("track")
    assert payload["ok"] is True
    assert payload["pending"] is True
    assert tracking.TRACKER_TARGET in payload["pending_reason"]
    assert payload["claim_required"] is True
    assert {row["name"] for row in payload["params"]} == \
        {"inputs", "folder", "claim", "tracker_options"}

    check = knowledge.validate("track", {"inputs": "x", "folder": "y"})
    assert check["ok"] is True and check["pending"] is True
    assert tracking.TRACKER_TARGET in check["note"]


def test_cli_describe_track_reports_pending(capsys):
    from pymicroglia.cli import main

    assert main(["describe", "track"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pending"] is True
    assert tracking.TRACKER_TARGET in payload["pending_reason"]


def test_an_ordinary_module_is_not_a_seam():
    assert registry.seam_status("cosmic.remove_cosmic_rays") is None
    assert registry.pending_reason("cosmic.remove_cosmic_rays") == ""
    assert registry.pending_reason("nowhere.at_all").startswith("pymicroglia.nowhere")


def test_a_tracker_that_resolves_makes_the_seam_ready(monkeypatch, tmp_path):
    """Swapping the tracker in is changing one dotted name."""
    labels = tmp_path / "out" / "s.tif"
    labels.parent.mkdir()
    labels.write_bytes(b"labels")
    raw = tmp_path / "raw.tif"
    raw.write_bytes(b"raw")
    asked = []

    def fake_run(inputs, folder, **options):
        asked.append((inputs, folder, options))
        return {"stem": "s", "labels": labels, "raw": raw}

    fake = types.ModuleType("fake_tracker")
    fake.run = fake_run
    monkeypatch.setitem(sys.modules, "fake_tracker", fake)
    monkeypatch.setattr(tracking, "TRACKER_TARGET", "fake_tracker:run")

    assert tracking.status() == ("ready", "")
    assert "track" not in registry.pending()
    assert registry.REGISTRY.resolve("track") is tracking.run
    assert knowledge.describe("track")["pending"] is False

    result = tracking.run("inputs.json", tmp_path, tracker_options={"gap": 2})
    assert asked == [("inputs.json", tmp_path, {"gap": 2})]
    assert result.labels == labels and result.raw == raw
    assert len(result.sha256["labels"]) == 64


# ── the handoff says what it expects back ──────────────────────────────────
def test_motion_inputs_carries_what_the_tracker_is_expected_to_write(tmp_path):
    """Exit gate 5: ``expects`` names the five output files and the
    decision-table root, per stem."""
    stack = tmp_path / "well_A1.tif"
    stack.write_bytes(b"registered")
    out = motion_handoff.write([{"path": str(stack)}], {}, tmp_path,
                               dataset="d", hashes=True)
    payload = json.loads(Path(out["outputs"]["motion_inputs"]).read_text(
        encoding="utf-8"))
    # The keys Motion already reads are untouched.
    assert {"dataset", "stems", "input_space", "pinned_files"} <= set(payload)
    expects = payload["expects"]["well_A1"]
    assert set(expects) == {*contract.ROLES, "decisions"}
    assert expects["labels"] == "out/well_A1.tif"
    assert expects["unclaimed"] == "out/well_A1_unclaimed_original_ids.tif"
    assert expects["provenance"] == "out/well_A1_provenance.tif"
    assert expects["evidence"] == "qc/well_A1_motion_evidence.tif"
    assert expects["decisions"] == "mid/accepted_history"
    assert expects["raw"] == payload["pinned_files"]["well_A1"]["registered_raw"]["path"]
    assert expects == contract.expected_files("well_A1", stack)
