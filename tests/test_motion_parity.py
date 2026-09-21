"""The parity net for the Motion port: does a run folder match the frozen record?

``tests/fixtures/motion_parity/expected.json`` is what ``Motion/analysis``
wrote on one synthetic tracked movie (``tests/fixtures.py::tracked_movie``)
at the Motion commit its README names. Every later stage of the port points
``compare_run`` at the folder its ported code produces and expects an empty
list back. At stage 01 the only run folder is the frozen one, so these tests
check the record against itself and against its own promises: every table the
manifest lists is frozen, every figure has a plotted-data hash or a recorded
refusal, every pipeline demo verified, and the fixture is small enough to
commit.

Numbers compare at nine significant figures, the precision Motion writes.
Column order is free; row order is not, because every Motion table is
written sorted on its grain.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

FIXTURE = Path(__file__).parent / "fixtures" / "motion_parity"
EXPECTED = FIXTURE / "expected.json"
FROZEN_RUN = FIXTURE / "run"

SIGNIFICANT_FIGURES = 9
#: A committed fixture has to stay small enough that nobody minds it.
SIZE_LIMIT_BYTES = 20 * 1024 * 1024

#: The six result-dependent workflows, as the plan names them. The state
#: demo is not a pipeline run (no runner, no verification.json) and is frozen
#: under ``extra.state_demo`` beside the ``states`` step's own output.
PIPELINES = ("rhythm-discovery", "method-audit", "measurement-relationships",
             "behaviour-states", "spatial-coordination", "intervention-response")

#: Boolean fields of a demo's ``verification.json`` that describe the design
#: rather than assert a check. Every other boolean must be true, and
#: ``biological_result`` must be false: no synthetic demo claims a finding.
DESIGN_FLAGS = {"biological_result", "synthetic", "deliberately_permissive_policy"}

#: Counted evidence a demo may record instead of a boolean: links that
#: resolved, scientific steps reused, confirmation cases checked. A demo whose
#: proof carries no true boolean must carry at least one of these, non-zero.
EVIDENCE_COUNTS = {"relative_links", "verified_links", "verified_source_artifacts",
                   "scientific_steps_reused", "fresh_confirmation_cases"}


@functools.lru_cache(maxsize=1)
def expected() -> dict:
    return json.loads(EXPECTED.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_number(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _same_value(left: str, right: str) -> bool:
    if left == right:
        return True
    a, b = _as_number(left), _as_number(right)
    if a is None or b is None:
        return False
    if math.isnan(a) and math.isnan(b):
        return True
    if a == b:
        return True
    return f"{a:.{SIGNIFICANT_FIGURES}g}" == f"{b:.{SIGNIFICANT_FIGURES}g}"


def _row_differences(frame: "pd.DataFrame", record: dict, rel: str) -> list[str]:
    """Where a table departs from its frozen head, tail, shape and columns."""
    problems = []
    if len(frame) != record["rows"]:
        problems.append(f"{rel}: {len(frame)} rows, frozen {record['rows']}")
    missing = sorted(set(record["columns"]) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(record["columns"]))
    if missing:
        problems.append(f"{rel}: missing columns {missing}")
    if extra:
        problems.append(f"{rel}: unexpected columns {extra}")
    if problems:
        return problems
    # Pipeline tables are frozen by hash and shape alone (no samples), so a
    # changed hash there is reported by its shape and columns only.
    head, tail = record.get("head", []), record.get("tail", [])
    if not head and not tail and record["rows"]:
        problems.append(f"{rel}: bytes differ from the frozen table (no row sample frozen)")
    samples = list(enumerate(head))
    tail_start = len(frame) - len(tail)
    samples += [(tail_start + i, row) for i, row in enumerate(tail)]
    for index, frozen in samples:
        if index < 0 or index >= len(frame):
            continue
        actual = frame.iloc[index]
        for column, value in frozen.items():
            if not _same_value(str(actual[column]), str(value)):
                problems.append(
                    f"{rel} row {index} {column}: {actual[column]!r} != frozen {value!r}")
    return problems


def _frame_differences(actual: "pd.DataFrame", frozen: "pd.DataFrame", rel: str) -> list[str]:
    """Every cell of a whole table against its frozen copy, at 9 s.f."""
    problems = []
    if list(sorted(actual.columns)) != list(sorted(frozen.columns)):
        return [f"{rel}: columns differ"]
    if len(actual) != len(frozen):
        return [f"{rel}: {len(actual)} rows, frozen {len(frozen)}"]
    for column in frozen.columns:
        left = actual[column].astype(str).to_numpy()
        right = frozen[column].astype(str).to_numpy()
        for index in (left != right).nonzero()[0]:
            if not _same_value(left[index], right[index]):
                problems.append(
                    f"{rel} row {index} {column}: {left[index]!r} != frozen {right[index]!r}")
                if len(problems) > 50:
                    return problems
    return problems


def read_as_written(path: Path) -> "pd.DataFrame":
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def compare_run(run: Path, *, section: str = "measure",
                ignore: tuple[str, ...] = ()) -> list[str]:
    """Every difference between a run folder and the frozen record; empty means parity.

    ``section`` is ``measure`` for the measurement run, a pipeline name for a
    demo folder, or ``extra:<name>`` for the states or cluster output. A table
    whose bytes match is accepted at once; otherwise its whole frozen copy is
    compared when the fixture holds one, and its head and tail when not.
    ``ignore`` lists relative-path prefixes to leave out; a ported run should
    pass with none.
    """
    document = expected()
    if section.startswith("extra:"):
        tables = document["extra"][section.partition(":")[2]]["tables"]
    elif section == "measure":
        tables = document["measure"]["tables"]
    else:
        tables = document["pipelines"][section]["tables"]
    problems = []
    for rel, record in tables.items():
        if rel.startswith(ignore):
            continue
        path = run / rel
        if not path.exists():
            problems.append(f"missing {rel}")
            continue
        if sha256(path) == record["sha256"]:
            continue
        frame = read_as_written(path)
        frozen_copy = FROZEN_RUN / rel if section == "measure" else None
        if frozen_copy is not None and frozen_copy.exists():
            problems.extend(_frame_differences(frame, read_as_written(frozen_copy), rel))
        else:
            problems.extend(_row_differences(frame, record, rel))
    return problems


# --------------------------------------------------------------------------
# The record against itself and its own promises
# --------------------------------------------------------------------------

def test_frozen_run_matches_its_record():
    """The committed tables are the ones the hashes describe.

    The pooled tables are not copied into the fixture: with one movie each is
    byte-identical to its per-movie table, which the second assertion proves
    from the hashes, so nothing is lost by leaving them out.
    """
    assert compare_run(FROZEN_RUN, section="measure", ignore=("pooled/tables/",)) == []
    tables = expected()["measure"]["tables"]
    pooled = {rel for rel in tables if rel.startswith("pooled/tables/")}
    assert pooled
    for rel in pooled:
        twin = "parity_A1/tables/" + rel.rpartition("/")[2]
        assert tables[rel]["sha256"] == tables[twin]["sha256"], rel


def test_record_lists_every_table_the_manifest_names():
    """Exit gate 2: the manifest's table entries and the frozen tables agree."""
    document = expected()["measure"]
    manifest = document["manifest"]
    named = set()
    for movie in manifest["movies"]:
        for name, entry in movie["tables"].items():
            named.add(f"{movie['stem']}/{entry['folder']}/{entry['path']}")
    for name, entry in manifest["pooled"]["tables"].items():
        named.add(f"pooled/{entry['folder']}/{entry['path']}")
    if manifest.get("statistics", {}).get("path"):
        named.add(manifest["statistics"]["path"])
    frozen = set(document["tables"])
    assert named <= frozen, sorted(named - frozen)
    assert len(named) == document["manifest_table_count"]
    # The manifest's own hashes are the frozen ones.
    for movie in manifest["movies"]:
        for entry in movie["tables"].values():
            rel = f"{movie['stem']}/{entry['folder']}/{entry['path']}"
            assert document["tables"][rel]["sha256"] == entry["sha256"], rel


def test_every_table_record_is_complete():
    for rel, record in expected()["measure"]["tables"].items():
        assert set(record) >= {"sha256", "rows", "columns", "head", "tail"}, rel
        assert len(record["sha256"]) == 64, rel
        assert len(record["head"]) == min(3, record["rows"]), rel
        assert len(record["tail"]) == min(3, record["rows"]), rel
        for row in (*record["head"], *record["tail"]):
            assert sorted(row) == sorted(record["columns"]), rel


def test_a_changed_number_is_reported(tmp_path):
    """The comparison notices one digit moving in one cell."""
    document = expected()["measure"]["tables"]
    rel = "parity_A1/tables/cell_summary.csv"
    source = FROZEN_RUN / rel
    target = tmp_path / rel
    target.parent.mkdir(parents=True)
    frame = read_as_written(source)
    column = next(c for c in frame.columns if _as_number(frame.iloc[0][c]) not in (None, 0.0))
    original = float(frame.iloc[0][column])
    frame.loc[0, column] = f"{original * (1 + 1e-6):.9g}"
    frame.to_csv(target, index=False)
    problems = compare_run(tmp_path)
    assert any(rel in p and column in p for p in problems), problems
    assert len(problems) <= len(document), "only the edited cell should be reported"


def test_a_rewritten_table_with_the_same_numbers_passes(tmp_path):
    """Re-serialising a table (new column order, new float spelling) is parity."""
    rel = "parity_A1/tables/cell_summary.csv"
    frame = read_as_written(FROZEN_RUN / rel)
    target = tmp_path / rel
    target.parent.mkdir(parents=True)
    frame[list(reversed(frame.columns))].to_csv(target, index=False)
    problems = [p for p in compare_run(tmp_path) if rel in p]
    assert problems == []


def test_counts():
    counts = expected()["counts"]
    figures = expected()["figures"]
    assert counts["figures"] == 83 == len(figures)
    assert counts["review"] == 12 == sum(1 for f in figures.values() if f["review"])
    assert counts["results"] == 71 == sum(1 for f in figures.values() if not f["review"])
    assert counts["modules"] == 24 == len(expected()["measure"]["manifest"]["registered_modules"])
    assert counts["measured_tables"] == len(expected()["measure"]["tables"])


def test_every_figure_has_a_plotted_hash_or_a_recorded_refusal():
    """Exit gate 3, in the form the synthetic movie allows.

    A figure that declined this movie is recorded with its refusal text; that
    refusal is the expected outcome, and a port that draws it instead has
    changed behaviour just as surely as one that refuses a figure that drew.
    """
    for slug, figure in expected()["figures"].items():
        if figure["returncode"] == 0:
            assert figure.get("plotted_sha256"), f"{slug} drew but no plotted CSV was hashed"
            for rel, digest in figure["plotted_sha256"].items():
                assert len(digest) == 64, (slug, rel)
        else:
            assert figure.get("refusal"), f"{slug} failed without a recorded reason"


def _proofs(verification) -> list[dict]:
    """A demo writes one proof, or one per case; either way a list of dicts."""
    if isinstance(verification, dict):
        return [verification]
    return [item for item in verification if isinstance(item, dict)]


def test_every_pipeline_verified_and_every_step_recorded():
    """Exit gate 4: the demos ran, verified, and every step finished."""
    pipelines = expected()["pipelines"]
    assert set(pipelines) == set(PIPELINES), sorted(pipelines)
    for name, pipeline in pipelines.items():
        assert pipeline["verification"] is not None, f"{name}: no verification.json"
        assert pipeline["steps"], f"{name}: no execution records"
        for step, entries in pipeline["steps"].items():
            for entry in entries:
                assert entry["status"] in {"completed", "reused", "skipped-empty"}, (
                    name, step, entry["status"], entry["reason"])
                for artefact, record in entry["artifacts"].items():
                    assert record["sha256"], (name, step, artefact)
                    assert record["matches_record"] in (True, None), (name, step, artefact)
        proofs = _proofs(pipeline["verification"])
        assert proofs, name
        checks = 0
        for proof in proofs:
            for key, value in proof.items():
                if isinstance(value, bool) and key not in DESIGN_FLAGS:
                    assert value is True, (name, key)
                    checks += 1
                elif key in EVIDENCE_COUNTS:
                    assert value, (name, key)
                    checks += 1
        assert checks, f"{name}: verification.json asserts nothing"


def test_no_demo_claims_a_biological_result():
    for name, pipeline in expected()["pipelines"].items():
        for proof in _proofs(pipeline["verification"]):
            if "biological_result" in proof:
                assert proof["biological_result"] is False, name


def test_state_demo_and_states_step_are_frozen():
    extra = expected()["extra"]
    for name in ("states", "state_demo"):
        assert extra[name]["tables"], name
        assert "frame_states.csv" in extra[name]["tables"], name


def test_fixture_is_small_enough_to_commit():
    """Exit gate 5."""
    total = sum(p.stat().st_size for p in FIXTURE.rglob("*") if p.is_file())
    assert total < SIZE_LIMIT_BYTES, f"{total / 1e6:.1f} MB"


def test_readme_names_the_motion_commit_and_workbench_version():
    """Exit gate 6."""
    text = (FIXTURE / "README.md").read_text(encoding="utf-8")
    environment = expected()["environment"]
    assert environment["motion_commit"] in text
    assert environment["circadian_workbench"] in text
    assert len(environment["motion_commit"]) == 40


def test_inputs_are_the_ones_the_config_pins():
    """The synthetic stacks on disk are the ones the record was made from."""
    config = json.loads((FIXTURE / "config.json").read_text(encoding="utf-8"))
    movie = config["movies"][0]
    for role, digest in movie["sha256"].items():
        assert sha256(FIXTURE / movie[role]) == digest, role
    for block in ("channels", "objects", "side_tables"):
        for entry in movie[block]:
            assert sha256(FIXTURE / entry["path"]) == entry["sha256"], (block, entry["name"])


def test_builder_reproduces_the_inputs(tmp_path):
    """``tracked_movie`` is deterministic, so the fixture can be rebuilt anywhere."""
    from fixtures import tracked_movie

    written = tracked_movie(tmp_path)
    for role, path in written.items():
        assert sha256(path) == sha256(FIXTURE / "inputs" / path.name), role
