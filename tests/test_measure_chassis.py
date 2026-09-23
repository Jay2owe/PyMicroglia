"""The measure chassis end to end, over the stage-01 fixture with stand-ins.

Stage 03's exit gate, as tests: the action describes itself with every
parameter, validates the fixture's ``movies`` block verbatim, writes the
declared layout with one ledger per folder, and every table it wrote can be
found again through the store by the same key it was stored under. The
science is stage 04's; here a stand-in module measures pixel counts so that
the join, the folds, the roll-ups, the windows and the pooling have
something to carry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pymicroglia import knowledge, registry, store
from pymicroglia.measure import declare
from pymicroglia.measure.run import (MEASURE_FOLDER, METHOD_VERSION,
                                     STACKS_FOLDER, TRACKER_FOLDER,
                                     WINDOWS_FOLDER, read_manifest)

from measure_stubs import FIXTURE, STUB_NAMES, fixture_copy, measured, registered

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

STEM = "parity_A1"


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.setenv("PYMICROGLIA_INDEX", str(tmp_path / "index"))
    return tmp_path


@pytest.fixture
def run(isolated):
    folder, manifest, config = measured(isolated)
    return folder, manifest, config


# ------------------------------------------------------ gate 2: describe

def test_describe_lists_every_parameter_with_units_and_a_live_default() -> None:
    """The catalogue entry and the bound function agree, parameter for parameter."""
    import inspect

    from pymicroglia.measure import measure

    described = knowledge.describe("measure")
    assert described["ok"] and not described["pending"]
    rows = {row["name"]: row for row in described["params"]}
    signature = inspect.signature(measure)
    assert set(rows) == set(signature.parameters)
    for name, parameter in signature.parameters.items():
        row = rows[name]
        assert row["units"], name
        assert row["description"].strip(), name
        if parameter.default is not inspect.Parameter.empty:
            assert row["default"] == parameter.default, name
        if row["required"]:
            assert row["default"] is None, name
    # Movies and interval can be supplied together through analysis_config.
    # The action checks these conditional requirements before reading inputs.
    assert {n for n, r in rows.items() if r["required"]} == {"output_dir"}
    assert rows["if_exists"]["default"] == "version"
    # Mechanical: the claim is filled from a template unless the caller writes one.
    assert described["claim_required"] is False
    assert described["claim_template"] == "measured every configured module over {source}"
    assert described["method_version"] == METHOD_VERSION


def test_the_three_follow_on_actions_are_described_too() -> None:
    for action in ("pool", "window", "contrasts"):
        described = knowledge.describe(action)
        assert described["ok"], action
        assert {row["name"] for row in described["params"]} >= {"run_dir"}, action
    assert knowledge.describe("contrasts")["pending"] is False


def test_validate_accepts_the_fixture_movies_block_verbatim() -> None:
    """The stage-01 ``movies`` block, exactly as the file has it."""
    raw = json.loads((FIXTURE / "config.json").read_text(encoding="utf-8"))
    result = knowledge.validate("measure", {"movies": raw["movies"],
                                            "output_dir": "outputs", "claim": "x"})
    assert result["ok"] is True, result
    assert result["unknown_params"] == []
    assert result["pending"] is False


def test_load_config_reads_the_fixture_movies_block_verbatim(tmp_path) -> None:
    """``load_config`` on the copied file yields the movie the block describes."""
    from pymicroglia.measure import load_config

    config = load_config(fixture_copy(tmp_path))
    raw = json.loads((FIXTURE / "config.json").read_text(encoding="utf-8"))["movies"][0]
    (movie,) = config.movies
    assert movie.stem == raw["stem"]
    assert movie.labels == (tmp_path / "parity" / raw["labels"])
    assert [c.name for c in movie.channels] == [c["name"] for c in raw["channels"]]
    assert [o.name for o in movie.objects] == [o["name"] for o in raw["objects"]]
    assert [s.name for s in movie.side_tables] == [s["name"] for s in raw["side_tables"]]
    assert movie.expected_sha256 == raw["sha256"]
    assert [w.name for w in config.windows] == ["first_half", "second_half"]
    assert [c.name for c in config.contrasts] == ["second_half_against_first"]


def test_the_registry_knows_the_four_actions() -> None:
    assert {"measure", "measure.pool", "measure.windows", "measure.contrasts"} \
        <= set(registry.MODULE_NAMES)
    assert registry.claim_for("measure", params={"movies": [{"stem": "a"}, {"stem": "b"}]}) \
        == ("measured every configured module over a and 1 more movie(s)", False)
    assert registry.claim_for("pool", params={"run_dir": "outputs/measure/x"}) \
        == ("pooled every movie's tables in x", False)
    assert registry.claim_for("contrasts") == ("", True)


# ---------------------------------------------------------- gate 3: layout

def test_the_run_writes_the_declared_layout(run) -> None:
    folder, manifest, _ = run
    assert folder == Path(manifest["run"]["folder"])
    assert folder.parent.name == "outputs" and folder.name == "chassis"

    measure_dir = folder / MEASURE_FOLDER / STEM
    assert {p.name for p in measure_dir.glob("*.csv")} == {
        "cell_frame.csv", "cell_summary.csv", "frame_summary.csv",
        "stub_frames.csv", "stub_channels.csv"}
    assert {p.name for p in (folder / TRACKER_FOLDER / STEM).glob("*.csv")} == {
        "history_stub_copy.csv"}
    assert {p.name for p in (folder / STACKS_FOLDER / STEM).iterdir()
            if p.is_file()} == {"stub_stack.npz"}
    assert {p.name for p in (folder / WINDOWS_FOLDER / STEM).glob("*.csv")} == {
        "cell_summary_windowed.csv", "frame_summary_windowed.csv", "window_change.csv"}
    assert not (folder / "pooled").exists()


def test_the_manifest_records_the_run_in_full(run) -> None:
    folder, manifest, _ = run
    assert read_manifest(folder) == manifest
    assert manifest["run"]["method_version"] == METHOD_VERSION
    assert manifest["run"]["claim"] == "measured the fixture with stand-ins"
    assert manifest["settings"]["frame_interval_min"] == 30.0
    assert manifest["design"]["synthetic"] is True
    assert manifest["pooled"] is None and manifest["statistics"] is None
    assert {m["name"] for m in manifest["registered_modules"]} == set(STUB_NAMES)

    (movie,) = manifest["movies"]
    assert movie["stem"] == STEM
    assert movie["condition"]["condition"] == "control"
    assert movie["modules_unregistered"] == []
    done = {m["module"]: m for m in movie["modules"] if m["status"] == "done"}
    assert set(done) == set(STUB_NAMES)
    assert done["stub_area"]["parameters"] == {"scale": 1.0}
    assert [w["frames"] for w in movie["windows"]] == [24, 24]
    assert movie["summary"]["identities"] == 3
    for name, record in movie["tables"].items():
        assert record["rows"] > 0, name
        assert record["sha256"] and record["digest"], name
    assert movie["tables"]["history_stub_copy"]["origin"] == "tracker"
    assert movie["tables"]["cell_frame"]["grain"] == ["identity", "frame_index"]
    assert movie["provenance"]["inputs"]["labels"]["verified"] is True


def test_every_written_table_resolves_through_the_store(run) -> None:
    """The key it was stored under is the key it is found by."""
    folder, manifest, config = run
    (movie,) = manifest["movies"]
    labels = config.movie(STEM).labels
    for name, record in movie["tables"].items():
        found = store.resolve("measure", labels,
                              params={"run": "chassis", "stem": STEM, "table": name})
        assert found.kind == "table", name
        assert found.path == folder / record["folder"] / record["path"] \
            or found.path.name == record["path"], name
        assert found.path.is_file(), name
        loaded = found.load()
        assert set(loaded) >= {"stem", "condition", "subject"}, name
    for name in movie["stacks"]:
        found = store.resolve("measure", labels,
                              params={"run": "chassis", "stem": STEM, "stack": name})
        assert found.kind == "mask", name


def test_a_table_is_written_to_nine_significant_figures(run) -> None:
    import pandas as pd

    folder, _, _ = run
    frame = pd.read_csv(folder / MEASURE_FOLDER / STEM / "cell_frame.csv")
    assert list(frame.columns[:6]) == ["stem", "condition", "subject", "identity",
                                       "frame_index", "imagej_frame"]
    text = (folder / MEASURE_FOLDER / STEM / "cell_frame.csv").read_text(encoding="utf-8")
    longest = max((len(cell.replace("-", "").replace(".", "").lstrip("0"))
                   for line in text.splitlines()[1:]
                   for cell in line.split(",") if cell.replace("-", "").replace(".", "").isdigit()),
                  default=0)
    assert longest <= 9


def test_a_stand_in_option_reaches_the_module_and_the_manifest(isolated) -> None:
    import pandas as pd

    folder, manifest, _ = measured(isolated, module_options={"stub_area": {"scale": 2.0}})
    (movie,) = manifest["movies"]
    done = {m["module"]: m for m in movie["modules"]}
    assert done["stub_area"]["parameters"] == {"scale": 2.0}
    frame = pd.read_csv(folder / MEASURE_FOLDER / STEM / "cell_frame.csv")
    assert frame["area_px"].iloc[0] == 128.0       # 64 pixels, twice


def test_an_option_no_module_declares_is_refused_before_anything_runs(isolated) -> None:
    with pytest.raises(ValueError, match="does not declare"):
        measured(isolated, module_options={"stub_area": {"scal": 2.0}})
    with pytest.raises(ValueError, match="not a registered module"):
        measured(isolated, module_options={"morphology": {"min_area": 2}})
    assert not (isolated / "outputs").exists()


def test_a_requested_module_nobody_registered_is_recorded_not_run(isolated) -> None:
    _, manifest, _ = measured(isolated, modules=("stub_area", "morphology"))
    (movie,) = manifest["movies"]
    assert movie["modules_unregistered"] == ["morphology"]
    assert [m["module"] for m in movie["modules"]] == ["stub_area"]


def test_if_exists_governs_a_second_run_with_the_same_label(isolated) -> None:
    folder, first, _ = measured(isolated)
    with pytest.raises(FileExistsError, match="already exists"):
        measured(isolated / "again", if_exists="error",
                 output_dir=isolated / "outputs")
    skipped_folder, skipped, _ = measured(isolated / "skip", if_exists="skip",
                                          output_dir=isolated / "outputs")
    assert skipped_folder == folder and skipped == first
    versioned, _, _ = measured(isolated / "version", output_dir=isolated / "outputs")
    assert versioned == folder.parent / "chassis_v2"


def test_a_design_that_cannot_be_resolved_stops_the_run(isolated) -> None:
    from pymicroglia.measure.conditions import ConditionSet

    with pytest.raises(ValueError, match="design cannot be resolved"):
        measured(isolated, conditions=ConditionSet.from_config({"treated": r"_Z\d"}))


def test_a_changed_input_is_refused_by_name(isolated) -> None:
    path = fixture_copy(isolated / "changed")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["movies"][0]["sha256"]["labels"] = "0" * 64
    path.write_text(json.dumps(data), encoding="utf-8")
    from pymicroglia.measure import load_config, measure

    config = load_config(path)
    with registered():
        with pytest.raises(ValueError, match="labels"):
            measure(config.movies, output_dir=isolated / "outputs", run_label="x",
                    enabled_modules=list(STUB_NAMES), frame_interval_min=30, claim="x")


# ---------------------------------------------------------- pool and window

def test_pool_writes_the_pooled_folder_and_the_manifest_section(run) -> None:
    from pymicroglia.measure.pool import pool

    folder, _, _ = run
    fragment = pool(folder)
    assert fragment["movies"] == [STEM]
    assert set(fragment["tables"]) == {
        "cell_frame", "cell_summary", "frame_summary", "stub_frames",
        "stub_channels", "history_stub_copy", "cell_summary_windowed",
        "frame_summary_windowed", "window_change"}
    assert fragment["tables"]["history_stub_copy"]["origin_folder"] == TRACKER_FOLDER
    assert fragment["tables"]["window_change"]["origin_folder"] == WINDOWS_FOLDER
    assert (folder / "pooled" / "cell_frame.csv").is_file()
    assert read_manifest(folder)["pooled"] == fragment
    with pytest.raises(FileExistsError, match="immutable"):
        pool(folder)


def test_window_refuses_a_run_that_already_has_windows(run) -> None:
    from pymicroglia.measure.windows import window

    folder, _, _ = run
    with registered():
        with pytest.raises(FileExistsError, match="immutable"):
            window(folder)


def test_window_rolls_up_a_run_measured_without_windows(isolated) -> None:
    import pandas as pd

    from pymicroglia.measure.windows import window

    folder, manifest, _ = measured(isolated, windows=())
    assert not (folder / WINDOWS_FOLDER).exists()
    with registered():
        result = window(folder, windows=[
            {"name": "early", "from_frame": 0, "to_frame": 10},
            {"name": "late", "from_frame": 10, "to_frame": 48, "baseline": "early"}])
    assert [w["name"] for w in result["windows"]] == ["early", "late"]
    windowed = pd.read_csv(folder / WINDOWS_FOLDER / STEM / "cell_summary_windowed.csv")
    assert set(windowed["window"]) == {"early", "late"}
    assert windowed.loc[windowed["window"] == "early", "observed_frames"].max() == 10
    (movie,) = read_manifest(folder)["movies"]
    assert [w["frames"] for w in movie["windows"]] == [10, 38]
    assert "window_change" in movie["tables"]


def test_the_stand_ins_leave_no_trace_after_a_run(run) -> None:
    """The stand-ins are gone and the real modules are back where they were."""
    from pymicroglia.measure.modules import MODULE_NAMES

    names = {m.name for m in (*declare.list_modules(), *declare.list_derived())}
    assert not names & set(STUB_NAMES)
    assert names == set(MODULE_NAMES)


def test_measure_requires_movies_or_a_configuration_before_writing(tmp_path):
    from pymicroglia.measure import measure
    with pytest.raises(ValueError, match="movies or analysis_config"):
        measure(output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_measure_requires_an_interval_without_a_configuration(tmp_path):
    from pymicroglia.measure import measure
    with pytest.raises(ValueError, match="frame_interval_min is required"):
        measure(movies=[], output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()



def test_description_exposes_nested_rhythm_choices_and_independent_defaults():
    rows = {r["name"]: r for r in knowledge.describe("measure")["params"]}
    settings = rows["module_options"]["properties"]["rhythms"]
    assert settings["period_search_hours"]["default"] == [2., 48.]
    assert settings["min_observations"]["default"] == 24
    assert settings["min_cycles_for_confident_period"]["default"] == 3.
    assert "fft_nlls" in settings["period_estimation_method"]["choices"]
    assert "cosinor" not in settings["primary_rhythm_test"]["choices"]
    assert settings["daily_profile_measures"]["default"] is False
