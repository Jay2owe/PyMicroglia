"""What makes a stored artefact reusable, and what must make it miss.

The store's whole promise is that a hit is indistinguishable from a cold run.
These tests are that promise written down: the things that must miss, the things
that must not, and the requirement that a miss explains itself rather than
saying "cache miss" and costing somebody six hours of re-registration.
"""

from __future__ import annotations

import os
import shutil

import pytest

from pymicroglia import store
from pymicroglia.store import keys, manifest, tier_a


METHOD = "2026-08-16-matched-line-neighbour-blend"

SHIFTS = {"frame": [0, 1, 2], "dy": [0.0, -1.5, -2.25], "dx": [0.0, 0.5, 1.0]}
PARAMS = {"reference_channel": 3, "downsample": 4, "margin_px": 2}


@pytest.fixture
def store_root(tmp_path, monkeypatch):
    """A disposable store, so no test can see another's index."""
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.delenv("PYMICROGLIA_DECISIONS", raising=False)
    return tmp_path / "cache"


@pytest.fixture
def source(tmp_path):
    """A stand-in for a time-lapse: bytes on disk with a .tif name.

    Nothing here reads pixels, so a real stack would only make the suite slow.
    """
    path = tmp_path / "raw" / "VID52_C1_timestack.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(range(256)) * 40)
    return path


@pytest.fixture
def run_folder(tmp_path):
    return tmp_path / "raw" / "AI_Exports" / "registered_2026-08-18"


def store_shifts(source, run_folder, params=None, method=METHOD, **kwargs):
    return store.put("registration", source, params or PARAMS,
                     kind="table", value=SHIFTS,
                     name="registration_shifts_and_qc",
                     output_dir=run_folder, method_version=method, **kwargs)


# ------------------------------------------------------------------- hits
def test_an_identical_rerun_hits(store_root, source, run_folder):
    written = store_shifts(source, run_folder)
    hit = store.get("registration", source, PARAMS, method_version=METHOD)

    assert hit is not None
    assert hit.path == written.path
    assert hit.load() == SHIFTS


def test_a_table_comes_back_in_the_types_it_went_in_as(store_root, source,
                                                       run_folder):
    """A CSV is text. Without the recorded column types a hit would hand back
    strings where a cold run handed back floats, and the two would not be
    interchangeable."""
    store_shifts(source, run_folder)
    back = store.load("registration", source, PARAMS, method_version=METHOD)

    assert back["frame"] == [0, 1, 2]
    assert back["dy"] == [0.0, -1.5, -2.25]
    assert all(isinstance(v, float) for v in back["dx"])


def test_moving_the_source_to_another_folder_still_hits(store_root, source,
                                                        run_folder, tmp_path):
    """Dropbox reorganises folders. Identity is the content, not the path."""
    store_shifts(source, run_folder)
    moved = tmp_path / "reorganised" / "elsewhere" / source.name
    moved.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(moved))

    assert store.get("registration", moved, PARAMS, method_version=METHOD) is not None


def test_touching_the_source_still_hits(store_root, source, run_folder):
    """A Dropbox re-sync rewrites a file without changing a byte of it. That
    must cost one 24 MB re-fingerprint, not a re-registration of 10.8 GB."""
    first = store.fingerprint(source)
    store_shifts(source, run_folder)

    os.utime(source, ns=(first.mtime_ns + 10_000_000_000,
                         first.mtime_ns + 10_000_000_000))
    again = store.fingerprint(source)

    assert again.mtime_ns != first.mtime_ns
    assert again.sample == first.sample
    assert store.get("registration", source, PARAMS, method_version=METHOD) is not None


def test_neither_path_nor_mtime_reaches_the_digest(source):
    one = keys.SourceId(size=10, sample="abc", path="A:/one.tif", mtime_ns=1)
    two = keys.SourceId(size=10, sample="abc", path="Z:/other/two.tif",
                        mtime_ns=999)
    key_one = keys.Key("registration", one, PARAMS, METHOD)
    key_two = keys.Key("registration", two, PARAMS, METHOD)

    assert key_one.digest() == key_two.digest()


def test_parameter_order_does_not_change_the_digest(source):
    sid = keys.SourceId(size=10, sample="abc")
    forwards = {"a": 1, "b": {"y": 2, "x": 1}, "c": {3, 1, 2}}
    backwards = {"c": {2, 3, 1}, "b": {"x": 1, "y": 2}, "a": 1}

    assert (keys.Key("s", sid, forwards).digest()
            == keys.Key("s", sid, backwards).digest())


def test_a_list_keeps_its_order_because_the_order_can_be_meaningful():
    sid = keys.SourceId(size=10, sample="abc")
    assert (keys.Key("s", sid, {"channels": [1, 2]}).digest()
            != keys.Key("s", sid, {"channels": [2, 1]}).digest())


# ------------------------------------------------------------------ misses
def test_changing_one_parameter_misses_and_names_it(store_root, source,
                                                    run_folder):
    store_shifts(source, run_folder, params={**PARAMS, "margin_px": 2})
    changed = {**PARAMS, "margin_px": 8}

    assert store.get("registration", source, changed, method_version=METHOD) is None

    reason = store.explain("registration", source, changed, method_version=METHOD)
    assert "margin_px" in reason
    assert "stored 2" in reason and "requested 8" in reason
    assert "METHOD_VERSION" in reason        # says what did match, too


def test_changing_method_version_misses(store_root, source, run_folder):
    store_shifts(source, run_folder)

    assert store.get("registration", source, PARAMS,
                     method_version="2026-09-01-new-blend") is None
    reason = store.explain("registration", source, PARAMS,
                           method_version="2026-09-01-new-blend")
    assert "METHOD_VERSION" in reason


def test_changing_the_source_content_misses(store_root, source, run_folder):
    store_shifts(source, run_folder)
    source.write_bytes(b"a different acquisition entirely" * 100)

    assert store.get("registration", source, PARAMS, method_version=METHOD) is None


def test_a_change_inside_a_sampled_window_is_seen(tmp_path):
    """The fingerprint reads the first, middle and last window and nothing
    between them. That is the trade being made, so it is tested as a trade."""
    path = tmp_path / "big.bin"
    path.write_bytes(bytes(200))
    before = keys.fingerprint(path, window=16)

    data = bytearray(path.read_bytes())
    data[2] = 99                                   # inside the first window
    path.write_bytes(bytes(data))
    assert keys.fingerprint(path, window=16).sample != before.sample

    data = bytearray(bytes(200))
    data[95] = 99                                  # inside the middle window
    path.write_bytes(bytes(data))
    assert keys.fingerprint(path, window=16).sample != before.sample

    data = bytearray(bytes(200))
    data[30] = 99                                  # between two windows
    path.write_bytes(bytes(data))
    assert keys.fingerprint(path, window=16).sample == before.sample


def test_a_small_file_is_hashed_whole(tmp_path):
    path = tmp_path / "small.bin"
    path.write_bytes(bytes(30))
    before = keys.fingerprint(path, window=16)
    path.write_bytes(bytes(15) + b"\x01" + bytes(14))

    assert keys.fingerprint(path, window=16).sample != before.sample


def test_a_miss_with_nothing_stored_says_so(store_root, source):
    reason = store.explain("registration", source, PARAMS, method_version=METHOD)
    assert "Nothing is stored" in reason


# -------------------------------------------------------------- resolution
def test_resolution_finds_the_single_match_without_being_told_the_parameters(
        store_root, source, run_folder):
    """The divergence from the copied script: an upstream artefact resolves
    from the key, so nobody has to remember which dated folder held it."""
    written = store_shifts(source, run_folder)
    found = store.resolve("registration", source)

    assert found.path == written.path


def test_two_matches_raise_and_name_both_with_what_differs(store_root, source,
                                                           tmp_path):
    first = tmp_path / "raw" / "AI_Exports" / "registered_v2"
    second = tmp_path / "raw" / "AI_Exports" / "registered_v3"
    store_shifts(source, first, params={**PARAMS, "downsample": 4})
    store_shifts(source, second, params={**PARAMS, "downsample": 2})

    with pytest.raises(store.AmbiguousArtefact) as raised:
        store.resolve("registration", source)

    message = str(raised.value)
    assert "registered_v2" in message and "registered_v3" in message
    assert "downsample" in message
    assert "4" in message and "2" in message


def test_a_narrower_request_resolves_where_the_broad_one_was_ambiguous(
        store_root, source, tmp_path):
    store_shifts(source, tmp_path / "a", params={**PARAMS, "downsample": 4})
    store_shifts(source, tmp_path / "b", params={**PARAMS, "downsample": 2})

    found = store.resolve("registration", source, params={"downsample": 2})
    assert "b" in str(found.path.parent)


def test_zero_matches_raises_and_explains_what_was_looked_for(store_root,
                                                              source):
    with pytest.raises(store.ArtefactMissing) as raised:
        store.resolve("registration", source)
    assert "registration" in str(raised.value)
    assert "pymicroglia scan" in str(raised.value)


def test_zero_matches_is_allowed_when_the_caller_says_it_is(store_root, source):
    assert store.resolve("registration", source, required=False) is None


def test_an_explicit_path_wins_over_resolution(store_root, source, tmp_path):
    store_shifts(source, tmp_path / "a", params={**PARAMS, "downsample": 4})
    store_shifts(source, tmp_path / "b", params={**PARAMS, "downsample": 2})
    named = tmp_path / "a" / "registration_shifts_and_qc.csv"

    found = store.resolve("registration", source, explicit=named)
    assert found.path == named


def test_an_explicit_path_that_does_not_exist_is_an_error(store_root, source,
                                                          tmp_path):
    with pytest.raises(store.ArtefactMissing):
        store.resolve("registration", source, explicit=tmp_path / "nope.csv")


def test_display_only_output_is_not_resolved_into_a_measurement(store_root,
                                                                source,
                                                                run_folder):
    """The house rule, enforced by the store rather than by remembering it."""
    written = store.put("display", source, {"gamma": 0.6}, kind="table",
                        value=SHIFTS, name="display_curve",
                        output_dir=run_folder, method_version=METHOD,
                        display_only=True)

    assert "_DISPLAY_ONLY" in written.path.name
    assert store.resolve("display", source, required=False) is None
    assert store.resolve("display", source, display_only=True) is not None


# --------------------------------------------------------------- decisions
def test_a_decision_survives_parameters_versions_and_a_lost_index(
        store_root, source, run_folder):
    """A person answered a question once. A version bump is not a reason to ask
    them again, and neither is losing the local cache."""
    store.decision("channel_assignment", source,
                   value={"dluc": 2, "structural": 1},
                   note="bright field is flat in this batch")

    assert store.decision("channel_assignment", source) == {"dluc": 2,
                                                            "structural": 1}

    manifest.path().unlink()                     # the whole local index, gone
    assert store.decision("channel_assignment", source) == {"dluc": 2,
                                                            "structural": 1}


def test_a_decision_is_keyed_on_the_source_alone(store_root, source):
    store.decision("time_window", source, value=[12, 480])
    sid = store.fingerprint(source)
    other = keys.Key(store.DECISION_STAGE, sid, {"decision": "time_window"})

    assert other.digest() == keys.Key(store.DECISION_STAGE, sid,
                                      {"decision": "time_window"}).digest()
    assert store.decisions_for(source) == {"time_window": [12, 480]}


def test_an_unanswered_decision_reads_as_none(store_root, source):
    assert store.decision("roi", source) is None


# ------------------------------------------------------------ the sidecars
def test_every_artefact_carries_its_key_beside_it(store_root, source,
                                                  run_folder):
    written = store_shifts(source, run_folder)
    side = tier_a.read_sidecar(written.path)

    assert side["stage"] == "registration"
    assert side["method_version"] == METHOD
    assert side["params"]["downsample"] == 4
    assert side["source"]["sample"] == store.fingerprint(source).sample
    assert side["kind"] == "table"
    assert side["digest"] == written.digest
