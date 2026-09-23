"""The two tiers: what is permanent, what is disposable, and where each lives.

Tier A is the analysis and is never evicted. Tier B is tens of gigabytes of
rebuildable pixels — capped, evictable, and as of 2026-08-20 kept in the
project folder rather than on the local disk, because rebuilding costs hours
and a second machine should find them rather than re-derive them.

That move traded one failure for another. The old one was syncing pixels
nobody needed; the new one is a **dehydrated** array — one the sync client has
freed to online-only, which is a hit by every test the store can make and a
download when it is read. The tests below pin the new location and that check.

The rest is atomicity: a run that dies half way through must leave nothing the
next run will trust.
"""

from __future__ import annotations

import gc
import json

import numpy as np
import pytest

from pymicroglia import config, store
from pymicroglia.store import budget, manifest, tier_a, tier_b


@pytest.fixture
def store_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.delenv("PYMICROGLIA_DECISIONS", raising=False)
    return tmp_path / "cache"


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "raw" / "VID52_C1_timestack.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(range(256)) * 40)
    return path


def fill_with(value):
    def fill(array):
        array[...] = value
    return fill


# ------------------------------------------------------- tier B, in the project
def test_the_rebuildable_tier_accepts_a_synced_folder(tmp_path, monkeypatch):
    """It used to refuse one. It now lives in one on purpose.

    Asserted rather than assumed, because the refusal was load-bearing for
    three years of this package's short life and re-introducing it would break
    every default install rather than failing visibly here.
    """
    synced = tmp_path / "UK Dementia Research Institute Dropbox" / "store"
    monkeypatch.setenv("PYMICROGLIA_STORE", str(synced))

    assert config.inside_dropbox(tier_b.root())
    assert tier_b.root() == synced


def test_the_default_root_is_the_projects_own_store(monkeypatch, tmp_path):
    """Beside the analysis, named for what it holds.

    A person who finds twenty gigabytes in their Dropbox has to be able to tell
    what it is and that deleting it costs time and nothing else. That is what
    the folder name is for.
    """
    monkeypatch.delenv("PYMICROGLIA_STORE", raising=False)
    monkeypatch.delenv("AUTO_ORGANOTYPIC_STORE", raising=False)
    project = tmp_path / "Microglia Project"
    source = project / "PyMicroglia" / "src" / "pymicroglia" / "__init__.py"
    source.parent.mkdir(parents=True)
    root = config.store_root(start=source, projects=(project.name,))
    assert root.name == config.STORE_DIRNAME
    assert root.parent == config.project_root(start=source, names=(project.name,))


def test_every_tier_b_path_sits_under_the_configured_root(store_root):
    """One root, so evicting it or pinning it offline reaches all of it."""
    for path in (tier_b.array_path("registered", "abc"),
                 tier_b.sidecar_path("registered", "abc"),
                 tier_b.partial_path("registered", "abc"),
                 manifest.path()):
        assert store_root in path.parents, path


def test_a_dehydrated_array_is_reported_rather_than_read(store_root, monkeypatch):
    """The failure a synced store has and a local one does not.

    An online-only placeholder passes every test the store makes — it is the
    right size on disk, it has its sidecar, its digest matches — and reading it
    faults twenty-one gigabytes back down the network. ``doctor`` is where that
    surfaces, because the fix is a Dropbox setting and not a code path.
    """
    from pymicroglia import knowledge

    with tier_b.open_write("registered", "abc", (4, 8, 8), "float32") as array:
        array[...] = 1.0

    assert knowledge.doctor()["store_dehydrated"] == 0

    monkeypatch.setattr(config, "is_placeholder", lambda path: True)
    report = knowledge.doctor()

    assert report["store_dehydrated"] == 1
    assert report["ok"] is False
    assert any("online-only" in complaint for complaint in report["complaints"])


# ------------------------------------------------------------------- atomicity
def test_a_completed_write_becomes_readable_and_a_hit(store_root):
    with tier_b.open_write("registered", "abc", (4, 8, 8), "float32") as array:
        array[...] = 3.5

    got = tier_b.read("registered", "abc", (4, 8, 8), "float32")
    assert got is not None
    assert got.dtype == np.float32 and got.shape == (4, 8, 8)
    assert float(got[0, 0, 0]) == 3.5
    tier_b.release("registered", "abc")


def test_an_interrupted_write_never_becomes_a_hit(store_root):
    with pytest.raises(RuntimeError):
        with tier_b.open_write("registered", "abc", (4, 8, 8), "float32") as array:
            array[0] = 1.0
            raise RuntimeError("the acquisition was truncated")

    assert not tier_b.array_path("registered", "abc").exists()
    assert tier_b.read("registered", "abc", (4, 8, 8), "float32") is None


def test_a_partial_left_by_a_dead_process_is_never_read(store_root):
    """A crash cannot roll back its own write, so the guarantee has to be that
    a partial file is invisible rather than that it is removed."""
    folder = store_root / "registered"
    folder.mkdir(parents=True, exist_ok=True)
    partial = tier_b.partial_path("registered", "abc")
    array = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32,
                                      shape=(4, 8, 8))
    array[0] = 1.0
    del array
    gc.collect()

    assert tier_b.read("registered", "abc", (4, 8, 8), "float32") is None
    assert [e["digest"] for e in tier_b.entries()] == []


def test_a_stored_array_of_the_wrong_shape_is_not_a_hit(store_root):
    with tier_b.open_write("registered", "abc", (4, 8, 8), "float32") as array:
        array[...] = 1.0
    assert tier_b.read("registered", "abc", (4, 8, 9), "float32") is None
    assert tier_b.read("registered", "abc", (4, 8, 8), "uint16") is None


def test_the_partial_sits_beside_its_target_not_in_a_temp_folder(store_root):
    """``os.replace`` is only atomic within one volume."""
    assert (tier_b.partial_path("registered", "abc").parent
            == tier_b.array_path("registered", "abc").parent)


# ------------------------------------------------------------ budget and eviction
def _make(digest, value, root):
    with tier_b.open_write("registered", digest, (16, 16), "float64") as array:
        array[...] = value
    side = tier_b.sidecar_path("registered", digest)
    return side


def _set_last_read(digest, when):
    path = tier_b.sidecar_path("registered", digest)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["last_read"] = when
    path.write_text(json.dumps(data), encoding="utf-8")


def test_eviction_takes_the_least_recently_used_first(store_root):
    for index, digest in enumerate(("old", "middle", "new")):
        _make(digest, index, store_root)
    _set_last_read("old", 1000.0)
    _set_last_read("middle", 2000.0)
    _set_last_read("new", 3000.0)

    one_array = tier_b.entries()[0]["bytes"]
    report = budget.evict(cap=one_array * 2)

    remaining = {entry["digest"] for entry in tier_b.entries()}
    assert remaining == {"middle", "new"}
    assert report["freed_bytes"] == one_array
    assert any("old" in path for path in report["evicted"])


def test_eviction_never_takes_an_array_that_is_mapped(store_root):
    for index, digest in enumerate(("old", "new")):
        _make(digest, index, store_root)
    _set_last_read("old", 1000.0)
    _set_last_read("new", 2000.0)

    live = tier_b.read("registered", "old", (16, 16), "float64")
    assert live is not None

    report = budget.evict(cap=0)
    assert {entry["digest"] for entry in tier_b.entries()} == {"old"}
    assert any("old" in path for path in report["held"])

    del live
    gc.collect()
    tier_b.release("registered", "old")
    budget.evict(cap=0)
    assert tier_b.entries() == []


def test_room_is_made_before_a_long_write_starts(store_root):
    for index, digest in enumerate(("old", "new")):
        _make(digest, index, store_root)
    _set_last_read("old", 1000.0)
    _set_last_read("new", 2000.0)
    one_array = tier_b.entries()[0]["bytes"]

    budget.room_for(one_array, cap=one_array * 2)
    assert {entry["digest"] for entry in tier_b.entries()} == {"new"}


def test_the_report_says_what_is_held_and_what_the_cap_is(store_root):
    _make("only", 1, store_root)
    report = budget.report()

    assert report["arrays"] == 1
    assert report["cap_gb"] == 64.0
    assert report["over_cap"] is False


def test_materialise_builds_once_and_reuses_after(store_root, source):
    calls = []

    def fill(array):
        calls.append(1)
        array[...] = 7.0

    first = store.materialise("registered", source, {"downsample": 4},
                              shape=(4, 4), dtype="float32", fill=fill,
                              method_version="v1")
    second = store.materialise("registered", source, {"downsample": 4},
                               shape=(4, 4), dtype="float32", fill=fill,
                               method_version="v1")

    assert len(calls) == 1
    assert float(first[0, 0]) == 7.0 and float(second[0, 0]) == 7.0

    third = store.materialise("registered", source, {"downsample": 2},
                              shape=(4, 4), dtype="float32", fill=fill,
                              method_version="v1")
    assert len(calls) == 2
    assert third is not None


# ------------------------------------------------------------------- tier A kinds
def test_a_mask_round_trips_as_the_boolean_it_was(store_root, source, tmp_path):
    mask = np.zeros((3, 5, 5), dtype=bool)
    mask[1, 2, 3] = True
    written = store.put("cosmic_rays", source, {"threshold_sigma": 12.0},
                        kind="mask", value=mask, name="cosmic_ray_mask",
                        output_dir=tmp_path / "out", method_version="v1")

    back = written.load()
    assert back.dtype == bool and back.shape == (3, 5, 5)
    assert np.array_equal(back, mask)


def test_labels_keep_their_integer_dtype(store_root, source, tmp_path):
    labels = np.arange(36, dtype=np.uint16).reshape(6, 6)
    written = store.put("segmentation", source, {"min_area_px": 12},
                        kind="labels", value=labels, name="cell_labels",
                        output_dir=tmp_path / "out", method_version="v1")

    back = written.load()
    assert back.dtype == np.uint16
    assert np.array_equal(back, labels)


def test_a_float_array_is_not_quietly_narrowed(store_root, source, tmp_path):
    """A hit that rounded where a cold run did not is not a valid artefact."""
    values = np.array([[0.1, 0.25], [1e-7, 3.5]], dtype=np.float32)
    written = store.put("tracing", source, {"detrend": "none"}, kind="array",
                        value=values, name="trace_matrix",
                        output_dir=tmp_path / "out", method_version="v1")

    back = written.load()
    assert back.dtype == np.float32
    assert np.array_equal(back, values)


def test_a_mask_that_is_not_boolean_is_refused(store_root, source, tmp_path):
    with pytest.raises(TypeError):
        store.put("cosmic_rays", source, {}, kind="mask",
                  value=np.zeros((2, 2), dtype=np.uint8), name="wrong",
                  output_dir=tmp_path / "out")


def test_scalars_are_json_a_human_can_open(store_root, source, tmp_path):
    written = store.put("registration", source, {}, kind="scalars",
                        value={"max_drift_px": 4.25, "frames": 480},
                        name="registration_summary",
                        output_dir=tmp_path / "out", method_version="v1")

    assert written.path.suffix == ".json"
    assert json.loads(written.path.read_text(encoding="utf-8"))["frames"] == 480


def test_nothing_is_stored_as_a_pickle(store_root, source, tmp_path):
    """A stored artefact has to be readable in five years by something that is
    not this package.

    Only files are artefacts. Auto-Organotypic keeps its own bookkeeping in a
    dotted folder beside them -- one ``artefacts.json`` for the directory rather
    than a ``.artefact.json`` twin per file, which it changed on the way to
    0.6 -- and where it puts that is its business, not this assertion's. What is
    this assertion's business is the extension of everything a reader would
    open.
    """
    store.put("tracing", source, {}, kind="array", value=np.zeros(4),
              name="t", output_dir=tmp_path / "out")
    store.put("registration", source, {}, kind="table",
              value={"frame": [0, 1]}, name="s", output_dir=tmp_path / "out")

    written = sorted(p.suffix for p in (tmp_path / "out").glob("*")
                     if p.is_file()
                     and not p.name.endswith(tier_a.SIDECAR_SUFFIX))
    assert written == [".csv", ".npz"]


# ------------------------------------------------------------------------ scan
def test_scan_indexes_one_folder_and_reads_nothing_outside_it(store_root,
                                                              source, tmp_path):
    here = tmp_path / "exports" / "run_a"
    below = tmp_path / "exports" / "run_a" / "nested"
    beside = tmp_path / "exports" / "run_b"
    for folder, name in ((here, "here"), (below, "below"), (beside, "beside")):
        store.put("registration", source, {"tag": name}, kind="table",
                  value={"frame": [0]}, name=name, output_dir=folder,
                  method_version="v1")

    manifest.path().unlink()                      # forget everything
    report = store.scan(here)

    assert report["indexed"] == 1
    indexed = {entry["params"]["tag"] for entry in manifest.find()}
    assert indexed == {"here"}


def test_scan_drops_records_whose_artefact_has_gone(store_root, source,
                                                    tmp_path):
    folder = tmp_path / "exports" / "run_a"
    written = store.put("registration", source, {}, kind="table",
                        value={"frame": [0]}, name="shifts",
                        output_dir=folder, method_version="v1")
    assert len(manifest.find()) == 1

    # The artefact goes. The per-file ``.artefact.json`` twin is deleted only if
    # this folder has one: Auto-Organotypic moved to a single ``artefacts.json``
    # per directory on 2026-09-14 and still reads the old twins, so both layouts
    # pass here.
    written.path.unlink()
    tier_a.sidecar_path(written.path).unlink(missing_ok=True)
    store.scan(folder)

    # What matters is that nothing hands back a path to a file that is not
    # there. Not asserted on scan's ``dropped`` count any more: that counts
    # claims which vanished from the folder ledger, and a file deleted by hand
    # leaves its claim behind. The guarantee moved to every lookup instead --
    # ``find`` takes ``require_file=True`` by default -- which is the stronger
    # place for it, because it no longer waits for somebody to run a scan.
    assert manifest.find() == []
    assert manifest.find(require_file=False), (
        "the claim should still be on record; it is the *lookup* that refuses "
        "to return an artefact whose file has gone")


def test_scanning_something_that_is_not_a_folder_says_so(store_root, tmp_path):
    report = store.scan(tmp_path / "nowhere")
    assert report["ok"] is False and report["error"] == "not_a_folder"


def test_status_reports_both_tiers(store_root, source, tmp_path):
    store.put("registration", source, {}, kind="table", value={"frame": [0]},
              name="shifts", output_dir=tmp_path / "out", method_version="v1")
    with tier_b.open_write("registered", "abc", (4, 4), "float32") as array:
        array[...] = 0.0

    status = store.status()
    assert status["tier_a"]["artefacts"] == 1
    assert status["tier_a"]["stages"] == {"registration": 1}
    assert status["tier_b"]["arrays"] == 1


def test_the_full_hash_is_opt_in_and_recorded_once(store_root, source):
    plain = store.fingerprint(source)
    assert plain.full_sha256 is None

    verified = store.verify_source(source)
    assert verified.full_sha256 and len(verified.full_sha256) == 64
    assert verified.sample == plain.sample          # identity does not move
    assert store.fingerprint(source).full_sha256 == verified.full_sha256


def test_the_index_stays_local_when_the_pixels_do_not(tmp_path, monkeypatch):
    """The one file that must not follow the pixels into Dropbox.

    A few hundred kilobytes rewritten after every run, by both machines. A sync
    client's answer to that is a "conflicted copy" and two divergent indexes,
    neither of which reports itself as wrong. It is rebuildable by scanning, so
    keeping it per-machine costs nothing.
    """
    monkeypatch.delenv("PYMICROGLIA_INDEX", raising=False)

    local = tmp_path / "scratch"
    monkeypatch.setenv("PYMICROGLIA_STORE", str(local))
    assert manifest.path().parent == local

    synced = tmp_path / "UK Dementia Research Institute Dropbox" / "PixelStore"
    monkeypatch.setenv("PYMICROGLIA_STORE", str(synced))
    assert manifest.path().parent == config.local_cache()
    assert not config.inside_dropbox(manifest.path())
