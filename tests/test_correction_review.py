"""Review agent changes page: review sets, decisions, edits and the pictures behave as the Motion reviewer did."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from pymicroglia.ui import correction_review as cr


def disc(shape, cy, cx, r):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def make_bundle(root: Path) -> Path:
    """A Motion-style bundle: cell 1 becomes 2 at frame 5 (one applied join); cell 3 sits still."""
    T, shape = 10, (60, 80)
    lab = np.zeros((T,) + shape, np.uint16)
    for f in range(T):
        lab[f][disc(shape, 20, 10 + 2 * f, 6)] = 1 if f < 5 else 2
        lab[f][disc(shape, 45, 60, 6)] = 3
    raw = (lab > 0).astype(np.uint16) * 900 + 100
    folder = root / "position_01"
    folder.mkdir(parents=True)
    tifffile.imwrite(folder / "labels.tif", lab)
    tifffile.imwrite(folder / "raw.tif", raw)
    sha = hashlib.sha256((folder / "labels.tif").read_bytes()).hexdigest()
    (folder / "motion-editing-bundle.json").write_text(json.dumps({"stem": "position_01", "assets": {
        "canonical_labels": {"path": "labels.tif", "sha256": sha},
        "registered_raw": {"path": "raw.tif", "label_frame_to_source": list(range(T))}},
        "time": {"frame_interval_min": 30}}))
    return folder


@pytest.fixture()
def review(tmp_path):
    source = make_bundle(tmp_path)
    made = cr.create_review(tmp_path, source)
    return tmp_path, made["stem"]


def test_review_set_lists_the_join_and_never_touches_the_source(review):
    root, stem = review
    before = (root / "position_01/labels.tif").read_bytes()
    state = cr.load(root, stem)
    (item,) = state["items"]
    assert item["kind"] == "applied" and item["frame"] == 5
    assert item["old"] == [2] and item["new"] == [1] and state["hours"][2] == "1"
    assert cr.list_reviews(root)[0]["applied"] == 1
    assert (root / "position_01/labels.tif").read_bytes() == before


def first_key(root, stem):
    return cr.load(root, stem)["items"][0]["key"]


def test_denying_the_join_restores_before_and_approving_keeps_it(review):
    root, stem = review
    key = first_key(root, stem)
    labels, after, _ = cr.composed(root, stem)
    assert set(np.unique(after[5:][labels[5:] == 2])) == {1}
    assert cr.decide(root, stem, key, "deny")["ok"]
    labels, after, _ = cr.composed(root, stem)
    assert np.array_equal(labels, after)
    assert cr.decide(root, stem, key, None)["decisions"] == {}
    with pytest.raises(ValueError):
        cr.save(root, stem, decisions={"not-a-key": "approve"})


def test_click_reports_the_number_now_and_the_end_of_its_run(review):
    root, stem = review
    key = first_key(root, stem)
    hit = cr.pick(root, stem, key, 6, 20, 22, "Before")
    assert hit["cell"] == 2 and hit["now"] == 1 and hit["to"] == 9
    assert cr.cell_now(root, stem, 6, 20, 22) == {"frame": 6, "y": 20, "x": 22, "before": 2, "now": 1, "to": 9}


def test_edit_buttons_swap_give_delete_piece_and_undo(review):
    root, stem = review
    key = first_key(root, stem)
    one = cr.pick(root, stem, key, 0, 20, 10, "Before")
    done = cr.edit(root, stem, action="swap", picked=one, frame=0, to_frame=2, key=key, other=3)
    assert done["ok"] and done["count"] == 1
    _, after, _ = cr.composed(root, stem)
    assert after[0][20, 10] == 3 and after[0][45, 60] == 1 and after[3][20, 16] == 1
    three = cr.pick(root, stem, key, 7, 45, 60, "Before")
    assert cr.edit(root, stem, action="delete", picked=three, frame=7, to_frame=9, key=key)["ok"]
    _, after, _ = cr.composed(root, stem)
    assert not (after[7:] == 3).any() and (after[6] == 3).any()
    nothing = cr.edit(root, stem, action="delete", picked={**three, "now": 3}, frame=8, to_frame=8, key=key)
    assert not nothing["ok"] and nothing["message"].startswith("Nothing to change")
    piece = cr.pick(root, stem, key, 4, 45, 60, "Before")
    assert cr.edit(root, stem, action="give", picked=piece, frame=4, to_frame=9, key=key, number=1,
                   piece_only=True)["ok"]
    saved = cr.load(root, stem)["edits"]
    assert [e["op"] for e in saved] == ["swap", "delete", "piece"]
    # the piece is followed to the 'to' frame while it lasts (3 is deleted from frame 7)
    assert (saved[2]["first"], saved[2]["last"]) == (4, 6) and saved[0]["by"] == "jamie" and saved[0]["item"] == key
    assert not cr.undo_edit(root, stem, 2)["ok"]                    # the list changed since edit 2
    assert cr.undo_edit(root, stem, 3)["ok"] and len(cr.load(root, stem)["edits"]) == 2


def test_drawn_split_cuts_the_cell_into_two_numbers(review):
    root, stem = review
    key = first_key(root, stem)
    done = cr.edit(root, stem, action="split", picked=None, frame=2, to_frame=2, key=key,
                   points=[[38, 60], [45, 60], [52, 60]])
    assert done["ok"], done
    _, after, _ = cr.composed(root, stem)
    left, right = after[2][45, 56], after[2][45, 64]
    assert left and right and left != right and 3 in (left, right)


def test_notes_keep_the_reviewers_format(review):
    root, stem = review
    key = first_key(root, stem)
    hit = cr.pick(root, stem, key, 6, 20, 22, "After")
    assert cr.add_note(root, stem, key=key, frame=6, text="is 1", tag="Wrong number", picked=hit)["ok"]
    (note,) = cr.load(root, stem)["notes"]
    assert note["text"] == "Wrong number: is 1" and note["cell"] == 1 and note["tag"] == "Wrong number"
    assert note["item"] == key and (note["y"], note["x"], note["panel"]) == (20, 22, "After")


def test_frame_pictures_carry_numbers_and_outline_colours(review):
    root, stem = review
    key = first_key(root, stem)
    window = cr.item_view(root, stem, key)
    view = cr.frame_view(root, stem, key, 5, 4)
    assert set(view["panels"]) == set(cr.PANELS) and view["k"] == 4
    assert window["lo"] == view["lo"] and window["hi"] == view["hi"]
    import base64, io
    from PIL import Image
    before = np.asarray(Image.open(io.BytesIO(base64.b64decode(view["panels"]["Before"].split(",")[1]))))
    assert before.shape[:2] == tuple(4 * n for n in view["crop"])
    assert (before == cr.MAGENTA).all(-1).any()                     # the renamed cell is marked
    assert {t for _, _, t, _ in view["nb"]} == {"2", "3"} and {t for _, _, t, _ in view["na"]} == {"1", "3"}
    assert [c for _, _, t, c in view["na"] if t == "1"] == ["#a0ffaa"]   # joined number in green


def test_export_needs_the_motion_service(review):
    root, stem = review
    with pytest.raises(ValueError, match="Motion service"):
        cr.export(root, stem, None)


def test_page_and_routes_are_served(review):
    from fastapi.testclient import TestClient
    from pymicroglia.ui.server import create_app
    root, stem = review
    client = TestClient(create_app())
    cr.PREWARM = False   # the Workbench's background warm-up would only slow the tests after this one
    assert "correction-review.js" in client.get("/correction-review").text
    assert client.get("/assets/correction-review.js").status_code == 200
    where = {"root": str(root), "stem": stem}
    listed = client.get("/api/correction-review", params={"root": str(root)}).json()
    assert listed["reviews"][0]["stem"] == stem
    key = client.get("/api/correction-review/set", params=where).json()["items"][0]["key"]
    frame = client.get("/api/correction-review/frame", params={**where, "key": key, "frame": 5, "k": 3}).json()
    assert frame["k"] == 3 and frame["panels"]["Before"].startswith("data:image/png")
    hit = client.get("/api/correction-review/pick", params={**where, "key": key, "frame": 6, "y": 20, "x": 22}).json()
    assert hit["now"] == 1
    edit = client.post("/api/correction-review/edit", json={**where, "action": "delete", "picked": hit, "frame": 6,
                                                          "to_frame": 6, "key": key}).json()
    assert edit["ok"] and client.post("/api/correction-review/undo-edit", json={**where, "count": 1}).json()["ok"]
    assert client.post("/api/correction-review/note", json={**where, "key": key, "frame": 6, "tag": "Swap"}).json()["ok"]
    assert client.post("/api/correction-review/decide", json={**where, "key": key, "value": "deny"}).json()["ok"]
    bad = client.put("/api/correction-review/set", json={**where, "decisions": {"x": "no"}})
    assert bad.status_code in (400, 422)


def test_page_assistant_follows_jamies_rules(review, monkeypatch):
    pytest.importorskip("aiify")
    monkeypatch.setenv("PYMICROGLIA_AI", "1")
    from fastapi.testclient import TestClient
    from pymicroglia.ui.server import create_app
    root, stem = review
    app = create_app()
    client = TestClient(app)
    assert "/aiify-corrections" in client.get("/correction-review").text      # its own assistant here
    assert "/aiify-corrections" not in client.get("/").text                    # and not elsewhere
    import asyncio
    host = app.state.correction_agent.actions

    def run(name, params, *, confirm=False):
        return asyncio.run(host.run(name, params, confirm=confirm))
    where = {"root": str(root), "stem": stem}
    cr.add_note(root, stem, key=None, frame=6, text="this is 1", tag="Wrong number",
                picked={"frame": 6, "y": 45, "x": 60, "cell": 3, "panel": "Before"})
    assert [n["index"] for n in run("read_notes", where)["result"]["notes"]] == [0]
    refused = run("edit_cell", {**where, "action": "give", "frame": 6, "y": 45, "x": 60, "last": 6, "number": 77},
                  confirm=True)
    assert not refused["ok"] and "no new numbers" in refused["error"]
    added = run("edit_cell", {**where, "action": "give", "frame": 6, "y": 45, "x": 60, "last": 6, "number": 1,
                              "note": 0, "why": "this is 1"}, confirm=True)
    assert added["ok"], added
    made = cr.load(root, stem)["edits"][0]
    assert {k: made[k] for k in ("op", "from", "to", "first", "last", "by", "why", "note")} == {
        "op": "renumber", "from": 3, "to": 1, "first": 6, "last": 6, "by": "agent", "why": "this is 1", "note": 0}
    assert run("read_notes", where)["result"]["notes"] == []                  # the note is handled
    run("ask_jamie", {**where, "frame": 6, "question": "Is this one cell?", "options": "A) yes B) no",
                      "y": 20, "x": 22}, confirm=True)
    question = cr.load(root, stem)["notes"][-1]
    assert question["tag"] == "Question" and question["cell"] == 1 and question["by"] == "agent"
    hit = cr.pick(root, stem, first_key(root, stem), 0, 45, 60, "Before")
    assert cr.edit(root, stem, action="delete", picked=hit, frame=0, to_frame=0)["ok"]       # Jamie's edit
    assert not run("remove_agent_edit", where, confirm=True)["ok"]            # Jamie's edit stays
    cr.undo_edit(root, stem, 2)
    assert run("remove_agent_edit", where, confirm=True)["ok"]
    assert cr.load(root, stem)["edits"] == []


def test_pictures_show_the_numbers_a_click_reports(review):
    """Regression (Jamie, 2026-11-11): after a swap the picture kept the old number while a click said the new
    one, and a deleted cell stayed drawn but could not be clicked."""
    root, stem = review
    key = first_key(root, stem)
    one = cr.pick(root, stem, key, 0, 20, 10, "Before")
    assert cr.edit(root, stem, action="swap", picked=one, frame=0, to_frame=9, key=key, other=3)["ok"]
    view = cr.frame_view(root, stem, key, 2, 2)
    drawn = {t: (y, x) for y, x, t, _ in view["nb"]}
    y0, x0 = view["origin"]
    for t, (y, x) in drawn.items():   # every number drawn on Before is the number a click there reports
        assert cr.pick(root, stem, key, 2, int(round(y)) + y0, int(round(x)) + x0, "Before")["now"] == int(t)
    three = cr.pick(root, stem, key, 2, 45, 60, "Before")
    assert three["now"] == 1
    assert cr.edit(root, stem, action="delete", picked=three, frame=2, to_frame=2, key=key)["ok"]
    view = cr.frame_view(root, stem, key, 2, 2)
    assert {t for _, _, t, _ in view["nb"]} == {"3"} and {t for _, _, t, _ in view["na"]} == {"3"}


def test_edits_follow_their_cell_when_a_later_decision_renumbers_it(review):
    """Regression (slice 8, 2026-11-11): approving a join after editing one of its cells voided the edit."""
    root, stem = review
    key = first_key(root, stem)
    assert cr.decide(root, stem, key, "deny")["ok"]                 # the cell after frame 5 is 2 again
    hit = cr.pick(root, stem, key, 6, 20, 22, "Before")
    assert hit["now"] == 2
    assert cr.edit(root, stem, action="delete", picked=hit, frame=6, to_frame=7, key=key)["ok"]
    _, after, _ = cr.composed(root, stem)
    assert not after[6:8][:, 20, 22].any()
    done = cr.decide(root, stem, key, "approve")                     # renumbers that cell to 1
    assert done["ok"] and done["edits"][0]["cell"] == 1 and done["edits"][0]["as_made"]["cell"] == 2
    _, after, _ = cr.composed(root, stem)
    assert not after[6:8][:, 20, 22].any() and after[8][20, 26] == 1  # still deleted, and only there
    cr.decide(root, stem, key, "deny")                               # and back again
    assert cr.load(root, stem)["edits"][0]["cell"] == 2


def test_give_and_delete_change_only_the_clicked_cell_when_its_number_is_shared(tmp_path):
    """Regression (2026-11-12): a renumber caught a second, distant cell that carried the same number."""
    source = make_bundle(tmp_path)
    lab = tifffile.imread(source / "labels.tif")
    for f in range(len(lab)):
        lab[f][disc(lab.shape[1:], 8, 70, 5)] = 3                    # a far piece also numbered 3
    tifffile.imwrite(source / "labels.tif", lab)
    manifest = json.loads((source / "motion-editing-bundle.json").read_text())
    manifest["assets"]["canonical_labels"]["sha256"] = hashlib.sha256((source / "labels.tif").read_bytes()).hexdigest()
    (source / "motion-editing-bundle.json").write_text(json.dumps(manifest))
    stem = cr.create_review(tmp_path, source)["stem"]
    key = first_key(tmp_path, stem)
    hit = cr.pick(tmp_path, stem, key, 2, 45, 60, "Before")
    done = cr.edit(tmp_path, stem, action="delete", picked=hit, frame=2, to_frame=4, key=key)
    assert done["ok"] and "another cell" in done["message"]
    _, after, _ = cr.composed(tmp_path, stem)
    assert not after[2:5][:, 45, 60].any() and (after[2:5][:, 8, 70] == 3).all() and after[5][45, 60] == 3
    hit = cr.pick(tmp_path, stem, key, 6, 8, 70, "Before")
    assert cr.edit(tmp_path, stem, action="give", picked=hit, frame=6, to_frame=9, key=key, number=2)["ok"]
    _, after, _ = cr.composed(tmp_path, stem)
    assert (after[6:][:, 8, 70] == 2).all() and (after[6:][:, 45, 60] == 3).all()


def test_an_assistant_question_opens_at_its_own_cell_and_frame(review):
    """Jamie 2026-11-17: a question must take him to the cell it asks about, at that frame, not to the open join."""
    root, stem = review
    cr.add_note(root, stem, key=None, frame=8, text="Question: is 3 one cell?  A) yes  B) no", tag="Question",
                by="agent", picked={"frame": 8, "y": 45, "x": 60, "cell": 3, "panel": "After"})
    state = cr.load(root, stem)
    question, join = state["items"]
    assert question["kind"] == "question" and question["frame"] == 8 and question["old"] == [3]
    assert join["kind"] == "applied"
    view = cr.item_view(root, stem, question["key"])
    assert view["lo"] <= 8 <= view["hi"] and view["hi"] == 9
    shown = cr.frame_view(root, stem, question["key"], 8, 2)
    y0, x0 = shown["origin"]
    h, w = shown["crop"]
    assert y0 <= 45 < y0 + h and x0 <= 60 < x0 + w
    assert any(t == "3" for _, _, t, _ in shown["na"])
    assert cr.pick(root, stem, question["key"], 8, 45, 60, "After")["now"] == 3
    with pytest.raises(ValueError):
        cr.decide(root, stem, question["key"], "approve")
    notes = [dict(n, answer="yes") for n in state["notes"]]
    cr.save(root, stem, notes=notes)
    assert [it["kind"] for it in cr.load(root, stem)["items"]] == ["applied"]


# ---------------------------------------------------------------- Jamie 2026-11-17: follow, timeline, draw, seed, grow
from pymicroglia.tracking import correction_rules as rd   # noqa: E402


def test_follow_view_keeps_the_followed_cell_in_one_crop_over_its_frames(review):
    root, stem = review
    key = cr.follow_key([1])
    view = cr.item_view(root, stem, key)
    assert view["kind"] == "follow" and (view["lo"], view["hi"]) == (0, 9)
    y0, y1, x0, x1 = view["box"]          # cell 1 moves from x=10 to x=28 (radius 6) on row 20
    assert y0 <= 14 and y1 >= 27 and x0 <= 4 and x1 >= 35
    shown = cr.frame_view(root, stem, key, 9, 2, follow=[1])
    assert shown["names"][0] == "Current" and "Before" not in shown["names"]
    assert any(t == "1" and col == shown["colours"]["1"] for _, _, t, col in shown["nb"])
    assert cr.pick(root, stem, key, 9, 20, 28, "Current")["now"] == 1
    every = cr.frame_view(root, stem, key, 9, 2, every=True)
    assert every["panels"]["Current"] != shown["panels"]["Current"]


def test_timeline_flags_a_number_that_ends_and_one_that_starts(review):
    root, stem = review
    quiet = cr.cells_view(root, stem)
    assert [c["n"] for c in quiet["cells"]] == [1, 3] and not any(c["events"] for c in quiet["cells"])
    assert quiet["cells"][0]["runs"] == [[0, 9]] and quiet["joins"][0]["frame"] == 5
    cr.decide(root, stem, first_key(root, stem), "deny")
    events = {c["n"]: [e[1] for e in c["events"]] for c in cr.cells_view(root, stem)["cells"]}
    assert events == {1: ["end"], 2: ["start"], 3: []}


def test_drawn_outline_takes_only_free_pixels_erases_and_exports_as_a_new_number(review):
    root, stem = review
    key = first_key(root, stem)
    square = [(2, 40), (2, 60), (12, 60), (12, 40)]
    assert cr.edit(root, stem, action="draw", picked=None, frame=3, key=key, points=square)["ok"]
    labels, after, _ = cr.composed(root, stem)
    n = int(after[3, 6, 50])
    assert n > 3 and (after[3] == n).sum() > 150 and not after[2, 6, 50]
    runs = [r for fr in rd.patch_from(labels, after)["frames"] if fr["imagej_frame"] == 4 for r in fr["runs"]]
    assert runs and all(source == 0 and target == n for _, _, _, source, target in runs)
    picked = {"frame": 3, "y": 6, "x": 50, "now": n}
    assert cr.edit(root, stem, action="erase", picked=picked, frame=3, key=key, points=square)["ok"]
    assert not (cr.composed(root, stem)[1][3] == n).any()


def signal_review(tmp_path):
    """As make_bundle, plus unnumbered signal: a blob at (50, 20) and a ring round cell 3 (signal r 9, outline r 6)."""
    source = make_bundle(tmp_path)
    lab = tifffile.imread(source / "labels.tif")
    raw = np.full(lab.shape, 100, np.uint16)
    for f in range(len(lab)):
        raw[f][lab[f] > 0] = 900
        raw[f][disc(lab.shape[1:], 50, 20, 5)] = 900
        raw[f][disc(lab.shape[1:], 45, 60, 9)] = 900
    tifffile.imwrite(source / "raw.tif", raw)
    return tmp_path, cr.create_review(tmp_path, source)["stem"]


def test_seed_takes_the_free_signal_at_the_click_and_follows_it_both_ways(tmp_path):
    """Jamie 2026-11-17: seeding frame by frame gave each frame a new number; a seed now follows its signal back
    and forward through the frames shown (forward only to a later 'to' frame)."""
    root, stem = signal_review(tmp_path)
    key = cr.follow_key([3])                       # the whole recording
    T = cr._rec(root, stem).data["T"]
    picked = cr.cell_now(root, stem, 2, 50, 20)
    assert not picked["now"]
    done = cr.edit(root, stem, action="seed", picked=picked, frame=2, to_frame=6, key=key)
    assert done["ok"], done
    _, after, _ = cr.composed(root, stem)
    n = int(after[2, 50, 20])
    assert n > 3 and all(after[f, 50, 20] == n for f in range(0, 7)) and not after[7, 50, 20]
    assert not cr.edit(root, stem, action="seed", picked=cr.cell_now(root, stem, 2, 45, 60), frame=2, key=key)["ok"]


def test_seed_continues_the_number_of_a_track_that_vanished_where_it_starts(tmp_path):
    root, stem = signal_review(tmp_path)
    key = cr.follow_key([3])
    T = cr._rec(root, stem).data["T"]
    square = [(42, 12), (42, 28), (58, 28), (58, 12)]
    for f in (0, 1):                               # the tracker had it as 9, then lost it
        assert cr.edit(root, stem, action="draw", picked=None, frame=f, key=key, points=square, number=9)["ok"]
    done = cr.edit(root, stem, action="seed", picked=cr.cell_now(root, stem, 4, 50, 20), frame=4, to_frame=4, key=key)
    assert done["ok"] and "continues" in done["message"], done
    _, after, _ = cr.composed(root, stem)
    assert all(after[f, 50, 20] == 9 for f in range(T))


def test_grow_adds_only_the_free_signal_touching_the_cell(tmp_path):
    root, stem = signal_review(tmp_path)
    key = first_key(root, stem)
    labels, before, _ = cr.composed(root, stem)
    hit = cr.pick(root, stem, key, 0, 45, 60, "Before")
    assert cr.edit(root, stem, action="grow", picked=hit, frame=0, to_frame=9, key=key)["ok"]
    _, after, _ = cr.composed(root, stem)
    assert ((after == 3).sum(axis=(1, 2)) > (before == 3).sum(axis=(1, 2)) + 50).all()
    assert np.array_equal(after == 1, before == 1) and not (after[:, 50, 20]).any()


def test_delete_whole_track_and_tower_of_followed_cells(review):
    root, stem = review
    key = first_key(root, stem)
    hit = cr.pick(root, stem, key, 6, 45, 60, "Before")
    assert cr.edit(root, stem, action="delete", picked=hit, frame=6, to_frame=6, key=key, whole=True)["ok"]
    assert not (cr.composed(root, stem)[1] == 3).any()
    # Jamie 2026-11-17: the full tower, every cell and line; followed cells only emphasised, none needed
    T = cr._rec(root, stem).data["T"]
    for cells in ([], [1]):
        tower = cr.tower_view(root, stem, cells)
        assert tower["image"].startswith("data:image/png;base64,") and (tower["lo"], tower["hi"]) == (0, T - 1)
        assert tower["frames"][0] == 0 and set(tower["identities"]) >= {1} and 3 not in tower["identities"]


def test_drawn_outline_takes_the_typed_number(review):
    """Drawing a missed cell frame by frame keeps one number when it is typed in the give box."""
    root, stem = review
    key = first_key(root, stem)
    square = [(2, 40), (2, 60), (12, 60), (12, 40)]
    for f in (3, 4):
        assert cr.edit(root, stem, action="draw", picked=None, frame=f, key=key, points=square, number=77)["ok"]
    after = cr.composed(root, stem)[1]
    assert after[3, 6, 50] == 77 and after[4, 6, 50] == 77


def test_follow_view_shows_frames_before_a_cell_appears(review):
    """Jamie 2026-11-17: when a cell suddenly appears he must be able to look before it, to see why."""
    root, stem = review
    cr.decide(root, stem, first_key(root, stem), "deny")   # number 2 now starts at frame 5
    key = cr.follow_key([2])
    view = cr.item_view(root, stem, key)
    assert (view["lo"], view["hi"]) == (0, 9) and view["present"] == [5, 9]
    early = cr.frame_view(root, stem, key, 2, 2, follow=[2])
    assert early["names"][0] == "Current" and not any(t == "2" for _, _, t, _ in early["nb"])
    y0, x0 = early["origin"]
    assert y0 <= 20 < y0 + early["crop"][0] and x0 <= 20 < x0 + early["crop"][1]   # 2 first appears at x=20
    assert cr.item_view(root, stem, cr.follow_key([2], 6, 8))["present"] == [5, 9]


def test_a_wider_field_shows_more_and_edits_in_it_are_stored_under_the_item(review):
    """Jamie 2026-11-17: zoom out to find a cell that left the crop after a wrong identity swap."""
    root, stem = review
    key = first_key(root, stem)
    normal = cr.frame_view(root, stem, key, 3, 1)
    H, W = normal["shape"]
    wide = cr.frame_view(root, stem, key + cr.FIELD + f"0,{H},0,{W}", 3, 1)
    assert wide["crop"] == [H, W] and wide["origin"] == [0, 0]
    assert normal["crop"][0] * normal["crop"][1] <= H * W
    square = [(2, 40), (2, 60), (12, 60), (12, 40)]
    assert cr.edit(root, stem, action="draw", picked=None, frame=3, key=key + cr.FIELD + f"0,{H},0,{W}",
                   points=square, number=77)["ok"]
    assert cr._ops(cr._rec(root, stem).folder)[-1]["item"] == key


def test_piece_only_give_on_a_seeded_cell_covers_the_following_frames(tmp_path):
    """Jamie 2026-11-17: giving a seeded cell a number had to be repeated frame by frame."""
    root, stem = signal_review(tmp_path)
    key = cr.follow_key([3])
    assert cr.edit(root, stem, action="seed", picked=cr.cell_now(root, stem, 2, 50, 20), frame=2, to_frame=2, key=key)["ok"]
    hit = cr.cell_now(root, stem, 2, 50, 20)
    seeded = hit["now"]
    assert seeded and hit["to"] > 2
    done = cr.edit(root, stem, action="give", picked=hit, frame=2, to_frame=6, key=key, number=40, piece_only=True)
    assert done["ok"], done
    _, after, _ = cr.composed(root, stem)
    assert all(after[f, 50, 20] == 40 for f in range(2, 7)) and after[7, 50, 20] == seeded and after[1, 50, 20] == seeded


def test_current_labels_updated_edit_by_edit_match_a_full_rebuild(tmp_path):
    """Speed (Jamie 2026-11-17): after an edit only that edit is applied; after an undo only its frames redone."""
    from pymicroglia.tracking import correction_rules as rd
    from pymicroglia.ui import correction_cells as cc
    root, stem = signal_review(tmp_path)
    key = cr.follow_key([3])
    rec = cr._rec(root, stem)
    full = lambda: rd.compose(rec.labels, rec.data, cr._decisions(rec.folder), cr._ops(rec.folder))
    cc.current_stack(rec)
    assert cr.edit(root, stem, action="seed", picked=cr.cell_now(root, stem, 2, 50, 20), frame=2, to_frame=2, key=key)["ok"]
    assert cr.edit(root, stem, action="give", picked=cr.cell_now(root, stem, 4, 45, 60), frame=4, to_frame=7, key=key, number=9)["ok"]
    entry = cc.current_stack(rec)
    assert np.array_equal(entry["stack"], full())
    assert entry["per"] == [cc._frame_summary(f) for f in full()]       # summaries updated number by number
    assert cr.undo_edit(root, stem, 2)["ok"]
    entry = cc.current_stack(rec)
    assert np.array_equal(entry["stack"], full()) and entry["per"] == [cc._frame_summary(f) for f in full()]
    frame = cr.current_frame_fn(rec)
    with pytest.raises(ValueError):
        frame(3)[0, 0] = 1                     # cached frames are read-only


def test_queued_edits_run_in_order_each_on_the_cell_there_then(tmp_path):
    """Jamie 2026-11-17: queue several actions, then run them all at once."""
    root, stem = signal_review(tmp_path)
    key = cr.follow_key([3])
    T = cr._rec(root, stem).data["T"]
    items = [dict(action="seed", frame=2, y=50, x=20),
             dict(action="give", frame=4, y=45, x=60, number=9),
             dict(action="give", frame=4, y=45, x=60, number=9),            # nothing left to change: reported
             dict(action="draw", frame=3, points=[[2, 40], [2, 60], [12, 60], [12, 40]], number=77),
             dict(action="swap", frame=3, y=45, x=60)]
    r = cr.edit_batch(root, stem, items, key=key)
    assert [x["ok"] for x in r["results"]] == [True, True, False, True, False] and r["done"] == 3
    _, after, _ = cr.composed(root, stem)
    assert after[2, 50, 20] > 3 and all(after[f, 45, 60] == 9 for f in range(4, T)) and after[3, 45, 60] == 3
    assert after[3, 6, 50] == 77 and len(cr._ops(cr._rec(root, stem).folder)) == 3


def test_personal_ticks_and_flags_are_kept_apart_from_notes_and_edits(review):
    """Jamie 2026-11-17: tick cells off as good, flag a cell at a frame to revisit with a note to self."""
    root, stem = review
    assert cr.marks(root, stem) == {"ticks": {}, "flags": [], "resolved": []}
    cr.mark(root, stem, op="tick", cell=3, frame=4)
    m = cr.mark(root, stem, op="flag", cell=1, frame=6, note="check the split")
    assert list(m["ticks"]) == ["3"] and m["flags"][0]["cell"] == 1 and m["flags"][0]["note"] == "check the split"
    fid = m["flags"][0]["id"]
    assert cr.mark(root, stem, op="note", id=fid, note="fine")["flags"][0]["note"] == "fine"
    m = cr.mark(root, stem, op="untick", cell=3)
    m = cr.mark(root, stem, op="unflag", id=fid)
    assert m == {"ticks": {}, "flags": [], "resolved": []} == cr.marks(root, stem)
    with pytest.raises(ValueError):
        cr.mark(root, stem, op="flag", cell=None, frame=2)
    assert cr.load(root, stem)["edits"] == [] and not cr._read(cr._rec(root, stem).folder / "notes.json", [])


def test_grow_preview_shows_what_seed_and_grow_would_take_and_saves_nothing(tmp_path):
    """Jamie 2026-11-17: preview the pixels a seed / grow would include at the grow level."""
    root, stem = signal_review(tmp_path)
    key = cr.follow_key([3])
    p = cr.grow_preview(root, stem, 2, 50, 20, 0.4, "seed")
    picked = cr.cell_now(root, stem, 2, 50, 20)
    assert cr.edit(root, stem, action="seed", picked=picked, frame=2, to_frame=2, key=key, level=0.4)["ok"]
    _, after, _ = cr.composed(root, stem)
    seeded = {(y, x + i) for y, x, n in p["take"] for i in range(n)}
    n = int(after[2, 50, 20])
    assert p["count"] == len(seeded) > 0 and seeded == set(zip(*np.nonzero(after[2] == n)))   # preview == result
    assert cr.grow_preview(root, stem, 2, 45, 60, 0.4, "seed")["count"] == 0                 # on a cell: nothing
    low, high = (cr.grow_preview(root, stem, 2, 50, 20, lv, "grow")["count"] for lv in (0.1, 3.0))
    assert low >= high                                                                   # lower level takes more
    box = cr.grow_preview(root, stem, 2, 50, 20, 0.4, "seed", (40, 60, 10, 30))
    assert all(40 <= y < 60 and 10 <= x and x + k <= 30 for y, x, k in box["take"] + box["free"])
    assert len(cr._ops(cr._rec(root, stem).folder)) == 1


def test_ground_truth_snapshot_copies_the_state_and_says_what_is_checked(tmp_path):
    """Jamie 2026-11-17: save my edits as ground truth for improving the base tracking."""
    import json, tifffile
    from pymicroglia.ui import correction_truth as ct
    root, stem = signal_review(tmp_path)
    key = cr.follow_key([3])
    assert cr.edit(root, stem, action="seed", picked=cr.cell_now(root, stem, 2, 50, 20), frame=2, to_frame=2, key=key)["ok"]
    cr.mark(root, stem, op="tick", cell=3, frame=2)
    folder = cr._rec(root, stem).folder
    before = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
    m = ct.snapshot(root, stem, tmp_path / "truth" / "snap1", held_out_note="none")
    out = tmp_path / "truth" / "snap1"
    assert {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()} == before          # read-only
    assert json.loads((out / "edits.json").read_text()) == cr._ops(folder)
    _, after, _ = cr.composed(root, stem)
    assert np.array_equal(tifffile.imread(out / "truth_labels.tif"), after)
    assert m["edits"]["total"] == 1 and m["cells_by_status"]["ticked_good"] == 1
    status = {r.split(",")[0]: r.split(",")[5] for r in (out / "cells.csv").read_text().splitlines()[1:]}
    assert status["3"] == "ticked_good" and "edited" in status.values()
    with pytest.raises(FileExistsError):
        ct.snapshot(root, stem, tmp_path / "truth" / "snap1")                                  # never overwritten


def test_timeline_suspects_lists_flagged_events_and_leaves_out_ticked_cells(tmp_path):
    from pymicroglia.ui import correction_truth as ct
    root, stem = signal_review(tmp_path)
    T = cr._rec(root, stem).data["T"]
    assert cr.edit(root, stem, action="delete", picked=cr.cell_now(root, stem, 5, 45, 60), frame=5, to_frame=T - 1,
                   key=cr.follow_key([3]))["ok"]                                      # 3 now ends mid-recording
    every = ct.suspects(root, stem, limit=999)
    assert any(e["cell"] == 3 and e["kind"] == "end" and e["jamie_edited_here"] for e in every["events"])
    assert every["total"] == len(every["events"]) and every["total"] > 0
    order = [ct.SUSPECT_ORDER.index(e["kind"]) for e in every["events"] if not e["jamie_edited_here"]]
    assert order == sorted(order)
    n = every["events"][0]["cell"]
    cr.mark(root, stem, op="tick", cell=n)
    assert all(e["cell"] != n for e in ct.suspects(root, stem, limit=999)["events"])
    assert any(e["cell"] == n for e in ct.suspects(root, stem, limit=999, include_ticked=True)["events"])


def test_export_button_saves_dated_truth_and_points_latest_at_it(tmp_path):
    """Jamie 2026-11-17: one button that exports and sends to the truth folder, any number of times."""
    import json, time
    from pathlib import Path
    from pymicroglia.ui import correction_truth as ct
    root, stem = signal_review(tmp_path)
    with pytest.raises(ValueError):
        ct.save_truth(root, stem)                                                   # no folder set yet
    (Path(root) / cr.FOLDER / ct.SETTINGS).write_text(json.dumps({"folder": str(tmp_path / "truth")}))
    a = ct.save_truth(root, stem)
    assert a["ok"] and "not connected" in a["export_note"] and Path(a["snapshot"], "truth_labels.tif").is_file()
    time.sleep(1.1)
    b = ct.save_truth(root, stem)
    assert a["snapshot"] != b["snapshot"] and Path(a["snapshot"]).is_dir()                    # nothing overwritten
    latest = json.loads((tmp_path / "truth" / "latest.json").read_text())
    assert Path(tmp_path / "truth" / latest[stem]["snapshot"]) == Path(b["snapshot"])


def test_seed_runs_over_the_frames_set_and_takes_the_number_set(tmp_path):
    """Jamie 2026-11-18: set the frames the seed runs for and the number it gets."""
    root, stem = signal_review(tmp_path)
    key = cr.follow_key([3])
    done = cr.edit(root, stem, action="seed", picked=cr.cell_now(root, stem, 4, 50, 20), frame=4, to_frame=4,
                   key=key, number=42, range_from=3, range_to=6)
    assert done["ok"], done
    _, after, _ = cr.composed(root, stem)
    assert [int(after[f, 50, 20]) for f in range(10)] == [0, 0, 0, 42, 42, 42, 42, 0, 0, 0]


def test_split_preview_shows_the_sides_and_flip_chooses_which_keeps_the_number(tmp_path):
    """Jamie 2026-11-18: know which side of a split gets which number."""
    line = [[37, 58], [45, 58], [53, 58]]                       # off-centre: the right side is larger
    results = []
    for flip in (False, True):
        root, stem = signal_review(tmp_path / str(flip))
        key = cr.follow_key([3])
        p = cr.split_preview(root, stem, 2, line, flip=flip)
        keep = {(y, x + i) for y, x, n in p["keep"] for i in range(n)}
        assert p["host"] == 3 and keep and p["cut"]
        done = cr.edit(root, stem, action="split", picked=None, frame=2, to_frame=2, key=key, points=line,
                       flip=flip, number=77)
        assert done["ok"], done
        _, after, _ = cr.composed(root, stem)
        assert all(after[2][y, x] == 3 for y, x in keep)              # the previewed side keeps 3
        assert 77 in after[2]                                          # the typed number goes to the cut side
        results.append(int(after[2][45, 64]))
    assert results == [3, 77]                                          # flip swaps the sides


def test_track_on_carries_the_number_forward_where_the_tracker_switched(tmp_path):
    """Jamie 2026-11-18: run the tracking on from here after making room for a cell to continue."""
    root, stem = signal_review(tmp_path)
    key = cr.follow_key([3])
    assert cr.edit(root, stem, action="give", picked=cr.cell_now(root, stem, 6, 45, 60), frame=6, to_frame=9,
                   key=key, number=8)["ok"]                            # the 'tracker' renamed 3 as 8 from frame 6
    done = cr.edit(root, stem, action="track_on", picked=cr.cell_now(root, stem, 5, 45, 60), frame=5, key=key)
    assert done["ok"], done
    _, after, _ = cr.composed(root, stem)
    assert all(after[f, 45, 60] == 3 for f in range(10)) and 8 not in after
    again = cr.edit(root, stem, action="track_on", picked=cr.cell_now(root, stem, 5, 45, 60), frame=5, key=key)
    assert not again["ok"]


def test_timeline_events_can_be_marked_resolved_and_box_select_finds_cells(review):
    root, stem = review
    m = cr.mark(root, stem, op="resolve", cell=3, frame=4, note="gap")
    assert [r[:3] for r in m["resolved"]] == [[3, 4, "gap"]]
    assert cr.mark(root, stem, op="unresolve", cell=3, frame=4, note="gap")["resolved"] == []
    found = cr.cells_in_box(root, stem, 2, 0, 60, 0, 80)
    assert {c["n"] for c in found} == {1, 3}
    lab = cr.current_frame_fn(cr._rec(root, stem))(2)
    assert all(lab[c["y"], c["x"]] == c["n"] for c in found)


def test_fill_gaps_seeds_at_the_last_centroid_with_the_cells_number(tmp_path):
    """Jamie 2026-11-18: fill a cell's gaps by seeding at its last centroid at the grow level."""
    root, stem = signal_review(tmp_path)
    key = cr.follow_key([3])
    assert cr.edit(root, stem, action="delete", picked=cr.cell_now(root, stem, 4, 45, 60), frame=4, to_frame=6,
                   key=key)["ok"]                                    # 3 missing in frames 4-6, back at 7
    _, after, _ = cr.composed(root, stem)
    assert not (after[4:7] == 3).any()
    done = cr.edit(root, stem, action="fill_gaps", picked=cr.cell_now(root, stem, 2, 45, 60), frame=2, key=key)
    assert done["ok"], done
    _, after, _ = cr.composed(root, stem)
    assert all(after[f, 45, 60] == 3 for f in range(10))
    assert not cr.edit(root, stem, action="fill_gaps", picked=cr.cell_now(root, stem, 2, 45, 60), frame=2,
                       key=key)["ok"]                                # no gaps left


def test_seed_keeps_following_a_small_dim_cell():
    """Jamie 2026-11-18: seeds of 30-56 px cells stopped after one frame under a fixed 60-px floor."""
    from pymicroglia.tracking import correction_grow as cg
    assert cg.keeps_going(47, 47, 56) and cg.keeps_going(47, 56, 30)
    assert not cg.keeps_going(47, 47, 10)            # shrank to under a third of the cell
    assert not cg.keeps_going(47, 20, 70)            # jumped more than three-fold in one frame
    assert not cg.keeps_going(47, 47, 0)


def test_trim_erases_the_cut_off_side_without_a_new_number(tmp_path):
    """Jamie 2026-11-18: cut back a cell that grew too far; the cut-off part takes no identity."""
    root, stem = signal_review(tmp_path)
    _, before, _ = cr.composed(root, stem)
    line = [(37, 58), (41, 58), (45, 58), (49, 58), (53, 58)]
    done = cr.edit(root, stem, action="trim", picked=None, frame=2, to_frame=0, key=cr.follow_key([3]), points=line)
    assert done["ok"] and done["message"].startswith("Trimmed 3"), done
    _, after, _ = cr.composed(root, stem)
    assert set(np.unique(after[2]).tolist()) <= set(np.unique(before[2]).tolist())   # no new number
    assert 0 < (after[2] == 3).sum() < (before[2] == 3).sum()                            # 3 lost one side
    assert ((before[2] == 3) & (after[2] == 0)).sum() == (before[2] == 3).sum() - (after[2] == 3).sum()


def test_fence_keeps_a_cell_out_of_an_area_in_every_later_frame(tmp_path):
    """Jamie 2026-11-18: a boundary the cell never goes into, instead of splitting every frame."""
    root, stem = signal_review(tmp_path)
    _, before, _ = cr.composed(root, stem)
    ys, xs = np.nonzero(before[3] == 3)
    cx = int(xs.mean())
    area = [(0, cx), (0, 200), (200, 200), (200, cx)]          # everything right of the cell's centre
    done = cr.edit(root, stem, action="fence", picked=None, frame=3, cells=[3], points=area, key=cr.follow_key([3]))
    assert done["ok"], done
    _, after, _ = cr.composed(root, stem)
    assert not (after[3:, :, cx:] == 3).any()                    # never right of the line from frame 3 on
    assert (after[3:, :, :cx] == 3).any(axis=(1, 2)).all()      # still there on the left
    assert (after[:3] == before[:3]).all()                       # earlier frames untouched
    assert not cr.edit(root, stem, action="fence", picked=None, frame=3, cells=[3], points=area,
                       key=cr.follow_key([3]))["ok"]             # nothing left to fence


def test_keep_in_removes_the_cell_outside_the_area_in_every_later_frame(tmp_path):
    """Jamie 2026-11-18: the opposite of a fence - the cell has to stay within an area."""
    root, stem = signal_review(tmp_path)
    _, before, _ = cr.composed(root, stem)
    ys, xs = np.nonzero(before[3] == 3)
    cx = int(xs.mean())
    area = [(0, 0), (0, cx), (200, cx), (200, 0)]               # everything left of the cell's centre
    done = cr.edit(root, stem, action="keep_in", picked=None, frame=3, cells=[3], points=area, key=cr.follow_key([3]))
    assert done["ok"] and done["message"].startswith("Kept 3 inside"), done
    _, after, _ = cr.composed(root, stem)
    assert not (after[3:, :, cx + 1:] == 3).any()                # nothing right of the area from frame 3 on
    assert (after[3:] == 3).any(axis=(1, 2)).all()               # still there inside it
    assert (after[:3] == before[:3]).all()                        # earlier frames untouched
    other = (before[3:] != 3) & (before[3:] > 0)
    assert (after[3:][other] == before[3:][other]).all()          # other cells untouched


def test_seeds_queued_together_share_the_signal_between_them():
    """Jamie 2026-11-18: two seeds in one patch of signal each get their own side, not first-come-takes-all."""
    from pymicroglia.tracking import correction_grow as cg
    H, W, T = 30, 60, 3
    yy, xx = np.indices((H, W))
    raw = np.zeros((T, H, W), np.float32)
    for f in range(T):   # two blobs joined by a dim bridge, drifting right one pixel a frame
        raw[f] = (np.exp(-((yy - 15) ** 2 + (xx - 18 - f) ** 2) / 30) + np.exp(-((yy - 15) ** 2 + (xx - 40 - f) ** 2) / 30)
                  + 0.25 * ((abs(yy - 15) < 3) & (xx > 18 + f) & (xx < 40 + f)))
    empty = lambda f: np.zeros((H, W), np.int32)
    sig = lambda f: raw[f] > 0.15
    alone = cg.seed_track(empty, lambda f: raw[f], 0, T - 1, 15, 18, sig=sig)
    a, b = cg.seed_many(empty, lambda f: raw[f], 0, T - 1, [(15, 18), (15, 40)], sig=sig)
    assert alone[0][15, 40]                                  # one seed alone floods both blobs
    for f in range(T):
        assert a[f][15, 18 + f] and not a[f][15, 40 + f]     # together: each keeps its own blob
        assert b[f][15, 40 + f] and not b[f][15, 18 + f]
        assert not (a[f] & b[f]).any()
        assert (a[f] | b[f]).sum() == sig(f).sum()             # and between them they take all of it


def test_grows_queued_together_share_the_signal_between_the_cells():
    """Jamie 2026-11-18: "by seed i meant seed and growing cells" - two cells grown at once split the signal."""
    from pymicroglia.tracking import correction_grow as cg
    H, W, T = 30, 60, 2
    yy, xx = np.indices((H, W))
    raw = np.zeros((T, H, W), np.float32)
    raw[:] = (np.exp(-((yy - 15) ** 2 + (xx - 18) ** 2) / 30) + np.exp(-((yy - 15) ** 2 + (xx - 40) ** 2) / 30)
              + 0.25 * ((abs(yy - 15) < 3) & (xx > 18) & (xx < 40)))
    cores = np.zeros((H, W), np.int32)
    cores[(yy - 15) ** 2 + (xx - 18) ** 2 <= 4] = 1
    cores[(yy - 15) ** 2 + (xx - 40) ** 2 <= 4] = 2
    frame = lambda f: cores
    sig = lambda f: raw[f] > 0.15
    alone = cg.grow_track(frame, lambda f: raw[f], 1, 0, 1, sig=sig)
    assert alone[0][15, 44]                                   # cell 1 grown alone reaches round cell 2
    seeds, (g1, g2) = cg.seed_many(frame, lambda f: raw[f], 0, 1, [], sig=sig, grows=[(1, 1), (2, 1)])
    for f in range(T):
        assert g1[f][15, 22] and not g1[f][15, 44] and g2[f][15, 44] and not g2[f][15, 22]
        assert not (g1[f] & g2[f]).any() and not (g1[f] & (cores > 0)).any()


def test_trace_grid_measures_the_current_outlines_and_render_makes_the_movie(tmp_path):
    """Jamie 2026-11-18: the Render / Movie buttons and a trace grid button."""
    import imageio_ffmpeg, time
    from pymicroglia.ui import correction_outputs as co
    root, stem = signal_review(tmp_path)
    tr = co.traces(root, stem)
    _, after, _ = cr.composed(root, stem)
    three = next(c for c in tr["cells"] if c["n"] == 3)
    assert [v is not None for v in three["total"]] == [(after[f] == 3).any() for f in range(len(after))]
    assert co.render_movie(root, stem)["ok"]
    for _ in range(300):
        if co.render_status(root, stem)["job"]["state"] != "running":
            break
        time.sleep(0.1)
    assert co.render_status(root, stem)["job"]["state"] == "done"
    assert imageio_ffmpeg.count_frames_and_secs(str(co.movie_file(root, stem)))[0] == len(after)
