"""What a film is allowed to show, before any pixels are encoded.

The expensive half of ``analysis.cell_videos`` is rendering; the half that can
be wrong without anyone noticing is the arithmetic that decides which frames the
film covers, where the window sits, and what happens when the window runs off
the edge of the field. That is what these check, on hand-drawn labels.

Run with ``python -m pytest analysis/test_cell_videos.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymicroglia.figure_tables import follow as cell_videos


FRAMES, HEIGHT, WIDTH = 12, 40, 40


def _labels(*, first: int = 2, last: int = 9, step: int = 2) -> np.ndarray:
    """One 6x6 cell that walks to the right two pixels a frame."""
    labels = np.zeros((FRAMES, HEIGHT, WIDTH), dtype=np.uint16)
    for frame in range(first, last + 1):
        left = 5 + step * (frame - first)
        labels[frame, 15:21, left:left + 6] = 1
    return labels


def _cell_frame(labels: np.ndarray) -> pd.DataFrame:
    rows = []
    for frame in range(labels.shape[0]):
        found = np.nonzero(labels[frame] == 1)
        if found[0].size:
            rows.append({"identity": 1, "frame_index": frame, "hours": frame * 0.5,
                         "centroid_y": float(found[0].mean()),
                         "centroid_x": float(found[1].mean())})
    return pd.DataFrame(rows)


def _plan(labels, options=None):
    settings = {**cell_videos.DEFAULTS, **(options or {})}
    return cell_videos._plan(_cell_frame(labels), 1, labels, settings)


# ------------------------------------------------------------------ the frames

def test_the_default_film_covers_the_whole_recording() -> None:
    """'Over the duration of the recording' means the frames before and after it too."""
    frames, _, named, _ = _plan(_labels())
    assert frames == list(range(FRAMES))
    assert named.sum() == 8 and (~named).sum() == 4


def test_a_lifespan_film_is_trimmed_to_the_frames_the_name_was_on() -> None:
    frames, _, named, _ = _plan(_labels(), {"span": "lifespan"})
    assert frames == list(range(2, 10))
    assert named.all()


def test_an_unknown_span_is_refused() -> None:
    with pytest.raises(ValueError, match="recording or lifespan"):
        _plan(_labels(), {"span": "everything"})


def test_a_missing_cell_is_refused_rather_than_filmed_empty() -> None:
    labels = _labels()
    with pytest.raises(ValueError, match="not in this run"):
        cell_videos._plan(_cell_frame(labels), 99, labels, dict(cell_videos.DEFAULTS))


# ------------------------------------------------------------------ the window

def test_the_window_keeps_one_size_for_the_whole_film() -> None:
    """A window that resized would make every change of shape ambiguous."""
    _, _, _, window = _plan(_labels())
    assert window == 2 * (3 + cell_videos.DEFAULTS["crop_margin_px"]) + 1


def test_an_explicit_window_size_wins() -> None:
    _, _, _, window = _plan(_labels(), {"crop_px": 30})
    assert window == 31


def test_the_window_follows_the_cell_and_holds_still_where_it_is_missing() -> None:
    _, centres, named, _ = _plan(_labels())
    walking = centres[named]
    assert np.all(np.diff(walking[:, 1]) > 0)         # it moves right while named
    assert centres[0, 1] == walking[0, 1]             # held at the first known place
    assert centres[-1, 1] == walking[-1, 1]           # and at the last


def test_a_window_over_the_edge_is_padded_rather_than_slid_back() -> None:
    """A cell at the edge is exactly the cell whose ending is being judged."""
    image = np.ones((20, 20))
    out = cell_videos._window(image, (1.0, 1.0), 11)
    assert out.shape == (11, 11)
    assert out[0, 0] == 0.0 and out[5, 5] == 1.0      # centre is still the cell
    assert (out == 0.0).sum() == 11 * 11 - 7 * 7


def test_the_region_covers_every_window_the_film_will_take() -> None:
    frames, centres, _, window = _plan(_labels())
    top, bottom, left, right = cell_videos._region(centres, window, (HEIGHT, WIDTH))
    half = window // 2
    assert top <= centres[:, 0].min() - half or top == 0
    assert right >= centres[:, 1].max() + half or right == WIDTH
    assert (bottom - top) <= HEIGHT and (right - left) <= WIDTH


# ---------------------------------------------------------------- the captions

def test_an_event_is_captioned_on_its_own_frame_and_held() -> None:
    events = pd.DataFrame([{"identity": 1, "frame_index": 4, "event": "died",
                            "partner_identity": pd.NA},
                           {"identity": 1, "frame_index": 2, "event": "born",
                            "partner_identity": 7}])
    written = cell_videos._captions(events, 1, hold=3)
    assert written[4].startswith("ground went dark")
    assert written[6] == written[4]                   # held for three frames
    assert 7 not in written
    assert "cell 7" in written[2]                     # the partner is named


def test_another_cell_s_events_are_never_captioned_on_this_film() -> None:
    events = pd.DataFrame([{"identity": 2, "frame_index": 4, "event": "died",
                            "partner_identity": pd.NA}])
    assert cell_videos._captions(events, 1, hold=3) == {}


def test_a_run_without_the_lifecycle_table_still_films() -> None:
    assert cell_videos._captions(None, 1, hold=3) == {}


# ----------------------------------------------------------------- the choices

def test_event_clips_keep_original_frame_numbers_and_held_centres():
    from pymicroglia.figure_tables.film_action import plans
    labels = _labels()
    labels[7] = 0
    frame = _cell_frame(labels)
    events = pd.DataFrame([{'identity':1,'frame_index':8,'event':'divided'},
                           {'identity':1,'frame_index':11,'event':'lost'}])
    clips = plans(frame,events,labels,[1],span='event',event_hours=.5,interval_min=30,
                  options=cell_videos.DEFAULTS)
    assert [c['frames'] for c in clips] == [[7,8,9],[10,11]]
    assert clips[0]['named'].tolist() == [False,True,True]
    assert clips[0]['centres'][0].tolist() == frame.loc[frame.frame_index.eq(6),['centroid_y','centroid_x']].iloc[0].tolist()
    assert clips[1]['hours'].tolist() == [5,5.5]


def _run(tmp_path):
    from pymicroglia._results import write_document
    import tifffile
    labels = _labels()
    stack = tmp_path/'labels.tif'
    tifffile.imwrite(stack,labels)
    folder = tmp_path/'measure'/'movie'
    folder.mkdir(parents=True)
    _cell_frame(labels).to_csv(folder/'cell_frame.csv',index=False)
    pd.DataFrame([{'identity':1,'frame_index':5,'event':'divided','event_status':'candidate'}]).to_csv(folder/'lifecycle_events.csv',index=False)
    write_document(tmp_path/'manifest.json',{'movies':[{'stem':'movie','provenance':{
        'inputs':{'labels':{'path':str(stack)}},'scale':{'minutes_per_frame':30}}}]})
    return tmp_path


def test_dry_run_writes_nothing_and_event_filter_is_checked(tmp_path):
    from pymicroglia.figure_tables.film_action import follow
    run = _run(tmp_path)
    before = sorted(str(p) for p in run.rglob('*'))
    plan = follow(run,events=['divided'],span='event',event_hours=.5,dry_run=True)
    assert plan['clips'][0]['frames'] == [4,5,6]
    assert plan['clips'][0]['captions'][5].startswith('a second name')
    assert sorted(str(p) for p in run.rglob('*')) == before
    with pytest.raises(ValueError,match='Unknown lifecycle'):
        follow(run,events=['invented'],dry_run=True)


def test_event_film_encodes_three_frames_with_one_ledger_and_no_zip(tmp_path):
    from pymicroglia.figure_tables.film_action import follow
    import imageio.v3 as imageio
    run = _run(tmp_path)
    result = follow(run,events=['divided'],span='event',event_hours=.5,
                    display_options={'frame_px':160})
    from pathlib import Path
    path = Path(result['clips'][0]['path'])
    assert path.is_file()
    assert sum(1 for _ in imageio.imiter(path)) == 3
    from auto_organotypic.store.ledger import ledger_path
    assert ledger_path(path.parent).is_file()
    assert not list(path.parent.rglob('*.png')) + list(path.parent.rglob('*.zip'))
    assert len(list(path.parent.rglob('*.json'))) == 1
