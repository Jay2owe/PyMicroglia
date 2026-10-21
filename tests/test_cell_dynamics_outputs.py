"""Independent action wiring, native saved-run alignment and all output types."""
import json
from pathlib import Path
import numpy as np
import pytest
from test_cell_review_data import recording
from pymicroglia import catalogue
from pymicroglia.registry import resolve, CLAIM_TEMPLATES
from pymicroglia.visualisation.figures import load
from pymicroglia.figure_tables.saved import Tables


@pytest.fixture
def saved_recording(recording, tmp_path):
    raw, labels, _, _ = recording
    run = tmp_path / "saved"
    run.mkdir()
    manifest = {"movies": [{"stem": "sample", "spec": {"source_frame_offset": 1},
        "summary": {"scale": {"minutes_per_frame": 15}},
        "provenance": {"inputs": {"raw": {"path": str(raw)}, "labels": {"path": str(labels)}}}}]}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


def test_every_method_has_its_own_registered_entry_point():
    names = ("mask_change_map", "cell_time_sculpture", "cell_trace_card", "recording_barcode", "cell_card_video")
    for name in names:
        entry = next(row for row in catalogue.actions() if row["name"] == name)
        assert callable(resolve(entry["method"])) and "{source}" in CLAIM_TEMPLATES[name]
        assert entry["display_only"]


@pytest.mark.parametrize("name,overrides", [
    ("mask_change_map", {"identity": 19, "source_frame_index": 2}),
    ("cell_time_sculpture", {"identity": 19}),
    ("cell_trace_card", {"identity": 19, "source_frame_index": 4}),
    ("recording_barcode", {}),
])
def test_individual_figures_prepare_from_saved_inputs(name, overrides, saved_recording):
    import importlib
    spec = load()[name]
    module, function = spec.prepare.split(":")
    settings = {option.name: option.default for option in spec.options} | overrides
    data, statistics = getattr(importlib.import_module(module), function)(Tables(saved_recording), settings)
    assert statistics is None and set(data) == {view.key for view in spec.views}
    assert not any("html" in str(key).lower() for key in data)


def test_independent_video_stream_has_every_native_frame_and_a_receipt(saved_recording, tmp_path, monkeypatch):
    from pymicroglia.video.cell_cards import cell_card_video
    from auto_organotypic.video import encode
    captured = []
    def encode_frames(path, frames, *, fps):
        captured.extend(frames)
        Path(path).write_bytes(b"encoded-test-video")
        return path
    monkeypatch.setattr(encode, "write_video", encode_frames)
    result = cell_card_video(saved_recording, identities=[19, 26], frame_width_px=640,
                             output_dir=tmp_path / "movie")
    assert len(captured) == result["frames"] == 6
    assert all(frame.shape == (480, 640, 3) for frame in captured)
    assert result["identities"] == [19, 26]
    assert result["source_frame_offset"] == 1 and result["frame_interval_h"] == .25
    assert result["display_filter"] is None and not result["motion_trails"]
    assert Path(result["table"]).is_file() and Path(result["receipt"]).is_file()
    assert not list((tmp_path / "movie").glob("*.html"))
    with pytest.raises(FileExistsError):
        cell_card_video(saved_recording, identities=[19, 26], output_dir=tmp_path / "movie")
