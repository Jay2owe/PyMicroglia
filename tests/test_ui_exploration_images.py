import base64
import hashlib
import json
from io import BytesIO
import numpy as np
import pytest
import tifffile
from PIL import Image
from pymicroglia.ui.exploration_images import load_cell_frame, load_cell_images, load_change, load_frame, load_frame_image, load_history


def test_image_lookup_tables_match_shared_export_colours():
    from auto_organotypic.render import luts
    from pymicroglia.ui.exploration_images import image_luts
    result = image_luts()
    assert result["default"] == "grays"
    assert {entry["key"] for entry in result["maps"]} == set(luts.lut_names())
    for entry in result["maps"]:
        table = np.asarray(entry["table"])
        assert table.shape == (256, 3)
        np.testing.assert_array_equal(table, luts.paint(np.arange(256, dtype=np.float32) / 255, entry["key"]))
    gray = next(entry["table"] for entry in result["maps"] if entry["key"] == "grays")
    assert gray[0] == [0, 0, 0] and gray[255] == [255, 255, 255]


def test_moved_recording_reloads_exact_cell_media(tmp_path):
    original = tmp_path / "original"
    sources = original / "sources"
    run = original / "results" / "measurement"
    sources.mkdir(parents=True)
    (run / ".auto-organotypic").mkdir(parents=True)
    raw = np.arange(3 * 24 * 24, dtype=np.uint16).reshape(3, 24, 24)
    labels = np.zeros_like(raw)
    labels[:, 7:11, 8:12] = 19
    spec = {"source_frame_offset": 1, "sha256": {}}
    for role, pixels in (("raw", raw), ("labels", labels[:2])):
        path = sources / (role + ".tif")
        tifffile.imwrite(path, pixels, photometric="minisblack")
        spec[role] = str(path)
        spec["sha256"][role] = hashlib.sha256(path.read_bytes()).hexdigest()
    saved = {"run": {"folder": str(run)}, "movies": [{"stem": "test", "spec": spec,
             "display": {"contrast": [0, 2000]}}]}
    manifest = run / ".auto-organotypic" / "manifest.json"
    manifest.write_text(json.dumps(saved))
    before = manifest.read_bytes()
    moved = tmp_path / "moved"
    original.rename(moved)
    run = moved / "results" / "measurement"

    frame = load_frame(run, "test", 0)
    shown = np.array(Image.open(BytesIO(base64.b64decode(frame["image"].split(",")[1]))))
    np.testing.assert_array_equal(shown, (raw[1].astype(float) * 255 / 2000).astype(np.uint8))
    assert frame["layers"]["accepted"][0]["identity"] == 19
    assert load_history(run, "test", 19)["count"] == 2
    assert load_cell_images(run, "test", 19)["available"]
    assert load_cell_frame(run, "test", 19, 1)["observation"]
    assert load_change(run, "test", 19, 1)["change"]
    assert (run / ".auto-organotypic" / "manifest.json").read_bytes() == before

    # A different recording at the relocated path must never be substituted.
    tifffile.imwrite(moved / "sources" / "raw.tif", raw + 1, photometric="minisblack")
    with pytest.raises(ValueError, match="relocated raw source does not match"):
        load_frame_image(run, "test", 0)


def test_native_pixels_offsets_holes_and_missingness(tmp_path):
    raw = np.arange(6 * 12 * 12, dtype=np.uint16).reshape(6, 12, 12)
    labels = np.zeros((4, 12, 12), dtype=np.uint16)
    labels[:, 2:8, 2:8] = 19
    labels[:, 4:6, 4:6] = 0
    labels[1] = 0
    labels[:, 10, 10] = 26
    for name, value in (("raw", raw), ("labels", labels)):
        tifffile.imwrite(tmp_path / (name + ".tif"), value, photometric="minisblack")
    movie = {"stem": "test", "spec": {"raw": "raw.tif", "labels": "labels.tif", "source_frame_offset": 2}, "display": {"contrast": [0, 1000]}}
    (tmp_path / "manifest.json").write_text(json.dumps({"movies": [movie]}))
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    frame = load_frame(tmp_path, "test", 0)
    image_frame = load_frame_image(tmp_path, "test", 0)
    png = np.array(Image.open(BytesIO(base64.b64decode(frame["image"].split(",")[1]))))
    np.testing.assert_array_equal(png, (raw[2].astype(float) * 255 / 1000).astype(np.uint8))
    assert image_frame["image"] == frame["image"]
    assert image_frame["source_frame_index"] == 2
    assert "layers" not in image_frame
    assert len(frame["layers"]["accepted"][0]["contours"]) == 2
    assert load_history(tmp_path, "test", 19, 1)["observations"][1]["observation"] is None
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()} == before
    with pytest.raises(ValueError, match="outside"):
        load_frame(tmp_path, "test", 8)


def test_history_keeps_exact_outlines_between_display_samples(tmp_path):
    labels = np.zeros((105, 12, 12), dtype=np.uint16)
    for frame in range(len(labels)):
        x = 2 + frame % 5
        labels[frame, 3:6, x:x + 3] = 19
    labels[55] = 0
    tifffile.imwrite(tmp_path / "labels.tif", labels, photometric="minisblack")
    (tmp_path / "manifest.json").write_text(json.dumps({"movies": [{"stem": "test", "spec": {"labels": "labels.tif"}}]}))

    history = load_history(tmp_path, "test", 19, 53)
    assert history["stride"] == 2
    assert history["exact"] is True
    assert [row["frame_index"] for row in history["observations"]] == list(range(105))
    assert history["observations"][55]["observation"] is None
    assert history["observations"][54]["observation"]["centre"] != history["observations"][56]["observation"]["centre"]


def test_cell_image_strip_uses_native_source_pixels_and_keeps_missing_frames(tmp_path):
    raw = np.arange(4 * 48 * 48, dtype=np.uint16).reshape(4, 48, 48) % 1000
    labels = np.zeros((3, 48, 48), dtype=np.uint16)
    labels[0, 8:12, 9:13] = 19
    labels[2, 9:13, 12:16] = 19
    tifffile.imwrite(tmp_path / "raw.tif", raw, photometric="minisblack")
    tifffile.imwrite(tmp_path / "labels.tif", labels, photometric="minisblack")
    movie = {"stem": "test", "spec": {"raw": "raw.tif", "labels": "labels.tif", "source_frame_offset": 1}, "display": {"contrast": [0, 1000]}}
    (tmp_path / "manifest.json").write_text(json.dumps({"movies": [movie]}))

    strip = load_cell_images(tmp_path, "test", 19)
    assert strip["available"] and len(strip["frames"]) == 3
    assert strip["frames"][1]["image_bounds"] == strip["frames"][0]["image_bounds"]
    for frame in strip["frames"]:
        x0, y0, x1, y1 = frame["image_bounds"]
        shown = np.array(Image.open(BytesIO(base64.b64decode(frame["image"].split(",")[1]))))
        expected = (raw[frame["source_frame_index"], y0:y1, x0:x1].astype(float) * 255 / 1000).astype(np.uint8)
        np.testing.assert_array_equal(shown, expected)


def test_signed_change_uses_saved_identity_pixels_and_preserves_absence(tmp_path):
    raw = np.ones((3, 20, 20), dtype=np.uint16) * 100
    labels = np.zeros((3, 20, 20), dtype=np.uint16)
    labels[0, 4:7, 4:7] = 19
    labels[1, 4:7, 6:9] = 19
    labels[1, 6, 8] = 26
    tifffile.imwrite(tmp_path / "raw.tif", raw, photometric="minisblack")
    tifffile.imwrite(tmp_path / "labels.tif", labels, photometric="minisblack")
    (tmp_path / "manifest.json").write_text(json.dumps({"movies": [{"stem": "test", "spec": {
        "raw": "raw.tif", "labels": "labels.tif"}, "display": {"contrast": [0, 200]}}]}))
    strip = load_cell_images(tmp_path, "test", 19)
    for frame in range(3):
        row = strip["frames"][frame]
        x0, y0, x1, y1 = row["change_bounds"]
        shown = np.array(Image.open(BytesIO(base64.b64decode(row["change"].split(",")[1]))))
        expected = np.zeros((20, 20, 4), dtype=np.uint8)
        if frame:
            now, before = labels[frame] == 19, labels[frame - 1] == 19
            expected[now & ~before] = (114, 222, 211, 240)
            expected[before & ~now] = (239, 199, 123, 240)
        np.testing.assert_array_equal(shown, expected[y0:y1, x0:x1])
        direct = load_change(tmp_path, "test", 19, frame)
        assert direct["change"] == row["change"]
        combined = load_cell_frame(tmp_path, "test", 19, frame)
        expected_positions = [{"identity": item["identity"], "x": item["centre"][0], "y": item["centre"][1]}
                              for item in load_frame(tmp_path, "test", frame)["layers"]["accepted"]]
        assert combined["positions"] == expected_positions
        assert load_cell_frame(tmp_path, "test", 999, frame)["positions"] == expected_positions
        assert combined["change"] == direct["change"]
        assert combined["change_bounds"] == direct["change_bounds"]
        assert combined["image_bounds"] == direct["change_bounds"]
        cx0, cy0, cx1, cy1 = combined["image_bounds"]
        image = np.array(Image.open(BytesIO(base64.b64decode(combined["image"].split(",")[1]))))
        np.testing.assert_array_equal(image, (raw[frame, cy0:cy1, cx0:cx1].astype(float) * 255 / 200).astype(np.uint8))
    assert strip["frames"][2]["observation"] is None
    assert load_cell_frame(tmp_path, "test", 19, 1)["observation"] is not None
    assert not load_cell_frame(tmp_path, "test", 19, 2)["observation"]
