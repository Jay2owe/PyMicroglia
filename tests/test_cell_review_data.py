"""Frozen identity and original-pixel contracts for the interactive review."""
import base64
import hashlib
from io import BytesIO
import json

import numpy as np
from PIL import Image
import pytest
import tifffile

from pymicroglia.figure_tables.cell_review_data import prepare, measurement_rows
from pymicroglia.visualisation.cell_review import cell_review


@pytest.fixture
def recording(tmp_path):
    raw = (np.arange(6 * 12 * 14).reshape(6, 12, 14) % 193).astype(np.uint16)
    labels = np.zeros((5, 12, 14), np.uint16)
    labels[0, 3:5, 2:5] = 19
    labels[1, 3:5, 3:6] = 19
    labels[2, 3:5, 3:6] = 19
    labels[2, 2, 4] = 19
    labels[4, 5:7, 7:10] = 19
    labels[0, 9, 11] = 26
    labels[0, 10, 12] = 26
    rp, lp = tmp_path / "raw.tif", tmp_path / "labels.tif"
    tifffile.imwrite(rp, raw, photometric="minisblack", metadata={"axes": "TYX"})
    tifffile.imwrite(lp, labels, photometric="minisblack", metadata={"axes": "TYX"})
    return rp, lp, raw, labels


def test_native_frames_original_sums_and_gaps(recording):
    rp, lp, raw, labels = recording
    payload = prepare(rp, lp, source_frame_offset=1, frame_interval_h=.25)
    assert payload["ids"] == [19, 26]
    assert payload["times_h"] == [0, .25, .5, .75, 1, 1.25]
    cell = payload["cells"]["19"]
    assert cell["area"] == [None, 6., 6., 7., None, 6.]
    for native in (1, 2, 3, 5):
        assert cell["signal_raw"][native] == raw[native][labels[native - 1] == 19].sum()
    assert cell["signal_mean3"][2] == np.mean(cell["signal_raw"][1:4])
    assert all(value is None for value in cell["signal_mean3"][:2] + cell["signal_mean3"][3:])
    assert cell["observations"][0] is None and cell["observations"][4] is None
    for native in (1, 2, 3, 5):
        decoded = np.zeros(labels.shape[1:], bool)
        runs = cell["observations"][native]["runs"]
        for y, x0, x1 in np.array(runs).reshape(-1, 3):
            decoded[y, x0:x1] = True
        np.testing.assert_array_equal(decoded, labels[native - 1] == 19)
    assert len(list(measurement_rows(payload))) == 12


def test_exact_display_mapping_not_denoised(recording):
    rp, lp, raw, _ = recording
    data = prepare(rp, lp, display_range=(0, 192))
    first = np.asarray(Image.open(BytesIO(base64.b64decode(data["frames"][0].split(",")[1]))))
    expected = np.round(raw[0].astype(float) / 192 * 255).astype(np.uint8)
    np.testing.assert_array_equal(first[..., 0], expected)
    assert not first[..., 1:].any()
    assert data["display"]["filter"] is None
    assert data["display"]["show_motion_trails"] is False


def test_holes_and_disconnected_components_are_retained(tmp_path):
    labels = np.zeros((2, 12, 12), np.uint16)
    labels[:, 1:8, 1:8] = 8
    labels[:, 3:6, 3:6] = 0
    labels[:, 10, 10] = 8
    rp, lp = tmp_path / "raw.tif", tmp_path / "labels.tif"
    tifffile.imwrite(rp, np.ones_like(labels), photometric="minisblack")
    tifffile.imwrite(lp, labels, photometric="minisblack")
    data = prepare(rp, lp)
    assert data["cells"]["8"]["area"] == [41., 41.]
    assert len(data["cells"]["8"]["observations"][0]["contours"]) == 3


def test_portable_export_does_not_mutate_or_replace_inputs(recording, tmp_path):
    rp, lp, _, _ = recording
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (rp, lp)]
    output = tmp_path / "review.html"
    result = cell_review(rp, lp, output=output, source_frame_offset=1,
                         title="unsafe </script><h1> title")
    assert before == [hashlib.sha256(path.read_bytes()).hexdigest() for path in (rp, lp)]
    receipt = json.loads((tmp_path / "review_review.json").read_text())
    assert receipt["source_sha256"] == dict(zip(("raw", "labels"), before))
    assert receipt["table_rows"] == 12 and result["cells"] == 2
    html = output.read_text(encoding="utf-8")
    assert "data:image/png;base64," in html and "unsafe </script>" not in html
    assert "src=\"http" not in html and "href=\"http" not in html
    with pytest.raises(FileExistsError):
        cell_review(rp, lp, output=output)


@pytest.mark.parametrize("kwargs", [
    {"source_frame_offset": -1}, {"source_frame_offset": 2},
    {"source_frame_offset": 1.5}, {"frame_interval_h": 0},
    {"display_lut": "filtered"}, {"display_range": [2, 1]},
])
def test_incompatible_settings_are_rejected(recording, kwargs):
    with pytest.raises(ValueError):
        prepare(recording[0], recording[1], **kwargs)
