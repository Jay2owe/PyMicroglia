"""Does a region survive being written, and does Fiji's own file read back?

An ImageJ ``.roi`` is a fixed binary header and a ``RoiSet.zip`` is a zip of
them. Two things have to be true: what this package writes must come back
unchanged, and what the existing pipeline wrote must be readable — the second
being the one that would break silently, because a reader that agrees with its
own writer proves nothing.

**Nothing here opens Fiji.** The binary is checked field by field instead. A
modal ImageJ dialog blocks everything until somebody dismisses it by hand, and
the headless bridge that could open one arrives in stage 13.
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from pymicroglia import roi


@pytest.fixture
def store_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.setenv("PYMICROGLIA_DECISIONS", str(tmp_path / "decisions"))
    return tmp_path / "cache"


@pytest.fixture
def blob_mask():
    mask = np.zeros((120, 120), bool)
    yy, xx = np.mgrid[0:120, 0:120]
    mask[((yy - 55) ** 2 + (xx - 62) ** 2) < 30 ** 2] = True
    return mask


# ------------------------------------------------------ the layout, by field
def test_the_header_is_the_imagej_header(blob_mask):
    """Every field the format defines, read back with ``struct`` directly.

    Written out rather than delegated to the decoder, so that a decoder bug
    cannot hide an encoder bug — the two would agree with each other and this
    would still fail.
    """
    polygon = roi.polygon_from_mask(blob_mask, name="scn_left")
    blob = roi.encode_roi(polygon.x, polygon.y, name="scn_left")

    assert blob[0:4] == b"Iout"
    assert struct.unpack_from(">h", blob, 4)[0] == 227
    assert blob[6] == 0                                    # polygon
    top, left, bottom, right = struct.unpack_from(">hhhh", blob, 8)
    count = struct.unpack_from(">H", blob, 16)[0]
    assert struct.unpack_from(">h", blob, 34)[0] == 1      # stroke width
    header2 = struct.unpack_from(">I", blob, 60)[0]

    assert count == len(polygon)
    assert (top, left, bottom, right) == polygon.bounds
    assert header2 == 64 + 4 * count

    # the header2 block is not optional in practice: readers assume it is there
    assert len(blob) >= header2 + 64
    name_offset = struct.unpack_from(">I", blob, header2 + 16)[0]
    name_length = struct.unpack_from(">I", blob, header2 + 20)[0]
    assert name_offset == header2 + 64
    assert name_length == len("scn_left")
    assert blob[name_offset:name_offset + 2 * name_length].decode("utf-16-be") \
        == "scn_left"


def test_coordinates_are_stored_relative_to_the_bounding_box(blob_mask):
    """int16, and relative — which is what keeps a 500 px ROI under a kilobyte."""
    polygon = roi.polygon_from_mask(blob_mask, name="scn")
    blob = roi.encode_roi(polygon.x, polygon.y, name="scn")
    count = struct.unpack_from(">H", blob, 16)[0]
    top, left = struct.unpack_from(">hh", blob, 8)

    xs = np.array(struct.unpack_from(">%dh" % count, blob, 64))
    ys = np.array(struct.unpack_from(">%dh" % count, blob, 64 + 2 * count))

    assert xs.min() == 0 and ys.min() == 0
    np.testing.assert_array_equal(xs, np.rint(np.asarray(polygon.x) - left))
    np.testing.assert_array_equal(ys, np.rint(np.asarray(polygon.y) - top))


# --------------------------------------------------------------- round trips
def test_a_polygon_round_trips_to_the_same_points(blob_mask):
    polygon = roi.polygon_from_mask(blob_mask, name="scn_left")
    back = roi.decode_roi(roi.encode_roi(polygon.x, polygon.y, name="scn_left"))

    assert back.name == "scn_left"
    assert len(back) == len(polygon)
    # coordinates are stored as integers, so the round trip is exact only after
    # the first pass rounds them. Both halves are asserted below.
    np.testing.assert_allclose(back.x, np.asarray(polygon.x), atol=1.0)
    np.testing.assert_allclose(back.y, np.asarray(polygon.y), atol=1.0)


def test_the_encoding_is_a_fixed_point_after_one_pass(blob_mask):
    """Write, read, write again — and the bytes stop moving.

    They are not identical on the *first* re-encode, and that is the format
    rather than a fault: coordinates are int16, so a contour at x=27.5 becomes
    27 and the bounding box's right edge follows it in by one. Once integer,
    nothing moves again — which is the property that matters, because it means
    a ROI cannot drift by being opened and saved repeatedly.
    """
    polygon = roi.polygon_from_mask(blob_mask, name="scn")
    first = roi.encode_roi(polygon.x, polygon.y, name="scn")
    once = roi.decode_roi(first)
    second = roi.encode_roi(once.x, once.y, name=once.name)
    twice = roi.decode_roi(second)
    third = roi.encode_roi(twice.x, twice.y, name=twice.name)

    assert second == third
    np.testing.assert_array_equal(once.x, twice.x)
    np.testing.assert_array_equal(once.y, twice.y)


def test_a_roiset_round_trips_through_a_file(tmp_path, blob_mask):
    second = np.zeros_like(blob_mask)
    second[10:40, 70:110] = True

    polygons = [roi.polygon_from_mask(blob_mask, name="scn left"),
                roi.polygon_from_mask(second, name="scn right")]
    target = roi.write_roi_zip(tmp_path / "RoiSet.zip", polygons)

    back = roi.read_roi_zip(target)
    assert [p.name for p in back] == ["scn_left", "scn_right"]
    for before, after in zip(polygons, back):
        np.testing.assert_allclose(after.x, np.asarray(before.x), atol=1.0)
        np.testing.assert_allclose(after.y, np.asarray(before.y), atol=1.0)

    # and the mask it describes is still the same region, to within the half
    # pixel the format costs: ImageJ stores int16 coordinates, so a contour
    # point at x=38.5 comes back as 38. On a 30 px circle that half-pixel edge
    # shift is about 2 % of the area, and it is the format rather than the
    # round trip. What is exact is the coordinates, asserted above, and the
    # fixed point, asserted below.
    assert roi.dice(polygons[0].to_mask(blob_mask.shape),
                    back[0].to_mask(blob_mask.shape)) > 0.97
    assert roi.dice(blob_mask, back[0].to_mask(blob_mask.shape)) > 0.95


def test_a_mask_mapping_is_accepted_as_well_as_polygons(tmp_path, blob_mask):
    """Both are what a caller has: segmentation makes masks, a person draws one."""
    target = roi.write_roi_zip(tmp_path / "RoiSet.zip",
                               {"cell 1": blob_mask})
    assert [p.name for p in roi.read_roi_zip(target)] == ["cell_1"]


def test_writing_over_an_existing_roiset_is_refused(tmp_path, blob_mask):
    target = tmp_path / "RoiSet.zip"
    roi.write_roi_zip(target, {"cell": blob_mask})
    with pytest.raises(FileExistsError):
        roi.write_roi_zip(target, {"cell": blob_mask})
    roi.write_roi_zip(target, {"cell": blob_mask}, overwrite=True)


def test_a_parametric_roi_is_refused_rather_than_approximated():
    """A rectangle silently becoming its bounding box would change every number.

    ImageJ's rectangle, oval and line are a few *parameters*, not an outline,
    and turning one into vertices is a guess about what somebody meant. This
    refuses and says which types it can read instead.
    """
    for kind, name in ((1, "rectangle"), (2, "oval"), (3, "line")):
        blob = bytearray(roi.encode_roi([0, 10, 10, 0], [0, 0, 10, 10],
                                        name="r"))
        blob[6] = kind
        with pytest.raises(ValueError) as raised:
            roi.decode_roi(bytes(blob))
        assert name in str(raised.value)
        assert "polygon" in str(raised.value)


def test_a_freehand_outline_reads_like_a_polygon():
    """The commonest hand tracing in Fiji is a freehand, not a polygon.

    Polygon, freehand and traced differ only in how a person drew them and are
    stored identically, as a list of vertices. Refusing type 7 refused the
    reference dataset's own hand-drawn region.
    """
    for kind in (roi.TYPE_POLYGON, roi.TYPE_FREEHAND, roi.TYPE_TRACED):
        blob = bytearray(roi.encode_roi([0, 10, 10, 0], [0, 0, 10, 10],
                                        name="SCN"))
        blob[6] = kind
        outline = roi.decode_roi(bytes(blob))
        assert outline.name == "SCN"
        assert len(outline) == 4
        assert outline.position["type"] == kind


def test_something_that_is_not_a_roi_is_refused():
    with pytest.raises(ValueError) as raised:
        roi.decode_roi(b"PK\x03\x04 this is a zip, not a roi")
    assert "not an ImageJ ROI" in str(raised.value)


# ---------------------------------- against a RoiSet the existing pipeline wrote
REFERENCE = Path("reference-data") / "dluc-pipeline"


@pytest.fixture
def engine_roiset():
    target = REFERENCE / "RoiSet_auto.zip"
    if not target.is_file():
        pytest.skip(f"the reference RoiSet is not here: {target.name}")
    return target


def test_the_pipelines_own_roiset_reads_back(engine_roiset):
    """The half that a self-consistent reader would not catch.

    ``RoiSet_auto.zip`` was written by ``dluc_pipeline.py`` on another machine
    in August. If this package's reader could only read this package's writer,
    it would pass every other test in this file and still fail the first time
    somebody pointed it at a real file.
    """
    polygons = roi.read_roi_zip(engine_roiset)

    assert [p.name for p in polygons] == ["SCN_left_lobe", "SCN_right_lobe",
                                          "SCN_both_lobes"]
    for polygon in polygons:
        assert len(polygon) > 20
        assert np.asarray(polygon.x).min() >= 0
        assert np.asarray(polygon.y).min() >= 0
        assert polygon.position["version"] == roi.VERSION

    both = polygons[2]
    assert len(both) > len(polygons[0])          # both lobes, more outline


def test_re_encoding_the_pipelines_rois_changes_nothing_that_matters(
        engine_roiset):
    """Coordinates and name identical; only a fractional bounding edge moves.

    The engine computes the bounding box from the float contour before rounding
    the coordinates, so an outline whose lowest point was at y=384.5 records a
    bottom of 386 while the stored integers only reach 385. Re-encoding brings
    the box in by one. Nothing that indexes a pixel changes, and a second pass
    changes nothing at all.
    """
    with zipfile.ZipFile(engine_roiset) as archive:
        for name in archive.namelist():
            original = archive.read(name)
            polygon = roi.decode_roi(original)
            again = roi.encode_roi(polygon.x, polygon.y, name=polygon.name)

            assert len(again) == len(original)
            differing = [i for i in range(len(original))
                         if original[i] != again[i]]
            assert all(8 <= offset < 16 for offset in differing), (
                f"{name} differs outside the bounding box at {differing}")

            # coordinates and name survive exactly
            second = roi.decode_roi(again)
            np.testing.assert_array_equal(second.x, polygon.x)
            np.testing.assert_array_equal(second.y, polygon.y)
            assert second.name == polygon.name

            # and it is stable from here on
            assert roi.encode_roi(second.x, second.y, name=second.name) == again


def test_the_pipelines_rois_describe_the_masks_it_also_wrote(engine_roiset):
    """Polygon and mask are two views of one region; they must agree.

    ``scn_mask_combined.tif`` is the same region as ``SCN_both_lobes``, written
    by the same run in a different form. If the polygon reader were off by a
    row or a column, this is where it would show.
    """
    tifffile = pytest.importorskip("tifffile")
    mask_path = REFERENCE / "scn_mask_combined.tif"
    if not mask_path.is_file():
        pytest.skip("scn_mask_combined.tif is not here")

    mask = tifffile.imread(mask_path) > 0
    both = [p for p in roi.read_roi_zip(engine_roiset)
            if p.name == "SCN_both_lobes"][0]

    assert roi.dice(mask, both.to_mask(mask.shape)) > 0.97


# --------------------------------------------------- the ROI as a decision
def test_a_hand_drawn_roi_survives_a_parameter_change_and_a_version_bump(
        tmp_path, store_root, blob_mask):
    """Gate 6, and the reason a decision is not an artefact.

    Somebody drew this. A parameter change is not a reason to ask them again,
    and neither is a ``METHOD_VERSION`` bump — which is exactly what would
    happen if the ROI were keyed like a derived thing.
    """
    from tests_support import two_channel_stack

    from pymicroglia import segmentation, store

    source = two_channel_stack(tmp_path)
    drawn = roi.polygon_from_mask(blob_mask, name="both")
    roi.scn_roi(source, value={"both": drawn}, note="traced by hand")

    # a parameter change
    segmentation.segment(source, output_dir=tmp_path / "a",
                         channels="dluc=0,struct=1", k_mask=6.0,
                         stationarity_check=False)
    assert store.decision(roi.ROI_DECISION, source)["both"]["name"] == "both"

    # and a version bump
    original = segmentation.METHOD_VERSION
    try:
        segmentation.METHOD_VERSION = "9999-01-01-something-else"
        stored = store.decision(roi.ROI_DECISION, source)
    finally:
        segmentation.METHOD_VERSION = original

    assert stored is not None
    assert len(stored["both"]["x"]) == len(drawn)


def test_an_automatic_roi_is_recorded_and_then_never_re_derived(
        tmp_path, store_root):
    """``if_absent``: automate when nobody has answered, defer the moment one does.

    This is what keeps an unattended run moving without ever overwriting
    somebody's answer.
    """
    from tests_support import two_channel_stack

    source = two_channel_stack(tmp_path)
    yy, xx = np.mgrid[0:200, 0:200]
    structural = 800.0 * np.exp(
        -((yy - 100) ** 2 + (xx - 100) ** 2) / (2 * 40.0 ** 2))

    first = roi.scn_roi(source, structural=structural)
    assert first["source"] == "automatic"
    assert first["value"]["automated"] is True

    # second time it is read, not re-derived
    again = roi.scn_roi(source, structural=structural)
    assert again["source"] == "decision"
    assert again["value"]["waist"] == first["value"]["waist"]

    # and a person's answer wins from then on
    drawn = roi.Polygon(name="both", x=[10, 90, 90, 10], y=[10, 10, 90, 90])
    roi.scn_roi(source, value={"both": drawn})
    third = roi.scn_roi(source, structural=structural)
    assert third["source"] == "decision"
    assert third["value"].get("automated") is not True


def test_export_writes_a_roiset_from_stored_objects_and_the_stored_region(
        tmp_path, store_root):
    from tests_support import two_channel_stack

    from pymicroglia import segmentation

    source = two_channel_stack(tmp_path)
    found = segmentation.segment(source, output_dir=tmp_path / "out",
                                 channels="dluc=0,struct=1",
                                 stationarity_check=False)
    assert len(found) > 0

    drawn = roi.Polygon(name="both", x=[10, 90, 90, 10], y=[10, 10, 90, 90])
    roi.scn_roi(source, value={"both": drawn, "left": drawn, "right": drawn})

    result = roi.export_roi(source, output_dir=tmp_path / "out")
    assert result["ok"] is True
    assert result["objects"] == len(found)
    assert result["scn"] == 3

    names = [p.name for p in roi.read_roi_zip(result["output"])]
    assert names[:1] == ["object_1"]
    assert "scn_both" in names


def test_export_says_what_is_missing_rather_than_writing_an_empty_zip(
        tmp_path, store_root):
    from tests_support import two_channel_stack

    source = two_channel_stack(tmp_path)
    with pytest.raises(FileNotFoundError) as raised:
        roi.export_roi(source, output_dir=tmp_path / "out")
    assert "segment()" in str(raised.value)
