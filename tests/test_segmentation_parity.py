"""Two mistakes that cost real time, written as tests that fail if undone.

``dluc_pipeline.py``'s header lists five refusals. Two of them are
segmentation's, and a comment in a header does not stop the next person
reintroducing what it warns about. These do.

The synthetic cases are the load-bearing ones: each reproduces the *shape* of a
real failure — a small faint cell, and a background read off the wrong pixels —
on data small enough to reason about by hand. They do not need a real
acquisition and they run everywhere.

The comparison against the reference dataset's own label image is the third
test here and needs the registered, windowed, cosmic-cleaned array that
``dluc_pipeline.py`` builds before it segments. That chain is stage 11's; until
it exists this skips and says so, rather than comparing against something else
and calling it parity.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from pymicroglia import segmentation


@pytest.fixture
def store_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.delenv("PYMICROGLIA_DECISIONS", raising=False)
    return tmp_path / "cache"


def _profile_with(spots, shape=(120, 120), sigma=10.0, seed=5):
    """A background-subtracted profile image with Gaussian spots planted in it.

    ``spots`` is ``(y, x, peak_in_sigma, width_px)``. Built so the answer is
    known: each spot's peak is exactly ``peak_in_sigma`` times the background
    sigma, which is what every threshold in this module is expressed in.
    """
    rng = np.random.default_rng(seed)
    # A field whose robust sigma is very close to `sigma`, so "9.3 sigma" in a
    # test means 9.3 sigma to the code as well.
    image = rng.normal(0.0, sigma, size=shape).astype(np.float32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    for y, x, height, width in spots:
        image += (height * sigma * np.exp(
            -((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * width ** 2))
        ).astype(np.float32)
    return image


def _measured_sigma(image, mask=None):
    values = image if mask is None else image[mask]
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


# ------------------------------------- regression: the small faint cell
def test_a_small_faint_cell_is_detected():
    """FINDINGS sections 22 and 24, as a test.

    A 40 px minimum area silently discarded a real 9.3-sigma cell that had
    30 px above threshold. It failed on size, not on brightness. This plants a
    cell of exactly that description and asserts it comes back.

    If anybody reintroduces a minimum area, this is what stops them — and it
    fails loudly rather than quietly returning one cell fewer, which is how the
    original mistake went unnoticed.
    """
    sigma_target = 10.0
    # width 5.6 px puts about thirty pixels above 8 sigma at a 9.3 sigma
    # peak — the cell the 40 px floor threw away, to the pixel.
    profile = _profile_with([(60, 60, 9.3, 5.6)], sigma=sigma_target)
    sigma = _measured_sigma(profile)

    above = int((profile > segmentation.K_MASK * sigma).sum())
    assert 15 <= above <= 60, (
        f"the planted cell should have a few tens of pixels above threshold, "
        f"not {above}; the fixture no longer reproduces the failure")

    # at the mask threshold it is exactly the cell that was thrown away: too
    # small for a 40 px floor, and plenty bright
    _, strict, _ = segmentation.segment_cells(profile, sigma, relax_below=0)
    assert len(strict) == 1, f"the faint cell was lost: {strict}"
    assert strict[0]["area_px"] < 40, (
        "this test is only meaningful while the strict mask is smaller than "
        f"the 40 px floor that once discarded it; it is "
        f"{strict[0]['area_px']} px")
    assert strict[0]["peak_sigma"] > 8.0

    # and with the rescue on, it is not merely kept but regrown to a size a
    # trace can be extracted from — what the reference analysis did by hand
    labels, records, _ = segmentation.segment_cells(profile, sigma)
    assert len(records) == 1
    assert records[0]["area_px"] > strict[0]["area_px"]
    assert labels.max() == 1


def test_no_minimum_area_parameter_exists_anywhere():
    """Gate 7, checked on the signatures rather than by trying a value.

    The failure being guarded against is somebody *adding* the parameter, so
    the assertion is about what the functions accept, not about what happens
    when they are called.
    """
    forbidden = ("min_area", "minimum_area", "min_px", "min_cell_px",
                 "area_min", "min_size")
    for name in ("segment_cells", "somata_by_prominence", "segment"):
        function = getattr(segmentation, name)
        parameters = set(inspect.signature(function).parameters)
        offending = parameters & set(forbidden)
        assert not offending, f"{name} grew a minimum-area parameter: {offending}"

    # and the sweep's floor is on candidates, not cells
    sweep = set(inspect.signature(segmentation.sweep_candidates).parameters)
    assert "min_cand_px" in sweep
    assert not (set(inspect.signature(segmentation.segment_cells).parameters)
                & {"min_cand_px"})


def test_relax_below_rescues_rather_than_discards():
    """The 40 that is allowed to be 40.

    ``RELAX_BELOW`` is also 40 and is the opposite setting: a component below
    it is regrown at a lower threshold. Same number, opposite direction, and
    the coincidence is exactly the sort of thing a later tidy-up folds
    together. This asserts the direction.
    """
    profile = _profile_with([(60, 60, 9.3, 5.6)], sigma=10.0)
    sigma = _measured_sigma(profile)

    _, strict, _ = segmentation.segment_cells(profile, sigma, relax_below=0)
    _, relaxed, _ = segmentation.segment_cells(profile, sigma, relax_below=40)

    assert len(relaxed) == len(strict) == 1
    assert relaxed[0]["area_px"] > strict[0]["area_px"], (
        "relax_below must grow a small component, not drop it")


# ---------------------------------- regression: where the background comes from
def test_background_comes_from_off_tissue_and_not_from_the_corners():
    """FINDINGS: the corners inflated the noise estimate by about 28 %.

    The corners sit on the low-frequency instrumental gradient. Every mask
    downstream is drawn at a multiple of the sigma read off them, so a corner
    estimate shrinks all of them at once — quietly, because the masks still
    look reasonable.

    Both halves of gate 3 are asserted: that the two estimates genuinely differ
    on data shaped like the real thing, and that this module contains no path
    that reads the corners at all.
    """
    shape = (200, 200)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]

    # a slice in the middle, and an instrumental gradient rising towards the
    # corners — the arrangement the finding describes
    rng = np.random.default_rng(11)
    gradient = 40.0 * (((yy - 100) ** 2 + (xx - 100) ** 2) ** 0.5 / 140.0)
    structural = (600.0 * np.exp(-((yy - 100) ** 2 + (xx - 100) ** 2) / (2 * 45 ** 2))
                  + gradient + rng.normal(0, 5, shape))

    outside = segmentation.off_tissue(structural)
    assert outside is not None and outside.any()

    profile = (gradient + rng.normal(0, 10, shape)).astype(np.float32)
    off_tissue_sigma = segmentation.background_sigma(profile, outside)

    corner = np.zeros(shape, bool)
    corner[:40, :40] = corner[:40, -40:] = True
    corner[-40:, :40] = corner[-40:, -40:] = True
    corner_sigma = segmentation.background_sigma(profile, corner)

    assert corner_sigma != off_tissue_sigma
    assert corner_sigma > off_tissue_sigma, (
        "on data shaped like the real thing the corners should read noisier; "
        f"got corners {corner_sigma:.1f} against off-tissue "
        f"{off_tissue_sigma:.1f}")

    # and no code path that reads a corner. The word itself appears in this
    # module — in the comments saying not to — so the scan is for the shapes a
    # corner estimate actually takes.
    import re

    code = " ".join(line for line in inspect.getsource(segmentation).splitlines()
                    if not line.strip().startswith("#"))
    for pattern in (r"\[\s*:\s*\d+\s*,\s*:\s*\d+\s*\]",   # image[:40, :40]
                    r"\[\s*-\d+\s*:\s*,",                 # image[-40:, ...
                    r"corner\w*\s*="):                    # corner_mask = ...
        assert not re.search(pattern, code), (
            f"segmentation.py has something shaped like a corner estimate: "
            f"{re.search(pattern, code).group(0)!r}")


def test_a_higher_sigma_shrinks_every_mask():
    """Why the background estimate matters at all.

    The mechanism behind the finding, stated directly: masks are drawn at a
    multiple of sigma, so reading sigma off the wrong pixels moves every mask
    at once.
    """
    profile = _profile_with([(40, 40, 20.0, 3.0), (90, 90, 12.0, 3.0)],
                            sigma=10.0)
    sigma = _measured_sigma(profile)

    # relax_below=0, so the rescue does not compensate: the point here is the
    # threshold moving, not what the code then does about it.
    _, honest, _ = segmentation.segment_cells(profile, sigma, relax_below=0)
    _, inflated, _ = segmentation.segment_cells(profile, sigma * 1.28,
                                                relax_below=0)

    honest_area = sum(row["area_px"] for row in honest)
    inflated_area = sum(row["area_px"] for row in inflated)
    assert inflated_area < honest_area


# --------------------------------------------------- what the store records
def test_labels_and_the_off_tissue_mask_are_stored_with_their_lineage(
        tmp_path, store_root):
    """Gate 4: both artefacts, both carrying the cosmic-ray digest upstream.

    Empty upstream here because nothing cleaned this synthetic file, which is
    honest — the assertion is that the field is carried into the key, so a
    re-clean produces a different key rather than silently reusing labels
    derived from other pixels.
    """
    from tests_support import two_channel_stack           # noqa: F401

    source = two_channel_stack(tmp_path)
    found = segmentation.segment(source, output_dir=tmp_path / "out",
                                 channels="dluc=0,struct=1",
                                 stationarity_check=False)

    assert "labels" in found.artefacts
    labels_record = found.artefacts["labels"].record
    assert labels_record["stage"] == segmentation.SEGMENTATION_STAGE
    assert "upstream" in labels_record
    assert labels_record["extra"]["no_minimum_cell_area"] is True
    assert (labels_record["extra"]["background_sigma_counts"]
            == pytest.approx(found.reference.sigma))

    from pymicroglia import store

    # keyed on its own, smaller parameter set: the background does not depend
    # on any mask threshold, so changing one must not make it miss.
    stored_mask = store.resolve(segmentation.BACKGROUND_STAGE, source,
                                required=False)
    assert stored_mask is not None
    assert stored_mask.record["extra"]["source_of_background"].startswith(
        "structural channel")


def test_a_second_identical_run_reads_the_stored_labels(tmp_path, store_root):
    from tests_support import two_channel_stack

    source = two_channel_stack(tmp_path)
    options = {"output_dir": tmp_path / "out", "channels": "dluc=0,struct=1",
               "stationarity_check": False}
    first = segmentation.segment(source, **options)
    again = segmentation.segment(source, **options)

    assert again.artefacts.get("cached") is True
    np.testing.assert_array_equal(again.labels, first.labels)


def test_a_changed_threshold_misses(tmp_path, store_root):
    from tests_support import two_channel_stack

    source = two_channel_stack(tmp_path)
    options = {"output_dir": tmp_path / "out", "channels": "dluc=0,struct=1",
               "stationarity_check": False}
    segmentation.segment(source, **options)
    other = segmentation.segment(source, k_mask=6.0, **options)

    assert other.artefacts.get("cached") is not True


# ------------------------------------- against the reference dataset's labels
REFERENCE = Path("reference-data") / "dluc-pipeline"

#: What ``pipelines.registered.parity_bundle`` writes, and what this needs from
#: it. The bioluminescence array alone is not enough: masks are placed on the
#: background-subtracted profile at a multiple of a sigma read off the
#: **structural** channel's off-tissue mask, so a test that re-derived either
#: from the bioluminescence would compare something else and agree with itself.
BUNDLE = ("dluc_registered_windowed_clean.npy", "off_tissue.npy",
          "tissue.npy", "profile.npy")

HOW_TO_BUILD = (
    "Set PYMICROGLIA_DLUC_REGISTERED to the .npy in a bundle written by\n"
    "    from pymicroglia.pipelines.registered import parity_bundle\n"
    "    parity_bundle(r'<...>/MCG_04 - 1 - 595.tif', r'<folder>')\n"
    "which takes about two minutes on the reference file and is cached after.")


def reference_bundle():
    """The bundle folder named by the environment, or ``None`` with a reason."""
    override = os.environ.get("PYMICROGLIA_DLUC_REGISTERED")
    if not override:
        return None, ("this gate needs the registered, windowed, "
                      "cosmic-cleaned dLuc array the pipeline segments. "
                      + HOW_TO_BUILD)
    path = Path(override)
    folder = path if path.is_dir() else path.parent
    missing = [name for name in BUNDLE if not (folder / name).is_file()]
    if missing:
        return None, (f"the bundle at {folder} is missing {missing}. "
                      + HOW_TO_BUILD)
    if not REFERENCE.is_dir():
        return None, f"the reference run is not here: {REFERENCE.name}"
    return folder, ""


def test_the_reference_labels_match(store_root):
    """Gate 1, and the one thing stage 07 could not close on its own.

    ``dluc_pipeline.py`` does not segment ``MCG_04 - 1 - 595.tif``. It segments
    a 250-frame window of it, registered on brightfield with whole-pixel
    shifts, cropped by 7 px and cosmic-cleaned — and that chain is stage 11's
    pipeline. Segmenting the raw file instead and calling the result parity
    would be comparing two different things, so this asks for the array the
    engine actually segmented.

    The off-tissue sigma is asserted first and on purpose. It is one number
    that summarises the whole chain above it — registration, the window, the
    cosmic-ray removal and the background — so a label image that matched by
    way of a different background would fail here rather than pass quietly.
    """
    import json

    import tifffile

    folder, why = reference_bundle()
    if folder is None:
        pytest.skip(why)

    from pymicroglia import controls
    from pymicroglia.pipelines import StackView

    expected_labels = tifffile.imread(REFERENCE / "cell_masks_labels.tif")
    summary = json.loads((REFERENCE / "summary.json").read_text(encoding="utf-8"))

    stack = np.asarray(np.load(folder / BUNDLE[0], mmap_mode="r"))
    outside = np.load(folder / "off_tissue.npy")
    tissue = np.load(folder / "tissue.npy")
    profile = np.load(folder / "profile.npy")
    frames, height, width = stack.shape

    assert int(outside.sum()) == summary["off_tissue_px"]
    sigma = segmentation.background_sigma(profile, outside)
    assert sigma == pytest.approx(summary["off_tissue_sigma"], rel=1e-9)

    # the whole object chain, in the engine's order
    labels, cells, _ = segmentation.segment_cells(profile, sigma)
    objects = [{**row, "_mask": labels == row["label"]} for row in cells]
    candidates, _ = segmentation.sweep_candidates(profile, sigma, labels > 0,
                                                  tissue)
    objects += [dict(row) for row in candidates]
    objects.sort(key=lambda row: (0 if row["kind"] == "cell" else 1,
                                  -row["peak_sigma"]))
    for index, row in enumerate(objects, 1):
        row["label"] = index
        row["soma"] = (int(row["soma_y"]), int(row["soma_x"]))

    view = StackView([stack])
    for row in objects:
        motion = segmentation.stationarity(view, 0, row["soma"])
        row["still"] = bool(max(motion["y_range"], motion["x_range"])
                            <= segmentation.MOVE_MAX)

    every = np.zeros((height, width), bool)
    for row in objects:
        every |= row["_mask"]
    occupied = ndimage.binary_dilation(every, np.ones((3, 3), bool),
                                       iterations=3)
    numbered = np.zeros((height, width), np.int32)
    for row in objects:
        numbered[row["_mask"]] = row["label"]

    times = np.asarray(json.loads(
        (folder / "reference.json").read_text(encoding="utf-8"))["times_h"],
        float)
    verdicts = {row["label"]: row for row in controls.decoy_test(
        None, 0, numbered, tissue, times, baseline_h=max(summary["baselines"]),
        count=summary["params"]["NDECOY"], alpha=summary["params"]["DECOY_P"],
        objects=objects, rng=np.random.default_rng(163), occupied=occupied,
        planes=[stack[index] for index in range(frames)]).records}

    engine_rows = {int(row["label"]): row
                   for row in summary["objects"] + summary["rejected"]}
    for row in objects:
        mine = verdicts[row["label"]]
        theirs = engine_rows[row["label"]]
        assert row["area_px"] == theirs["area"], f"object {row['label']} area"
        assert mine["p_counts"] == pytest.approx(theirs["p_counts"], rel=1e-9), \
            f"object {row['label']} decoy p"
        assert mine["amp_counts"] == pytest.approx(theirs["amp_counts"],
                                                   rel=1e-9)

    final = np.zeros((height, width), np.uint16)
    for row in objects:
        if verdicts[row["label"]]["admissible"] and row["still"]:
            final[row["_mask"]] = row["label"]

    assert int(final.max()) == int(expected_labels.max())
    np.testing.assert_array_equal(final, expected_labels)
