"""End-to-end linked selection checks in an isolated local browser context."""
from pathlib import Path
import urllib.request

import pytest

from test_cell_review_data import recording  # shared translated/gapped mask fixture
from pymicroglia.visualisation.cell_review import cell_review


@pytest.fixture(scope="module")
def browser_context():
    playwright = pytest.importorskip("playwright.sync_api")
    endpoint = "http://127.0.0.1:9333"
    try:
        urllib.request.urlopen(endpoint + "/json/version", timeout=1).close()
    except OSError:
        pytest.skip("Start the dedicated browser-use-auto profile for local browser verification")
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.connect_over_cdp(endpoint)
        context = browser.new_context(viewport={"width": 1440, "height": 1040})
        yield context
        context.close()  # only the test context; keep the user's other tabs open


@pytest.fixture
def review_page(browser_context, recording, tmp_path):
    output = tmp_path / "review.html"
    cell_review(recording[0], recording[1], output=output, source_frame_offset=1,
                frame_interval_h=.25, title="Synthetic saved-cell review")
    page = browser_context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(output.as_uri())
    page.wait_for_function("window.CellReview?.ready === true")
    yield page
    assert not errors
    page.close()


def test_translation_and_gained_pixel(review_page):
    page = review_page
    page.evaluate("async()=>{await CellReview.select({cellId:19,frame:2,changeMode:'centered'});}")
    assert page.evaluate("[CellReview.changes.last.gained.length,CellReview.changes.last.lost.length]") == [0, 0]
    page.select_option("#change-mode", "original")
    page.wait_for_function("CellReview.changes.last?.gained.length === 2")
    assert page.evaluate("CellReview.changes.last.lost.length") == 2
    page.evaluate("async()=>{await CellReview.select({frame:3});}")
    assert page.evaluate("[CellReview.changes.last.gained.length,CellReview.changes.last.lost.length]") == [1, 0]
    assert page.evaluate("CellReview.state.cellId") == 19


def test_no_comparison_across_missing_masks(review_page):
    page = review_page
    for frame in (0, 4, 5):
        page.evaluate("async f=>{await CellReview.select({cellId:19,frame:f});}", frame)
        assert page.evaluate("CellReview.changes.last") is None
    assert "preceding frame" in page.locator("#change-status").inner_text()
    assert page.evaluate("[...document.querySelectorAll('canvas.movie')].every(c=>c.getContext('2d').imageSmoothingEnabled === false)")


def test_controls_keep_native_time(review_page):
    page = review_page
    page.select_option("#cell-select", "26")
    page.wait_for_function("CellReview.renderedCell === 26")
    page.locator("#frame-slider").fill("1")
    page.wait_for_function("CellReview.renderedFrame === 1")
    assert "Frame 2 / 6" in page.locator("#time-label").inner_text()
    assert "0.25 h" in page.locator("#time-label").inner_text()
    page.click("#step-forward")
    page.wait_for_function("CellReview.renderedFrame === 2")
    assert page.evaluate("CellReview.state.cellId") == 26


def test_time_sculpture_observations_and_rotation(review_page):
    page = review_page
    page.evaluate("async()=>{await CellReview.select({cellId:19,frame:3});}")
    assert page.evaluate("CellReview.sculpture.observedFrames") == [1, 2, 3, 5]
    assert page.evaluate("CellReview.sculpture.geometry.slices.map(s=>s.time)") == [.25, .5, .75, 1.25]
    before = page.evaluate("CellReview.sculpture.yaw")
    box = page.locator("#sculpture").bounding_box()
    page.mouse.move(box["x"] + 100, box["y"] + 100)
    page.mouse.down()
    page.mouse.move(box["x"] + 180, box["y"] + 125, steps=4)
    page.mouse.up()
    assert page.evaluate("CellReview.sculpture.yaw") != before
    page.click("#sculpture-reset")
    assert page.evaluate("CellReview.sculpture.yaw") == before
    assert page.evaluate("[CellReview.state.cellId,CellReview.state.frame]") == [19, 3]
    page.evaluate("async()=>{await CellReview.select({frame:4});}")
    assert "no outline at the selected time" in page.locator("#sculpture-status").inner_text()


def test_card_trace_click_and_cursor_link(review_page):
    page = review_page
    page.evaluate("async()=>{await CellReview.select({cellId:19,frame:1});}")
    signal = page.locator(".cell-card[data-cell='19'] svg.trace").first
    signal.scroll_into_view_if_needed()
    box = signal.bounding_box()
    # Frame 3 of the six-frame fixture: x is 42 + 3/5*(350-42-8).
    page.mouse.click(box["x"] + 222 / 350 * box["width"], box["y"] + box["height"] / 2)
    page.wait_for_function("CellReview.renderedFrame === 3")
    assert page.evaluate("[...document.querySelectorAll('.cursor')].every(c=>c.dataset.frame==='3')")
    assert page.evaluate("CellReview.sculpture.observedFrames") == [1, 2, 3, 5]
    page.click("#pin-cell")
    page.select_option("#cell-select", "26")
    page.wait_for_function("CellReview.renderedCell === 26")
    assert page.locator(".cell-card").count() == 2
    assert page.locator(".cell-card").first.get_attribute("data-cell") == "26"
    page.evaluate("async()=>{await CellReview.select({cellId:19,frame:4});}")
    assert "No saved mask" in page.locator(".cell-card[data-cell='19'] .card-missing").inner_text()
    assert page.evaluate("[...document.querySelectorAll('canvas.movie')].every(c=>c.getContext('2d').imageSmoothingEnabled === false)")


def test_barcode_click_selects_existing_id_and_native_frame(review_page):
    page = review_page
    geometry = page.evaluate("CellReview.barcode.geometry")
    canvas = page.locator("#barcode")
    canvas.scroll_into_view_if_needed()
    box = canvas.bounding_box()
    x = geometry["left"] + 1.5 * geometry["columnWidth"]
    y = geometry["top"] + 1.5 * geometry["rowHeight"]
    page.mouse.click(box["x"] + x / geometry["width"] * box["width"],
                     box["y"] + y / geometry["height"] * box["height"])
    page.wait_for_function("CellReview.renderedCell === 26 && CellReview.renderedFrame === 1")
    assert page.evaluate("CellReview.state.cellId") == 26
    assert page.locator("#barcode").get_attribute("data-current-frame") == "1"
    assert page.locator(".cell-card").first.get_attribute("data-cell") == "26"
    assert page.evaluate("[...document.querySelectorAll('.cursor')].every(c=>c.dataset.frame==='1')")
    assert "Cell 26" in page.locator("#change-status").inner_text() or "preceding frame" in page.locator("#change-status").inner_text()
    ids = page.evaluate("CellReview.data.ids")
    page.select_option("#barcode-metric", "area")
    page.select_option("#barcode-scale", "recording")
    page.wait_for_function("CellReview.barcode.metric === 'area' && CellReview.barcode.scale === 'recording'")
    assert page.evaluate("CellReview.data.ids") == ids
    assert page.evaluate("CellReview.barcode.finiteCount") == 5
    assert page.evaluate("CellReview.data.cells['26'].area[2]") is None
    page.evaluate("async()=>{await CellReview.select({frame:2});}")
    assert "no saved mask" in page.locator("#barcode-tooltip").inner_text()


def test_first_view_uses_an_observed_identity_without_renaming(browser_context, tmp_path):
    import numpy as np
    import tifffile

    raw = np.ones((5, 12, 12), np.uint16)
    labels = np.zeros_like(raw)
    labels[:3, 3:6, 3:6] = 19
    labels[4, 3:6, 3:6] = 1  # smallest ID appears much later
    rp, lp = tmp_path / "raw.tif", tmp_path / "labels.tif"
    tifffile.imwrite(rp, raw, photometric="minisblack")
    tifffile.imwrite(lp, labels, photometric="minisblack")
    output = tmp_path / "late_low_id.html"
    cell_review(rp, lp, output=output)
    page = browser_context.new_page()
    try:
        page.goto(output.as_uri())
        page.wait_for_function("CellReview.ready")
        assert page.evaluate("CellReview.data.ids") == [1, 19]
        assert page.evaluate("[CellReview.state.cellId,CellReview.state.frame]") == [19, 1]
        assert page.evaluate("CellReview.changes.last.retained") == 9
        page.select_option("#cell-select", "1")
        page.wait_for_function("CellReview.renderedCell===1")
        assert page.evaluate("CellReview.observed()") is None
    finally:
        page.close()
