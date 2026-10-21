"""Focused browser checks for this standalone synthetic prototype.

Connects to the dedicated browser-use automation profile, never a personal
Chrome profile. Open the demo with browser-use-auto before running this.
"""
from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parent


def verify() -> None:
    checks: list[str] = []
    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://127.0.0.1:9333")
        context = browser.contexts[0]
        page = next((p for p in context.pages if "/workbench-layout/demo/index.html" in p.url), None)
        if page is None:
            page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.goto(ROOT.joinpath("index.html").as_uri())
        page.wait_for_function("document.documentElement.dataset.demoReady === 'true'")

        def state() -> dict:
            return page.evaluate("window.demo.getState()")

        def linked(cell: int, frame: int) -> None:
            page.wait_for_function(
                "([cell,frame]) => Array.from(document.querySelectorAll('#workspace-extras canvas')).every(c => Number(c.dataset.cell) === cell && Number(c.dataset.frame) === frame)",
                arg=[cell, frame],
            )
            assert state()["cell"] == cell and state()["frame"] == frame

        def choose(identifier: str, value: str) -> None:
            control = page.locator(f"#{identifier}")
            disclosure = control.locator("xpath=ancestor::details[1]")
            if disclosure.count() and disclosure.get_attribute("open") is None:
                disclosure.locator("summary").click()
            control.select_option(value)
            # The browser paints after requestAnimationFrame; wait for that
            # observable selection and a subsequent frame before comparing it.
            page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")

        def frame(number: int) -> None:
            page.locator("#frame-slider").evaluate(
                "(e,n) => {e.value=n; e.dispatchEvent(new Event('input',{bubbles:true}));}", number,
            )
            linked(state()["cell"], number)

        def image(identifier: str) -> str:
            return page.locator(f"#{identifier}").evaluate("e => e.toDataURL()")

        plots = page.evaluate("window.demo.getDiagnostics().plots")
        assert len(plots) == 9 == len(set(plots)), plots
        assert page.locator("#workspace-extras").is_visible()
        assert page.locator("#show-trails").is_checked() is False
        assert page.evaluate("window.demo.getDiagnostics().immutableObservations")
        linked(19, 72)
        checks.append("Nine unique plot families; initial views autofill; trails off")

        source_before = page.evaluate("JSON.stringify(window.demo.getObservation(19,72))")
        trace_before, barcode_before = image("trace"), image("barcode")
        choose("barcode-metric", "area")
        assert image("barcode") != barcode_before
        assert image("trace") == trace_before
        assert page.locator("#trace-metric").input_value() == "light"
        choose("barcode-scale", "recording")
        choose("barcode-order", "phase")
        assert page.locator("[data-plot-type='matrix']").count() == 1
        choose("barcode-order", "identity")
        checks.append("Matrix configuration changes in place without changing the trace")

        barcode = page.locator("#barcode")
        box = barcode.bounding_box()
        assert box is not None
        # Row seven, frame 82: a declared synthetic observation gap.
        barcode.click(position={
            "x": (47 + (81.5 / 240) * (1280 - 47 - 20)) / 1280 * box["width"],
            "y": (10 + (6.5 / 24) * (370 - 10 - 29)) / 370 * box["height"],
        })
        linked(7, 81)
        assert page.evaluate("window.demo.getDiagnostics().selectionMissing")
        assert "missing" in page.locator("#selection-status").inner_text().lower()
        assert page.evaluate("window.demo.getValue(7,81,'light','total',3)") is None
        choose("distribution-time", "frame")
        assert not any(hit["id"] == 7 for hit in page.evaluate("window.demo.getDiagnostics().hitTargets.distribution"))
        checks.append("Barcode selects cell and time; gaps remain missing in every view")

        trace = page.locator("#trace")
        box = trace.bounding_box()
        assert box is not None
        trace.click(position={"x": (66 + 145 / 239 * (1280 - 66 - 22)) / 1280 * box["width"], "y": box["height"] / 2})
        linked(7, 145)
        assert page.evaluate("window.demo.getDiagnostics().selectionMissing") is False
        checks.append("Trace clicks move the same shared time selection")

        choose("cell-select", "19")
        frame(72)
        sculpture = page.locator("#outline-history")
        sculpture_before = image("outline-history")
        sculpture.focus()
        page.keyboard.press("ArrowRight")
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        assert image("outline-history") != sculpture_before
        assert state()["frame"] == 72
        page.locator("#reset-rotation").click()
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        assert image("outline-history") == sculpture_before
        choose("map-layer", "light")
        choose("network-scope", "all")
        choose("distribution-metric", "speed")
        choose("distribution-time", "recording")
        checks.append("Outline rotation, reset and local map/network/comparison controls work")

        # Select another object through a complementary view, not the selector.
        hit = next(h for h in page.evaluate("window.demo.getDiagnostics().hitTargets.map") if h["id"] == 4)
        canvas = page.locator("#tissue-map")
        box = canvas.bounding_box()
        assert box is not None
        canvas.click(position={"x": hit["x"] / 660 * box["width"], "y": hit["y"] / 370 * box["height"]})
        linked(4, 72)
        assert page.locator("#cell-select").input_value() == "4"
        checks.append("Tissue map selection updates image, trace, matrix and shape history")

        for metric in ("light", "area", "speed", "turnover"):
            choose("trace-metric", metric)
            choose("barcode-metric", metric)
            choose("distribution-metric", metric)
        choose("trace-metric", "light")
        choose("trace-aggregation", "mean")
        choose("trace-smoothing", "7")
        assert page.evaluate("window.demo.getDiagnostics().settings.trace.aggregation") == "mean"
        total = page.evaluate("window.demo.getValue(19,72,'light','total')")
        mean = page.evaluate("window.demo.getValue(19,72,'light','mean')")
        area = page.evaluate("window.demo.getValue(19,72,'area')")
        assert abs(total - mean * area) < 1e-8
        assert source_before == page.evaluate("JSON.stringify(window.demo.getObservation(19,72))")
        checks.append("All metric selectors work; display settings preserve original observations")

        page.locator("#tab-motion").click()
        assert page.locator("#workspace-motion").is_visible()
        assert page.locator("#sidebar-motion").is_visible()
        assert page.locator("#workspace-extras").is_visible() is False
        page.get_by_role("button", name="Cell 8", exact=True).click()
        assert state()["cell"] == 8
        page.locator("#tab-motion").focus()
        page.keyboard.press("ArrowLeft")
        assert page.locator("#tab-auto").get_attribute("aria-selected") == "true"
        assert page.locator("#workspace-auto").is_visible()
        page.keyboard.press("End")
        linked(8, 72)
        assert page.locator("#trace-aggregation").input_value() == "mean"
        assert page.locator("#trace-smoothing").input_value() == "7"
        page.locator("#open-pipelines").click()
        assert page.locator("#pipeline-dialog").is_visible()
        assert page.locator("#pipeline-dialog button:disabled").count() == 3
        page.keyboard.press("Escape")
        checks.append("Full workspace tabs, their sidebars, keyboard navigation and explicit pipeline boundary")

        frame(20)
        page.locator("#play").click()
        page.wait_for_function("window.demo.getState().frame > 20")
        page.locator("#play").click()
        assert state()["playing"] is False
        checks.append("Playback advances shared time and pauses")

        # Return to the demonstrated starting state for visual inspection.
        choose("cell-select", "19")
        frame(72)
        for identifier, option in {
            "trace-metric": "light", "trace-aggregation": "total", "trace-smoothing": "3",
            "barcode-metric": "light", "barcode-aggregation": "total", "barcode-smoothing": "3", "barcode-scale": "cell",
            "distribution-metric": "area", "map-layer": "phase", "network-scope": "neighbours",
        }.items():
            choose(identifier, option)
        page.locator("details[open]").evaluate_all("items => items.forEach(e => e.open=false)")
        page.locator("#main").evaluate("e => e.scrollTop=0")
        page.screenshot(path=str(ROOT / "desktop-check.png"))
        for width, height in [(1440, 1000), (1072, 768), (600, 900)]:
            page.set_viewport_size({"width": width, "height": height})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1"), width
            assert page.locator("#frame-slider").is_visible()
            if width == 600:
                page.screenshot(path=str(ROOT / "narrow-check.png"))
        checks.append("No horizontal page overflow at desktop, small desktop or narrow width")
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.locator("#main").evaluate("e => e.scrollTop=0")
        assert not errors, errors
        report = {"status": "passed", "checks": checks, "javascript_errors": errors,
                  "dataset": {"kind": "synthetic", "cells": 24, "frames": 240, "frame_interval_hours": 0.5},
                  "backend": "none; standalone UI prototype"}
        ROOT.joinpath("verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    verify()
